"""挑戰的選擇權價值 C 與最佳門檻 p*：序列建構、逆向動態規劃、近似政策迭代、前向傳遞。

對應研究步驟：06（主結果、敏感度、bootstrap），並被 07、10、11 共用。

每支球隊每場比賽 = 一條「對自己不利的判決球」序列（打擊時：原判好球，守備時：原判壞球），依時間排序，
另在每個延長局開始插入「補挑戰」（k = 0 → 1）。
對序列做逆向遞迴（對感知誤差 ε 取期望，不必模擬）：
  在節點 j、手上 k 次：
    球員看到 m̃ = m + σ_e ε，主觀機率 p̃(m̃)，當 p̃ > p*_j(k) = C_j(k) / (G_j + C_j(k)) 時挑戰
    → 挑戰機率 q_j(k) = λ · [1 − Φ((m̃*_j(k) − m_j) / σ_e)]，m̃* = p̃⁻¹(p*)
    F_j(k) = q·[1(m>0)·(G_j + F_{j+1}(k)) + 1(m≤0)·F_{j+1}(k−1)] + (1 − q)·F_{j+1}(k)
  C_j(k) 不能用該場實際的未來（那是預知），而是用「同狀態格」的條件期望：
    Ĉ(格, k) = 平均[F_{j+1}(k) − F_{j+1}(k−1)]，格 = 局(1..9, 10+) × 打擊/守備 × 出局 × 比分差(−5..5) × 主/客
    樣本少的格往上層（去掉出局 → 去掉主客與打守）收縮（κ = 50）
  重複「給定 Ĉ 評估 F → 由 F 更新 Ĉ」直到收斂。
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import norm

from src.config import (
    C_INIT_WP,
    CALLS_G,
    D_CLIP,
    EDGES,
    ITERS_DEFAULT,
    KAPPA,
    OPPS,
    SAVANT_COST_RUNS,
)
from src.perception import SideParams, count_dummies, inning_group, log_g, posterior_table
from src.zone import edge_index, misjudgment, nearest_edge

SIDES: tuple[tuple[str, bool], ...] = (("off", True), ("def", False))


def build_sequences(calls: pd.DataFrame | None = None) -> pd.DataFrame:
    """由 calls_2026_G 建立每隊每場的「不利判決球」序列（只用合併品質良好的球）。

    新增欄位：team_home（對誰不利的那隊是否主隊）、team_bat、side（off／def）、
    d_team（該隊觀點比分差）、m（誤判幅度）、edge（最近的邊）、seq（場內排序鍵）、
    replenish（延長局第一個節點）、actual_gain（實際成功挑戰的 G）、actual_fail。
    """
    df = pd.read_parquet(CALLS_G) if calls is None else calls
    df = df[df.merge_ok].copy()
    home_bat = df.half.eq("bottom")
    # 對誰不利：原判好球 → 打擊方，原判壞球 → 守備方
    off = df.orig_strike.values
    df["team_home"] = np.where(off, home_bat, ~home_bat)
    df["team_bat"] = off
    df["side"] = np.where(off, "off", "def")
    hd = np.where(home_bat, df.bat_score - df.fld_score, df.fld_score - df.bat_score)
    df["d_team"] = np.where(df.team_home, hd, -hd)
    df["m"] = misjudgment(off, df.signed_dist_in)
    df["edge"] = nearest_edge(df.plate_x, df.plate_z, df.sz_top, df.sz_bot)
    # 場內順序：打席編號 × 1000 + 逐球編號（一個打席不會超過 1000 球）
    df["seq"] = df.at_bat_number * 1000 + df.pitch_number
    df = df.sort_values(["game_pk", "team_home", "seq"])
    # 延長局補挑戰節點：每場、每隊，在第 10 局以後每局第一個節點前標記
    df["inn_key"] = df.inning.clip(upper=10)
    first_in_inning = df.groupby(["game_pk", "team_home", "inning"]).cumcount() == 0
    df["replenish"] = first_in_inning & (df.inning >= 10)
    ov = df.overturned.astype("boolean").fillna(False).astype(bool)
    df["actual_gain"] = np.where(df.challenged & ov, df.G, 0.0)
    df["actual_fail"] = df.challenged & ~ov
    return df.reset_index(drop=True)


def cell_keys(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """回傳三層狀態格代碼 (完整格, 父格, 祖父格)，供 Ĉ 的階層收縮使用。

    完整格 = 局 × 打/守 × 出局 × 比分差 × 主/客，父格去掉出局，祖父格只剩局 × 比分差。
    """
    d = df.d_team.clip(-D_CLIP, D_CLIP).astype(int)
    inn = df.inning.clip(upper=10).astype(int)
    nd = 2 * D_CLIP + 1
    k3 = (inn * 2 + df.team_bat.astype(int)) * 3 + df.outs_when_up.astype(int)
    full = ((k3 * nd + (d + D_CLIP)) * 2 + df.team_home.astype(int)).values
    par = ((inn * 2 + df.team_bat.astype(int)) * nd + (d + D_CLIP)).values * 2 + df.team_home.astype(int).values
    gpar = (inn * nd + (d + D_CLIP)).values
    return full, par, gpar


def load_prior() -> dict[tuple[str, str], np.ndarray]:
    """讀 05 的機會樣本，回傳各 (打/守, 邊) 的 m 分布（σ 改變時重算主觀機率表用）。"""
    o = pd.read_parquet(OPPS, columns=["sideflag", "edge", "m"])
    return {(s, e): o[(o.sideflag == s) & (o.edge == e)].m.values for s in ("off", "def") for e in EDGES}


class Seq:
    """把序列排成 [球隊-場次, 位置] 的矩陣（左對齊＋有效遮罩 valid），供逆向遞迴使用。"""

    def __init__(
        self,
        df: pd.DataFrame,
        perc: dict,
        sigma_scale: float = 1.0,
        prior: dict[tuple[str, str], np.ndarray] | None = None,
    ) -> None:
        g = df.groupby(["game_pk", "team_home"], sort=False)
        self.tg = g.ngroup().values
        self.pos = g.cumcount().values
        self.ntg, self.L = self.tg.max() + 1, self.pos.max() + 1
        self.n = len(df)
        self.G = self._mat(df.G.values)
        self.m = self._mat(df.m.values)
        self.valid = self._mat(np.ones(self.n), 0.0).astype(bool)
        self.rep = self._mat(df.replenish.values.astype(float)).astype(bool)
        self.dRE = self._mat(df.dRE.values)
        self.side_off = self._mat(df.side.eq("off").values.astype(float)).astype(bool)
        self.edge = self._mat(edge_index(df.edge).astype(float)).astype(int)
        full, par, gpar = cell_keys(df)
        self.cell = self._mat(full, -1).astype(int)
        self.cell_flat, self.par_flat, self.gpar_flat = full, par, gpar
        # 感知：每個節點的 σ 與主觀機率表 p̃（σ 放大縮小時，用先驗重算對照表）
        self.sig = np.zeros_like(self.m)
        self.post: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]] = {}
        for s, flag in SIDES:
            for ei, e in enumerate(EDGES):
                msk = (self.side_off == flag) & (self.edge == ei)
                self.sig[msk] = perc[s]["sigma"][e] * sigma_scale
                grid, post = perc[s]["post"][e]
                if sigma_scale != 1.0:
                    if prior is None:
                        raise ValueError("sigma_scale ≠ 1 時必須提供 prior（load_prior()）")
                    grid, post = posterior_table(prior[(s, e)], perc[s]["sigma"][e] * sigma_scale)
                # p̃ 應隨 m̃ 遞增，先驗直方圖的抽樣雜訊可能造成小幅不單調 → 取累積最大值
                self.post[(s, e)] = (grid, np.maximum.accumulate(post))
        self.lam = {s: perc[s]["lambda"] for s in ("off", "def")}

    def _mat(self, v: np.ndarray, fill: float = 0.0) -> np.ndarray:
        """把節點層級的一維陣列攤成 [球隊-場次, 位置] 矩陣。"""
        M = np.full((self.ntg, self.L), fill, dtype=float)
        M[self.tg, self.pos] = v
        return M

    def cutoff(self, pstar: np.ndarray) -> np.ndarray:
        """p* → 感知幅度門檻 m̃*（依打擊/守備 × 邊，查主觀機率表的反函數）。"""
        out = np.zeros_like(pstar)
        for s, flag in SIDES:
            for ei, e in enumerate(EDGES):
                msk = (self.side_off == flag) & (self.edge == ei)
                grid, post = self.post[(s, e)]
                # 嚴格遞增化（加極小斜率）後做反函數插值，p* 超出表的範圍 → ±∞（永遠挑／永遠不挑）
                ps = np.clip(pstar[msk], 0, 1)
                c = np.interp(ps, post + np.arange(len(post)) * 1e-12, grid)
                c = np.where(ps >= post[-1], np.inf, np.where(ps <= post[0], -np.inf, c))
                out[msk] = c
        return out

    def qprob(self, pstar: np.ndarray, lam_on: bool) -> np.ndarray:
        """給定門檻 p*，每個節點的挑戰機率 q = P(m̃ > m̃*)（lam_on 時再乘上注意力 λ）。"""
        c = self.cutoff(pstar)
        with np.errstate(invalid="ignore"):
            z = (c - self.m) / np.where(self.sig > 0, self.sig, 1e-9)
        q = 1 - norm.cdf(z)
        q = np.where(np.isinf(c), np.where(c > 0, 0.0, 1.0), q)
        if lam_on:
            lam = np.where(self.side_off, self.lam["off"], self.lam["def"])
            q = q * lam
        return np.where(self.valid, q, 0.0)

    def backward(self, q1: np.ndarray, q2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """逆向遞迴。

        給定每節點在 k=1、k=2 的挑戰機率，回傳 F[k]（k = 0,1,2，節點 j 之前、手上 k 次時的期望收益）
        與續值 Fn[k]（節點 j 之後）。
        """
        ntg, L = self.ntg, self.L
        F = np.zeros((3, ntg, L + 1))
        Fn = np.zeros((3, ntg, L))
        win = self.m > 0  # 真的判錯 → 挑戰成功，不扣次數
        for j in range(L - 1, -1, -1):
            nxt = F[:, :, j + 1]
            Fn[:, :, j] = nxt
            v = self.valid[:, j]
            g = self.G[:, j]
            w = win[:, j]
            cur = np.empty((3, ntg))
            cur[0] = nxt[0]
            for k, q in ((1, q1[:, j]), (2, q2[:, j])):
                # 挑戰：成功得 G 且保留 k 次，失敗少一次
                chal = np.where(w, g + nxt[k], nxt[k - 1])
                cur[k] = q * chal + (1 - q) * nxt[k]
            cur = np.where(v[None, :], cur, nxt)
            # 延長局補挑戰：此節點之前 k=0 視同 k=1
            rep = self.rep[:, j] & v
            cur[0] = np.where(rep, cur[1], cur[0])
            F[:, :, j] = cur
        return F, Fn

    def flat(self, M: np.ndarray) -> np.ndarray:
        """矩陣 → 節點層級一維陣列（與輸入 df 的列順序相同）。"""
        return M[self.tg, self.pos]


def estimate_C(seq: Seq, Fn: np.ndarray, kappa: float = KAPPA) -> list[np.ndarray]:
    """Ĉ(格, k) = E[F_{j+1}(k) − F_{j+1}(k−1) | 格]，三層階層收縮。回傳每節點的 C1、C2（矩陣）。

    收縮：祖父格平均 = (總和 + κ·全體平均)/(n + κ)，父格向祖父格收縮，完整格向父格收縮。
    κ 相當於 κ 筆虛擬樣本，樣本少的格就主要用上層的平均。
    """
    out = []
    for k in (1, 2):
        diff = seq.flat(Fn[k] - Fn[k - 1])
        s = pd.DataFrame({"f": seq.cell_flat, "p": seq.par_flat, "g": seq.gpar_flat, "y": diff})
        gm = s.y.mean()
        g_ = s.groupby("g").y.agg(["sum", "size"])
        g_mean = (g_["sum"] + kappa * gm) / (g_["size"] + kappa)
        p_ = s.groupby("p").y.agg(["sum", "size"])
        p_par = s.groupby("p").g.first().map(g_mean)
        p_mean = (p_["sum"] + kappa * p_par) / (p_["size"] + kappa)
        f_ = s.groupby("f").y.agg(["sum", "size"])
        f_par = s.groupby("f").p.first().map(p_mean)
        f_mean = (f_["sum"] + kappa * f_par) / (f_["size"] + kappa)
        c = s.f.map(f_mean).values
        out.append(seq._mat(np.maximum(c, 0.0)))
    return out


def run_policy(
    seq: Seq,
    C1: np.ndarray,
    C2: np.ndarray,
    lam_on: bool = False,
    iters: int = 0,
    tol: float = 1e-6,
    verbose: bool = True,
    kappa: float = KAPPA,
) -> dict[str, Any]:
    """近似政策迭代。

    iters = 0：給定 C 只評估一次。iters > 0：每輪由 F 重估 Ĉ，直到 C 的最大變動 < tol。
    p* = C / (G + C)：挑戰成功得 G、失敗失去選擇權 C，主觀機率超過 p* 時挑戰的期望值為正。
    """
    hist = []
    for it in range(max(iters, 1)):
        ps1 = np.where(seq.valid, C1 / (seq.G + C1 + 1e-12), 1.0)
        ps2 = np.where(seq.valid, C2 / (seq.G + C2 + 1e-12), 1.0)
        q1, q2 = seq.qprob(ps1, lam_on), seq.qprob(ps2, lam_on)
        F, Fn = seq.backward(q1, q2)
        if iters == 0:
            break
        C1n, C2n = estimate_C(seq, Fn, kappa)
        delta = max(np.abs(C1n - C1)[seq.valid].max(), np.abs(C2n - C2)[seq.valid].max())
        hist.append((it, F[2][:, 0].mean(), delta))
        if verbose:
            print(f"    iter {it}: 每場期望收益 V2 = {F[2][:, 0].mean():.5f}，ΔC 最大 {delta:.2e}", flush=True)
        C1, C2 = C1n, C2n
        if delta < tol:
            break
    return dict(F=F, Fn=Fn, C1=C1, C2=C2, q1=q1, q2=q2, hist=hist)


def modeled_actual_q(df: pd.DataFrame, seq: Seq, perc: dict) -> tuple[np.ndarray, np.ndarray]:
    """05 估計的「實際行為」模型 → 每節點在 k=1、k=2 時的挑戰機率。

    用來驗證框架：把實際行為模型放進同一套動態規劃評估，應接近實際挑戰的收益。
    """
    ig = inning_group(df.inning).values
    player = np.where(df.side.eq("off"), df.batter, df.fielder_2)
    Z = count_dummies(df)
    lg = log_g(df.G).values
    qs = {1: np.zeros(len(df)), 2: np.zeros(len(df))}
    for s_ in ("off", "def"):
        P = SideParams.from_pickle(perc[s_])
        lam, sig = P.lam, P.sigma
        msk = df.side.eq(s_).values
        a = P.player_effect(player[msk])
        e_i = edge_index(df.edge[msk])
        for k in (1, 2):
            ci = P.cell_index(ig[msk], k)
            b = np.where(ci >= 0, P.beta[np.clip(ci, 0, None)], np.nan)
            c = b + P.gam * (lg[msk] - P.lg_mean) + Z[msk] @ P.delta + a
            q = lam * norm.cdf((df.m.values[msk] - c) / sig[e_i])
            qs[k][msk] = np.nan_to_num(q)
    return seq._mat(qs[1]), seq._mat(qs[2])


def savant_pstar(seq: Seq) -> np.ndarray:
    """Savant 規則：失敗代價固定 0.2 分 → p* = 0.2 / (0.2 + ΔRE)。"""
    return np.where(seq.valid, SAVANT_COST_RUNS / (SAVANT_COST_RUNS + seq.dRE), 1.0)


def evaluate(
    df: pd.DataFrame,
    perc: dict,
    iters: int = ITERS_DEFAULT,
    sigma_scale: float = 1.0,
    kappa: float | None = None,
    verbose: bool = False,
    prior: dict[tuple[str, str], np.ndarray] | None = None,
) -> pd.DataFrame:
    """回傳每隊-場次在各政策下的期望勝率收益（DataFrame），供敏感度與 bootstrap 使用。

    政策：opt（最佳門檻、λ = 1）、opt_lam（最佳門檻、維持觀察到的 λ）、static（C 固定為 opt 的節點平均）、
    savant（Savant 規則）、actual_model（05 的實際行為模型）、perfect（σ = 0 的上限）、actual（實際）。
    """
    kp = KAPPA if kappa is None else kappa
    seq = Seq(df, perc, sigma_scale, prior)
    init = np.where(seq.valid, C_INIT_WP, 0.0)
    out = {}
    r = run_policy(seq, init.copy(), init.copy(), iters=iters, verbose=verbose, kappa=kp)
    out["opt"] = r["F"][2][:, 0]
    rl = run_policy(seq, init.copy(), init.copy(), lam_on=True, iters=iters, verbose=verbose, kappa=kp)
    out["opt_lam"] = rl["F"][2][:, 0]
    c1bar, c2bar = r["C1"][seq.valid].mean(), r["C2"][seq.valid].mean()
    out["static"] = run_policy(seq, np.where(seq.valid, c1bar, 0), np.where(seq.valid, c2bar, 0))["F"][2][:, 0]
    q = seq.qprob(savant_pstar(seq), False)
    out["savant"] = seq.backward(q, q)[0][2][:, 0]
    q1a, q2a = modeled_actual_q(df, seq, perc)
    out["actual_model"] = seq.backward(q1a, q2a)[0][2][:, 0]
    out["perfect"] = np.where(seq.valid & (seq.m > 0), seq.G, 0).sum(1)
    out["actual"] = seq._mat(df.actual_gain.values).sum(1)
    return pd.DataFrame(out)


def resample_games(df0: pd.DataFrame, by: dict, games: np.ndarray, rng: np.random.Generator) -> pd.DataFrame:
    """以比賽為單位的 bootstrap 重抽：抽出的每場比賽給新的 game_pk（0, 1, 2, …）避免重複場次合併。"""
    draw = rng.choice(games, len(games), replace=True)
    parts = []
    for i, g in enumerate(draw):
        d = by[g].copy()
        d["game_pk"] = i
        parts.append(d)
    return pd.concat(parts, ignore_index=True)


def forward_pass(seq: Seq, q1: np.ndarray, q2: np.ndarray, start_k: int = 2) -> tuple[np.ndarray, np.ndarray]:
    """前向機率傳遞：追蹤每節點手上 k 次的機率（含延長局補挑戰），回傳期望挑戰次數與期望成功次數矩陣。"""
    ntg, L = seq.ntg, seq.L
    Pk = np.zeros((3, ntg))
    Pk[start_k] = 1.0
    exp_ch = np.zeros((ntg, L))
    exp_win = np.zeros((ntg, L))
    win = seq.m > 0
    for j in range(L):
        v = seq.valid[:, j]
        rep = seq.rep[:, j] & v
        Pk[1] = np.where(rep, Pk[1] + Pk[0], Pk[1])
        Pk[0] = np.where(rep, 0.0, Pk[0])
        a1, a2 = Pk[1] * q1[:, j], Pk[2] * q2[:, j]
        exp_ch[:, j] = np.where(v, a1 + a2, 0)
        exp_win[:, j] = np.where(v & win[:, j], a1 + a2, 0)
        # 失敗 → k−1
        fail = ~win[:, j] & v
        new = Pk.copy()
        new[2] = Pk[2] - np.where(fail, a2, 0)
        new[1] = Pk[1] - np.where(fail, a1, 0) + np.where(fail, a2, 0)
        new[0] = Pk[0] + np.where(fail, a1, 0)
        Pk = new
    return exp_ch, exp_win


def cutoff_inch(seq: Seq, pstar_flat: np.ndarray) -> np.ndarray:
    """每節點：p* → 感知英吋門檻（依打/守 × 邊）。"""
    return seq.flat(seq.cutoff(seq._mat(pstar_flat, 1.0)))
