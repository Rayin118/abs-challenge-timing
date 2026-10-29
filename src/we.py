"""含球數的勝率（WE）與得分期望（RE）模型：2026 ABS 年代。

對應研究步驟：04（建模、驗證、改判收益 G），並被 05–11 透過 we_model.pkl 使用。

結構（全部由 2026 例行賽估計，可傳入子樣本做樣本外檢查）
1. 半局剩餘得分分布 P(r | 出局, 壘況, 上/下半)：以打席開始狀態估計，只用 1–8 局
   （9 局下半以後有再見截斷，1 場雨天提前結束的比賽排除其最後半局）。
2. 比賽層級遞迴（主隊觀點，D = 主 − 客）：
   - 9 局下以後：主隊領先則不打，打到領先即結束（再見），平手進入延長賽。
   - 延長賽每半局由二壘有人、無人出局開始（突破僵局制），各延長局相同 → 以不動點解出 W_x。
3. 球數層級：WE(狀態, 球數) = Σ_type P(type | 球數) Σ P(出局', 壘況', 得分 | type, 出局, 壘況) · V(之後)
   - P(type | 球數)：經過該球數的打席，其最終結果類型（每個打席每個球數只計一次）。
   - 轉移：打席開始狀態 → 下一打席開始狀態（或三出局），得分 = 兩者的打擊方得分差，
     來源為所有半局，但排除 9 局下以後（再見截斷）。
4. 改判收益：ball 結果 = 球數 (b+1, s)，若 b = 3 則保送，strike 結果 = (b, s+1)，若 s = 2 則三振。
   G = |WE_打擊方(ball 結果) − WE_打擊方(strike 結果)|（對打者或守備方挑戰皆同）。

注意：data/processed/we_model.pkl 是 WEModel 物件的 pickle，類別路徑 src.we.WEModel
與屬性名稱（P、T、WEc、NV、MV、RE24、RE288 …）都不能改，否則舊檔讀不回來。
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd

from src.config import DM, RMAX, SPARSE_TRANSITION_N, TRMAX
from src.zone import bases_code

ND = 2 * DM + 1  # 比分差網格點數
COUNTS: list[tuple[int, int]] = [(b, s) for b in range(4) for s in range(3)]
CIDX: dict[tuple[int, int], int] = {c: i for i, c in enumerate(COUNTS)}
#: 球數索引 k：0–11 = 12 種球數（b*3+s），12 = 三振，13 = 保送
K_STRIKEOUT, K_WALK = 12, 13
TYPES: list[str] = ["K", "BB", "HBP", "1B", "2B", "3B", "HR", "OUT", "ROE", "TRUNC"]
#: Statcast events → 打席結果類型
EVMAP: dict[str, str] = {
    "strikeout": "K",
    "strikeout_double_play": "K",
    "walk": "BB",
    "intent_walk": "BB",
    "hit_by_pitch": "HBP",
    "single": "1B",
    "double": "2B",
    "triple": "3B",
    "home_run": "HR",
    "field_out": "OUT",
    "force_out": "OUT",
    "grounded_into_double_play": "OUT",
    "double_play": "OUT",
    "sac_fly": "OUT",
    "sac_bunt": "OUT",
    "fielders_choice_out": "OUT",
    "sac_fly_double_play": "OUT",
    "triple_play": "OUT",
    "field_error": "ROE",
    "fielders_choice": "ROE",
    "catcher_interf": "ROE",
    "truncated_pa": "TRUNC",
}
GHOST = 0 * 8 + 2  # 狀態索引 = 出局*8 + 壘況，延長賽開局：無人出局、二壘有人


def build_pa(sc: pd.DataFrame) -> pd.DataFrame:
    """把 Statcast 逐球資料彙整成每個打席一列。

    欄位：開始狀態（局、上下半、出局、壘況、打擊方得分）、結果類型、
    下一打席開始狀態（outs1、bases1，半局結束則 outs1 = 3）、本打席得分 r_pa、
    到半局結束的剩餘得分 r_rest、經過的球數 cmask（12 個球數的位元遮罩）。
    """
    sc = sc.sort_values(["game_pk", "at_bat_number", "pitch_number"])
    sc = sc.assign(bases=bases_code(sc), half=np.where(sc.inning_topbot.eq("Top"), 0, 1))
    g = sc.groupby(["game_pk", "at_bat_number"], sort=False)
    pa = g.agg(
        inning=("inning", "first"),
        half=("half", "first"),
        outs=("outs_when_up", "first"),
        bases=("bases", "first"),
        bat0=("bat_score", "first"),
        event=("events", "last"),
        post_bat=("post_bat_score", "last"),
    ).reset_index()
    # 經過的球數：每個打席把出現過的球數記成位元（bit k = 球數 k），同一球數只計一次
    ci = (sc.balls.clip(upper=3) * 3 + sc.strikes.clip(upper=2)).astype(int).values
    u = sc[["game_pk", "at_bat_number"]].assign(cbit=np.left_shift(1, ci)).drop_duplicates()
    cm = u.groupby(["game_pk", "at_bat_number"], sort=False).cbit.sum()
    pa["cmask"] = cm.reindex(pd.MultiIndex.from_frame(pa[["game_pk", "at_bat_number"]])).values
    pa["type"] = pa.event.map(EVMAP)
    # 半局結束時打擊方的得分
    hk = ["game_pk", "inning", "half"]
    half_end = sc.groupby(hk).post_bat_score.max().rename("half_end")
    pa = pa.merge(half_end, left_on=hk, right_index=True, how="left")
    # 下一打席（同半局）的開始狀態，沒有下一打席 = 三出局
    pa = pa.sort_values(["game_pk", "at_bat_number"]).reset_index(drop=True)
    same = (pa.game_pk.shift(-1) == pa.game_pk) & (pa.inning.shift(-1) == pa.inning) & (pa.half.shift(-1) == pa.half)
    pa["outs1"] = np.where(same, pa.outs.shift(-1), 3).astype(int)
    pa["bases1"] = np.where(same, pa.bases.shift(-1), 0).astype(int)
    pa["r_pa"] = (np.where(same, pa.bat0.shift(-1), pa.half_end) - pa.bat0).astype(int)
    pa["r_rest"] = (pa.half_end - pa.bat0).astype(int)
    # 少於 9 局就結束（雨天）的比賽，最後一個半局不完整
    maxinn = pa.groupby("game_pk").inning.transform("max")
    pa["short_game_last_half"] = (maxinn < 9) & (pa.inning == maxinn)
    return pa


def _cum_ge(p: np.ndarray, k: int) -> float:
    """P(r ≥ k)，p 為 0..RMAX 的機率向量。"""
    return p[min(k, RMAX) :].sum() if k <= RMAX else 0.0


class WEModel:
    """含球數的勝率與得分期望模型（主隊觀點）。

    建構時估計 P（半局剩餘得分）、T（打席轉移）、Ptype（各球數的結果類型分布），
    並解出比賽層級的半局結束值 S_top／S_bot。呼叫 table() 得到完整查表 WEc，
    re_table() 得到 RE24 與 RE288。
    """

    def __init__(self, pa: pd.DataFrame, rest_innings: tuple[int, int] = (1, 8)) -> None:
        self.pa = pa
        lo, hi = rest_innings
        ok = ~pa.short_game_last_half
        # 1. 半局剩餘得分分布：只用 1–8 局，避開 9 局下半的再見截斷
        self.P = np.zeros((2, 24, RMAX + 1))
        d = pa[ok & pa.inning.between(lo, hi)]
        for (h, o, b), s in d.groupby(["half", "outs", "bases"]).r_rest:
            cnt = np.bincount(np.clip(s.values, 0, RMAX), minlength=RMAX + 1)
            self.P[h, o * 8 + b] = cnt / cnt.sum()
        self.n_rest = d.groupby(["half", "outs", "bases"]).size()
        # 3. 打席轉移（排除 9 局下以後）：T[type, 狀態, 出局', 壘況', 得分]
        dt = pa[ok & pa.type.notna() & ~((pa.half == 1) & (pa.inning >= 9))]
        self.T = np.zeros((len(TYPES), 24, 4, 8, TRMAX + 1))
        for (t, o, b), s in dt.groupby(["type", "outs", "bases"]):
            ti = TYPES.index(t)
            for o1, b1, r in zip(s.outs1.values, s.bases1.values, np.clip(s.r_pa.values, 0, TRMAX), strict=False):
                self.T[ti, o * 8 + b, o1, (b1 if o1 < 3 else 0), r] += 1
        self.n_trans = self.T.sum(axis=(2, 3, 4))
        # 稀疏格（< 20 筆）→ 用同 type、同壘況、其他出局數的轉移（出局數平移）補強
        self._Traw = self.T.copy()
        for ti in range(len(TYPES)):
            for i in range(24):
                n = self.n_trans[ti, i]
                # TRUNC（打席因跑壘出局結束半局而中斷）只可能發生在兩出局 → 不做出局平移的合併
                if n < SPARSE_TRANSITION_N and TYPES[ti] != "TRUNC":
                    self.T[ti, i] = self._pooled(ti, i, n)
                tot = self.T[ti, i].sum()
                if tot > 0:
                    self.T[ti, i] /= tot
        # 該狀態下沒有任何轉移資料的 type（例如 0、1 出局時的 TRUNC）→ 在該狀態把 P(type|count) 重新正規化
        self.valid = self.T.sum(axis=(2, 3, 4)) > 0
        da = pa[ok & pa.type.notna()]
        self.Ptype = np.zeros((12, len(TYPES)))
        tidx = da.type.map({t: i for i, t in enumerate(TYPES)}).values
        for ci in range(12):
            m = (da.cmask.values >> ci) & 1 == 1
            self.Ptype[ci] = np.bincount(tidx[m], minlength=len(TYPES)) / m.sum()
        self.n_count = {COUNTS[ci]: int(((da.cmask.values >> ci) & 1).sum()) for ci in range(12)}
        self._solve()

    def _pooled(self, ti: int, i: int, n: float) -> np.ndarray:
        """稀疏轉移：把其他出局數的同壘況轉移平移到本出局數後加總（出局增量、壘況、得分不變）。"""
        o, b = divmod(i, 8)
        out = self._Traw[ti, i].copy()
        for o2 in range(3):
            if o2 == o:
                continue
            src = self._Traw[ti, o2 * 8 + b]
            for o1 in range(4):
                for b1 in range(8):
                    for r in range(TRMAX + 1):
                        c = src[o1, b1, r]
                        if c == 0:
                            continue
                        no = min(o + (o1 - o2), 3)
                        out[no, b1 if no < 3 else 0, r] += c
        return out

    # ---------- 比賽層級 ----------
    @staticmethod
    def _shift(arr: np.ndarray, k: int) -> np.ndarray:
        """arr 定義於 D∈[−DM,DM]，回傳 f(D+k)，超出邊界取端點值。"""
        idx = np.clip(np.arange(ND) + k, 0, ND - 1)
        return arr[idx]

    def _solve(self) -> None:
        """逆向歸納出每局上／下半開始時的主隊勝率 S_top、S_bot（比分差 D 的函數）。

        從第 10 局（代表所有延長局）往第 1 局倒推：
        下半局開始值 = Σ_r P(下半得 r 分) · 之後的值，上半局開始值 = Σ_r P(上半失 r 分) · 下半局開始值。
        9 局以後下半局：主隊領先就不打（勝率 1），打到領先即再見。
        """
        P = self.P
        R = np.arange(RMAX + 1)
        Dg = np.arange(-DM, DM + 1)
        # 延長賽不動點：每個延長局都從 GHOST 開始，客隊先攻得 r 分、主隊需 ≥ r+1 分才贏、剛好 r 分則再一局
        # W_x = A + B·W_x → W_x = A / (1 − B)
        pt, pb = P[0, GHOST], P[1, GHOST]
        A = sum(pt[r] * _cum_ge(pb, r + 1) for r in R)
        B = sum(pt[r] * pb[r] for r in R)
        self.Wx = A / (1 - B)
        # 半局開始值（主隊觀點），局數索引 1..10（10 = 延長）
        self.S_top = np.zeros((11, ND))
        self.S_bot = np.zeros((11, ND))
        for i in range(10, 0, -1):
            # 下半局開始值
            sb = np.zeros(ND)
            start = GHOST if i >= 10 else 0
            for di, D in enumerate(Dg):
                if i >= 9 and D > 0:
                    sb[di] = 1.0  # 主隊已領先，下半局不打
                    continue
                v = 0.0
                for r in R:
                    nd = D + r
                    if i >= 9:
                        val = 1.0 if nd > 0 else (self.Wx if nd == 0 else 0.0)
                    else:
                        val = self.S_top[i + 1, np.clip(nd + DM, 0, ND - 1)]
                    v += P[1, start, r] * val
                sb[di] = v
            self.S_bot[i] = sb
            st = np.zeros(ND)
            for di, D in enumerate(Dg):
                st[di] = sum(P[0, start, r] * sb[np.clip(D - r + DM, 0, ND - 1)] for r in R)
            self.S_top[i] = st
        # 檢查不動點一致：S_top[10][D=0] 應 = Wx
        self.fixed_point_gap = self.S_top[10, DM] - self.Wx

    def next_value(self, i: int, h: int, nd: np.ndarray | int) -> np.ndarray:
        """半局結束（三出局）後、比分差 nd（主隊觀點，可為陣列）的主隊勝率。"""
        i = min(i, 10)
        nd = np.asarray(nd)
        if h == 0:
            return self.S_bot[i][np.clip(nd + DM, 0, ND - 1)]
        if i >= 9:
            return np.where(nd > 0, 1.0, np.where(nd == 0, self.Wx, 0.0))
        return self.S_top[i + 1][np.clip(nd + DM, 0, ND - 1)]

    def mid_value(self, i: int, h: int, o: int, b: int, D: np.ndarray) -> np.ndarray:
        """半局中（出局 o、壘況 b、打席開始）的主隊勝率，D 可為陣列。"""
        i = min(i, 10)
        D = np.asarray(D)
        p = self.P[h, o * 8 + b]
        sgn = 1 if h == 1 else -1
        v = np.zeros(D.shape)
        for r in range(RMAX + 1):
            nd = D + sgn * r
            if h == 1 and i >= 9:
                val = np.where(nd > 0, 1.0, self.next_value(i, h, nd))
            else:
                val = self.next_value(i, h, nd)
            v = v + p[r] * val
        if h == 1 and i >= 9:
            v = np.where(D > 0, 1.0, v)
        return v

    def _precompute(self) -> None:
        """預先算好 NV（半局結束值）與 MV（半局中值）的整張網格。"""
        Dg = np.arange(-DM, DM + 1)
        self.NV = np.zeros((11, 2, ND))
        self.MV = np.zeros((11, 2, 3, 8, ND))
        for i in range(1, 11):
            for h in (0, 1):
                self.NV[i, h] = self.next_value(i, h, Dg)
                for o in range(3):
                    for b in range(8):
                        self.MV[i, h, o, b] = self.mid_value(i, h, o, b, Dg)

    def _ptype(self, ci: int, si: int) -> np.ndarray:
        """球數 ci、狀態 si 下的結果類型分布（去掉該狀態沒有轉移資料的類型後重新正規化）。"""
        pt = self.Ptype[ci] * self.valid[:, si]
        return pt / pt.sum()

    def table(self) -> np.ndarray:
        """完整查表 WEc[局(1..10), 上下半, 出局, 壘況, D, k]（主隊觀點）。

        k = 12 種球數 + 12：三振 + 13：保送。
        """
        self._precompute()
        out = np.zeros((11, 2, 3, 8, ND, 14))
        Dg = np.arange(-DM, DM + 1)
        for i in range(1, 11):
            for h in (0, 1):
                sgn = 1 if h == 1 else -1
                walk_possible = h == 1 and i >= 9  # 9 局下以後打到領先即再見
                for o in range(3):
                    for b in range(8):
                        si = o * 8 + b
                        # tv[type]：打席以該類型結束後的期望勝率
                        tv = np.zeros((len(TYPES), ND))
                        for ti in range(len(TYPES)):
                            Tm = self.T[ti, si]
                            acc = np.zeros(ND)
                            for o1, b1, r in np.argwhere(Tm > 0):
                                nd = Dg + sgn * r
                                idx = np.clip(nd + DM, 0, ND - 1)
                                val = self.NV[i, h, idx] if o1 == 3 else self.MV[i, h, o1, b1, idx]
                                if walk_possible:
                                    val = np.where(nd > 0, 1.0, val)
                                acc += Tm[o1, b1, r] * val
                            tv[ti] = acc
                        # 球數 k 的值 = Σ_type P(type | k) · tv[type]
                        for ci in range(12):
                            out[i, h, o, b, :, ci] = self._ptype(ci, si) @ tv
                        out[i, h, o, b, :, K_STRIKEOUT] = tv[TYPES.index("K")]
                        out[i, h, o, b, :, K_WALK] = tv[TYPES.index("BB")]
                        if walk_possible:
                            out[i, h, o, b, Dg > 0, :] = 1.0
        self.WEc = out
        return out

    def lookup(
        self,
        inning: np.ndarray,
        half: np.ndarray,
        outs: np.ndarray,
        bases: np.ndarray,
        home_diff: np.ndarray,
        k: np.ndarray,
    ) -> np.ndarray:
        """向量化查表（主隊觀點），局數 ≥ 10 一律視為延長局，比分差超出 ±DM 取端點。"""
        i = np.clip(np.asarray(inning), 1, 10)
        d = np.clip(np.asarray(home_diff) + DM, 0, ND - 1)
        return self.WEc[i, half, outs, bases, d, k]

    # ---------- 得分期望（RE288） ----------
    def re_table(self) -> np.ndarray:
        """RE24（由 P）與 RE288[上下半, 出局, 壘況, k]（k 同 table()）。"""
        re24 = np.zeros((2, 4, 8))
        for h in (0, 1):
            for o in range(3):
                for b in range(8):
                    re24[h, o, b] = (self.P[h, o * 8 + b] * np.arange(RMAX + 1)).sum()
        re = np.zeros((2, 3, 8, 14))
        for h in (0, 1):
            for o in range(3):
                for b in range(8):
                    tv = np.zeros(len(TYPES))
                    for ti in range(len(TYPES)):
                        Tm = self.T[ti, o * 8 + b]
                        for o1, b1, r in np.argwhere(Tm > 0):
                            tv[ti] += Tm[o1, b1, r] * (r + re24[h, o1, b1])
                    for ci in range(12):
                        re[h, o, b, ci] = self._ptype(ci, o * 8 + b) @ tv
                    re[h, o, b, K_STRIKEOUT] = tv[TYPES.index("K")]
                    re[h, o, b, K_WALK] = tv[TYPES.index("BB")]
        self.RE24, self.RE288 = re24, re
        return re


def outcome_index(balls: np.ndarray, strikes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """回傳 (判 ball 之後的 k, 判 strike 之後的 k)：三壞球再一壞 = 保送 13，兩好球再一好 = 三振 12。"""
    balls, strikes = np.asarray(balls), np.asarray(strikes)
    kb = np.where(balls >= 3, K_WALK, (balls + 1).clip(max=3) * 3 + strikes)
    ks = np.where(strikes >= 2, K_STRIKEOUT, balls * 3 + (strikes + 1).clip(max=2))
    return kb, ks


def home_view(df: pd.DataFrame, half_col: str = "half") -> tuple[np.ndarray, np.ndarray]:
    """由判決球表（half = "top"/"bottom"）算出 (上下半 0/1, 主隊觀點比分差)。"""
    half = np.where(df[half_col].eq("bottom"), 1, 0)
    hd = np.where(half == 1, df.bat_score - df.fld_score, df.fld_score - df.bat_score)
    return half, hd


def to_batting(half: np.ndarray) -> Callable[[np.ndarray], np.ndarray]:
    """回傳把主隊觀點勝率換成打擊方觀點的函式（上半局打擊方是客隊 → 1 − w）。"""
    return lambda w: np.where(half == 1, w, 1 - w)
