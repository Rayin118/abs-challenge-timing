"""05_perception.py：球員的「感知誤差」與「實際門檻」（結構式挑戰選擇模型）。

目的
    從實際挑戰行為反推：球員看判決有多準（σ）、多常注意到可挑戰的球（λ）、以及他們實際用的門檻 c。
    結果供 06 評估「實際行為」與最佳策略，並把實際門檻換算成主觀機率 τ 與最佳門檻 p* 比較。
研究步驟
    感知模型（實際行為的結構式估計）。
輸入
    data/processed/calls_2026_G.parquet（04 gain）
輸出
    data/processed/opps_2026.parquet（挑戰機會樣本，06 敏感度、08 f1）
    data/processed/perception.pkl（參數＋主觀機率對照表 p̃(m̃)，06、07、10、11）
    outputs/tables/perc_params.csv、perc_fit_bins.csv、perc_thresholds.csv
執行
    python scripts/05_perception.py            # 約 10 秒

設定（分守備方 def／打擊方 off 各自估計）
    m_i  = 真實誤判幅度（英吋，> 0 表示原判錯誤，由重建 ABS 好球帶而來）
    球員看到 m̃_i = m_i + σ_e·ε，ε ~ N(0,1)，σ_e 依最近的好球帶邊（上緣／下緣／左右）不同
    若 m̃_i > c_i 則挑戰，但只有比例 λ 的機會會被「考慮」（注意力／其他未建模因素）
    → P(挑戰_i) = λ · Φ((m_i − c_i) / σ_e(i))
    c_i = Σ_cell β_cell（局數區段 × 剩餘次數）＋ γ·log G_i ＋ δ·球數 ＋ α_player
          （守備：捕手，打擊：打者，α 以 ridge 懲罰 τ = 1 英吋）
樣本
    挑戰方仍有挑戰可用的判決球，|m| ≤ 6 英吋（m > 6 的「誤判」幾乎不被挑戰，疑為追蹤誤差，另外報告）
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import OptimizeResult, minimize
from scipy.stats import norm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import CALLS_G, EDGES, M_MAX_IN, OPPS, PERCEPTION, TAB, TAU_PLAYER_IN  # noqa: E402
from src.data_io import dump_pickle  # noqa: E402
from src.perception import (  # noqa: E402
    N_COUNT_DUMMIES,
    count_dummies,
    inning_group,
    log_g,
    posterior_table,
    unpack_theta,
)
from src.zone import misjudgment, nearest_edge  # noqa: E402

#: 起始值：門檻 2 英吋、σ 1.5 英吋、logit λ = 1（λ ≈ 0.73），其餘 0
TH0_C, TH0_SIGMA, TH0_LOGIT_LAM = 2.0, 1.5, 1.0
#: 適配檢查用的 m 分箱（英吋）
M_BINS = [-6, -3, -2, -1, -0.5, 0, 0.5, 1, 1.5, 2, 3, 4, 6]


def build_opps() -> pd.DataFrame:
    """挑戰機會樣本：挑戰方還有挑戰可用（rem_eligible > 0）的判決球，加上模型需要的欄位。"""
    df = pd.read_parquet(CALLS_G)
    o = df[df.rem_eligible > 0].copy()
    o["m"] = misjudgment(o.orig_strike, o.signed_dist_in)
    o["edge"] = nearest_edge(o.plate_x, o.plate_z, o.sz_top, o.sz_bot)
    o["sideflag"] = np.where(o.orig_strike, "off", "def")
    o["ig"] = inning_group(o.inning)
    o["cell"] = o.ig + "|k" + o.rem_eligible.clip(upper=2).astype(str)
    o["player"] = np.where(o.orig_strike, o.batter, o.fielder_2)
    o["logG"] = log_g(o.G)
    o.to_parquet(OPPS, index=False)
    return o


class Probit:
    """λ·Φ((m − c)/σ_e) 的最大概似估計（含球員隨機效果的 ridge 懲罰），提供解析梯度。"""

    def __init__(self, d: pd.DataFrame) -> None:
        self.cells = sorted(d.cell.unique())
        self.ci = d.cell.map({c: i for i, c in enumerate(self.cells)}).values
        self.ei = d.edge.map({e: i for i, e in enumerate(EDGES)}).values
        self.players = np.sort(d.player.unique())
        self.pi = np.searchsorted(self.players, d.player.values)
        self.m = d.m.values
        self.y = d.challenged.values.astype(float)
        # log G 置中，讓 β 解讀為「平均 G 下」的門檻
        self.lg_mean = d.logG.mean()
        self.lg = d.logG.values - self.lg_mean
        # 球數虛擬變數：一好球、兩好球、三壞球（超出 G 之外的球數慣性，例如打者多半只在兩好球時挑戰）
        self.Z = count_dummies(d)
        self.nz = self.Z.shape[1]
        self.nc, self.ne, self.np_ = len(self.cells), len(EDGES), len(self.players)

    def unpack(self, th: np.ndarray) -> tuple:
        """參數向量 → (β, γ, logσ, logitλ, δ, α)。"""
        return unpack_theta(th, self.nc)

    def nll(self, th: np.ndarray) -> tuple[float, np.ndarray]:
        """負對數概似（含 α 的 ridge 懲罰）與其梯度。"""
        beta, gam, ls, ll, delta, alpha = self.unpack(th)
        sig = np.exp(ls)[self.ei]
        lam = 1 / (1 + np.exp(-ll))
        c = beta[self.ci] + gam * self.lg + self.Z @ delta + alpha[self.pi]
        z = (self.m - c) / sig
        Phi = norm.cdf(z)
        phi = norm.pdf(z)
        p = np.clip(lam * Phi, 1e-12, 1 - 1e-12)
        y = self.y
        ll_ = y * np.log(p) + (1 - y) * np.log1p(-p)
        # 鏈鎖律：ℓ → p → z → (c, σ)
        dp = y / p - (1 - y) / (1 - p)  # dℓ/dp
        dz = dp * lam * phi  # dℓ/dz
        g_c = -dz / sig  # dℓ/dc
        g_beta = np.bincount(self.ci, g_c, self.nc)
        g_gam = (g_c * self.lg).sum()
        g_ls = np.bincount(self.ei, -dz * z, self.ne)  # dz/dlogσ = −z
        g_ll = (dp * Phi * lam * (1 - lam)).sum()
        g_delta = self.Z.T @ g_c
        g_alpha = np.bincount(self.pi, g_c, self.np_) - alpha / TAU_PLAYER_IN**2
        f = -(ll_.sum() - 0.5 * (alpha**2).sum() / TAU_PLAYER_IN**2)
        g = -np.concatenate([g_beta, [g_gam], g_ls, [g_ll], g_delta, g_alpha])
        return f, g

    def fit(self) -> OptimizeResult:
        """以 L-BFGS-B 估計（解析梯度）。"""
        th0 = np.concatenate(
            [
                np.full(self.nc, TH0_C),
                [0.0],
                np.log([TH0_SIGMA] * 3),
                [TH0_LOGIT_LAM],
                np.zeros(self.nz),
                np.zeros(self.np_),
            ]
        )
        r = minimize(self.nll, th0, jac=True, method="L-BFGS-B", options={"maxiter": 5000, "maxfun": 20000})
        self.res, self.th = r, r.x
        return r

    def core_se(self, h: float = 1e-4) -> np.ndarray:
        """核心參數（β、γ、logσ、logitλ、δ）的標準誤：固定 α，對梯度做中央差分得 Hessian。"""
        k = self.nc + 1 + self.ne + 1 + self.nz
        H = np.zeros((k, k))
        for j in range(k):
            e = np.zeros_like(self.th)
            e[j] = h
            H[:, j] = (self.nll(self.th + e)[1][:k] - self.nll(self.th - e)[1][:k]) / (2 * h)
        H = (H + H.T) / 2
        return np.sqrt(np.diag(np.linalg.inv(H)))

    def predict(self, th: np.ndarray | None = None) -> np.ndarray:
        """每個機會的預測挑戰機率 λ·Φ((m − c)/σ)。"""
        beta, gam, ls, ll, delta, alpha = self.unpack(self.th if th is None else th)
        sig = np.exp(ls)[self.ei]
        lam = 1 / (1 + np.exp(-ll))
        c = beta[self.ci] + gam * self.lg + self.Z @ delta + alpha[self.pi]
        return lam * norm.cdf((self.m - c) / sig)


def main() -> None:
    """分守備方、打擊方估計感知模型，輸出參數表、適配表、門檻表與 perception.pkl。"""
    t0 = time.time()
    o = build_opps()
    out = {}
    rows_p, rows_fit, rows_thr = [], [], []
    for side in ["def", "off"]:
        d_all = o[o.sideflag == side]
        far = d_all[d_all.m > M_MAX_IN]
        print(f"[{side}] 機會 {len(d_all):,}，m > 6 英吋（排除）{len(far)} 顆，挑戰率 {far.challenged.mean():.3f}")
        d = d_all[d_all.m.abs() <= M_MAX_IN].copy()
        mod = Probit(d)
        r = mod.fit()
        se = mod.core_se()
        beta, gam, ls, ll, delta, alpha = mod.unpack(mod.th)
        lam = 1 / (1 + np.exp(-ll))
        print(f"  收斂：{r.success}（{r.message}），n = {len(d):,}，球員 {mod.np_}，{time.time() - t0:.0f}s")
        names = (
            [f"c[{c}]" for c in mod.cells]
            + ["gamma_logG"]
            + [f"log_sigma[{e}]" for e in EDGES]
            + ["logit_lambda", "d_strike1", "d_strike2", "d_ball3"]
        )
        assert len(names) == mod.nc + 1 + len(EDGES) + 1 + N_COUNT_DUMMIES
        k = len(names)
        for nm, v, s in zip(names, mod.th[:k], se, strict=False):
            rows_p.append({"side": side, "param": nm, "est": v, "se": s})
        sig = np.exp(ls)
        print(
            "  σ（英吋）：",
            dict(zip(EDGES, sig.round(3), strict=False)),
            " λ =",
            round(lam, 3),
            " γ =",
            round(gam, 3),
            "，球數 δ（1好、2好、3壞）=",
            delta.round(2),
            "，球員效果 SD =",
            round(alpha.std(), 3),
        )
        print("  門檻 c（英吋，G 為平均 log）：", dict(zip(mod.cells, beta.round(2), strict=False)))
        # 適配：依 m 分箱、狀態格、邊、m × 好球數
        d["pred"] = mod.predict()
        d["mb"] = pd.cut(d.m, M_BINS).astype(str)
        d["cnt"] = d.balls_pre.astype(str) + "-" + d.strikes_pre.astype(str)
        d["mb_cnt"] = np.where(d.m > 1.5, "m>1.5|", "m<=1.5|") + np.where(
            d.strikes_pre == 2, "2s", np.where(d.strikes_pre == 1, "1s", "0s")
        )
        for key in ["mb", "cell", "edge", "mb_cnt"]:
            g = d.groupby(key).agg(n=("challenged", "size"), actual=("challenged", "mean"), pred=("pred", "mean"))
            for idx, rr in g.iterrows():
                rows_fit.append({"side": side, "by": key, "group": idx, **rr.to_dict()})
        # 主觀機率對照表（先驗用全部機會，含 |m| > 6）與「實際門檻 τ」
        tabs = {}
        for ei, e in enumerate(EDGES):
            grid, post = posterior_table(d_all[d_all.edge == e].m.values, sig[ei])
            tabs[e] = (grid, post)
        # τ = 實際門檻 c 在主觀機率表上的值，依各邊的機會比例加權
        ew = d.edge.value_counts(normalize=True)
        for ci_, cell in enumerate(mod.cells):
            c_med = beta[ci_]
            tau = sum(ew[e] * np.interp(c_med, *tabs[e]) for e in EDGES)
            dd = d[d.cell == cell]
            c2 = c_med + delta[1]  # 兩好球時的門檻
            tau2 = sum(ew[e] * np.interp(c2, *tabs[e]) for e in EDGES)
            rows_thr.append(
                {
                    "side": side,
                    "cell": cell,
                    "n": len(dd),
                    "chal": int(dd.challenged.sum()),
                    "c_inch_0strike": c_med,
                    "c_se": se[ci_],
                    "tau_0strike": tau,
                    "c_inch_2strike": c2,
                    "tau_2strike": tau2,
                    "G_median": dd.G.median(),
                }
            )
        out[side] = {
            "th": mod.th,
            "cells": mod.cells,
            "players": mod.players,
            "lg_mean": mod.lg_mean,
            "sigma": dict(zip(EDGES, sig, strict=False)),
            "lambda": lam,
            "gamma": gam,
            "delta": delta,
            "post": tabs,
        }
    pd.DataFrame(rows_p).to_csv(TAB / "perc_params.csv", index=False)
    pd.DataFrame(rows_fit).to_csv(TAB / "perc_fit_bins.csv", index=False)
    thr = pd.DataFrame(rows_thr)
    thr.to_csv(TAB / "perc_thresholds.csv", index=False)
    dump_pickle(out, PERCEPTION)
    pd.set_option("display.width", 200)
    print(thr.round(3).to_string())
    fit = pd.DataFrame(rows_fit)
    print(fit[fit.by.isin(["mb", "mb_cnt", "edge"])].round(4).to_string())


if __name__ == "__main__":
    main()
