"""模糊斷點迴歸（fuzzy RD）的共用部分：樣本建構與 rdrobust 包裝。

對應研究步驟：09（主結果與檢查），並被 10 --part rd（分球數）使用。

樣本：被挑戰的判決球（merge_ok）。
跑變數 m：真實誤判幅度（英吋，> 0 = 原判錯誤 → ABS 改判）。
處理：是否改判（m > 0 幾乎必然改判 → 近乎 sharp，以 fuzzy RD 估計，第一階段約 1）。
為什麼用 RD：被挑戰的球中，剛好擦到邊被改判與剛好沒擦到被維持的球，其他條件幾乎相同，
所以斷點上的結果跳躍就是「改判」的因果效果，可以拿來驗證勝率模型預測的 G。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from rdrobust import rdrobust

from src.config import CALLS_G, DM, PA, WE_MODEL
from src.data_io import load_pickle
from src.we import ND
from src.zone import misjudgment


def build_rd_sample() -> pd.DataFrame:
    """建立 RD 樣本與結果變數（皆以「挑戰方」觀點，正 = 對挑戰方有利）。

    Y_pa   ：打席結束時的勝率 − 這球之前的勝率（WE 用 04 的模型套在「實際發生的打席結果」上）
    Y_runs ：這球之後到半局結束，挑戰方的淨得分（打擊方挑戰 = 本隊得分，守備方挑戰 = −失分）
    Y_win  ：挑戰方最後是否贏球（0/1）
    另加模型預測（pred_G、pred_dRE）與預先決定變數（平衡檢定用）。
    """
    m = load_pickle(WE_MODEL)
    c = pd.read_parquet(CALLS_G)
    d = c[c.challenged & c.merge_ok].copy()
    pa = pd.read_parquet(PA)
    d = d.merge(
        pa[["game_pk", "at_bat_number", "bat0", "r_pa", "outs1", "bases1", "half_end"]],
        on=["game_pk", "at_bat_number"],
        how="left",
        validate="m:1",
    )
    # 比賽最終比分（上半 = 客隊打擊，下半 = 主隊打擊）
    hs = pa.groupby(["game_pk", "half"]).half_end.max().unstack()
    d = d.join(hs.rename(columns={0: "away_final", 1: "home_final"}), on="game_pk")
    half = np.where(d.half.eq("bottom"), 1, 0)
    inn = d.inning.values.astype(int)
    # 打席結束後狀態（主隊觀點比分差）→ 查 04 模型的半局結束值 NV 或半局中值 MV
    bat_after = d.bat0 + d.r_pa
    hd_after = np.where(half == 1, bat_after - d.fld_score, d.fld_score - bat_after)
    o1, b1 = d.outs1.values.astype(int), d.bases1.values.astype(int)
    i10 = np.clip(inn, 1, 10)
    idx = np.clip(hd_after + DM, 0, ND - 1).astype(int)
    we_after_home = np.where(o1 >= 3, m.NV[i10, half, idx], m.MV[i10, half, np.clip(o1, 0, 2), b1, idx])
    walk = (half == 1) & (inn >= 9) & (hd_after > 0)  # 再見：主隊已贏
    we_after_home = np.where(walk, 1.0, we_after_home)
    we_after_bat = np.where(half == 1, we_after_home, 1 - we_after_home)
    off = d.orig_strike.values  # 打擊方挑戰（原判好球）
    sgn = np.where(off, 1.0, -1.0)  # 挑戰方觀點
    d["m"] = misjudgment(off, d.signed_dist_in)
    d["treat"] = d.overturned.astype("boolean").fillna(False).astype(float)
    d["Y_pa"] = sgn * (we_after_bat - d.we_bat_pre)
    d["Y_runs"] = sgn * (d.half_end - d.bat_score)
    home_won = (d.home_final > d.away_final).astype(float)
    chal_home = np.where(off, half == 1, half == 0)
    d["Y_win"] = np.where(chal_home, home_won, 1 - home_won)
    d["we_pre_chal"] = np.where(off, d.we_bat_pre, 1 - d.we_bat_pre)
    d["side"] = np.where(off, "batter", "defense")
    # 模型預測的改判效果（挑戰方觀點）
    d["pred_G"] = d.G
    d["pred_dRE"] = d.dRE
    # 預先決定變數（不應在斷點跳躍）
    d["two_strike"] = (d.strikes_pre == 2).astype(float)
    d["three_ball"] = (d.balls_pre == 3).astype(float)
    d["runners_on"] = (d.bases > 0).astype(float)
    d["late"] = (d.inning >= 7).astype(float)
    d["absdiff"] = (d.bat_score - d.fld_score).abs()
    print(f"被挑戰球 {len(d):,}，|m| ≤ 1：{(d.m.abs() <= 1).sum():,}，打席結果缺值 {d.r_pa.isna().sum()}")
    return d


def kern_mean(x: np.ndarray, w: np.ndarray) -> float:
    """加權平均（用三角核權重算斷點附近的模型預測值）。"""
    return float(np.sum(x * w) / np.sum(w))


def triangular_weights(m: pd.Series, h: float) -> np.ndarray:
    """三角核權重 max(1 − |m|/h, 0)，與 rdrobust 預設核一致。"""
    return np.clip(1 - m.abs() / h, 0, None)


def rd(
    d: pd.DataFrame,
    y: str,
    h: float | None = None,
    fuzzy: bool = True,
    c: float = 0.0,
    donut: float = 0.0,
) -> dict:
    """以 rdrobust 估計斷點 c 的跳躍（以比賽聚類標準誤）。

    fuzzy 且 c = 0 時以 treat（是否改判）為處理變數，安慰劑斷點（c ≠ 0）一律 sharp。
    donut > 0 時排除 |m| < donut 的球（好球帶重建誤差最大的區域）。
    回傳：頻寬 h、係數、標準誤、穩健偏誤校正 95% 區間與 p 值、有效樣本數。
    """
    s = d[d.m.abs() >= donut] if donut else d
    kw = dict(y=s[y].values, x=s.m.values - c, c=0.0, cluster=s.game_pk.values)
    if fuzzy and c == 0.0:
        kw["fuzzy"] = s.treat.values
    if h is not None:
        kw["h"] = h
    r = rdrobust(**kw)
    hh = float(r.bws.iloc[0, 0])
    coef = float(r.coef.iloc[0, 0])
    se_c = float(r.se.iloc[0, 0])
    lo, hi = float(r.ci.iloc[2, 0]), float(r.ci.iloc[2, 1])  # 第 3 列 = robust bias-corrected
    pv = float(r.pv.iloc[2, 0])
    n = int(sum(r.N_h))
    return dict(y=y, h=hh, coef=coef, se=se_c, rb_lo=lo, rb_hi=hi, rb_p=pv, n_eff=n)
