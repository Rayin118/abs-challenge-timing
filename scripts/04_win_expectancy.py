"""04_win_expectancy.py：建立含球數的勝率（WE）／得分期望（RE）模型、驗證、計算每顆判決球的改判收益。

目的
    挑戰的價值取決於「改判能讓勝率變多少」，這一步建立能查到每個球數的勝率表，並算出每顆判決球的 G。
研究步驟
    勝率模型（逆向歸納）與改判收益 G。
輸入
    data/raw/statcast/statcast_2026_*.parquet（00_fetch_statcast.py 下載）（fit、valid）
    data/processed/calls_2026.parquet（gain，03）
輸出
    --part fit    data/processed/pa_2026.parquet、we_model.pkl
    --part valid  outputs/tables/we_calibration.csv、re288_vs_statcast.csv
    --part gain   data/processed/calls_2026_G.parquet（05、06、07、09、10、12 使用）
執行
    python scripts/04_win_expectancy.py --part fit     # 約 2 秒
    python scripts/04_win_expectancy.py --part valid   # 約 2 秒
    python scripts/04_win_expectancy.py --part gain    # 約 2 秒

驗證內容
    (a) 打席開始的勝率 vs Statcast bat_win_exp
    (b) 對實際勝負的校準與 Brier（含樣本外：前半季建模 → 後半季）
    (c) RE288 差 vs Statcast delta_run_exp（非終結的判決球）
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import CALLS, CALLS_G, DM, PA, TAB, WE_MODEL  # noqa: E402
from src.data_io import dump_pickle, load_pickle, load_statcast  # noqa: E402
from src.we import COUNTS, TYPES, WEModel, build_pa, home_view, outcome_index, to_batting  # noqa: E402
from src.zone import bases_code  # noqa: E402

STATCAST_COLS = [
    "game_pk",
    "game_date",
    "at_bat_number",
    "pitch_number",
    "inning",
    "inning_topbot",
    "outs_when_up",
    "on_1b",
    "on_2b",
    "on_3b",
    "bat_score",
    "fld_score",
    "post_bat_score",
    "events",
    "balls",
    "strikes",
    "description",
    "delta_run_exp",
    "bat_win_exp",
    "home_score",
    "away_score",
    "post_home_score",
    "post_away_score",
]
#: 樣本外檢查的切點：此日之前的比賽建模，之後的比賽評估
OOS_SPLIT_DATE = "2026-07-01"


def load_sc() -> pd.DataFrame:
    """讀 Statcast 並依比賽、打席、逐球排序。"""
    sc = load_statcast(STATCAST_COLS)
    return sc.sort_values(["game_pk", "at_bat_number", "pitch_number"]).reset_index(drop=True)


def fit() -> None:
    """建立打席表與 WEModel，存成 pa_2026.parquet 與 we_model.pkl，並印出摘要。"""
    t = time.time()
    sc = load_sc()
    pa = build_pa(sc)
    pa.to_parquet(PA, index=False)
    m = WEModel(pa)
    m.table()
    m.re_table()
    m.pa = None  # 不把打席表存進 pickle（另有 pa_2026.parquet）
    dump_pickle(m, WE_MODEL)
    print(f"打席 {len(pa):,}，type 缺值 {pa.type.isna().sum()}，耗時 {time.time() - t:.0f}s")
    print(f"延長賽 W_x（客隊開局平手時主隊勝率）= {m.Wx:.4f}，不動點落差 {m.fixed_point_gap:.2e}")
    print(f"開賽主隊勝率 = {m.S_top[1, DM]:.4f}")
    print("各球數經過打席數：", m.n_count)
    print("P(type | count)：")
    print(pd.DataFrame(m.Ptype, index=[f"{b}-{s}" for b, s in COUNTS], columns=TYPES).round(3).to_string())
    print("RE24（上半）：")
    print(pd.DataFrame(m.RE24[0, :3], index=["0out", "1out", "2out"]).round(3).to_string())
    print(
        "稀疏轉移格（<20，已合併出局數）：",
        int(((m.n_trans > 0) & (m.n_trans < 20)).sum()),
        "，完全沒有資料的格：",
        int((m.n_trans == 0).sum()),
    )


def state_cols(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """由 Statcast 欄位取 (壘況代碼, 上下半 0/1, 主隊觀點比分差)。"""
    bases = bases_code(df)
    half = np.where(df.inning_topbot.eq("Top"), 0, 1)
    home_diff = np.where(half == 1, df.bat_score - df.fld_score, df.fld_score - df.bat_score)
    return bases.values, half, home_diff


def valid() -> None:
    """驗證勝率與得分期望模型，寫出 we_calibration.csv 與 re288_vs_statcast.csv。"""
    m = load_pickle(WE_MODEL)
    sc = load_sc()

    # 比賽勝負
    last = sc.groupby("game_pk").last()
    home_win = (last.post_home_score > last.post_away_score).rename("home_win")
    print("平手結束的比賽：", int((last.post_home_score == last.post_away_score).sum()))
    sc = sc.join(home_win, on="game_pk")
    bases, half, hd = state_cols(sc)
    k = (sc.balls.clip(upper=3) * 3 + sc.strikes.clip(upper=2)).astype(int).values
    we_home = m.lookup(sc.inning.values, half, sc.outs_when_up.values, bases, hd, k)
    sc["we_bat"] = np.where(half == 1, we_home, 1 - we_home)
    sc["bat_won"] = np.where(half == 1, sc.home_win, ~sc.home_win).astype(float)
    first = sc[sc.pitch_number == 1]
    r = np.corrcoef(first.we_bat, first.bat_win_exp)[0, 1]
    print(
        f"(a) 打席開始：本模型 vs Statcast bat_win_exp  r = {r:.4f}，"
        f"MAE = {np.abs(first.we_bat - first.bat_win_exp).mean():.4f}，"
        f"平均差 = {(first.we_bat - first.bat_win_exp).mean():+.4f}"
    )

    # 校準（每打席一筆，Brier）
    br_m = ((first.we_bat - first.bat_won) ** 2).mean()
    br_s = ((first.bat_win_exp - first.bat_won) ** 2).mean()
    print(
        f"(b) Brier（打席開始，越小越好）：本模型 {br_m:.5f}，Statcast {br_s:.5f}，"
        f"常數 0.5 = {((0.5 - first.bat_won) ** 2).mean():.5f}"
    )
    bins = np.linspace(0, 1, 11)
    cal = (
        first.assign(bin=pd.cut(first.we_bat, bins, include_lowest=True))
        .groupby("bin", observed=True)
        .agg(n=("bat_won", "size"), pred=("we_bat", "mean"), actual=("bat_won", "mean"))
    )
    print(cal.round(3).to_string())
    cal.to_csv(TAB / "we_calibration.csv")

    # 樣本外：前半季建模 → 後半季（只看打席開始、0-0 球數）
    pa = pd.read_parquet(PA)
    dates = sc.groupby("game_pk").game_date.first()
    early = set(dates[dates < OOS_SPLIT_DATE].index)
    m1 = WEModel(pa[pa.game_pk.isin(early)])
    m1.table()
    late = first[~first.game_pk.isin(early)]
    b2, h2, d2 = state_cols(late)
    k2 = np.zeros(len(late), int)
    w1 = m1.lookup(late.inning.values, h2, late.outs_when_up.values, b2, d2, k2)
    w1 = np.where(h2 == 1, w1, 1 - w1)
    print(
        f"樣本外（前半季建模、後半季評估，{len(late):,} 打席）Brier：本模型 {((w1 - late.bat_won) ** 2).mean():.5f}，"
        f"全季模型 {((late.we_bat - late.bat_won) ** 2).mean():.5f}，"
        f"Statcast {((late.bat_win_exp - late.bat_won) ** 2).mean():.5f}"
    )

    # (c) RE288：非終結的判決球（called strike／ball，該球不是打席最後一球）
    nxt_same = (sc.game_pk.shift(-1) == sc.game_pk) & (sc.at_bat_number.shift(-1) == sc.at_bat_number)
    called = sc.description.isin(["ball", "called_strike", "blocked_ball"]) & nxt_same
    c = sc[called].copy()
    bb, hh, _ = state_cols(c)
    k0 = (c.balls * 3 + c.strikes).astype(int).values
    kb, ks = outcome_index(c.balls.values, c.strikes.values)
    k1 = np.where(c.description.eq("called_strike"), ks, kb)

    # 下一球前壘況、出局、比分若改變（盜壘、暴投）則排除：那樣的 delta_run_exp 不只反映判決
    nx = sc.shift(-1)[called]
    same_state = (
        (nx.outs_when_up.values == c.outs_when_up.values)
        & (state_cols(nx)[0] == bb)
        & (nx.bat_score.values == c.bat_score.values)
    )
    pred = m.RE288[hh, c.outs_when_up.values, bb, k1] - m.RE288[hh, c.outs_when_up.values, bb, k0]
    dre = c.delta_run_exp.values.astype(float)
    ok = same_state & np.isfinite(dre) & c.description.isin(["ball", "called_strike"]).values
    rr = np.corrcoef(pred[ok], dre[ok])[0, 1]
    mae = np.abs(pred[ok] - dre[ok]).mean()
    print(
        f"(c) 判決球（ball/called_strike）ΔRE288：本模型 vs Statcast delta_run_exp：r = {rr:.4f}，"
        f"MAE = {mae:.4f}，n = {ok.sum():,}，平均（本模型）{pred[ok].mean():+.4f} vs {dre[ok].mean():+.4f}"
    )
    cc = pd.DataFrame(
        {
            "count": [f"{b}-{s}" for b, s in zip(c.balls, c.strikes, strict=False)],
            "call": c.description.values,
            "model": pred,
            "statcast": c.delta_run_exp.values,
        }
    )[ok]
    tab = (
        cc.groupby(["count", "call"])
        .agg(n=("model", "size"), model=("model", "mean"), statcast=("statcast", "mean"))
        .round(4)
    )
    print(tab.to_string())
    tab.to_csv(TAB / "re288_vs_statcast.csv")


def gain() -> None:
    """每顆判決球的改判收益 G（勝率）與 ΔRE（得分），寫出 calls_2026_G.parquet。"""
    m = load_pickle(WE_MODEL)
    df = pd.read_parquet(CALLS)
    half, hd = home_view(df)
    o, b = df.outs_when_up.values.astype(int), df.bases.values.astype(int)
    b0, s0 = df.balls_pre.values.astype(int), df.strikes_pre.values.astype(int)
    kb, ks = outcome_index(b0, s0)
    k0 = b0 * 3 + s0
    inn = df.inning.values.astype(int)
    wb = m.lookup(inn, half, o, b, hd, kb)
    ws = m.lookup(inn, half, o, b, hd, ks)
    w0 = m.lookup(inn, half, o, b, hd, k0)
    to_bat = to_batting(half)
    df["we_bat_pre"] = to_bat(w0)
    df["we_bat_if_ball"] = to_bat(wb)
    df["we_bat_if_strike"] = to_bat(ws)
    df["G_raw"] = df.we_bat_if_ball - df.we_bat_if_strike

    # 少數大比分差（落後 7 分以上、九局兩出局，WE < 0.0002）出現 |G| < 1e-4 的負值：
    # 來自各球數經驗 P(type) 的抽樣雜訊，對決策沒有影響 → 截在 0
    df["G"] = df.G_raw.clip(lower=0)
    df["dRE"] = m.RE288[half, o, b, kb] - m.RE288[half, o, b, ks]
    neg = (df.G_raw < -1e-9).sum()
    print(
        f"G_raw < 0 的球：{neg}（最小 {df.G_raw.min():.2e}，已截為 0），G 中位數 {df.G.median():.4f}，"
        f"平均 {df.G.mean():.4f}，P90 {df.G.quantile(0.9):.4f}，最大 {df.G.max():.3f}"
    )
    print(f"ΔRE 中位數 {df.dRE.median():.3f}，平均 {df.dRE.mean():.3f}")
    df.to_parquet(CALLS_G, index=False)
    print("→ calls_2026_G.parquet")


PARTS = {"fit": fit, "valid": valid, "gain": gain}

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="勝率／得分期望模型：建模、驗證、改判收益")
    ap.add_argument("--part", required=True, choices=list(PARTS))
    a = ap.parse_args()
    TAB.mkdir(parents=True, exist_ok=True)
    PARTS[a.part]()
