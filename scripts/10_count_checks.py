"""10_count_checks.py：勝率表「球數層」的三項穩健性檢查。

目的
    勝率模型假設「同一個球數，不論怎麼走到，後續價值都一樣」，而且結果機率只依球數。
    這一步檢查這兩個假設，以及斷點迴歸的效果是否依球數不同。
研究步驟
    穩健性檢查（球數層）。
輸入
    data/processed/we_model.pkl、pa_2026.parquet（04 fit）、calls_2026_G.parquet（04 gain）、perception.pkl（05）
    data/raw/statcast/statcast_2026_*.parquet（00_fetch_statcast.py 下載）（path：逐球球數）
輸出
    --part path   outputs/tables/cc_path.csv
    --part state  outputs/tables/cc_state_G.csv、cc_state_policy.csv
    --part rd     outputs/tables/cc_rd_by_count.csv
執行
    python scripts/10_count_checks.py --part path    # 約 4 秒
    python scripts/10_count_checks.py --part state   # 約 17 秒
    python scripts/10_count_checks.py --part rd      # 約 5 秒

各項內容
    path   路徑相依：同一個球數，從不同前一球數走來，後續價值是否不同？
           比較 ΔRE（壞球結果 − 好球結果）的「路徑版」與「馬可夫版」，以比賽 bootstrap（500 次）
    state  結果機率改成依壘況分組：壘上無人／有人但一壘空／一壘有人。重建勝率表 → 新 G → 重跑最佳策略
    rd     斷點迴歸依球數分組：終結球數（兩好球或三壞球，改判 = 三振或保送）vs 其他球數
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import (  # noqa: E402
    CALLS_G,
    ITERS_MAIN,
    N_PATH_BOOT,
    PA,
    PERCEPTION,
    SEASON_GAMES,
    SEED_PATH_BOOT,
    TAB,
    WE_MODEL,
)
from src.data_io import load_pickle, load_statcast  # noqa: E402
from src.option_value import build_sequences, evaluate  # noqa: E402
from src.rd import build_rd_sample, rd, triangular_weights  # noqa: E402
from src.we import COUNTS, TYPES, WEModel, home_view, outcome_index, to_batting  # noqa: E402

#: 逐球球數欄位與型別（與當初存成 data/xfer/pitch_counts.parquet 的版本相同）
PITCH_COUNT_DTYPES = {
    "game_pk": "int32",
    "at_bat_number": "int16",
    "pitch_number": "int16",
    "balls": "int8",
    "strikes": "int8",
}
BASE_GROUP_LABELS = ["壘上無人", "有人一壘空", "一壘有人"]


def load_pitch_counts() -> pd.DataFrame:
    """Statcast 逐球的球數（含自動好壞球列），型別縮成整數。

    原版讀 data/xfer/pitch_counts.parquet，那是從 Statcast 取這 5 欄另存的暫存檔，沒有程式產生它。
    這裡改成直接從 Statcast 讀同樣的欄位與型別，內容逐列相同（已核對），流程不再依賴傳檔資料夾。
    """
    return load_statcast(list(PITCH_COUNT_DTYPES)).astype(PITCH_COUNT_DTYPES)


# ---------------------------------------------------------------- 1. 路徑相依
def part_path(B: int = N_PATH_BOOT) -> None:
    """路徑版 vs 馬可夫版的 ΔRE（以 RE24 計的打席價值 Y），比賽 bootstrap 區間。"""
    m = load_pickle(WE_MODEL)
    pa = pd.read_parquet(PA)
    pa = pa[pa.type.notna() & ~pa.short_game_last_half & ~((pa.half == 1) & (pa.inning >= 9))].copy()
    # 打席價值 Y = 本打席得分 + RE24(之後) − RE24(之前)，三出局的 RE24 = 0
    re24 = np.zeros((2, 4, 8))
    re24[:, :3, :] = m.RE24[:, :3, :]
    pa["Y"] = pa.r_pa + re24[pa.half, pa.outs1, np.where(pa.outs1 < 3, pa.bases1, 0)] - re24[pa.half, pa.outs, pa.bases]
    pc = load_pitch_counts().sort_values(["game_pk", "at_bat_number", "pitch_number"])
    pc["c"] = pc.balls.clip(upper=3).astype(int).astype(str) + "-" + pc.strikes.clip(upper=2).astype(int).astype(str)
    pc = pc[(pc.c != pc.groupby(["game_pk", "at_bat_number"]).c.shift(1))]  # 去掉連續重複（兩好球界外）
    pc["prev"] = pc.groupby(["game_pk", "at_bat_number"]).c.shift(1)
    pc = pc.merge(pa[["game_pk", "at_bat_number", "Y", "type"]], on=["game_pk", "at_bat_number"])
    # 終結結果作為「目的地」：最後一個球數之後若打席以 K／BB 結束，加一筆 last→K/BB 的轉移
    last = pc.groupby(["game_pk", "at_bat_number"]).tail(1)
    term = last[last.type.isin(["K", "BB"])].copy()
    term = term[term.type.eq("BB") & term.c.str.startswith("3") | term.type.eq("K") & term.c.str.endswith("2")]
    term = term.assign(prev=term.c, c=term.type)
    tr = pd.concat([pc[pc.prev.notna()][["game_pk", "prev", "c", "Y"]], term[["game_pk", "prev", "c", "Y"]]])
    visit = pd.concat([pc[["game_pk", "c", "Y"]], term[["game_pk", "c", "Y"]]])
    games = np.sort(pa.game_pk.unique())
    gi = {g: i for i, g in enumerate(games)}
    rows = []
    # bootstrap 以多項分配權重實作：W[b, g] = 第 b 次抽到比賽 g 的次數
    rng = np.random.default_rng(SEED_PATH_BOOT)
    W = rng.multinomial(len(games), np.ones(len(games)) / len(games), size=B).astype(float)

    def gsum(df: pd.DataFrame, mask: pd.Series) -> tuple[np.ndarray, np.ndarray]:
        """依比賽加總 Y 與筆數（長度 = 比賽數，供權重矩陣相乘）。"""
        s = df[mask].groupby("game_pk").Y.agg(["sum", "size"])
        S = np.zeros(len(games))
        N = np.zeros(len(games))
        idx = s.index.map(gi).values
        S[idx] = s["sum"].values
        N[idx] = s["size"].values
        return S, N

    for b, s_ in COUNTS:
        c = f"{b}-{s_}"
        db = "BB" if b == 3 else f"{b + 1}-{s_}"
        ds = "K" if s_ == 2 else f"{b}-{s_ + 1}"
        # 馬可夫版：只看目的球數，路徑版：只看「從 c 走到」目的球數的打席
        Sb_m, Nb_m = gsum(visit, visit.c.eq(db))
        Ss_m, Ns_m = gsum(visit, visit.c.eq(ds))
        Sb_p, Nb_p = gsum(tr, tr.prev.eq(c) & tr.c.eq(db))
        Ss_p, Ns_p = gsum(tr, tr.prev.eq(c) & tr.c.eq(ds))
        g_m = Sb_m.sum() / Nb_m.sum() - Ss_m.sum() / Ns_m.sum()
        g_p = Sb_p.sum() / Nb_p.sum() - Ss_p.sum() / Ns_p.sum()
        bm = (W @ Sb_m) / (W @ Nb_m) - (W @ Ss_m) / (W @ Ns_m)
        bp = (W @ Sb_p) / (W @ Nb_p) - (W @ Ss_p) / (W @ Ns_p)
        dd = bp - bm
        rows.append(
            {
                "count": c,
                "ball_dest": db,
                "strike_dest": ds,
                "n_ball_via": int(Nb_p.sum()),
                "n_strike_via": int(Ns_p.sum()),
                "dRE_markov": g_m,
                "dRE_path": g_p,
                "diff": g_p - g_m,
                "diff_lo": np.percentile(dd, 2.5),
                "diff_hi": np.percentile(dd, 97.5),
                "rel_diff": (g_p - g_m) / g_m,
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(TAB / "cc_path.csv", index=False)
    pd.set_option("display.width", 200)
    print(out.round(4).to_string(index=False))
    # 以判決球的實際分布加權的平均相對差
    calls = pd.read_parquet(CALLS_G, columns=["balls_pre", "strikes_pre", "merge_ok"])
    w = (
        calls[calls.merge_ok]
        .assign(count=lambda d: d.balls_pre.astype(str) + "-" + d.strikes_pre.astype(str))["count"]
        .value_counts(normalize=True)
    )
    o = out.set_index("count")
    print(
        f"判決球加權：馬可夫 ΔRE {np.sum(w * o.dRE_markov):.4f}，路徑 ΔRE {np.sum(w * o.dRE_path):.4f}，"
        f"加權相對差 {np.sum(w * o['diff']) / np.sum(w * o.dRE_markov):+.3f}"
    )


# ---------------------------------------------------------------- 2. 壘況分組
def grp_of(si: int) -> int:
    """狀態索引 → 壘況組：0 = 壘上無人，1 = 有人但一壘空，2 = 一壘有人。"""
    b = si % 8
    return 0 if b == 0 else (2 if b & 1 else 1)


class WEState(WEModel):
    """P(type | 球數, 壘況組) 版本的勝率模型：0 = 壘上無人，1 = 有人但一壘空，2 = 一壘有人。"""

    def __init__(self, pa: pd.DataFrame) -> None:
        self._pa_for_groups = pa
        super().__init__(pa)

    def _ptype(self, ci: int, si: int) -> np.ndarray:
        """覆寫：結果類型分布改依壘況組估計（第一次呼叫時計算並快取）。"""
        if not hasattr(self, "Ptype_g"):
            pa = self._pa_for_groups
            da = pa[~pa.short_game_last_half & pa.type.notna()]
            g = np.where(da.bases == 0, 0, np.where(da.bases % 2 == 1, 2, 1))
            tidx = da.type.map({t: i for i, t in enumerate(TYPES)}).values
            self.Ptype_g = np.zeros((3, 12, len(TYPES)))
            for gg in range(3):
                for c in range(12):
                    msk = (g == gg) & (((da.cmask.values >> c) & 1) == 1)
                    self.Ptype_g[gg, c] = np.bincount(tidx[msk], minlength=len(TYPES)) / msk.sum()
        pt = self.Ptype_g[grp_of(si), ci] * self.valid[:, si]
        return pt / pt.sum()


def part_state() -> None:
    """壘況分組的勝率表 → 新 G → 與原 G 比較，並重跑最佳策略。"""
    pa = pd.read_parquet(PA)
    m1 = WEState(pa)
    m1.table()
    m1.re_table()
    # 各組在關鍵球數的結果機率差
    rows = []
    for c in ["0-0", "2-0", "3-0", "3-1", "3-2", "0-2", "2-2"]:
        b, s = map(int, c.split("-"))
        ci = b * 3 + s
        for gg, lab in enumerate(BASE_GROUP_LABELS):
            p = m1.Ptype_g[gg, ci]
            rows.append({"count": c, "group": lab, "K": p[0], "BB": p[1], "HR": p[6], "OUT": p[7]})
    print(pd.DataFrame(rows).round(3).to_string(index=False))
    # 新 G
    calls = pd.read_parquet(CALLS_G)
    half, hd = home_view(calls)
    o, b = calls.outs_when_up.values.astype(int), calls.bases.values.astype(int)
    kb, ks = outcome_index(calls.balls_pre.values.astype(int), calls.strikes_pre.values.astype(int))
    inn = calls.inning.values.astype(int)
    to_bat = to_batting(half)
    G1 = (to_bat(m1.lookup(inn, half, o, b, hd, kb)) - to_bat(m1.lookup(inn, half, o, b, hd, ks))).clip(min=0)
    calls["G_state"] = G1
    calls["grp"] = np.where(calls.bases == 0, "壘上無人", np.where(calls.bases % 2 == 1, "一壘有人", "有人一壘空"))
    calls["cnt"] = calls.balls_pre.astype(str) + "-" + calls.strikes_pre.astype(str)
    ok = calls.merge_ok
    r = np.corrcoef(calls.G[ok], calls.G_state[ok])[0, 1]
    print(
        f"\nG 的相關 r = {r:.4f}，平均 {calls.G[ok].mean():.5f} → {calls.G_state[ok].mean():.5f}，"
        f"平均絕對差 {np.abs(calls.G - calls.G_state)[ok].mean():.5f}"
    )
    t = calls[ok].groupby(["grp", "cnt"]).agg(G=("G", "mean"), G_state=("G_state", "mean"), n=("G", "size"))
    t["rel"] = t.G_state / t.G - 1
    t.to_csv(TAB / "cc_state_G.csv")
    print(t.round(4).sort_values("rel").to_string())
    # 重跑最佳策略：兩種 G 各評估一次
    perc = load_pickle(PERCEPTION)
    df = build_sequences()
    key = ["game_pk", "at_bat_number", "pitch_number"]
    df = df.merge(calls[key + ["G_state"]], on=key, how="left")
    res = {}
    for lab, gcol in (("原模型", "G"), ("壘況分組", "G_state")):
        d = df.copy()
        d["G"] = d[gcol]
        d["actual_gain"] = np.where(d.challenged & d.overturned.astype("boolean").fillna(False), d.G, 0.0)
        v = evaluate(d, perc, iters=ITERS_MAIN).mean() * SEASON_GAMES
        res[lab] = v
        print(lab, v.round(3).to_dict())
    out = pd.DataFrame(res).T
    out["opt-actual"] = out.opt - out.actual
    out.to_csv(TAB / "cc_state_policy.csv")
    print(out.round(3).to_string())


# ---------------------------------------------------------------- 3. 斷點迴歸分球數
def part_rd() -> None:
    """斷點迴歸依球數分組（終結球數 vs 其他，3-2 vs 非 3-2）。"""
    warnings.filterwarnings("ignore")  # 與 09 相同：略過 rdrobust 的提示性警告
    d = build_rd_sample()
    d["term"] = np.where((d.strikes_pre == 2) | (d.balls_pre == 3), "終結球數（兩好或三壞）", "其他球數")
    d["full"] = np.where((d.strikes_pre == 2) & (d.balls_pre == 3), "3-2", "非 3-2")
    rows = []
    for key in ["term", "full"]:
        for g, s in d.groupby(key):
            for y, pred in (("Y_pa", "pred_G"), ("Y_runs", "pred_dRE")):
                r = rd(s, y)
                w = triangular_weights(s.m, r["h"])
                r["model_pred"] = float(np.sum(s[pred] * w) / np.sum(w))
                r["group"] = g
                r["n_total"] = len(s)
                rows.append(r)
    out = pd.DataFrame(rows)
    out.to_csv(TAB / "cc_rd_by_count.csv", index=False)
    pd.set_option("display.width", 200)
    cols = ["group", "y", "n_total", "h", "n_eff", "coef", "rb_lo", "rb_hi", "rb_p", "model_pred"]
    print(out[cols].round(4).to_string(index=False))


PARTS = {"path": part_path, "state": part_state, "rd": part_rd}

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="球數層的三項檢查")
    ap.add_argument("--part", required=True, choices=list(PARTS))
    a = ap.parse_args()
    PARTS[a.part]()
