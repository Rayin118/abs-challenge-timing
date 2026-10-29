"""12_savant_check.py：與 Baseball Savant ABS 挑戰排行榜（打者層級 CSV）核對。

目的
    確認本研究重建的挑戰紀錄、改判數與 Runs Gained，和官方 Savant 排行榜一致。
研究步驟
    外部核對。
輸入
    data/processed/calls_2026_G.parquet（04 gain）
    data/ext/savant_batter.csv（Savant 手動下載的 abs-challenges-2026-batter.csv）
    data/ext/player_names.csv（由 feed JSON 整理的 球員 ID → 姓名）
輸出
    outputs/tables/savant_compare_batter.csv（逐人比對）、savant_compare_summary.csv（加總與相關）
執行
    python scripts/12_savant_check.py           # 約 2 秒

比對：打者自己的挑戰（打擊方）與「對該打者」的挑戰（守備方），逐人比對次數、改判數、
Runs Gained（= 改判成功的 ΔRE 加總）。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import CALLS_G, EXT, TAB  # noqa: E402

SAVANT_COLS = [
    "team_abbr",
    "n_challenges",
    "n_overturns",
    "runs_gained",
    "n_challenges_against",
    "n_overturns_against",
    "runs_gained_against",
]
#: (Savant 欄位, 本研究欄位)
METRIC_PAIRS = [
    ("n_challenges", "o_n"),
    ("n_overturns", "o_ov"),
    ("runs_gained", "o_rg"),
    ("n_challenges_against", "d_n"),
    ("n_overturns_against", "d_ov"),
    ("runs_gained_against", "d_rg"),
]


def per_batter(d: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """依打者姓名彙總：挑戰次數、改判數、Runs Gained（改判成功的 ΔRE 加總）。"""
    return d.groupby("name").apply(
        lambda g: pd.Series({f"{prefix}n": len(g), f"{prefix}ov": g.ov.sum(), f"{prefix}rg": g.dRE[g.ov].sum()})
    )


def main() -> None:
    """逐人比對並輸出摘要。"""
    c = pd.read_parquet(CALLS_G)
    sv = pd.read_csv(EXT / "savant_batter.csv")
    nm = pd.read_csv(EXT / "player_names.csv").set_index("id").name
    # 注意：這裡沒有篩 merge_ok（目前被挑戰的球全部 merge_ok，不影響結果）
    ch = c[c.challenged].copy()
    ch["ov"] = ch.overturned.astype(bool)
    ch["name"] = ch.batter.map(nm)
    off, de = ch[ch.orig_strike], ch[~ch.orig_strike]
    m = sv.set_index("entity_name")[SAVANT_COLS].join(per_batter(off, "o_")).join(per_batter(de, "d_")).fillna(0)
    m.to_csv(TAB / "savant_compare_batter.csv")
    rows = []
    for a, b in METRIC_PAIRS:
        # 一致的定義：次數完全相同（差 < 0.5），Runs Gained 差 < 0.05
        tol = 0.05 if "runs" in a else 0.5
        rows.append(
            {
                "metric": a,
                "savant_sum": m[a].sum(),
                "ours_sum": m[b].sum(),
                "r": np.corrcoef(m[a], m[b])[0, 1],
                "share_exact_or_within_0.05": (np.abs(m[a] - m[b]) < tol).mean(),
            }
        )
    o = ch[ch.ov]
    rows += [
        {
            "metric": "batter K erased（打者挑掉三振）",
            "savant_sum": sv.n_strikeouts_flip.sum(),
            "ours_sum": ((o.orig_strike) & (o.strikes_pre == 2)).sum(),
        },
        {
            "metric": "batter BB created（打者挑出保送）",
            "savant_sum": sv.n_walks_flip.sum(),
            "ours_sum": ((o.orig_strike) & (o.balls_pre == 3)).sum(),
        },
        {
            "metric": "runs_gained per overturn（打擊）",
            "savant_sum": sv.runs_gained.sum() / sv.n_overturns.sum(),
            "ours_sum": off.dRE[off.ov].mean(),
        },
        {
            "metric": "runs_gained per overturn（守備）",
            "savant_sum": sv.runs_gained_against.sum() / sv.n_overturns_against.sum(),
            "ours_sum": de.dRE[de.ov].mean(),
        },
        {
            "metric": "runs_cost per failed challenge（打擊）",
            "savant_sum": sv.runs_cost.sum() / sv.n_confirms.sum(),
            "ours_sum": np.nan,
        },
        {
            "metric": "runs_cost per failed challenge（守備）",
            "savant_sum": sv.runs_cost_against.sum() / sv.n_confirms_against.sum(),
            "ours_sum": np.nan,
        },
    ]
    s = pd.DataFrame(rows)
    s.to_csv(TAB / "savant_compare_summary.csv", index=False)
    pd.set_option("display.width", 200)
    print(s.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
