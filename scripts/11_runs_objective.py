"""11_runs_objective.py：如果球隊改用「最大化得失分差期望值」來決定何時挑戰，會少贏多少？

目的
    Savant 與多數球隊用「得分」衡量挑戰價值。這一步比較「以得失分差為目標」與「以勝率為目標」的最佳策略，
    看看只用得分能拿到多少好處、差距出現在哪些局數與比分。
研究步驟
    延伸分析（目標函數的比較）。
輸入
    data/processed/calls_2026_G.parquet（04）、perception.pkl（05）
    outputs/tables/ro_boot_*.csv（只有 --summary 需要）
輸出
    預設        outputs/tables/ro_summary.csv、ro_decomp.csv、ro_by_inning.csv、ro_by_score.csv、ro_by_side.csv
    --boot S N  outputs/tables/ro_boot_{S:03d}.csv（第 S 到 S+N−1 次，種子 5000 + b）
    --summary   outputs/tables/ro_boot_summary.csv（彙整全部 ro_boot_*.csv，08 f10 使用）
執行
    python scripts/11_runs_objective.py              # 約 7 秒
    python scripts/11_runs_objective.py --boot 0 50  # 約 10 分鐘
    python scripts/11_runs_objective.py --summary    # 不到 1 秒

注意：「得失分差版」同時包含打擊與守備的挑戰：打擊時改判 = 本隊多得 ΔRE 分，守備時改判 = 對手少得 ΔRE 分。
兩者放在同一支球隊的序列、共用同一組挑戰次數。

做法
    得分版最佳策略：把每顆球的收益換成 ΔRE（RE288 差，單位：分），C 也以分為單位，
                    且 C 的狀態格不含比分差（得分目標本來就不看比分），其他與 06 完全相同。
    交叉評估：得分版策略的挑戰決定 → 用勝率評估（每季勝場），勝率版策略 → 用得分評估（每季分數）。
    拆解：前向傳遞下，每個節點的期望勝率收益 = Σ_k P(k) q(k) 1(m>0) G，依局數 × 比分差加總。
    不確定性：以比賽 bootstrap（--boot）重估兩種策略，得到「勝率版 − 得分版」的區間。
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import (  # noqa: E402
    C_INIT_RUNS,
    C_INIT_WP,
    IG_LAB,
    ITERS_MAIN,
    PERCEPTION,
    SEASON_GAMES,
    SEED_RO_BOOT,
    TAB,
)
from src.data_io import load_pickle  # noqa: E402
from src.option_value import Seq, build_sequences, forward_pass, resample_games, run_policy  # noqa: E402
from src.perception import inning_group  # noqa: E402

SCORE_BINS = [-1, 0, 1, 3, 99]
SCORE_LABELS = ["平手", "差 1 分", "差 2–3 分", "差 4 分以上"]
#: ro_boot_summary.csv 的列順序
BOOT_COLS = [
    "wp_policy_in_wins",
    "run_policy_in_wins",
    "wp_policy_in_runs",
    "run_policy_in_runs",
    "actual_in_wins",
    "actual_in_runs",
    "diff_w",
    "diff_r",
    "runopt_minus_actual",
    "share",
]


def run_df(df: pd.DataFrame) -> pd.DataFrame:
    """得分版的序列：收益改成 ΔRE，比分差設為 0（C 的狀態格不看比分）。"""
    d = df.copy()
    d["G_wp"] = d.G
    d["G"] = d.dRE
    d["d_team"] = 0  # C 的狀態格不含比分
    return d


def evaluate(df: pd.DataFrame, perc: dict, iters: int = ITERS_MAIN) -> tuple:
    """勝率版與得分版各自求最佳策略，並交叉評估。回傳 (每隊-場次結果, seq_wp, seq_run, wp, rn)。"""
    seq_wp = Seq(df, perc)
    seq_run = Seq(run_df(df), perc)
    init = np.where(seq_wp.valid, C_INIT_WP, 0.0)
    wp = run_policy(seq_wp, init.copy(), init.copy(), iters=iters, verbose=False)
    initr = np.where(seq_run.valid, C_INIT_RUNS, 0.0)
    rn = run_policy(seq_run, initr.copy(), initr.copy(), iters=iters, verbose=False)
    out = {
        "wp_policy_in_wins": wp["F"][2][:, 0],
        "run_policy_in_wins": seq_wp.backward(rn["q1"], rn["q2"])[0][2][:, 0],
        "wp_policy_in_runs": seq_run.backward(wp["q1"], wp["q2"])[0][2][:, 0],
        "run_policy_in_runs": rn["F"][2][:, 0],
        "actual_in_wins": seq_wp._mat(df.actual_gain.values).sum(1),
        "actual_in_runs": seq_run._mat(np.where(df.actual_gain > 0, df.dRE, 0.0)).sum(1),
    }
    return pd.DataFrame(out), seq_wp, seq_run, wp, rn


def main_part() -> None:
    """主結果：兩種目標的勝場與分數、差距拆解、打擊／守備拆解。"""
    t = time.time()
    perc = load_pickle(PERCEPTION)
    df = build_sequences()
    v, seq_wp, seq_run, wp, rn = evaluate(df, perc)
    s = v.mean() * SEASON_GAMES
    print("每隊每季：")
    print(f"  勝率版策略：{s.wp_policy_in_wins:.3f} 勝 ／ {s.wp_policy_in_runs:.2f} 分")
    print(f"  得失分差版：{s.run_policy_in_wins:.3f} 勝 ／ {s.run_policy_in_runs:.2f} 分")
    print(f"  實際：      {s.actual_in_wins:.3f} 勝 ／ {s.actual_in_runs:.2f} 分")
    print(
        f"  勝率版 − 得失分差版：{s.wp_policy_in_wins - s.run_policy_in_wins:+.3f} 勝，"
        f"{s.wp_policy_in_runs - s.run_policy_in_runs:+.2f} 分"
    )
    pd.DataFrame({"value": s}).to_csv(TAB / "ro_summary.csv")
    # 得分版的 C 與 p*
    df["C1_run"], df["C2_run"] = seq_run.flat(rn["C1"]), seq_run.flat(rn["C2"])
    df["ig"] = inning_group(df.inning)
    print("得失分差版 C（分）依局數：", df.groupby("ig")[["C1_run", "C2_run"]].mean().loc[IG_LAB].round(3).to_dict())
    # 拆解：前向傳遞的期望勝率收益
    for name, (q1, q2) in {"wp": (wp["q1"], wp["q2"]), "run": (rn["q1"], rn["q2"])}.items():
        ch, wn = forward_pass(seq_wp, q1, q2)
        df[f"ch_{name}"] = seq_wp.flat(ch)
        df[f"gain_{name}"] = seq_wp.flat(wn) * df.G
    df["score"] = pd.cut(df.d_team.abs(), SCORE_BINS, labels=SCORE_LABELS).astype(str)
    ntg = df.groupby(["game_pk", "team_home"]).ngroups
    tab = (
        df.groupby(["ig", "score"])
        .agg(
            chal_wp=("ch_wp", "sum"),
            chal_run=("ch_run", "sum"),
            gain_wp=("gain_wp", "sum"),
            gain_run=("gain_run", "sum"),
            n=("G", "size"),
        )
        .reset_index()
    )
    for c in ["chal_wp", "chal_run"]:
        tab[c] = tab[c] / ntg
    for c in ["gain_wp", "gain_run"]:
        tab[c] = tab[c] / ntg * SEASON_GAMES
    tab["diff_wins"] = tab.gain_wp - tab.gain_run
    tab.to_csv(TAB / "ro_decomp.csv", index=False)
    pd.set_option("display.width", 200)
    print(tab.round(4).to_string(index=False))
    print("拆解加總（勝）：", round(tab.diff_wins.sum(), 3))
    by_ig = tab.groupby("ig")[["chal_wp", "chal_run", "diff_wins"]].sum().loc[IG_LAB]
    by_sc = tab.groupby("score")[["chal_wp", "chal_run", "diff_wins"]].sum()
    print(by_ig.round(3))
    print(by_sc.round(3))
    by_ig.to_csv(TAB / "ro_by_inning.csv")
    by_sc.to_csv(TAB / "ro_by_score.csv")
    # 依打擊／守備拆解（每隊每季：挑戰次數、勝場、分數）
    rows = []
    act = df.actual_gain > 0
    sides = ((True, "打擊（本隊多得分）"), (False, "守備（對手少得分）"))
    for name, (q1, q2) in {"勝率版": (wp["q1"], wp["q2"]), "得失分差版": (rn["q1"], rn["q2"])}.items():
        ch, wn = forward_pass(seq_wp, q1, q2)
        w, c = seq_wp.flat(wn), seq_wp.flat(ch)
        for side, lab in sides:
            m = df.team_bat.values == side
            rows.append(
                {
                    "策略": name,
                    "方面": lab,
                    "每場挑戰": c[m].sum() / ntg,
                    "每季勝場": (w * df.G.values)[m].sum() / ntg * SEASON_GAMES,
                    "每季分數": (w * df.dRE.values)[m].sum() / ntg * SEASON_GAMES,
                }
            )
    for side, lab in sides:
        m = df.team_bat.values == side
        rows.append(
            {
                "策略": "實際",
                "方面": lab,
                "每場挑戰": df.challenged[m].sum() / ntg,
                "每季勝場": df.actual_gain[m].sum() / ntg * SEASON_GAMES,
                "每季分數": df.dRE[m & act].sum() / ntg * SEASON_GAMES,
            }
        )
    side_tab = pd.DataFrame(rows)
    side_tab.to_csv(TAB / "ro_by_side.csv", index=False)
    print(side_tab.round(3).to_string(index=False))
    print(f"{time.time() - t:.0f}s")


def boot(start: int, n: int) -> None:
    """以比賽為單位重抽，第 b 次用種子 SEED_RO_BOOT + b，輸出 ro_boot_{start:03d}.csv。"""
    perc = load_pickle(PERCEPTION)
    df0 = build_sequences()
    games = df0.game_pk.unique()
    by = {g: d for g, d in df0.groupby("game_pk")}
    rows = []
    for b in range(start, start + n):
        rng = np.random.default_rng(SEED_RO_BOOT + b)
        d = resample_games(df0, by, games, rng)
        v = evaluate(d, perc)[0].mean() * SEASON_GAMES
        rows.append({"b": b, **v.to_dict()})
        print(b, round(v.wp_policy_in_wins - v.run_policy_in_wins, 4), flush=True)
    pd.DataFrame(rows).to_csv(TAB / f"ro_boot_{start:03d}.csv", index=False)


def summary() -> None:
    """彙整全部 ro_boot_*.csv：各量的平均與 2.5%／97.5% 分位數，輸出 ro_boot_summary.csv。

    原本這個檔是手動彙整的，沒有對應程式，這裡補上，結果與既有檔案的差距 < 1e-14。
    share = (得分版 − 實際) / (勝率版 − 實際)：得分版拿到的改進占勝率版的比例（逐次計算再平均）。
    """
    bs = pd.concat(pd.read_csv(f) for f in sorted(TAB.glob("ro_boot_[0-9]*.csv")))
    bs["diff_w"] = bs.wp_policy_in_wins - bs.run_policy_in_wins
    bs["diff_r"] = bs.wp_policy_in_runs - bs.run_policy_in_runs
    bs["runopt_minus_actual"] = bs.run_policy_in_wins - bs.actual_in_wins
    bs["share"] = (bs.run_policy_in_wins - bs.actual_in_wins) / (bs.wp_policy_in_wins - bs.actual_in_wins)
    s = pd.DataFrame(
        {"mean": bs[BOOT_COLS].mean(), "lo": bs[BOOT_COLS].quantile(0.025), "hi": bs[BOOT_COLS].quantile(0.975)}
    )
    s.to_csv(TAB / "ro_boot_summary.csv")
    print(f"B = {len(bs)}")
    print(s.round(4).to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="得失分差目標 vs 勝率目標")
    ap.add_argument("--boot", type=int, nargs=2, metavar=("START", "N"), help="以比賽為單位的 bootstrap")
    ap.add_argument("--summary", action="store_true", help="彙整 ro_boot_*.csv → ro_boot_summary.csv")
    a = ap.parse_args()
    if a.summary:
        summary()
    elif a.boot:
        boot(*a.boot)
    else:
        main_part()
