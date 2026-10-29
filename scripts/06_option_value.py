"""06_option_value.py：挑戰的選擇權價值 C 與最佳門檻 p*，各挑戰策略的勝場比較。

目的
    用逆向動態規劃算出「失去一次挑戰的代價」C 與最佳門檻 p* = C / (G + C)，
    並以 2026 實際比賽序列評估各策略能帶來多少勝場。
研究步驟
    動態規劃（近似政策迭代）、敏感度、以比賽為單位的 bootstrap。
輸入
    data/processed/calls_2026_G.parquet（04 gain）
    data/processed/perception.pkl（05）
    data/processed/opps_2026.parquet（05，只有 --sens 的 σ 放大縮小需要）
輸出
    預設        data/processed/policy_values_teamgame.parquet、nodes_opt.parquet、dp_hist.pkl
    --sens      outputs/tables/dp_sensitivity.csv
    --boot S N  outputs/tables/dp_boot_{S:03d}.csv（第 S 到 S+N−1 次，種子 1000 + b）
執行
    python scripts/06_option_value.py --iters 60       # 主結果，約 10 秒
    python scripts/06_option_value.py --sens            # 敏感度，約 45 秒
    python scripts/06_option_value.py --boot 0 25       # bootstrap 25 次，約 4 分鐘

政策（全部用同一套感知模型評估，單位：勝率，最後換算成每隊每季勝場）
    opt          最佳門檻，λ = 1（完全注意）
    opt_lam      最佳門檻，但維持觀察到的注意力 λ
    static       門檻不隨局數／比分／k 變動：C 固定為 opt 的節點平均
    savant       Savant 規則 p̃ > 0.2 / (0.2 + ΔRE)
    actual_model 05 的實際行為模型
    perfect      σ = 0（只挑真正判錯的球，成功就保留 → 上限）
    actual       實際：被改判成功的挑戰的 G 總和
演算法細節見 src/option_value.py。
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
    C_INIT_WP,
    DP_HIST,
    ITERS_DEFAULT,
    NODES_OPT,
    PERCEPTION,
    POLICY_VALUES,
    SEASON_GAMES,
    SEED_DP_BOOT,
    TAB,
)
from src.data_io import dump_pickle, load_pickle  # noqa: E402
from src.option_value import (  # noqa: E402
    Seq,
    build_sequences,
    evaluate,
    load_prior,
    modeled_actual_q,
    resample_games,
    run_policy,
    savant_pstar,
)

POLICY_ORDER = ["actual", "actual_model", "savant", "static", "opt_lam", "opt", "perfect"]
NODE_COLS = [
    "game_pk",
    "team_home",
    "seq",
    "inning",
    "batter",
    "fielder_2",
    "pitch_type",
    "plate_x",
    "plate_z",
    "team_bat",
    "outs_when_up",
    "d_team",
    "side",
    "edge",
    "G",
    "dRE",
    "m",
    "challenged",
    "overturned",
    "rem_eligible",
    "balls_pre",
    "strikes_pre",
    "bases",
]


def sensitivity(iters: int) -> None:
    """敏感度：σ × 0.5／1.5、κ = 0／200、只用 |m| ≤ 6 的節點，輸出 dp_sensitivity.csv。"""
    perc = load_pickle(PERCEPTION)
    prior = load_prior()
    df0 = build_sequences()
    rows = []
    # (名稱, 資料, σ 倍數, κ)，κ = None 表示用預設 50
    specs = [
        ("main", df0, 1.0, None),
        ("sigma×0.5", df0, 0.5, None),
        ("sigma×1.5", df0, 1.5, None),
        ("kappa=0", df0, 1.0, 0.0),
        ("kappa=200", df0, 1.0, 200.0),
        ("|m|≤6 only", df0[df0.m.abs() <= 6].copy(), 1.0, None),
    ]
    for name, d, ss, kp in specs:
        t = time.time()
        v = evaluate(d, perc, iters, ss, kp, prior=prior).mean() * SEASON_GAMES
        rows.append({"spec": name, **v.round(3).to_dict()})
        print(name, v.round(2).to_dict(), f"{time.time() - t:.0f}s", flush=True)
    pd.DataFrame(rows).to_csv(TAB / "dp_sensitivity.csv", index=False)


def bootstrap(start: int, n: int, iters: int = ITERS_DEFAULT) -> None:
    """以比賽為單位重抽，第 b 次用種子 SEED_DP_BOOT + b，輸出 dp_boot_{start:03d}.csv。"""
    perc = load_pickle(PERCEPTION)
    df0 = build_sequences()
    games = df0.game_pk.unique()
    by = {g: d for g, d in df0.groupby("game_pk")}
    out = []
    for b in range(start, start + n):
        rng = np.random.default_rng(SEED_DP_BOOT + b)
        d = resample_games(df0, by, games, rng)
        v = evaluate(d, perc, iters).mean() * SEASON_GAMES
        out.append({"b": b, **v.to_dict()})
        print(b, v.round(3).to_dict(), flush=True)
    fn = TAB / f"dp_boot_{start:03d}.csv"
    pd.DataFrame(out).to_csv(fn, index=False)


def main_result(iters: int) -> None:
    """主結果：各政策的每隊每場期望收益，並存節點層級的 C、p*、挑戰機率。"""
    t0 = time.time()
    perc = load_pickle(PERCEPTION)
    df = build_sequences()
    seq = Seq(df, perc)
    print(f"節點 {seq.n:,}，球隊-場次 {seq.ntg:,}，最長序列 {seq.L}，{time.time() - t0:.0f}s")
    init = np.where(seq.valid, C_INIT_WP, 0.0)
    res = {}
    for name, lam_on in (("opt", False), ("opt_lam", True)):
        print(f"[{name}]")
        r = run_policy(seq, init.copy(), init.copy(), lam_on=lam_on, iters=iters)
        res[name] = r
        print(f"  {time.time() - t0:.0f}s")
    # static：C 固定為 opt 的節點平均
    c1bar = res["opt"]["C1"][seq.valid].mean()
    c2bar = res["opt"]["C2"][seq.valid].mean()
    res["static"] = run_policy(seq, np.where(seq.valid, c1bar, 0), np.where(seq.valid, c2bar, 0))
    # savant：p* = 0.2/(0.2+ΔRE)，與 k 無關
    q = seq.qprob(savant_pstar(seq), False)
    F, Fn = seq.backward(q, q)
    res["savant"] = dict(F=F, Fn=Fn)
    # 模型化的實際行為（05 的結構式模型）
    q1a, q2a = modeled_actual_q(df, seq, perc)
    F, Fn = seq.backward(q1a, q2a)
    res["actual_model"] = dict(F=F, Fn=Fn)
    # perfect：σ = 0 → 只挑 m > 0，必定成功、不會失去挑戰
    perfect = np.where(seq.valid & (seq.m > 0), seq.G, 0).sum(1)
    actual = seq._mat(df.actual_gain.values).sum(1)
    games = df.groupby(["game_pk", "team_home"], sort=False).size()
    summ = {k: v["F"][2][:, 0] for k, v in res.items()}
    summ["perfect"] = perfect
    summ["actual"] = actual
    out = pd.DataFrame(summ)
    out[["game_pk", "team_home"]] = games.index.to_frame(index=False).values
    out.to_parquet(POLICY_VALUES, index=False)
    print(f"\n每隊每場期望勝率收益（×{SEASON_GAMES} = 每季勝場）：")
    for k in POLICY_ORDER:
        print(f"  {k:8s} 每場 {out[k].mean():.5f}  → 每季 {out[k].mean() * SEASON_GAMES:.2f} 勝")
    # 存節點層級 C、p*（opt）
    o = res["opt"]
    node = df[NODE_COLS].copy()
    node["C1"], node["C2"] = seq.flat(o["C1"]), seq.flat(o["C2"])
    node["V0n"], node["V1n"], node["V2n"] = (seq.flat(o["Fn"][k]) for k in range(3))
    node["q1"], node["q2"] = seq.flat(o["q1"]), seq.flat(o["q2"])
    node["q1_act"], node["q2_act"] = seq.flat(q1a), seq.flat(q2a)
    node.to_parquet(NODES_OPT, index=False)
    dump_pickle({"hist": {k: v.get("hist") for k, v in res.items()}}, DP_HIST)
    print(f"完成 {time.time() - t0:.0f}s")


def main() -> None:
    """解析參數並執行主結果、敏感度或 bootstrap。"""
    ap = argparse.ArgumentParser(description="挑戰的選擇權價值與最佳門檻")
    ap.add_argument("--iters", type=int, default=ITERS_DEFAULT, help="政策迭代最多幾次（主結果用 60）")
    ap.add_argument("--sens", action="store_true", help="敏感度分析")
    ap.add_argument("--boot", type=int, nargs=2, metavar=("START", "N"), help="以比賽為單位的 bootstrap")
    a = ap.parse_args()
    if a.sens:
        return sensitivity(a.iters)
    if a.boot:
        return bootstrap(*a.boot, iters=a.iters)
    return main_result(a.iters)


if __name__ == "__main__":
    main()
