"""07_analysis.py：門檻卡、實際門檻 vs 最佳門檻（H2）、前向傳遞、各隊勝場（H3）。

目的
    把 06 的最佳策略整理成可用的門檻卡，並和 05 的實際行為比較：哪裡太保守、哪裡太積極、損失多少勝場。
研究步驟
    結果整理（假設 H1–H3 的檢驗）。
輸入
    data/processed/calls_2026_G.parquet（04）、perception.pkl（05）
    data/processed/policy_values_teamgame.parquet（06，teams）
    outputs/tables/dp_boot_*.csv（06 --boot，teams）
輸出
    --part card     outputs/tables/card_pstar.csv、card_inches.csv、card_C_by_inning.csv、card_C_by_score_inn78.csv
    --part h2       outputs/tables/h2_cells.csv、h2_coarse.csv、h2_by_count.csv，data/processed/h2_nodes.parquet
    --part forward  outputs/tables/h2b_forward.csv
    --part teams    outputs/tables/h3_teams.csv、h3_bootstrap_summary.csv
執行
    python scripts/07_analysis.py --part card       # 每個 part 約 1–7 秒
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import (  # noqa: E402
    C_INIT_WP,
    CALLS_G,
    COUNT_GROUP,
    EDGES,
    H2_NODES,
    IG_LAB,
    ITERS_MAIN,
    N_TEAM_BOOT,
    PERCEPTION,
    POLICY_VALUES,
    SAVANT_COST_RUNS,
    SEASON_GAMES,
    SEED_TEAM_BOOT,
    TAB,
)
from src.data_io import load_pickle  # noqa: E402
from src.option_value import Seq, build_sequences, cutoff_inch, forward_pass, modeled_actual_q, run_policy  # noqa: E402
from src.perception import SideParams, count_dummies, inning_group, log_g  # noqa: E402
from src.zone import edge_index  # noqa: E402

POLICIES = ["actual", "actual_model", "savant", "static", "opt_lam", "opt", "perfect"]
#: 比分接近程度（該隊觀點比分差的絕對值）
CLOSE_LABELS = ("close(±1)", "mid(2-3)", "blowout(≥4)")


def setup() -> tuple[dict, pd.DataFrame, Seq, dict, tuple[np.ndarray, np.ndarray]]:
    """共用前置：建序列、跑最佳政策（60 次迭代），並在節點表加上 p*、球數分組等欄位。"""
    perc = load_pickle(PERCEPTION)
    df = build_sequences()
    seq = Seq(df, perc)
    init = np.where(seq.valid, C_INIT_WP, 0.0)
    opt = run_policy(seq, init.copy(), init.copy(), iters=ITERS_MAIN, verbose=False)
    q1a, q2a = modeled_actual_q(df, seq, perc)
    df["ig"] = inning_group(df.inning)
    df["cnt"] = df.balls_pre.astype(int).astype(str) + "-" + df.strikes_pre.astype(int).astype(str)
    df["cgrp"] = df.cnt.map(COUNT_GROUP)
    df["C1"], df["C2"] = seq.flat(opt["C1"]), seq.flat(opt["C2"])
    # 最佳門檻 p* = C / (G + C)（1e-12 避免 0/0）
    df["ps1"] = df.C1 / (df.G + df.C1 + 1e-12)
    df["ps2"] = df.C2 / (df.G + df.C2 + 1e-12)
    absd = df.d_team.abs()
    df["close"] = np.where(absd <= 1, CLOSE_LABELS[0], np.where(absd >= 4, CLOSE_LABELS[2], CLOSE_LABELS[1]))
    return perc, df, seq, opt, (q1a, q2a)


def card() -> None:
    """門檻卡：各球數分組 × 局數區段的中位數 p*，以及換算成感知英吋的門檻。"""
    perc, df, seq, opt, _ = setup()
    df["cut1"] = cutoff_inch(seq, df.ps1.values)
    df["cut2"] = cutoff_inch(seq, df.ps2.values)
    rows = []
    for k in (2, 1):
        for close in ["all", CLOSE_LABELS[0], CLOSE_LABELS[2]]:
            d = df if close == "all" else df[df.close == close]
            t = d.groupby(["cgrp", "ig"])[f"ps{k}"].median().unstack()[IG_LAB]
            t.insert(0, "k", k)
            t.insert(1, "score", close)
            rows.append(t.reset_index())
    pst = pd.concat(rows)
    pst.to_csv(TAB / "card_pstar.csv", index=False)
    # 感知英吋：捕手（守備）與打者（打擊）
    rows = []
    for side in ["def", "off"]:
        for k in (2, 1):
            d = df[df.side == side]
            t = d.groupby(["cgrp", "ig"])[f"cut{k}"].median().unstack()[IG_LAB]
            t.insert(0, "side", side)
            t.insert(1, "k", k)
            rows.append(t.reset_index())
    inch = pd.concat(rows)
    inch.to_csv(TAB / "card_inches.csv", index=False)
    # C 與 Savant 固定 0.2 分（換算勝率：0.2 × 該狀態每分的勝率價值 ≈ 0.2 × G/ΔRE）
    df["wp_per_run"] = df.G / df.dRE.clip(lower=1e-6)
    cc = (
        df.groupby("ig")
        .agg(
            C1=("C1", "mean"),
            C2=("C2", "mean"),
            savant_wp=("wp_per_run", lambda s: SAVANT_COST_RUNS * s.median()),
            n=("G", "size"),
        )
        .loc[IG_LAB]
    )
    cc.to_csv(TAB / "card_C_by_inning.csv")
    cd = df[df.inning.between(7, 8)].groupby(df.d_team.clip(-5, 5)).agg(C1=("C1", "mean"), C2=("C2", "mean"))
    cd.to_csv(TAB / "card_C_by_score_inn78.csv")
    pd.set_option("display.width", 220)
    print(pst[pst.score == "all"].round(2).to_string(index=False))
    print(pst[pst.score != "all"].round(2).to_string(index=False))
    print(inch.round(1).to_string(index=False))
    print((cc * [1000, 1000, 1000, 1]).round(2))
    # H1（描述）：p* 對局數、k 的單調性
    m = df.groupby(["ig"])[["ps1", "ps2"]].median().loc[IG_LAB]
    print("p* 中位數：", m.round(3).to_dict())


def actual_thresholds(df: pd.DataFrame, perc: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """05 模型的實際門檻 c（英吋，含球員效果）、注意力 λ、感知誤差 σ（每節點，用實際剩餘次數 k）。"""
    c_act = np.full(len(df), np.nan)
    lam_v = np.full(len(df), np.nan)
    sig_v = np.full(len(df), np.nan)
    player = np.where(df.side.eq("off"), df.batter, df.fielder_2)
    Z = count_dummies(df)
    lg = log_g(df.G).values
    kk = df.rem_eligible.clip(lower=1, upper=2).astype(int).values
    for s_ in ("off", "def"):
        P = SideParams.from_pickle(perc[s_])
        msk = df.side.eq(s_).values
        ci = P.cell_index(df.ig.values[msk], kk[msk])
        a = P.player_effect(player[msk])
        # 注意：ci = −1（狀態格不存在）時 beta[−1] 會取到最後一格，目前 10 個格都存在所以不會發生
        c_act[msk] = P.beta[ci] + P.gam * (lg[msk] - P.lg_mean) + Z[msk] @ P.delta + a
        lam_v[msk] = P.lam
        sig_v[msk] = P.sigma[edge_index(df.edge[msk])]
    return c_act, lam_v, sig_v


def h2() -> None:
    """實際門檻 vs 最佳門檻，一步損失分解（門檻：挑太少／挑太多，注意力）。"""
    perc, df, seq, opt, (q1a, q2a) = setup()
    # 只看「實際上還有挑戰」的機會，用實際 k
    d = df[(df.rem_eligible > 0) & (df.m.abs() <= 6)].copy()
    k = d.rem_eligible.clip(upper=2).values
    d["k"] = k
    d["pstar"] = np.where(k == 2, d.ps2, d.ps1)
    d["C"] = np.where(k == 2, d.C2, d.C1)
    # 最佳門檻（感知英吋）
    idx = d.index.values
    ps_all = np.where(df.rem_eligible.clip(upper=2) == 2, df.ps2, df.ps1)
    cut_all = cutoff_inch(seq, ps_all)
    d["c_opt"] = cut_all[idx]
    # 實際門檻：05 模型（含球員效果）
    c_act, lam_v, sig_v = actual_thresholds(df, perc)
    d["c_act"], d["lam"], d["sig"] = c_act[idx], lam_v[idx], sig_v[idx]
    # 實際門檻換算成主觀機率 τ
    tau = np.zeros(len(d))
    for s_ in ("off", "def"):
        for e in EDGES:
            msk = (d.side.values == s_) & (d.edge.values == e)
            grid, post = perc[s_]["post"][e]
            tau[msk] = np.interp(d.c_act.values[msk], grid, np.maximum.accumulate(post))
    d["tau"] = tau
    # 一步損失分解（單位：勝率，正 = 實際比最佳少拿）
    # 挑戰的實現價值以真實 m 計：判錯 → +G，判對 → 失去一次挑戰 −C
    v = np.where(d.m > 0, d.G, -d.C)
    P_opt = 1 - norm.cdf((d.c_opt - d.m) / d.sig)
    P_opt = np.where(np.isinf(d.c_opt), np.where(d.c_opt > 0, 0.0, 1.0), P_opt)
    P_thr = 1 - norm.cdf((d.c_act - d.m) / d.sig)  # 實際門檻、完全注意
    P_act = d.lam * P_thr
    d["loss_thr"] = v * (P_opt - P_thr)  # 門檻造成的損失
    d["loss_att"] = v * (P_thr - P_act)  # 注意力造成的損失
    d["dir"] = np.where(d.c_act > d.c_opt, "too_conservative", "too_aggressive")
    d["obs_chal"] = d.challenged.astype(float)
    d["pred_chal"] = P_act
    tg_n = df.groupby(["game_pk", "team_home"]).ngroups
    per_season = SEASON_GAMES / tg_n  # 總和 → 每隊每季
    summ = (
        d.groupby(["side", "ig", "k", "cgrp"])
        .agg(
            n=("G", "size"),
            chal=("obs_chal", "sum"),
            pstar=("pstar", "median"),
            tau=("tau", "median"),
            c_opt=("c_opt", "median"),
            c_act=("c_act", "median"),
            share_conservative=("dir", lambda s: (s == "too_conservative").mean()),
            loss_thr=("loss_thr", "sum"),
            loss_att=("loss_att", "sum"),
        )
        .reset_index()
    )
    summ["loss_thr_wins_team_season"] = summ.loss_thr * per_season
    summ["loss_att_wins_team_season"] = summ.loss_att * per_season
    summ.to_csv(TAB / "h2_cells.csv", index=False)
    coarse = (
        d.groupby(["side", "ig", "k"])
        .agg(
            n=("G", "size"),
            pstar=("pstar", "median"),
            tau=("tau", "median"),
            share_conservative=("dir", lambda s: (s == "too_conservative").mean()),
            loss_thr=("loss_thr", "sum"),
            loss_att=("loss_att", "sum"),
        )
        .reset_index()
    )
    coarse["loss_thr_w"] = coarse.loss_thr * per_season
    coarse["loss_att_w"] = coarse.loss_att * per_season
    coarse.to_csv(TAB / "h2_coarse.csv", index=False)
    # 方向分解
    dirs = d.groupby(["side", "dir"]).loss_thr.sum() * per_season
    tot = pd.DataFrame(
        {
            "threshold": d.groupby("side").loss_thr.sum() * per_season,
            "attention": d.groupby("side").loss_att.sum() * per_season,
        }
    )
    pd.set_option("display.width", 220)
    print(coarse.round(3).to_string(index=False))
    print("\n門檻損失依方向（每隊每季勝場）：\n", dirs.round(3))
    print("\n總損失（每隊每季勝場）：\n", tot.round(3), "\n合計", round(tot.values.sum(), 3))
    by_cg = (
        d.groupby(["side", "cgrp"])
        .agg(pstar=("pstar", "median"), tau=("tau", "median"), loss_thr=("loss_thr", "sum"))
        .reset_index()
    )
    by_cg["loss_thr_w"] = by_cg.loss_thr * per_season
    by_cg.to_csv(TAB / "h2_by_count.csv", index=False)
    print(by_cg.round(3).to_string(index=False))
    keep = [
        "game_pk",
        "team_home",
        "seq",
        "side",
        "ig",
        "k",
        "cgrp",
        "m",
        "G",
        "C",
        "pstar",
        "tau",
        "c_opt",
        "c_act",
        "loss_thr",
        "loss_att",
        "challenged",
        "overturned",
    ]
    d[keep].to_parquet(H2_NODES, index=False)


def forward() -> None:
    """前向機率傳遞：最佳政策與實際行為模型下，各局的期望挑戰次數、成功率 vs 實際。"""
    perc, df, seq, opt, (q1a, q2a) = setup()
    out = []
    for name, (q1, q2) in {"opt": (opt["q1"], opt["q2"]), "actual_model": (q1a, q2a)}.items():
        ch, wn = forward_pass(seq, q1, q2)
        df[f"ch_{name}"], df[f"win_{name}"] = seq.flat(ch), seq.flat(wn)
    df["ch_actual"] = df.challenged.astype(float)
    df["win_actual"] = (df.challenged & df.overturned.astype("boolean").fillna(False)).astype(float)
    ntg = df.groupby(["game_pk", "team_home"]).ngroups
    pols = ["actual", "actual_model", "opt"]
    for key in ["ig", "side", "cgrp"]:
        g = df.groupby(key)
        t = pd.DataFrame({f"{p}_chal_per_game": g[f"ch_{p}"].sum() / ntg for p in pols})
        for p in pols:
            t[f"{p}_success"] = g[f"win_{p}"].sum() / g[f"ch_{p}"].sum()
        t.insert(0, "by", key)
        out.append(t.reset_index().rename(columns={key: "group"}))
    tot = pd.DataFrame({f"{p}_chal_per_game": [df[f"ch_{p}"].sum() / ntg] for p in pols})
    for p in pols:
        tot[f"{p}_success"] = df[f"win_{p}"].sum() / df[f"ch_{p}"].sum()
    tot.insert(0, "by", "all")
    tot.insert(1, "group", "all")
    res = pd.concat([tot] + out, ignore_index=True)
    res.to_csv(TAB / "h2b_forward.csv", index=False)
    pd.set_option("display.width", 220)
    print(res.round(3).to_string(index=False))


def teams() -> None:
    """各隊實際 vs 最佳（每季勝場）＋隊內 bootstrap 區間，彙整 06 的 DP bootstrap。"""
    pv = pd.read_parquet(POLICY_VALUES)
    calls = pd.read_parquet(CALLS_G, columns=["game_pk", "home_team", "away_team"]).drop_duplicates("game_pk")
    pv = pv.merge(calls, on="game_pk")
    pv["team"] = np.where(pv.team_home.astype(bool), pv.home_team, pv.away_team)
    g = pv.groupby("team")
    n = g.size()
    t = g[POLICIES].sum().div(n, axis=0) * SEASON_GAMES
    t["gap_opt_actual"] = t.opt - t.actual
    # 隊內 bootstrap：以該隊比賽重抽，單一亂數產生器依隊名順序連續使用
    rng = np.random.default_rng(SEED_TEAM_BOOT)
    lo, hi = [], []
    for _team, d in pv.groupby("team"):
        x = (d.opt - d.actual).values
        bs = [rng.choice(x, len(x)).mean() * SEASON_GAMES for _ in range(N_TEAM_BOOT)]
        lo.append(np.percentile(bs, 2.5))
        hi.append(np.percentile(bs, 97.5))
    t["gap_lo"], t["gap_hi"] = lo, hi
    t["games"] = n
    t = t.sort_values("gap_opt_actual")
    t.to_csv(TAB / "h3_teams.csv")
    pd.set_option("display.width", 220)
    print(t.round(2).to_string())
    print("隊間 gap SD", round(t.gap_opt_actual.std(), 2), "，實際勝場 SD", round(t.actual.std(), 2))
    # DP bootstrap 彙整（06 --boot 的全部檔案）
    bs = pd.concat(pd.read_csv(f) for f in sorted(TAB.glob("dp_boot_*.csv")))
    bs["opt-actual"] = bs.opt - bs.actual
    bs["opt-actual_model"] = bs.opt - bs.actual_model
    bs["static-actual"] = bs.static - bs.actual
    bs["opt-static"] = bs.opt - bs.static
    bs["savant-actual"] = bs.savant - bs.actual
    bs["perfect-opt"] = bs.perfect - bs.opt
    cols = POLICIES + ["opt-actual", "opt-actual_model", "static-actual", "opt-static", "savant-actual", "perfect-opt"]
    s = pd.DataFrame({"mean": bs[cols].mean(), "lo": bs[cols].quantile(0.025), "hi": bs[cols].quantile(0.975)})
    s["B"] = len(bs)
    s.to_csv(TAB / "h3_bootstrap_summary.csv")
    print(s.round(3))


PARTS = {"card": card, "h2": h2, "forward": forward, "teams": teams}

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="門檻卡、H2 損失分解、前向傳遞、各隊勝場")
    ap.add_argument("--part", required=True, choices=list(PARTS))
    a = ap.parse_args()
    PARTS[a.part]()
