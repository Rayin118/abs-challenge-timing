"""13_slide_extras.py：簡報後期新增的分析與舉例數字（第 16、22、23、26、27 頁等）。

研究步驟
    在 06（最佳政策）與 05（感知模型）之後執行，只讀既有的中間資料，不改動任何模型。
輸入
    data/processed/nodes_opt.parquet（06：每個決策點的 G、C1、C2）
    data/processed/perception.pkl（05：主觀機率對照表）
輸出（outputs/tables/）
    slide_examples.csv          第 16、22 頁的舉例球（G、C、p*）與第 9 局 3-2 的例子
    card_pstar_by_bases.csv     第 23 頁：依壘況拆開的門檻卡（還有 2 次）
    r1_context_tables.csv       第 23 頁：出局數、比分的門檻表（一般球數）
    f3_slide_data.csv           第 21 頁：各局失敗代價 C1、C2 與 Savant 換算值
    pstar_importance.csv        第 27 頁：各因素對最佳門檻的重要性（拿掉後 R² 下降多少）
    pstar_marginals.csv         第 27 頁：各因素的門檻中位數（所有情境）
    rule_accuracy.csv           第 28 頁：挑戰口訣和最佳門檻分級的吻合度
    perc_posterior_examples.csv 第 20、24 頁：看到差 1／2／3 英吋時，主審真的判錯的機率
執行
    python scripts/13_slide_extras.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import COUNT_GROUP, EDGES, IG_EDGES, IG_LAB, NODES_OPT, PERCEPTION, TAB  # noqa: E402
from src.data_io import ensure_dirs, load_pickle  # noqa: E402

SAVANT_COST_RUNS = 0.2  # Savant 公式的固定失敗代價（分）
CG_SHORT = {"a": "一般", "b": "打者有利", "c": "兩好球", "d": "2-2", "e": "3-2"}
BG_ORDER = ["壘上無人", "一壘有人", "得點圈有人", "滿壘"]


def load_nodes() -> pd.DataFrame:
    """讀取決策點，加上 p*、局數區段、球數分組、壘況分組、比分分組。"""
    cols = ["inning", "team_home", "outs_when_up", "bases", "d_team", "balls_pre", "strikes_pre",
            "G", "C1", "C2", "dRE", "side"]
    d = pd.read_parquet(NODES_OPT, columns=cols)
    d["ps1"] = d.C1 / (d.G + d.C1 + 1e-12)
    d["ps2"] = d.C2 / (d.G + d.C2 + 1e-12)
    d["ig"] = pd.cut(d.inning, IG_EDGES, labels=IG_LAB).astype(str)
    cnt = d.balls_pre.astype(int).astype(str) + "-" + d.strikes_pre.astype(int).astype(str)
    d["cg"] = cnt.map(COUNT_GROUP).str[0].map(CG_SHORT)
    # bases：一壘 = 1、二壘 = 2、三壘 = 4 的位元和
    d["bg"] = np.select([d.bases == 0, d.bases == 1, d.bases == 7], BG_ORDER[:2] + ["滿壘"], "得點圈有人")
    a = d.d_team.abs()
    d["sc"] = np.select([a <= 1, a <= 3], ["差 1 分以內", "差 2–3 分"], "差 4 分以上")
    d["outs"] = d.outs_when_up.astype(int).astype(str) + " 出局"
    return d


def examples(d: pd.DataFrame) -> pd.DataFrame:
    """第 16、22 頁的舉例球，以及第 22 頁第 9 局 3-2 的例子（取中位數）。"""
    rows = []
    m = (d.team_home & (d.inning == 5) & (d.outs_when_up == 0) & (d.bases == 0) & (d.d_team == 0)
         & (d.balls_pre == 1) & (d.strikes_pre == 1))
    x = d[m]
    rows.append({"例子": "5 局下平手、0 出局無人、1-1（第 16、22 頁）", "n": len(x),
                 "G": x.G.median(), "C2": x.C2.median(), "C1": x.C1.median()})
    y = d[(d.inning == 9) & (d.balls_pre == 3) & (d.strikes_pre == 2)]
    rows.append({"例子": "第 9 局、3-2（第 22 頁）", "n": len(y), "G": y.G.median(), "C2": y.C2.median(),
                 "C1": y.C1.median()})
    out = pd.DataFrame(rows)
    for k in ("C2", "C1"):
        out[f"p*_{k}"] = out[k] / (out.G + out[k])
    return out


def r1_context(d: pd.DataFrame) -> pd.DataFrame:
    """第 23 頁的表：還有 2 次、一般球數下，壘況、出局（第 1–3 局）與比分（各局）的門檻中位數。"""
    g = d[d.cg == "一般"]
    rows = []
    for k, v in g[g.ig == "1-3"].groupby("bg").ps2.median().items():
        rows.append({"表": "壘況（1–3 局）", "列": k, "欄": "", "p*": v})
    for k, v in g[g.ig == "1-3"].groupby("outs").ps2.median().items():
        rows.append({"表": "出局數（1–3 局）", "列": k, "欄": "", "p*": v})
    t = g[g.sc != "差 2–3 分"].groupby(["sc", "ig"]).ps2.median()
    for (sc, ig), v in t.items():
        rows.append({"表": "比分（各局）", "列": sc, "欄": ig, "p*": v})
    return pd.DataFrame(rows)


def cost_by_inning(d: pd.DataFrame) -> pd.DataFrame:
    """第 21 頁的圖：各局平均 C1、C2，以及 Savant 0.2 分換算成勝率（中位數）。"""
    n = d.assign(inn=d.inning.clip(upper=10), wp_per_run=d.G / d.dRE.clip(lower=1e-6))
    g = n.groupby("inn").agg(C1=("C1", "mean"), C2=("C2", "mean"),
                             sav=("wp_per_run", lambda s: SAVANT_COST_RUNS * s.median()))
    return (g * 100).reset_index()  # 單位：勝率百分點


def importance(d: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    """第 27 頁：把 logit(p*) 對 7 個因素做迴歸，逐一拿掉某個因素，看 R² 下降多少。

    剩 1 次、還有 2 次兩種狀態疊在一起（剩餘次數也是一個因素）；比分的效果在後段才明顯，
    所以另外放「局數 × 比分」交互作用，拿掉局數或比分時一併拿掉。
    """
    lg = lambda p: np.log(np.clip(p, 1e-4, 1 - 1e-4) / (1 - np.clip(p, 1e-4, 1 - 1e-4)))  # noqa: E731
    s = pd.concat([d.assign(k="還有 2 次", y=lg(d.ps2.values)), d.assign(k="只剩 1 次", y=lg(d.ps1.values))],
                  ignore_index=True)
    names = {"cg": "球數", "ig": "局數", "k": "剩餘挑戰次數", "bg": "壘上跑者", "sc": "比分", "outs": "出局數",
             "side": "攻守（捕手／打者）"}
    base = pd.get_dummies(s[list(names)], drop_first=True).astype(float)
    base.insert(0, "const", 1.0)
    inter = pd.get_dummies(s.ig + "|" + s.sc, drop_first=True).astype(float)
    y = s.y.values

    def r2(X: pd.DataFrame) -> float:
        b, *_ = np.linalg.lstsq(X.values, y, rcond=None)
        e = y - X.values @ b
        return 1 - e.var() / y.var()

    full = r2(pd.concat([base, inter], axis=1))
    rows = []
    for f, zh in names.items():
        keep = base[[c for c in base.columns if not c.startswith(f + "_")]]
        X = pd.concat([keep] + ([inter] if f not in ("ig", "sc") else []), axis=1)
        rows.append({"因素": zh, "R2下降": full - r2(X)})
    return pd.DataFrame(rows).sort_values("R2下降", ascending=False), full


def marginals(d: pd.DataFrame) -> pd.DataFrame:
    """第 27 頁的策略結論數字：所有情境下，各因素各類別的門檻中位數（還有 2 次）。"""
    rows = []
    for f, zh in [("cg", "球數"), ("ig", "局數"), ("bg", "壘況"), ("outs", "出局"), ("side", "攻守")]:
        for k, v in d.groupby(f).ps2.median().items():
            rows.append({"因素": zh, "類別": k, "p*": v})
    for (ig, sc), v in d.groupby(["ig", "sc"]).ps2.median().items():
        rows.append({"因素": f"比分（{ig} 局）", "類別": sc, "p*": v})
    rows.append({"因素": "剩餘次數", "類別": "還有 2 次", "p*": d.ps2.median()})
    rows.append({"因素": "剩餘次數", "類別": "只剩 1 次", "p*": d.ps1.median()})
    return pd.DataFrame(rows)


def rule_accuracy(d: pd.DataFrame) -> pd.DataFrame:
    """第 28 頁：用三級（保守 ≥ 40%、中立 20–40%、積極 < 20%）比較口訣和每個決策點的最佳門檻。

    口訣：第 1–6 局保守、第 7–8 局中立、第 9 局以後積極；兩好球、三壞球、滿壘時放寬一級；
    只剩 1 次時用前一段的標準（嚴一級）。
    """
    level = lambda p: np.select([p >= 0.40, p >= 0.20], [2, 1], 0)  # noqa: E731
    base = np.select([d.inning <= 6, d.inning <= 8], [2, 1], 0)
    loosen = (d.strikes_pre.eq(2) | d.balls_pre.eq(3) | d.bases.eq(7)).values
    rows = []
    for k, col, shift in (("還有 2 次", "ps2", 0), ("只剩 1 次", "ps1", 1)):
        t = level(d[col].values)
        for name, r in (("只看局數", base + shift), ("口訣（局數＋例外）", base - loosen + shift)):
            r = np.clip(r, 0, 2)
            rows.append({"剩餘次數": k, "規則": name, "吻合": np.mean(r == t),
                         "比最佳保守": np.mean(r > t), "比最佳積極": np.mean(r < t)})
    return pd.DataFrame(rows)


def posterior_examples() -> pd.DataFrame:
    """看到差 1、2、3 英吋時，主審真的判錯的機率（貝氏，依好球帶的邊）。"""
    p = load_pickle(PERCEPTION)
    rows = []
    for s, zh in (("def", "捕手（守備方）"), ("off", "打者")):
        for e in EDGES:
            grid, post = p[s]["post"][e]
            post = np.maximum.accumulate(post)
            rows.append({"方": zh, "邊": e, **{f"看到差{x}英吋": float(np.interp(x, grid, post)) for x in (1, 2, 3)}})
    return pd.DataFrame(rows)


def main() -> None:
    ensure_dirs()
    pd.set_option("display.width", 200)
    d = load_nodes()
    ex = examples(d)
    ex.to_csv(TAB / "slide_examples.csv", index=False)
    card = d.groupby(["bg", "cg", "ig"]).ps2.agg(median="median", size="size").reset_index()
    card.to_csv(TAB / "card_pstar_by_bases.csv", index=False)
    r1_context(d).to_csv(TAB / "r1_context_tables.csv", index=False)
    cost_by_inning(d).to_csv(TAB / "f3_slide_data.csv", index=False)
    imp, full = importance(d)
    imp.assign(全模型R2=full).to_csv(TAB / "pstar_importance.csv", index=False)
    marginals(d).to_csv(TAB / "pstar_marginals.csv", index=False)
    rule_accuracy(d).to_csv(TAB / "rule_accuracy.csv", index=False)
    posterior_examples().to_csv(TAB / "perc_posterior_examples.csv", index=False)
    print(ex.round(4).to_string(index=False))
    print(imp.round(3).to_string(index=False), "\n全模型 R² =", round(full, 3))


if __name__ == "__main__":
    main()
