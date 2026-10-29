"""08_figures.py：報告用靜態圖（PNG，淺色）。

目的
    把 05–11 的結果畫成簡報與報告用的圖。配色：藍 = 捕手／實際，橘 = 打者／最佳。
研究步驟
    結果呈現。
輸入
    data/processed/opps_2026.parquet（05）、nodes_opt.parquet（06）、rd_sample.parquet（09）
    outputs/tables/perc_fit_bins.csv、card_pstar.csv、h2b_forward.csv、h2_coarse.csv、h3_bootstrap_summary.csv、
    h3_teams.csv、rd_main.csv、ro_summary.csv、ro_boot_summary.csv、ro_by_inning.csv
輸出
    outputs/figures/abs_f1_perception.png、abs_f3_option_value.png、abs_f4_card.png、abs_f5_actual_vs_opt.png、
    abs_f6_thresholds.png、abs_f7_policies.png、abs_f8_teams.png、abs_f9_rd.png、abs_f10_runs_vs_wp.png
執行
    python scripts/08_figures.py f1_perception f4_card   # 只畫指定的圖
    python scripts/08_figures.py                         # 不給參數 = 全部（約 3 秒）

注意：圖上的標題與註解文字有些是寫死的（例如 f5 的「36%」、f9 的樣本數），資料更新後要人工確認。
中文字型：會載入系統中的 Noto Sans CJK（Regular／Medium），沒有時退回 sans-serif。
"""

from __future__ import annotations

import sys
from collections.abc import Callable
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib import font_manager as fm  # noqa: E402
from matplotlib.colors import BoundaryNorm, ListedColormap  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import FIG, IG_LAB, IG_LAB_ZH, NODES_OPT, OPPS, RD_SAMPLE, SAVANT_COST_RUNS, TAB  # noqa: E402

FIG.mkdir(parents=True, exist_ok=True)
for f in fm.findSystemFonts():
    if "NotoSansCJK" in f:
        try:
            fm.fontManager.addfont(f)
        except Exception:  # noqa: BLE001
            pass
plt.rcParams.update(
    {
        "font.family": ["Noto Sans CJK TC", "PingFang TC", "Heiti TC", "Noto Sans CJK JP",
                        "Microsoft JhengHei", "Arial Unicode MS", "sans-serif"],
        "font.size": 10.5,
        "figure.facecolor": "#fcfcfb",
        "axes.facecolor": "#fcfcfb",
        "savefig.facecolor": "#fcfcfb",
        "axes.edgecolor": "#c3c2b7",
        "axes.labelcolor": "#52514e",
        "xtick.color": "#52514e",
        "ytick.color": "#52514e",
        "axes.grid": True,
        "grid.color": "#e1e0d9",
        "grid.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.titlecolor": "#0b0b0b",
        "axes.titlelocation": "left",
        "legend.frameon": False,
        "text.color": "#0b0b0b",
        "axes.unicode_minus": False,
    }
)
BLUE, ORANGE, AQUA, VIOLET, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#6250d6", "#898781"
SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
IG = IG_LAB  # 局數區段代碼（表格的欄名／索引）
IG_TICKS = IG_LAB_ZH  # 圖上的中文標籤


def save(fig: plt.Figure, name: str) -> None:
    """存成 outputs/figures/{name}.png（200 dpi）並關閉圖。"""
    fig.savefig(FIG / f"{name}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("→", name)


def f1_perception() -> None:
    """f1：真實誤判幅度 vs 被挑戰比例（實際分箱與 05 模型預測），捕手與打者分開。"""
    o = pd.read_parquet(OPPS, columns=["sideflag", "m", "challenged"])
    fit = pd.read_csv(TAB / "perc_fit_bins.csv")
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    bins = np.arange(-4, 4.01, 0.5)
    for side, col, lab in (("def", BLUE, "捕手（守備方）"), ("off", ORANGE, "打者")):
        d = o[(o.sideflag == side) & o.m.between(-4, 4)]
        g = d.groupby(pd.cut(d.m, bins), observed=True).challenged.agg(["mean", "size"])
        mid = np.array([iv.mid for iv in g.index])
        se = np.sqrt(g["mean"] * (1 - g["mean"]) / g["size"])
        ax.errorbar(
            mid,
            g["mean"] * 100,
            yerr=1.96 * se * 100,
            fmt="o",
            ms=5,
            color=col,
            lw=1.2,
            capsize=0,
            label=f"{lab}：實際",
        )
        f = fit[(fit.side == side) & (fit.by == "mb")].copy()
        f["mid"] = f.group.str.extract(r"\(([-\d.]+), ([-\d.]+)\]").astype(float).mean(axis=1)
        f = f[f.mid.between(-4, 4)].sort_values("mid")
        ax.plot(f.mid, f.pred * 100, color=col, lw=2, alpha=0.85, label=f"{lab}：模型")
    ax.axvline(0, color="#52514e", lw=1, ls="--")
    ax.text(0.1, 66, "← 原判正確　｜　原判錯誤 →", color="#52514e", fontsize=9.5)
    ax.set_xlabel("真實誤判幅度 m（英吋；> 0 表示主審判錯）")
    ax.set_ylabel("被挑戰的比例（%）")
    ax.set_title("就算主審判錯 2 英吋，捕手也只挑戰約一半，打者約三成")
    ax.set_ylim(0, 70)
    ax.legend(loc="upper left", fontsize=9)
    save(fig, "abs_f1_perception")


def f3_option_value() -> None:
    """f3：挑戰失敗的代價 C（剩 1／2 次）隨局數變化，對照 Savant 固定 0.2 分換算的勝率。"""
    n = pd.read_parquet(NODES_OPT, columns=["inning", "C1", "C2", "G", "dRE"])
    n["inn"] = n.inning.clip(upper=10)
    n["wp_per_run"] = n.G / n.dRE.clip(lower=1e-6)
    g = n.groupby("inn").agg(
        C1=("C1", "mean"), C2=("C2", "mean"), sav=("wp_per_run", lambda s: SAVANT_COST_RUNS * s.median())
    )
    x = np.arange(1, 11)
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    ax.plot(x, g.sav * 100, color=GRAY, lw=2, ls="--", marker="s", ms=5, label="Savant：固定 0.2 分（換算勝率）")
    ax.plot(x, g.C1 * 100, color=BLUE, lw=2, marker="o", ms=6, label="本研究：只剩 1 次時")
    ax.plot(x, g.C2 * 100, color=ORANGE, lw=2, marker="o", ms=6, label="本研究：還有 2 次時")
    ax.set_xticks(x, [str(i) for i in range(1, 10)] + ["延長"])
    ax.set_xlabel("局數")
    ax.set_ylabel("挑戰失敗的代價（勝率百分點）")
    ax.set_title("挑戰失敗的代價隨比賽進行而下降，第 9 局只剩第 1 局的約 1/5")
    ax.set_ylim(0, None)
    ax.legend(fontsize=9)
    save(fig, "abs_f3_option_value")


def f4_card() -> None:
    """f4：門檻卡熱圖（球數分組 × 局數區段的中位數 p*），剩 2 次與剩 1 次並排。"""
    c = pd.read_csv(TAB / "card_pstar.csv")
    c = c[c.score == "all"]
    rows = ["a 0-0/1-0/0-1/1-1", "b 2-0/2-1/3-0/3-1", "c 0-2/1-2", "d 2-2", "e 3-2"]
    rlab = ["0-0、1-0、0-1、1-1", "2-0、2-1、3-0、3-1", "0-2、1-2", "2-2", "3-2"]
    cmap = ListedColormap(SEQ)
    norm = BoundaryNorm([0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7], cmap.N)
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.6), sharey=True)
    for ax, k, tt in ((axes[0], 2, "還有 2 次挑戰"), (axes[1], 1, "只剩 1 次挑戰")):
        m = c[c.k == k].set_index("cgrp").loc[rows, IG].values
        ax.imshow(m, cmap=cmap, norm=norm, aspect="auto")
        for i in range(m.shape[0]):
            for j in range(m.shape[1]):
                v = m[i, j]
                ax.text(
                    j,
                    i,
                    f"{v*100:.0f}%",
                    ha="center",
                    va="center",
                    fontsize=10,
                    color="#fcfcfb" if v >= 0.3 else "#0b0b0b",
                )
        ax.set_xticks(range(5), IG_TICKS)
        ax.set_yticks(range(5), rlab)
        ax.grid(False)
        ax.set_title(tt, fontsize=11)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.tick_params(length=0)
    fig.suptitle(
        "挑戰門檻卡：需要多少把握才值得挑戰（中位數 p*）", x=0.01, y=1.08, ha="left", fontsize=12.5, fontweight="bold"
    )
    fig.text(
        0.01,
        -0.03,
        "數字越小越該挑戰。p* = C ÷ (G + C)；G = 改判的勝率收益，C = 失去一次挑戰的選擇權價值。2026 例行賽 37 萬顆判決球。",
        fontsize=8.5,
        color="#52514e",
    )
    save(fig, "abs_f4_card")


def f5_forward() -> None:
    """f5：各局每場挑戰次數與成功率，實際 vs 最佳策略（07 前向傳遞）。"""
    r = pd.read_csv(TAB / "h2b_forward.csv")
    r = r[r.by == "ig"].set_index("group").loc[IG]
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.8))
    x = np.arange(5)
    w = 0.38
    ax = axes[0]
    ax.bar(x - w / 2 - 0.01, r.actual_chal_per_game, w, color=BLUE, label="實際")
    ax.bar(x + w / 2 + 0.01, r.opt_chal_per_game, w, color=ORANGE, label="最佳策略")
    ax.set_xticks(x, IG_TICKS)
    ax.set_ylabel("每隊每場挑戰次數")
    ax.set_title("最佳策略要多挑 36%")
    ax.legend(fontsize=9)
    ax.grid(axis="x", visible=False)
    ax = axes[1]
    ax.plot(x, r.actual_success * 100, color=BLUE, marker="o", lw=2, ms=6, label="實際")
    ax.plot(x, r.actual_model_success * 100, color=BLUE, ls=":", lw=1.5, label="實際行為模型")
    ax.plot(x, r.opt_success * 100, color=ORANGE, marker="o", lw=2, ms=6, label="最佳策略")
    ax.set_xticks(x, IG_TICKS)
    ax.set_ylabel("挑戰成功率（%）")
    ax.set_ylim(25, 65)
    ax.set_title("最佳策略的成功率反而更低，第 9 局只有約 33%")
    ax.legend(fontsize=9)
    save(fig, "abs_f5_actual_vs_opt")


def f6_thresholds() -> None:
    """f6：實際門檻 τ vs 最佳門檻 p*（各局、剩 1／2 次），捕手與打者分開。"""
    h = pd.read_csv(TAB / "h2_coarse.csv")
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.9), sharey=True)
    for ax, side, tt in ((axes[0], "def", "捕手（守備方）"), (axes[1], "off", "打者")):
        for k, mk in ((2, "o"), (1, "s")):
            d = h[(h.side == side) & (h.k == k)].set_index("ig").loc[IG]
            x = np.arange(5) + (-0.12 if k == 2 else 0.12)
            ax.vlines(x, d.pstar * 100, d.tau * 100, color="#c3c2b7", lw=2)
            ax.scatter(x, d.tau * 100, color=BLUE, marker=mk, s=40, zorder=3, label=f"實際門檻（剩 {k} 次）")
            ax.scatter(x, d.pstar * 100, color=ORANGE, marker=mk, s=40, zorder=3, label=f"最佳門檻（剩 {k} 次）")
        ax.set_xticks(range(5), IG_TICKS)
        ax.set_title(tt, fontsize=11)
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("需要的把握度（%，中位數）")
    axes[1].legend(fontsize=8.5, loc="upper right")
    fig.suptitle(
        "還有 2 次挑戰時，實際門檻都高於最佳門檻（太保守），而且越到後段差距越大",
        x=0.01,
        y=1.06,
        ha="left",
        fontsize=12.5,
        fontweight="bold",
    )
    save(fig, "abs_f6_thresholds")


def f7_policies() -> None:
    """f7：各策略每隊每季勝場與 95% bootstrap 區間（06 --boot 彙整）。"""
    s = pd.read_csv(TAB / "h3_bootstrap_summary.csv", index_col=0)
    order = [
        ("savant", "Savant 公式"),
        ("actual", "實際"),
        ("static", "固定門檻（調對水準）"),
        ("opt", "最佳（隨局數／比分／次數變動）"),
        ("perfect", "完美判斷（上限）"),
    ]
    fig, ax = plt.subplots(figsize=(7.6, 3.6))
    y = np.arange(len(order))[::-1]
    for yi, (k, _lab) in zip(y, order, strict=False):
        col = BLUE if k == "actual" else (ORANGE if k == "opt" else GRAY)
        ax.barh(yi, s.loc[k, "mean"], color=col, height=0.55)
        ax.errorbar(
            s.loc[k, "mean"],
            yi,
            xerr=[[s.loc[k, "mean"] - s.loc[k, "lo"]], [s.loc[k, "hi"] - s.loc[k, "mean"]]],
            color="#0b0b0b",
            lw=1.2,
            capsize=3,
        )
        ax.text(s.loc[k, "hi"] + 0.15, yi, f"{s.loc[k, 'mean']:.2f}", va="center", fontsize=10)
    ax.set_yticks(y, [l for _, l in order])
    ax.set_xlabel("挑戰帶來的勝場（每隊每季；95% bootstrap 區間）")
    ax.grid(axis="y", visible=False)
    ax.set_title(
        f"最佳策略每隊每季多 {s.loc['opt-actual','mean']:.1f} 勝 [{s.loc['opt-actual','lo']:.1f}, {s.loc['opt-actual','hi']:.1f}]"
    )
    save(fig, "abs_f7_policies")


def f8_teams() -> None:
    """f8：各隊「最佳 − 實際」勝場與 95% 區間。"""
    t = pd.read_csv(TAB / "h3_teams.csv").sort_values("gap_opt_actual")
    fig, ax = plt.subplots(figsize=(7.2, 7.0))
    y = np.arange(len(t))
    ax.hlines(y, t.gap_lo, t.gap_hi, color="#9ec5f4", lw=2.5)
    ax.scatter(t.gap_opt_actual, y, color=BLUE, s=28, zorder=3)
    ax.axvline(t.gap_opt_actual.mean(), color=GRAY, ls="--", lw=1)
    ax.axvline(0, color="#52514e", lw=1)
    ax.set_yticks(y, t.team)
    ax.set_xlabel("最佳 − 實際（每季勝場；95% 區間）")
    ax.grid(axis="y", visible=False)
    nsig = int((t.gap_lo > 0).sum())
    ax.set_title(
        f"{nsig}/30 隊的改進空間顯著大於 0：從 {t.gap_opt_actual.min():.1f} 勝（{t.team.iloc[0]}）到 {t.gap_opt_actual.max():.1f} 勝（{t.team.iloc[-1]}）"
    )
    save(fig, "abs_f8_teams")


def f9_rd() -> None:
    """f9：斷點迴歸圖（勝率、得分、輸贏），分箱平均＋兩側局部線性。"""
    d = pd.read_parquet(RD_SAMPLE)
    res = pd.read_csv(TAB / "rd_main.csv")
    main = res[res.spec == "main(MSE-opt h)"].set_index("y")
    specs = [
        ("Y_pa", 100, "打席結束時的勝率變化（百分點）", "勝率"),
        ("Y_runs", 1, "到半局結束的淨得分", "得分"),
        ("Y_win", 100, "最後贏球的比例（%）", "輸贏"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.9))
    bins = np.arange(-3, 3.001, 0.25)
    for ax, (y, sc, ylab, short) in zip(axes, specs, strict=False):
        s = d[d.m.between(-3, 3)]
        g = s.groupby(pd.cut(s.m, bins), observed=True)[y].agg(["mean", "size"])
        mid = np.array([iv.mid for iv in g.index])
        col = np.where(mid > 0, ORANGE, BLUE)
        ax.scatter(mid, g["mean"] * sc, c=col, s=np.clip(g["size"] / 8, 8, 60), zorder=3, edgecolor="none", alpha=0.9)
        h = 1.5
        for side, cc in ((-1, BLUE), (1, ORANGE)):
            ss = s[(np.sign(s.m) == side) & (s.m.abs() <= h)]
            w = 1 - ss.m.abs() / h
            X = np.column_stack([np.ones(len(ss)), ss.m])
            b = np.linalg.lstsq(X * np.sqrt(w.values)[:, None], ss[y].values * np.sqrt(w.values), rcond=None)[0]
            xs = np.linspace(0, side * h, 20)
            ax.plot(xs, (b[0] + b[1] * xs) * sc, color=cc, lw=2.2)
        ax.axvline(0, color="#52514e", lw=1, ls="--")
        r = main.loc[y]
        ax.set_title(
            f"{short}：跳躍 {r.coef*sc:.3g} [{r.rb_lo*sc:.2g}, {r.rb_hi*sc:.2g}]\n模型預測 {r.model_pred*sc:.3g}",
            fontsize=10.5,
        )
        ax.set_xlabel("真實誤判幅度 m（英吋）")
        ax.set_ylabel(ylab)
    fig.suptitle(
        "斷點迴歸：剛好擦到邊被改判 vs 剛好沒擦到被維持 —— 實測效果與模型預測一致",
        x=0.01,
        y=1.08,
        ha="left",
        fontsize=12.5,
        fontweight="bold",
    )
    fig.text(
        0.01,
        -0.06,
        "樣本：2026 例行賽被挑戰的 10,543 顆球；藍 = 原判維持（m < 0），橘 = 改判（m > 0）；點的大小 = 樣本數。"
        "跳躍為 rdrobust 的 fuzzy RD 估計（MSE 最適頻寬、穩健偏誤校正 95% 區間、以比賽聚類）。挑戰方觀點。",
        fontsize=8.5,
        color="#52514e",
    )
    save(fig, "abs_f9_rd")


def f10_runs_vs_wp() -> None:
    """f10：得失分差目標 vs 勝率目標的勝場，以及差距在各局的分布。"""
    s = pd.read_csv(TAB / "ro_summary.csv", index_col=0)["value"]
    bs = pd.read_csv(TAB / "ro_boot_summary.csv", index_col=0)
    by = pd.read_csv(TAB / "ro_by_inning.csv", index_col=0).loc[IG]
    sav = pd.read_csv(TAB / "h3_bootstrap_summary.csv", index_col=0).loc["savant", "mean"]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.9), gridspec_kw={"width_ratios": [1.15, 1]})
    ax = axes[0]
    items = [
        ("Savant（得失分、固定代價）", sav, GRAY),
        ("實際", s.actual_in_wins, BLUE),
        ("最佳：以得失分差為目標", s.run_policy_in_wins, AQUA),
        ("最佳：以勝率為目標", s.wp_policy_in_wins, ORANGE),
    ]
    y = np.arange(len(items))[::-1]
    for yi, (_lab, v, c) in zip(y, items, strict=False):
        ax.barh(yi, v, color=c, height=0.55)
        ax.text(v + 0.06, yi, f"{v:.2f}", va="center", fontsize=10)
    ax.set_yticks(y, [i[0] for i in items])
    ax.set_xlim(0, 5.2)
    ax.set_xlabel("挑戰帶來的勝場（每隊每季）")
    ax.grid(axis="y", visible=False)
    ax.set_title("以得失分差為目標，就能拿到約九成的好處")
    ax = axes[1]
    ax.bar(np.arange(5), by.diff_wins, color=ORANGE, width=0.6)
    for i, v in enumerate(by.diff_wins):
        ax.text(i, v + 0.001, f"{v:.3f}", ha="center", fontsize=9.5)
    ax.set_xticks(np.arange(5), IG_TICKS)
    ax.grid(axis="x", visible=False)
    ax.set_ylabel("勝率目標多拿的勝場（每季）")
    ax.set_title(f"差距 {s.wp_policy_in_wins - s.run_policy_in_wins:.2f} 勝，集中在後段")
    fig.suptitle(
        "得失分差目標 vs 勝率目標：主要貢獻來自「會變動的失敗代價」，勝率再加一小步",
        x=0.01,
        y=1.06,
        ha="left",
        fontsize=12.5,
        fontweight="bold",
    )
    fig.text(
        0.01,
        -0.06,
        f"勝率版 − 得失分差版：{s.wp_policy_in_wins - s.run_policy_in_wins:.2f} 勝（以比賽 bootstrap 100 次的 95% 區間 [{bs.loc['diff_w','lo']:.2f}, {bs.loc['diff_w','hi']:.2f}]）"
        f"。得失分差版同時計入打擊（多得分）與守備（少失分）的挑戰；其失敗代價隨局數與剩餘次數變動，但不看比分。",
        fontsize=8.5,
        color="#52514e",
    )
    save(fig, "abs_f10_runs_vs_wp")


FIGURES: dict[str, Callable[[], None]] = {
    "f1_perception": f1_perception,
    "f3_option_value": f3_option_value,
    "f4_card": f4_card,
    "f5_forward": f5_forward,
    "f6_thresholds": f6_thresholds,
    "f7_policies": f7_policies,
    "f8_teams": f8_teams,
    "f9_rd": f9_rd,
    "f10_runs_vs_wp": f10_runs_vs_wp,
}

if __name__ == "__main__":
    names = sys.argv[1:] or list(FIGURES)
    unknown = [n for n in names if n not in FIGURES]
    if unknown:
        sys.exit(f"不認得的圖：{unknown}，可用：{list(FIGURES)}")
    for name in names:
        FIGURES[name]()
