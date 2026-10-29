"""14_slide_figures.py：簡報用的圖（版面比 08 的報告圖大、字也大，並加上標示）。

研究步驟
    在 05、13 之後執行，只讀 outputs/tables 的結果表。
輸入
    outputs/tables/perc_fit_bins.csv（05）、f3_slide_data.csv、card_pstar_by_bases.csv、
    pstar_importance.csv（13）
輸出（outputs/figures/）
    slide_p20_perception.png   第 20 頁圖一：真實誤判幅度 vs. 被挑戰的比例
    slide_p20_by_count.png     第 20 頁圖二：判錯 1.5 英吋以上的球，依好球數分組
    slide_p21_cost.png         第 21 頁：失敗代價 C 隨局數下降（含 ①②③ 標示）
    slide_p23_by_bases.png     第 23 頁：依壘況拆開的門檻卡
    slide_p27_importance.png   第 27 頁：各因素的重要性
執行
    python scripts/14_slide_figures.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib import font_manager as fm  # noqa: E402
from matplotlib.colors import BoundaryNorm, ListedColormap  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import FIG, IG_LAB, TAB  # noqa: E402

FIG.mkdir(parents=True, exist_ok=True)
for f in fm.findSystemFonts():
    if "CJK" in f or "NotoSansTC" in f:
        try:
            fm.fontManager.addfont(f)
        except Exception:  # noqa: BLE001
            pass
plt.rcParams.update({"font.family": ["PingFang TC", "Heiti TC", "Noto Sans CJK TC", "Noto Sans TC",
                                     "Noto Sans CJK JP", "Noto Sans CJK SC", "Microsoft JhengHei",
                                     "Arial Unicode MS", "sans-serif"],
                     "font.size": 14, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": "#8a8984", "axes.labelcolor": "#3d4555", "xtick.color": "#52514e",
                     "ytick.color": "#52514e", "axes.unicode_minus": False})
BG, BLUE, ORANGE, GRAY, RED, INK = "#FBFBF8", "#2a78d6", "#eb6834", "#898781", "#b4441a", "#14213d"
SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]


def save(fig: plt.Figure, name: str) -> None:
    fig.tight_layout()
    fig.savefig(FIG / name, dpi=200, facecolor=BG)
    plt.close(fig)
    print("寫出", name)


def perception() -> None:
    """第 20 頁：依真實誤判幅度分箱的挑戰比例（點＝實際，線＝模型），以及依好球數分組的長條圖。"""
    fit = pd.read_csv(TAB / "perc_fit_bins.csv")
    fig, ax = plt.subplots(figsize=(8.8, 4.4))
    for side, col, lab in (("def", BLUE, "捕手（守備方）"), ("off", ORANGE, "打者")):
        b = fit[(fit.side == side) & (fit.by == "mb")].copy()
        lo_hi = b.group.str.strip("(]").str.split(",", expand=True).astype(float)
        b["mid"] = lo_hi.mean(axis=1)
        b = b[b.mid.abs() <= 4].sort_values("mid")
        se = np.sqrt(b.actual * (1 - b.actual) / b.n)
        ax.errorbar(b.mid, b.actual * 100, yerr=1.96 * se * 100, fmt="o", ms=6, color=col, label=f"{lab}：實際")
        ax.plot(b.mid, b.pred * 100, color=col, lw=2.4, alpha=0.85, label=f"{lab}：模型")
    ax.axvline(0, color="#52514e", lw=1.2, ls="--")
    ax.text(-0.15, 95, "← 原判正確", ha="right", color="#52514e", fontsize=12)
    ax.text(0.15, 95, "原判錯誤 →", ha="left", color="#52514e", fontsize=12)
    ax.set_xlim(-4.2, 4.6), ax.set_ylim(0, 100)
    ax.set_xlabel("真實誤判幅度 m（英吋，＞0 表示主審判錯）"), ax.set_ylabel("被挑戰的比例（%）")
    ax.grid(color="#e6e6e0", lw=0.8), ax.set_axisbelow(True)
    ax.legend(loc="upper left", fontsize=11.5, frameon=False)
    save(fig, "slide_p20_perception.png")

    fig, ax = plt.subplots(figsize=(7.4, 2.35))
    x, w = np.arange(3), 0.36
    for i, (side, col, lab) in enumerate((("def", BLUE, "捕手"), ("off", ORANGE, "打者"))):
        b = fit[(fit.side == side) & (fit.by == "mb_cnt") & fit.group.str.startswith("m>1.5")].sort_values("group")
        xs = x + (i - 0.5) * w
        a, n, p = b.actual.values, b.n.values, b.pred.values
        ax.bar(xs, a * 100, w * 0.92, color=col, alpha=0.85, label=f"{lab}：實際")
        ax.errorbar(xs, a * 100, yerr=1.96 * np.sqrt(a * (1 - a) / n) * 100, fmt="none", ecolor="#3d4555")
        ax.scatter(xs, p * 100, marker="_", s=900, color=INK, lw=3, zorder=4, label="模型" if i == 0 else None)
        for xx, aa in zip(xs, a, strict=True):
            ax.text(xx, aa * 50, f"{aa * 100:.0f}%", ha="center", va="center", color="white", fontweight="bold")
    ax.set_xticks(x, ["0 好球", "1 好球", "2 好球"]), ax.set_ylim(0, 90), ax.set_ylabel("挑戰比例（%）")
    ax.legend(loc="upper left", fontsize=12, frameon=False, ncol=3)
    save(fig, "slide_p20_by_count.png")


def cost() -> None:
    """第 21 頁：各局的失敗代價（只剩 1 次、還有 2 次）與 Savant 換算，標出 ①②③。"""
    g = pd.read_csv(TAB / "f3_slide_data.csv").set_index("inn")
    x = np.arange(1, 11)
    fig, ax = plt.subplots(figsize=(8.2, 4.52))
    ax.plot(x, g.sav, color=GRAY, lw=2.2, ls="--", marker="s", ms=6, label="Savant：固定 0.2 分（換算勝率）")
    ax.plot(x, g.C1, color=BLUE, lw=2.4, marker="o", ms=7, label="本研究：只剩 1 次時")
    ax.plot(x, g.C2, color=ORANGE, lw=2.4, marker="o", ms=7, label="本研究：還有 2 次時")
    ax.set_xticks(x, [str(i) for i in range(1, 10)] + ["延長"])
    ax.set_xlabel("局數"), ax.set_ylabel("挑戰失敗的代價（勝率百分點）"), ax.set_ylim(0, 4.3)
    ax.grid(color="#e6e6e0"), ax.legend(fontsize=11.5, loc="upper left", frameon=False)

    def tag(xx: float, yy: float, txt: str, dx: float, dy: float) -> None:
        ax.scatter([xx], [yy], s=260, facecolors="none", edgecolors=RED, linewidths=2.2, zorder=5)
        ax.annotate(txt, xy=(xx, yy), xytext=(xx + dx, yy + dy), fontsize=13, fontweight="bold", color=RED,
                    ha="center", va="center", arrowprops=dict(arrowstyle="-", color=RED, lw=1.5),
                    bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=RED, lw=1.2))

    tag(1, g.C1[1], f"① 第 1 局 {g.C1[1]:.2f}", 1.6, 0.95)
    tag(9, g.C1[9], f"② 第 9 局 {g.C1[9]:.2f}", -1.9, 0.95)
    tag(9, g.sav[9], f"③ Savant {g.sav[9]:.2f}", -1.5, 0.85)
    save(fig, "slide_p21_cost.png")


def by_bases() -> None:
    """第 23 頁：四種壘況各一張門檻卡（還有 2 次；列＝球數分組，欄＝局數），樣本 < 30 標星號。"""
    t = pd.read_csv(TAB / "card_pstar_by_bases.csv")
    cmap, norm = ListedColormap(SEQ), BoundaryNorm([0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7], len(SEQ))
    rows = ["一般", "打者有利", "兩好球", "2-2", "3-2"]
    fig, axes = plt.subplots(1, 4, figsize=(16.64, 4.1), sharey=True)
    for ax, b in zip(axes, ["壘上無人", "一壘有人", "得點圈有人", "滿壘"], strict=True):
        s = t[t.bg == b]
        M = s.pivot(index="cg", columns="ig", values="median").loc[rows, IG_LAB].values
        N = s.pivot(index="cg", columns="ig", values="size").loc[rows, IG_LAB].values
        ax.imshow(M, cmap=cmap, norm=norm, aspect="auto")
        for i in range(5):
            for j in range(5):
                v = M[i, j]
                lab = ("<1%" if v < 0.005 else f"{v * 100:.0f}%") + ("*" if N[i, j] < 30 else "")
                ax.text(j, i, lab, ha="center", va="center", fontsize=14.5, color="#fcfcfb" if v >= 0.3 else "#0b0b0b")
        ax.set_xticks(range(5), ["1–3局", "4–6局", "7–8局", "9局", "延長"], fontsize=13.5)
        ax.set_yticks(range(5), rows, fontsize=15)
        ax.set_title(b, fontsize=18, fontweight="bold", color=INK)
        ax.tick_params(length=0)
        for sp in ax.spines.values():
            sp.set_visible(False)
    save(fig, "slide_p23_by_bases.png")


def importance() -> None:
    """第 27 頁：七個因素拿掉後 R² 下降多少（越長越重要）。"""
    t = pd.read_csv(TAB / "pstar_importance.csv")
    cols = ["#b4441a", "#b4441a", "#e8673a", "#e8673a", "#9a9890", "#9a9890", "#c9c7c0"]
    fig, ax = plt.subplots(figsize=(9.0, 4.9))
    y = np.arange(len(t))[::-1]
    ax.barh(y, t.R2下降, color=cols[: len(t)], height=0.62)
    for yy, v in zip(y, t.R2下降, strict=True):
        ax.text(v + 0.003, yy, f"{v:.2f}", va="center", fontsize=15, color=INK, fontweight="bold")
    ax.set_yticks(y, t.因素, fontsize=16), ax.set_xlim(0, 0.19)
    ax.set_xlabel("拿掉這個因素後，解釋力（R²）下降多少")
    ax.grid(axis="x", color="#e6e6e0"), ax.set_axisbelow(True), ax.tick_params(length=0)
    save(fig, "slide_p27_importance.png")


if __name__ == "__main__":
    perception()
    cost()
    by_bases()
    importance()
