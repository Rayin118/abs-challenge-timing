"""09_rd.py：斷點迴歸，用 ABS 好球帶邊界驗證「改判 → 勝率」這一環。

目的
    勝率模型（04）預測改判能帶來 G 的勝率。被挑戰球在好球帶邊界兩側幾乎相同，
    只差有沒有被改判，所以斷點上的跳躍可以獨立檢驗 G 是否正確。
研究步驟
    模糊斷點迴歸（因果驗證）。
輸入
    data/processed/calls_2026_G.parquet（04 gain）、pa_2026.parquet、we_model.pkl（04 fit）
輸出
    outputs/tables/rd_main.csv、rd_balance.csv、rd_placebo.csv、rd_checks.csv
    data/processed/rd_sample.parquet（08 f9 使用）
執行
    python scripts/09_rd.py                     # 約 20 秒

結果變數（挑戰方觀點）、模型預測、跑變數與處理變數的定義見 src/rd.py。
檢查：第一階段、密度檢定（rddensity）、預先決定變數的平衡、安慰劑斷點、頻寬敏感度、甜甜圈 RD、分打擊／守備。
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import pandas as pd
from rddensity import rddensity

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import RD_SAMPLE, TAB  # noqa: E402
from src.rd import build_rd_sample, kern_mean, rd, triangular_weights  # noqa: E402

# rdrobust／rddensity 會對聚類與頻寬發出大量提示性警告，不影響估計
warnings.filterwarnings("ignore")

#: (結果變數, 對應的模型預測)
OUTCOMES = (("Y_pa", "pred_G"), ("Y_runs", "pred_dRE"), ("Y_win", "pred_G"))
#: 頻寬敏感度（英吋）
BANDWIDTHS = (0.75, 1.0, 1.5, 2.0, 3.0)
#: 甜甜圈：排除 |m| < 0.1 英吋（好球帶重建誤差最大的區域）
DONUT_IN = 0.1
#: 平衡檢定的預先決定變數
BALANCE_COVS = [
    "we_pre_chal",
    "pred_G",
    "pred_dRE",
    "two_strike",
    "three_ball",
    "runners_on",
    "late",
    "absdiff",
    "outs_when_up",
    "rem_eligible",
]
#: 安慰劑斷點（英吋），只用同一側的資料
PLACEBO_CUTOFFS = (-2.0, -1.0, 1.0, 2.0)
RD_SAMPLE_COLS = ["game_pk", "m", "treat", "side", "Y_pa", "Y_runs", "Y_win", "pred_G", "pred_dRE", "we_pre_chal"]


def main() -> None:
    """主結果、頻寬敏感度、甜甜圈、分打擊／守備、第一階段、密度、平衡、安慰劑。"""
    d = build_rd_sample()
    rows = []
    for y, pred in OUTCOMES:
        # 主結果：MSE 最適頻寬
        r = rd(d, y)
        w = triangular_weights(d.m, r["h"])
        r["model_pred"] = kern_mean(d[pred].values, w)
        # 預測值本身也做同樣的 RD（應 ≈ 預測均值，用來檢驗「在同樣加權下」的比較）
        r["pred_rd"] = rd(d, pred, h=r["h"])["coef"]
        r["spec"] = "main(MSE-opt h)"
        rows.append(r)
        # 頻寬敏感度
        for h in BANDWIDTHS:
            rr = rd(d, y, h=h)
            rr["spec"] = f"h={h}"
            rr["model_pred"] = kern_mean(d[pred].values, triangular_weights(d.m, h))
            rows.append(rr)
        # 甜甜圈
        rr = rd(d, y, donut=DONUT_IN)
        rr["spec"] = "donut 0.1"
        rows.append(rr)
        # 分打擊方／守備方
        for side in ("batter", "defense"):
            ds = d[d.side == side]
            rr = rd(ds, y)
            rr["spec"] = f"side={side}"
            rr["model_pred"] = kern_mean(ds[pred].values, triangular_weights(ds.m, rr["h"]))
            rows.append(rr)
    res = pd.DataFrame(rows)
    res.to_csv(TAB / "rd_main.csv", index=False)
    pd.set_option("display.width", 220)
    print(
        res[["y", "spec", "h", "n_eff", "coef", "rb_lo", "rb_hi", "rb_p", "model_pred"]].round(4).to_string(index=False)
    )

    # 第一階段：改判機率在斷點的跳躍（應接近 1）
    fs = rd(d, "treat", fuzzy=False)
    print(f"\n第一階段（改判機率在斷點的跳躍）：{fs['coef']:.4f} [{fs['rb_lo']:.4f}, {fs['rb_hi']:.4f}]")
    # 密度檢定：斷點兩側的樣本密度不應跳躍（否則表示挑戰選擇在斷點附近被操弄）
    dens = rddensity(X=d.m.values, c=0.0)
    p_dens = float(dens.test["p_jk"])
    print("密度檢定 p =", p_dens)
    # 共變數平衡（sharp RD on covariates）
    bal = []
    for cov in BALANCE_COVS:
        r = rd(d, cov, fuzzy=False)
        r["cov"] = cov
        bal.append(r)
    bal = pd.DataFrame(bal)
    bal.to_csv(TAB / "rd_balance.csv", index=False)
    print(bal[["cov", "h", "coef", "rb_lo", "rb_hi", "rb_p"]].round(4).to_string(index=False))
    # 安慰劑斷點（只用同一側資料）
    pl = []
    for c0 in PLACEBO_CUTOFFS:
        s = d[(d.m < 0)] if c0 < 0 else d[(d.m > 0)]
        for y in ("Y_pa", "Y_runs", "Y_win"):
            r = rd(s, y, fuzzy=False, c=c0)
            r["cutoff"] = c0
            pl.append(r)
    pl = pd.DataFrame(pl)
    pl.to_csv(TAB / "rd_placebo.csv", index=False)
    print(pl[["cutoff", "y", "h", "coef", "rb_lo", "rb_hi", "rb_p"]].round(4).to_string(index=False))
    pd.DataFrame({"first_stage": [fs["coef"]], "density_p": [p_dens]}).to_csv(TAB / "rd_checks.csv", index=False)
    d[RD_SAMPLE_COLS].to_parquet(RD_SAMPLE, index=False)


if __name__ == "__main__":
    main()
