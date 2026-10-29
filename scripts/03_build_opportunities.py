"""03_build_opportunities.py：合併 Statcast、重建 ABS 好球帶、算每顆判決球的挑戰機會。

目的
    把 API 的挑戰紀錄（02）和 Statcast 的位置、比賽狀態逐球合併，
    用 Statcast 位置重建 ABS 好球帶，判斷每顆判決球原判是否錯誤，並算出當下雙方剩幾次挑戰。
研究步驟
    資料清洗（合併錯位修正、球數）＋ 好球帶重判。
輸入
    data/processed/pitches_2026.parquet（02）
    data/raw/statcast/statcast_2026_*.parquet（00_fetch_statcast.py 下載）
輸出
    data/processed/calls_2026.parquet（只含判決球：called strike／ball，04 gain 使用）
執行
    python scripts/03_build_opportunities.py   # 約 5 秒

ABS 好球帶
    Statcast plate_x／plate_z 原值、球半徑 1.45 英吋、半寬 8.5 英吋、上下緣 = 2026 sz_top／sz_bot
    → 被挑戰球的分類正確率 99.71%。
    signed_dist_in：球心到「好球帶外擴一個球半徑」的矩形的帶號距離（英吋），> 0 = ABS 壞球，≤ 0 = ABS 好球。
剩餘挑戰
    每隊每場 2 次，失敗才扣，延長賽每局開始時若為 0 → 補成 1。
2026-10-27 修正
    合併前先用 renumber_statcast() 刪掉 Statcast 的自動好壞球列並在打席內重新編號，
    修正投球計時違規造成的逐球編號錯位（原編號存於 sc_pitch_number，有移動者 realigned = True）。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import (  # noqa: E402
    AUTO_CALL_DESCRIPTIONS,
    BALL_DESCRIPTIONS,
    CALLS,
    CHALLENGES_PER_GAME,
    EXTRA_INNING,
    MERGE_SPEED_TOL_MPH,
    PITCHES,
)
from src.data_io import load_statcast  # noqa: E402
from src.zone import bases_code, signed_dist  # noqa: E402

PITCH_KEY = ["game_pk", "at_bat_number", "pitch_number"]
STATCAST_COLS = [
    "game_pk",
    "at_bat_number",
    "pitch_number",
    "description",
    "plate_x",
    "plate_z",
    "sz_top",
    "sz_bot",
    "stand",
    "p_throws",
    "pitch_type",
    "home_team",
    "away_team",
    "inning_topbot",
    "outs_when_up",
    "on_1b",
    "on_2b",
    "on_3b",
    "bat_score",
    "fld_score",
    "home_score",
    "away_score",
    "bat_win_exp",
    "fielder_2",
    "umpire",
    "game_date",
    "balls",
    "strikes",
    "release_speed",
]


def remaining(pf: pd.DataFrame) -> pd.DataFrame:
    """逐球計算主隊／客隊在「這球之前」的剩餘挑戰數（rem_home、rem_away）。

    規則：每場開始各 2 次，挑戰失敗（overturned 為 False）才扣一次，
    第 10 局起每局開始時，剩 0 次的一方補成 1 次。
    """
    pf = pf.sort_values(PITCH_KEY).copy()
    # 挑戰方屬於主隊或客隊：打者挑戰 → 打擊方，捕手／投手挑戰 → 守備方
    bat_home = pf.half.eq("bottom")
    side_home = np.where(pf.chal_role.eq("batter"), bat_home, ~bat_home)
    pf["chal_side"] = np.where(pf.challenged, np.where(side_home, "home", "away"), None)
    rem_h, rem_a = np.empty(len(pf), int), np.empty(len(pf), int)
    n0 = CHALLENGES_PER_GAME
    g_prev, h, a, inn_prev = None, n0, n0, 0
    for i, (g, inn, cs, ov) in enumerate(
        zip(pf.game_pk.values, pf.inning.values, pf.chal_side.values, pf.overturned.values, strict=False)
    ):
        if g != g_prev:
            g_prev, h, a, inn_prev = g, n0, n0, 0
        if inn != inn_prev:
            if inn >= EXTRA_INNING:
                h, a = max(h, 1), max(a, 1)
            inn_prev = inn
        rem_h[i], rem_a[i] = h, a
        # overturned 欄是 object（None／True／False），用 `is False` 只在挑戰失敗時扣次數
        if cs is not None and ov is False:
            if cs == "home":
                h = max(h - 1, 0)
            else:
                a = max(a - 1, 0)
    pf["rem_home"], pf["rem_away"] = rem_h, rem_a
    return pf


def renumber_statcast(sc: pd.DataFrame) -> pd.DataFrame:
    """刪掉自動好壞球列，在打席內重新編號（2026-10-27）。

    有投球計時違規的打席，Statcast 把自動好壞球記成一列（description = automatic_ball／automatic_strike，
    沒有球速、沒有位置），API 則不編號（pitchNumber 0、isPitch False）→ 違規之後 Statcast 的
    pitch_number 比 API 多（違規次數）號。全季每個打席：Statcast 列數 − API 投球數 = 自動好壞球列數，
    沒有例外，所以刪掉這些列再依原順序重新編號，就和 API 的 pitchNumber 一一對應。
    原編號保留在 sc_pitch_number。
    """
    auto = sc.description.isin(AUTO_CALL_DESCRIPTIONS).fillna(False).astype(bool)
    out = sc[~auto].sort_values(PITCH_KEY).copy()
    out["sc_pitch_number"] = out.pitch_number
    out["pitch_number"] = out.groupby(["game_pk", "at_bat_number"]).cumcount() + 1
    n_pa = sc[auto][["game_pk", "at_bat_number"]].drop_duplicates().shape[0]
    print(
        f"Statcast 自動好壞球列：刪除 {int(auto.sum()):,} 列（{n_pa:,} 個打席），"
        f"編號有移動的列 {int(out.sc_pitch_number.ne(out.pitch_number).sum()):,}"
    )
    return out


def merge_quality(df: pd.DataFrame) -> pd.Series:
    """合併品質：API 與 Statcast 球速差 ≤ 0.2 mph（任一邊缺值視為通過），且最終判決與 description 一致。

    不一致多半是逐球編號錯位，這類球的位置不可信 → 標記為 False，分析中排除。
    """
    sp_ok = (
        (df.api_speed - df.release_speed).abs().le(MERGE_SPEED_TOL_MPH) | df.api_speed.isna() | df.release_speed.isna()
    )
    desc_ok = ((df.final_call == "S") & df.description.eq("called_strike")) | (
        (df.final_call == "B") & df.description.isin(BALL_DESCRIPTIONS)
    )
    return sp_ok & desc_ok


def main() -> None:
    """合併、重判、計算挑戰機會，寫出 calls_2026.parquet。"""
    pf = pd.read_parquet(PITCHES)
    pf = remaining(pf)  # 剩餘挑戰數用全部投球計算（含合併品質不佳者），避免漏扣
    sc = load_statcast(STATCAST_COLS)
    sc = renumber_statcast(sc)
    df = pf.merge(sc, on=PITCH_KEY, how="left", validate="1:1")
    df["realigned"] = df.sc_pitch_number.ne(df.pitch_number) & df.sc_pitch_number.notna()
    # 只留判決球（有最終判決）且有 Statcast 位置與好球帶上下緣
    df = df[df.final_call.notna() & df.plate_x.notna() & df.sz_top.notna()].copy()
    df["merge_ok"] = merge_quality(df)
    print(f"合併品質不佳而排除：{(~df.merge_ok).sum()} 顆判決球（其中被挑戰 {(~df.merge_ok & df.challenged).sum()}）")
    print(
        f"編號有移動（realigned）的判決球 {int(df.realigned.sum())} 顆"
        f"（其中被挑戰 {int((df.realigned & df.challenged).sum())}，"
        f"merge_ok {int((df.realigned & df.merge_ok).sum())}）"
    )
    # 投球前球數改用 Statcast：它把自動好壞球算進去才是正確的，API 由前一顆投球推算者會漏掉（保留供核對）
    df["balls_pre_api"], df["strikes_pre_api"] = df.balls_pre, df.strikes_pre
    df["balls_pre"], df["strikes_pre"] = df.balls.astype(int), df.strikes.astype(int)
    # ABS 重判
    df["signed_dist_in"] = signed_dist(df.plate_x.values, df.plate_z.values, df.sz_top.values, df.sz_bot.values)
    df["abs_strike"] = df.signed_dist_in <= 0
    df["orig_strike"] = df.orig_call.eq("S")
    df["missed"] = df.abs_strike != df.orig_strike
    # 誰受原判不利：原判好球 → 打者可挑，原判壞球 → 守備方可挑
    df["chal_side_eligible"] = np.where(df.orig_strike, "offense", "defense")
    bat_home = df.half.eq("bottom")
    off_rem = np.where(bat_home, df.rem_home, df.rem_away)
    def_rem = np.where(bat_home, df.rem_away, df.rem_home)
    df["rem_eligible"] = np.where(df.orig_strike, off_rem, def_rem)
    df.loc[~df.merge_ok, "rem_eligible"] = -1  # 合併品質不佳 → 不進入機會樣本
    df["bases"] = bases_code(df)
    df["score_diff_bat"] = df.bat_score - df.fld_score
    df.to_parquet(CALLS, index=False)
    c = df[df.challenged & df.merge_ok]
    # 正確率：被改判的球，ABS 結果應與原判相反
    truth = np.where(c.overturned, c.orig_call.eq("B"), c.orig_call.eq("S"))
    print(
        f"判決球 {len(df):,}，被挑戰 {len(c):,}，重建好球帶對被挑戰球的正確率 " f"{(c.abs_strike == truth).mean():.4f}"
    )
    print(
        f"誤判率（原判 vs ABS）{df.missed.mean():.4f}，"
        f"有剩餘挑戰時被挑戰者 rem>0 比例 {(c.rem_eligible > 0).mean():.4f}"
    )


if __name__ == "__main__":
    main()
