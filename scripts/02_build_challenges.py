"""02_build_challenges.py：把完整 playByPlay 攤平成「每一球」表，標記 ABS 挑戰（reviewType = MJ）。

目的
    從 MLB Stats API 的 feed 取出每一球的判決、球數、挑戰紀錄（誰挑戰、是否改判）。
研究步驟
    資料清洗：挑戰紀錄。
輸入
    data/raw/feed/2026/*.json.gz（01 下載）
輸出
    data/processed/pitches_2026.parquet（每一球一列，03 使用）
執行
    python scripts/02_build_challenges.py      # 約 20 秒（4 個行程平行）

挑戰標記的兩個位置
    1. playEvent.reviewDetails：打席繼續的挑戰
    2. play.reviewDetails（或其 additionalReviews）：挑戰直接結束打席（三振、保送）→ 掛到該打席最後一球
API 的判決與 count 都是「改判後」的結果
    - 投球前球數 = 同打席上一球之後的 count（第一球為 0-0）
    - 原判 = 若改判成功，與最終判決相反，否則等於最終判決
"""

from __future__ import annotations

import gzip
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import ABS_REVIEW_TYPE, API_BALL_CODES, API_STRIKE_CODES, PITCHES, feed_dir  # noqa: E402

N_WORKERS = 4


def mj(rd: dict | None) -> dict | None:
    """回傳 reviewDetails 中的 ABS 挑戰（reviewType = MJ，含 additionalReviews），最多一筆。"""
    if not rd:
        return None
    cands = [rd] + list(rd.get("additionalReviews") or [])
    for r in cands:
        if r.get("reviewType") == ABS_REVIEW_TYPE:
            return r
    return None


def parse(path: Path) -> list[dict]:
    """解析一場比賽的 feed，回傳每一球（isPitch）一筆紀錄。"""
    pk = int(path.name.split(".")[0])
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        js = json.load(fh)
    rows = []
    for play in js.get("allPlays") or []:
        ab = play.get("about") or {}
        mu = play.get("matchup") or {}
        bat = (mu.get("batter") or {}).get("id")
        pit = (mu.get("pitcher") or {}).get("id")
        # 只取真正的投球，投球計時違規的自動好壞球 isPitch = False，不在這裡（見 03 的重新編號）
        pitches = [e for e in play.get("playEvents") or [] if e.get("isPitch")]
        play_mj = mj(play.get("reviewDetails"))
        pb, ps = 0, 0  # 投球前球數：由上一顆投球之後的 count 推算
        for i, e in enumerate(pitches):
            d = e.get("details") or {}
            cnt = e.get("count") or {}
            r = mj(e.get("reviewDetails"))
            where = "event" if r else None
            if r is None and play_mj is not None and i == len(pitches) - 1:
                r, where = play_mj, "play"  # 挑戰結束打席 → 掛到最後一球
            pdat = e.get("pitchData") or {}
            co = pdat.get("coordinates") or {}
            call = d.get("call") or {}
            rows.append(
                {
                    "api_px": co.get("pX"),
                    "api_pz": co.get("pZ"),
                    "api_sz_top": pdat.get("strikeZoneTop"),
                    "api_sz_bot": pdat.get("strikeZoneBottom"),
                    "api_speed": pdat.get("startSpeed"),
                    "game_pk": pk,
                    "at_bat_number": ab.get("atBatIndex", -1) + 1,
                    "pitch_number": e.get("pitchNumber"),
                    "play_id": e.get("playId"),
                    "inning": ab.get("inning"),
                    "half": ab.get("halfInning"),
                    "batter": bat,
                    "pitcher": pit,
                    "balls_pre": pb,
                    "strikes_pre": ps,
                    "outs_post": cnt.get("outs"),
                    "call_code": call.get("code"),
                    "call_desc": call.get("description"),
                    "is_last_pitch": i == len(pitches) - 1,
                    "pa_event": (play.get("result") or {}).get("eventType"),
                    "challenged": r is not None,
                    "chal_where": where,
                    "overturned": None if r is None else bool(r.get("isOverturned")),
                    "chal_team": None if r is None else r.get("challengeTeamId"),
                    "chal_player": None if r is None else (r.get("player") or {}).get("id"),
                    "chal_in_progress": None if r is None else r.get("inProgress"),
                }
            )
            pb, ps = cnt.get("balls", pb), cnt.get("strikes", ps)
    return rows


def add_roles_and_calls(df: pd.DataFrame) -> pd.DataFrame:
    """加上挑戰者角色、最終判決（S／B）與原判。"""
    # 角色：挑戰者 = 打者 → batter，= 投手 → pitcher，其他 → catcher（守備方）
    df["chal_role"] = None
    c = df.challenged
    df.loc[c & (df.chal_player == df.batter), "chal_role"] = "batter"
    df.loc[c & (df.chal_player == df.pitcher), "chal_role"] = "pitcher"
    df.loc[c & df.chal_role.isna(), "chal_role"] = "catcher"
    # 最終判決 → 原判（改判成功的球，原判與最終相反）
    fin_strike = df.call_code.isin(API_STRIKE_CODES)
    fin_ball = df.call_code.isin(API_BALL_CODES)
    df["final_call"] = pd.Series(pd.NA, index=df.index, dtype="string")
    df.loc[fin_strike, "final_call"] = "S"
    df.loc[fin_ball, "final_call"] = "B"
    flip = {"S": "B", "B": "S"}
    df["orig_call"] = df.final_call
    m = df.overturned.astype("boolean").fillna(False).astype(bool)
    df.loc[m, "orig_call"] = df.loc[m, "final_call"].map(flip)
    return df


def main() -> None:
    """平行解析全部 feed，寫出 pitches_2026.parquet。"""
    files = sorted(feed_dir().glob("*.json.gz"))
    with ProcessPoolExecutor(N_WORKERS) as ex:
        rows = [r for part in ex.map(parse, files, chunksize=20) for r in part]
    df = add_roles_and_calls(pd.DataFrame(rows))
    PITCHES.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(PITCHES, index=False)
    print(f"{len(files)} 場、{len(df):,} 球 → {PITCHES}")


if __name__ == "__main__":
    main()
