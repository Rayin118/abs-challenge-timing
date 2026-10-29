"""00_fetch_statcast.py：從 Baseball Savant 下載 2026 例行賽 Statcast 逐球資料。

研究步驟
    資料取得（需要網路，約 20–40 分鐘）。後面所有步驟都需要這份資料。
做法
    逐「日」查詢 Savant 的 CSV 介面（單日約 4–5 千球，低於單次 25,000 列上限），
    每天存一個快取檔，中斷後重跑會自動跳過已完成的日期，最後依月份合併。
輸出
    data/raw/statcast/_daily/2026-MM-DD.parquet（日快取）
    data/raw/statcast/statcast_2026_MM.parquet（月檔，後續步驟讀這個）
執行
    python scripts/00_fetch_statcast.py                      # 整季
    python scripts/00_fetch_statcast.py --start 2026-04-01 --end 2026-04-03   # 測試幾天
備註
    若已有別處下載好的月檔，可設環境變數 ABS_STATCAST_DIR 指向該資料夾，跳過本步驟。
"""

from __future__ import annotations

import argparse
import io
import sys
import time
from datetime import date, timedelta
from pathlib import Path
from urllib.parse import urlencode

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import SEASON, STATCAST_DIR  # noqa: E402

SAVANT_CSV = "https://baseballsavant.mlb.com/statcast_search/csv"
ROW_LIMIT = 25_000
SEASON_START, SEASON_END = date(SEASON, 3, 15), date(SEASON, 10, 6)
STRING_COLS = {
    "pitch_type", "game_date", "player_name", "events", "description", "des", "game_type", "stand",
    "p_throws", "home_team", "away_team", "type", "bb_type", "inning_topbot", "pitch_name",
    "if_fielding_alignment", "of_fielding_alignment", "sv_id", "umpire", "spin_dir",
}
ID_COLS = ("game_pk", "pitcher", "batter", "at_bat_number", "pitch_number", "inning", "game_year")
CACHE = STATCAST_DIR / "_daily"


def day_url(d: date) -> str:
    """單日、例行賽、全部投手的 CSV 查詢網址。"""
    q = {
        "all": "true", "type": "details", "player_type": "pitcher", "hfGT": "R|", "hfSea": "",
        "game_date_gt": d.isoformat(), "game_date_lt": d.isoformat(), "min_pitches": "0",
        "min_results": "0", "group_by": "name", "sort_col": "pitches", "sort_order": "desc", "min_abs": "0",
    }
    return f"{SAVANT_CSV}?{urlencode(q)}"


def normalize(df: pd.DataFrame) -> pd.DataFrame:
    """統一型別，讓各日檔可以安全合併成 parquet。"""
    out = df.copy()
    for c in out.columns:
        if c in STRING_COLS:
            out[c] = out[c].astype("string")
        elif out[c].dtype == object:
            conv = pd.to_numeric(out[c], errors="coerce")
            ok = out[c].notna() & (out[c].astype("string").str.strip() != "")
            out[c] = conv if conv[ok].notna().all() else out[c].astype("string")
        elif out[c].dtype == bool:
            out[c] = out[c].astype("int8")
    for c in ID_COLS:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce").round().astype("Int64")
    if "game_date" in out.columns:
        out["game_date"] = pd.to_datetime(out["game_date"], errors="coerce")
    return out


def fetch_day(sess: requests.Session, d: date, tries: int = 5) -> pd.DataFrame:
    """下載一天；遇到限流或伺服器錯誤會退避重試。"""
    for k in range(tries):
        try:
            r = sess.get(day_url(d), timeout=180)
            if r.status_code in (429, 500, 502, 503, 504):
                raise requests.HTTPError(f"HTTP {r.status_code}")
            r.raise_for_status()
            text = r.content.decode("utf-8-sig", errors="replace").strip()
            if not text:
                return pd.DataFrame()
            if "<html" in text[:2000].lower():
                raise ValueError("Savant 回傳非 CSV 內容（可能被限流）")
            df = pd.read_csv(io.StringIO(text), low_memory=False)
            if len(df) >= ROW_LIMIT:
                raise ValueError(f"{d} 達 {ROW_LIMIT} 列上限，資料可能被截斷")
            return df
        except Exception as e:  # noqa: BLE001
            wait = 5 * 2**k
            print(f"  {d} 第 {k + 1} 次失敗：{e}，{wait} 秒後重試")
            time.sleep(wait)
    raise RuntimeError(f"{d} 下載失敗")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default=SEASON_START.isoformat())
    ap.add_argument("--end", default=SEASON_END.isoformat())
    a = ap.parse_args()
    d0, d1 = date.fromisoformat(a.start), min(date.fromisoformat(a.end), date.today() - timedelta(days=1))
    CACHE.mkdir(parents=True, exist_ok=True)
    sess = requests.Session()
    sess.headers["User-Agent"] = "abs-challenge-research (academic reproduction)"
    d = d0
    while d <= d1:
        f, empty = CACHE / f"{d}.parquet", CACHE / f"{d}.empty"
        if not (f.exists() or empty.exists()):
            df = fetch_day(sess, d)
            if df.empty:
                empty.touch()
            else:
                normalize(df).to_parquet(f, index=False)
            print(f"{d}：{len(df):,} 球")
            time.sleep(1.0)  # 對 Savant 客氣一點
        d += timedelta(days=1)
    # 依月份合併
    months = sorted({p.stem[:7] for p in CACHE.glob(f"{SEASON}-*.parquet")})
    for ym in months:
        parts = [pd.read_parquet(p) for p in sorted(CACHE.glob(f"{ym}-*.parquet"))]
        df = normalize(pd.concat(parts, ignore_index=True))
        df = df.drop_duplicates(["game_pk", "at_bat_number", "pitch_number"])
        df = df.sort_values(["game_date", "game_pk", "at_bat_number", "pitch_number"]).reset_index(drop=True)
        out = STATCAST_DIR / f"statcast_{ym.replace('-', '_')}.parquet"
        df.to_parquet(out, index=False)
        print(f"合併 {out.name}：{len(df):,} 球，{df.game_pk.nunique()} 場")


if __name__ == "__main__":
    main()
