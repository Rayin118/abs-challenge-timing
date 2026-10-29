"""讀寫工具：Statcast 載入、pickle 讀寫、輸出資料夾建立。"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path
from typing import Any

import pandas as pd

from src.config import FIG, PROC, STATCAST_DIR, STATCAST_GLOB, TAB


def statcast_files() -> list[Path]:
    """回傳本賽季 Statcast 月檔（依檔名排序），找不到就結束程式並說明位置。"""
    files = sorted(STATCAST_DIR.glob(STATCAST_GLOB))
    if not files:
        sys.exit(f"找不到 {STATCAST_DIR}/{STATCAST_GLOB}（請先執行 python scripts/00_fetch_statcast.py 下載）")
    return files


def load_statcast(columns: list[str]) -> pd.DataFrame:
    """讀取本賽季全部 Statcast 月檔的指定欄位並串接（保留各月檔內的原始列序）。"""
    return pd.concat(pd.read_parquet(f, columns=columns) for f in statcast_files())


def load_pickle(path: Path) -> Any:
    """讀取 pickle 檔（只用於本專案自己產生的檔案）。"""
    with open(path, "rb") as f:
        return pickle.load(f)


def dump_pickle(obj: Any, path: Path) -> None:
    """把物件寫成 pickle 檔。"""
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def ensure_dirs() -> None:
    """建立輸出資料夾（已存在則略過）。"""
    for d in (PROC, TAB, FIG):
        d.mkdir(parents=True, exist_ok=True)
