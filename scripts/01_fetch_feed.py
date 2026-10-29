"""01_fetch_feed.py：下載 2026 例行賽完整 playByPlay（含 ABS 挑戰欄位 reviewDetails）。

目的
    第一個研究（pitch_clock_fatigue_research）抓 playByPlay 時用了 fields 篩選，把 reviewDetails 濾掉了。
    這裡改抓「完整 JSON」，不篩欄位，作為 ABS 挑戰紀錄的來源。
研究步驟
    資料取得（唯一需要網路的步驟，run_abs.sh 預設不跑，加 --with-fetch 才跑）。
輸入
    data/raw/statcast/statcast_2026_*.parquet（00_fetch_statcast.py 下載）（取例行賽 game_pk，共 2,429 場）
    MLB Stats API：https://statsapi.mlb.com/api/v1/game/{game_pk}/playByPlay
輸出
    data/raw/feed/2026/{game_pk}.json.gz（快取，已存在的會跳過，可續傳）
    logs/fetch_failed.csv（只在有失敗時產生）
執行
    python scripts/01_fetch_feed.py --test 5      # 先試 5 場，印出挑戰欄位長相
    python scripts/01_fetch_feed.py               # 全部（約 15–30 分鐘）
"""

from __future__ import annotations

import argparse
import gzip
import json
import random
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import LOGS, RAW_FEED, SEASON  # noqa: E402
from src.data_io import statcast_files  # noqa: E402

API = "https://statsapi.mlb.com/api/v1/game/{pk}/playByPlay"
USER_AGENT = "DSbaseball-research/1.0"
#: 連線重試：最多 6 次、指數退避 1.5 秒，遇 429 與 5xx 重試並遵守 Retry-After
RETRY_TOTAL, RETRY_BACKOFF = 6, 1.5
RETRY_STATUS = (429, 500, 502, 503, 504)
REQUEST_TIMEOUT_S = 60


def make_session(pool: int = 12) -> requests.Session:
    """建立有自動重試與連線池的 requests Session。"""
    s = requests.Session()
    retry = Retry(
        total=RETRY_TOTAL,
        backoff_factor=RETRY_BACKOFF,
        status_forcelist=RETRY_STATUS,
        allowed_methods=frozenset(["GET"]),
        respect_retry_after_header=True,
    )
    ad = HTTPAdapter(max_retries=retry, pool_connections=pool, pool_maxsize=pool)
    s.mount("https://", ad)
    s.headers.update({"User-Agent": USER_AGENT, "Accept-Encoding": "gzip, deflate"})
    return s


class RateLimiter:
    """多執行緒共用的節流器：兩次請求至少間隔 gap 秒（另加最多 10% 的隨機抖動）。"""

    def __init__(self, gap: float) -> None:
        self.gap, self.lock, self.last = gap, threading.Lock(), 0.0

    def wait(self) -> None:
        """等到可以送出下一個請求為止。"""
        with self.lock:
            d = self.last + self.gap - time.monotonic()
            if d > 0:
                time.sleep(d + random.uniform(0, 0.1 * self.gap))
            self.last = time.monotonic()


def game_list(season: int) -> list[int]:
    """由 Statcast 取出該賽季例行賽（game_type = R）的 game_pk 清單。"""
    df = pd.concat(pd.read_parquet(f, columns=["game_pk", "game_type"]) for f in statcast_files())
    return sorted(df.loc[df.game_type == "R", "game_pk"].astype(int).unique())


def path_of(season: int, pk: int) -> Path:
    """某場比賽的快取檔路徑。"""
    return RAW_FEED / str(season) / f"{pk}.json.gz"


def save(js: dict, p: Path) -> None:
    """把 JSON 寫成 gzip（先寫暫存檔再改名，中斷時不會留下壞檔）。"""
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    with gzip.open(tmp, "wt", encoding="utf-8") as f:
        json.dump(js, f, separators=(",", ":"))
    tmp.replace(p)


def find_reviews(js: dict) -> list[tuple[str, dict]]:
    """回傳 (位置, reviewDetails) 清單：play 層級或 playEvent 層級。"""
    out = []
    for play in js.get("allPlays") or []:
        if play.get("reviewDetails"):
            out.append(("play", play["reviewDetails"]))
        for ev in play.get("playEvents") or []:
            if ev.get("reviewDetails"):
                out.append(("event", ev["reviewDetails"]))
    return out


def main() -> None:
    """下載缺少的比賽，最後彙總 reviewDetails 的數量與範例。"""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--season", type=int, default=SEASON)
    ap.add_argument("--test", type=int, default=0, help="只抓前 N 場並印出 reviewDetails 範例")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--interval", type=float, default=0.15, help="兩次請求的最小間隔（秒）")
    a = ap.parse_args()

    games = game_list(a.season)
    print(f"{a.season} 例行賽 {len(games):,} 場")
    if a.test:
        games = games[: a.test]
    todo = [pk for pk in games if not path_of(a.season, pk).exists()]
    print(f"待下載 {len(todo):,} 場")

    sess, lim, failed = make_session(), RateLimiter(a.interval), []

    def work(pk: int) -> None:
        """下載一場並存檔（失敗時拋出例外，由外層記錄）。"""
        lim.wait()
        r = sess.get(API.format(pk=pk), timeout=REQUEST_TIMEOUT_S)
        r.raise_for_status()
        save(r.json(), path_of(a.season, pk))

    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        futs = {ex.submit(work, pk): pk for pk in todo}
        for i, f in enumerate(as_completed(futs), 1):
            try:
                f.result()
            except Exception as e:  # noqa: BLE001  單場失敗不中斷，記下來之後續傳
                failed.append({"game_pk": futs[f], "error": repr(e)})
            if i % 100 == 0 or i == len(futs):
                print(f"  {i:,}/{len(futs):,}", flush=True)

    if failed:
        LOGS.mkdir(parents=True, exist_ok=True)
        out = LOGS / "fetch_failed.csv"
        pd.DataFrame(failed).to_csv(out, index=False)
        print(f"失敗 {len(failed)} 場 → {out}（重跑本檔即可續傳）")

    # 摘要：挑戰數量與欄位長相
    n_play = n_ev = 0
    example = None
    for pk in games:
        p = path_of(a.season, pk)
        if not p.exists():
            continue
        with gzip.open(p, "rt", encoding="utf-8") as fh:
            rv = find_reviews(json.load(fh))
        n_play += sum(1 for w, _ in rv if w == "play")
        n_ev += sum(1 for w, _ in rv if w == "event")
        if example is None and rv:
            example = rv[:3]
    print(f"reviewDetails：play 層級 {n_play}、playEvent 層級 {n_ev}（含非 ABS 的重播審查）")
    if example:
        print("範例：", json.dumps(example, ensure_ascii=False, indent=1)[:1500])
    print("完成")


if __name__ == "__main__":
    main()
