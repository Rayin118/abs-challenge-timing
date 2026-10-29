"""專案共用設定：路徑、檔名、研究常數與隨機種子。

所有步驟腳本都從這裡取路徑與常數，不在各腳本裡硬寫。
常數的來源與理由寫在旁邊的註解，改動任何一個都會改變研究結果，請先確認。
"""

from __future__ import annotations

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# 路徑
# ---------------------------------------------------------------------------
#: 專案根目錄（abs_challenge_research/）
ROOT: Path = Path(__file__).resolve().parents[1]

#: 賽季（本研究只用 2026 例行賽）
SEASON: int = 2026

DATA: Path = ROOT / "data"
RAW_FEED: Path = DATA / "raw" / "feed"  # 01 下載的 playByPlay：{SEASON}/{game_pk}.json.gz
EXT: Path = DATA / "ext"  # 外部手動下載的檔案（Savant 排行榜、球員姓名）
PROC: Path = DATA / "processed"  # 各步驟產生的中間資料（parquet／pkl）
XFER: Path = DATA / "xfer"  # 傳檔暫存（不屬於流程，程式不再讀取）
TAB: Path = ROOT / "outputs" / "tables"
FIG: Path = ROOT / "outputs" / "figures"
LOGS: Path = ROOT / "logs"

#: Statcast 逐球資料（00_fetch_statcast.py 下載到這裡）。
#: 若已經有別處下載好的檔案，可用環境變數 ABS_STATCAST_DIR 指定資料夾，不必重抓。
STATCAST_DIR: Path = Path(os.environ.get("ABS_STATCAST_DIR", str(DATA / "raw" / "statcast")))
STATCAST_GLOB: str = f"statcast_{SEASON}_*.parquet"


def feed_dir(season: int = SEASON) -> Path:
    """回傳某賽季 playByPlay 快取資料夾。"""
    return RAW_FEED / str(season)


# 各步驟的輸出檔（集中列出，方便查「誰寫、誰讀」）
PITCHES = PROC / f"pitches_{SEASON}.parquet"  # 02 → 03
CALLS = PROC / f"calls_{SEASON}.parquet"  # 03 → 04
PA = PROC / f"pa_{SEASON}.parquet"  # 04 fit → 04 valid、09、10
WE_MODEL = PROC / "we_model.pkl"  # 04 fit → 04 valid／gain、09、10
CALLS_G = PROC / f"calls_{SEASON}_G.parquet"  # 04 gain → 05、06、07、09、10、12
OPPS = PROC / f"opps_{SEASON}.parquet"  # 05 → 06（敏感度的先驗）、08
PERCEPTION = PROC / "perception.pkl"  # 05 → 06、07、10、11
POLICY_VALUES = PROC / "policy_values_teamgame.parquet"  # 06 → 07 teams
NODES_OPT = PROC / "nodes_opt.parquet"  # 06 → 08
DP_HIST = PROC / "dp_hist.pkl"  # 06（收斂紀錄）
H2_NODES = PROC / "h2_nodes.parquet"  # 07 h2（節點層級損失，供日後檢查）
RD_SAMPLE = PROC / "rd_sample.parquet"  # 09 → 08 f9

# ---------------------------------------------------------------------------
# 好球帶幾何（單位：英尺，Statcast 的 plate_x／plate_z 以英尺記錄）
# ---------------------------------------------------------------------------
#: 棒球半徑 1.45 英吋（MLB 規格球周長 9–9.25 英吋 → 直徑約 2.9 英吋）。
#: ABS 以「球的任何部分碰到好球帶」判定好球，所以好球帶要外擴一個球半徑。
#: 這個值也經過校準：用它重建的好球帶，對被挑戰球的分類正確率 99.71%（見 03）。
BALL_RADIUS_IN: float = 1.45
BALL_RADIUS_FT: float = BALL_RADIUS_IN / 12
#: 本壘板寬 17 英吋 → 半寬 8.5 英吋 = 17/24 英尺
PLATE_HALF_WIDTH_FT: float = 17 / 24
INCHES_PER_FOOT: float = 12.0
#: 好球帶最近的邊（感知誤差 σ 依邊估計）
EDGES: list[str] = ["bottom", "side", "top"]

# ---------------------------------------------------------------------------
# API 與 Statcast 合併（03）
# ---------------------------------------------------------------------------
#: 兩邊球速差 ≤ 0.2 mph 視為同一球（Statcast 與 API 的 startSpeed 四捨五入到 0.1 mph）
MERGE_SPEED_TOL_MPH: float = 0.2
#: 投球計時違規的自動好壞球（Statcast 記成一列，API 不編號）
AUTO_CALL_DESCRIPTIONS: list[str] = ["automatic_ball", "automatic_strike"]
#: Statcast description 中算「壞球」的判決
BALL_DESCRIPTIONS: list[str] = ["ball", "blocked_ball", "pitchout"]
#: API 判決代碼：C = called strike，B／*B = ball
API_STRIKE_CODES: list[str] = ["C"]
API_BALL_CODES: list[str] = ["B", "*B"]
#: ABS 挑戰在 reviewDetails 的 reviewType
ABS_REVIEW_TYPE: str = "MJ"

# ---------------------------------------------------------------------------
# 挑戰規則（2026 MLB ABS）
# ---------------------------------------------------------------------------
#: 每隊每場 2 次，只有失敗才扣
CHALLENGES_PER_GAME: int = 2
#: 延長賽（第 10 局起）每局開始時若剩 0 次 → 補成 1 次
EXTRA_INNING: int = 10

# ---------------------------------------------------------------------------
# 勝率／得分期望模型（src/we.py）
# ---------------------------------------------------------------------------
#: 半局剩餘得分上限：≥ 15 分併入 15（2026 半局最多得分遠低於此，截斷不影響結果）
RMAX: int = 15
#: 單一打席轉移的得分上限（一個打席最多 4 分，另加跑壘得分的保留空間）
TRMAX: int = 6
#: 比分差網格 −30..+30：超出部分取端點值（30 分差時勝率已是 0 或 1）
DM: int = 30
#: 稀疏轉移格的門檻：少於 20 筆就用其他出局數的同壘況轉移補強
SPARSE_TRANSITION_N: int = 20

# ---------------------------------------------------------------------------
# 局數分組、球數分組（05、06、07、11 共用）
# ---------------------------------------------------------------------------
IG_EDGES: list[int] = [0, 3, 6, 8, 9, 99]
IG_LAB: list[str] = ["1-3", "4-6", "7-8", "9", "10+"]
#: 圖上用的中文局數標籤（與 IG_LAB 一一對應）
IG_LAB_ZH: list[str] = ["1–3 局", "4–6 局", "7–8 局", "9 局", "延長"]
#: 門檻卡的球數分組
_CG_A, _CG_B = "a 0-0/1-0/0-1/1-1", "b 2-0/2-1/3-0/3-1"
# fmt: off
COUNT_GROUP: dict[str, str] = {
    "0-0": _CG_A, "1-0": _CG_A, "0-1": _CG_A, "1-1": _CG_A,
    "2-0": _CG_B, "2-1": _CG_B, "3-0": _CG_B, "3-1": _CG_B,
    "0-2": "c 0-2/1-2", "1-2": "c 0-2/1-2", "2-2": "d 2-2", "3-2": "e 3-2",
}
# fmt: on

# ---------------------------------------------------------------------------
# 感知模型（05）
# ---------------------------------------------------------------------------
#: 估計樣本只用 |m| ≤ 6 英吋（m > 6 的「誤判」幾乎不被挑戰，疑為追蹤誤差）
M_MAX_IN: float = 6.0
#: 球員效果 α 的 ridge 懲罰：α ~ N(0, τ²)，τ = 1 英吋
TAU_PLAYER_IN: float = 1.0
#: G 取 log 前的下限（避免 log 0）
G_FLOOR: float = 1e-4
#: 主觀機率對照表 p̃(m̃) 的網格（英吋）：np.arange(-12, 12.001, 0.05)
POST_GRID_ARANGE: tuple[float, float, float] = (-12, 12.001, 0.05)
#: 先驗直方圖：0.05 英吋一格，範圍 ±30 英吋，|m| 先截在 29.99 以內，避免落在最外側邊界之外
PRIOR_EDGES_ARANGE: tuple[float, float, float] = (-30, 30.001, 0.05)
PRIOR_CLIP: float = 29.99

# ---------------------------------------------------------------------------
# 選擇權價值／動態規劃（06）
# ---------------------------------------------------------------------------
#: 狀態格往上層收縮的強度（相當於 50 筆虛擬樣本）
KAPPA: float = 50.0
#: 比分差分格範圍（±5 分以上合併）
D_CLIP: int = 5
#: 政策迭代的 C 初始值（勝率）與得分版（分）
C_INIT_WP: float = 0.01
C_INIT_RUNS: float = 0.05
#: Savant 規則的固定失敗代價：0.2 分
SAVANT_COST_RUNS: float = 0.2
#: 一季 162 場：每場期望值 × 162 = 每隊每季
SEASON_GAMES: int = 162
#: 政策迭代次數（run_abs.sh 的主結果用 60，敏感度與 bootstrap 用預設 30，實際約 9 次就收斂）
ITERS_MAIN: int = 60
ITERS_DEFAULT: int = 30

# ---------------------------------------------------------------------------
# 隨機種子（全部集中在這裡）
# ---------------------------------------------------------------------------
SEED_DP_BOOT: int = 1000  # 06 --boot：第 b 次用 default_rng(1000 + b)
SEED_RO_BOOT: int = 5000  # 11 --boot：第 b 次用 default_rng(5000 + b)
SEED_TEAM_BOOT: int = 7  # 07 teams：各隊差距的 bootstrap
SEED_PATH_BOOT: int = 11  # 10 path：路徑相依的 bootstrap
N_TEAM_BOOT: int = 2000
N_PATH_BOOT: int = 500
