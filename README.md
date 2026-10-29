# ABS 挑戰時機研究（2026 MLB）

隊名：**志在參加**

2026 年 MLB 開始使用 ABS（自動好壞球系統）挑戰制：每隊每場 2 次，挑戰成功不扣次數。本研究用 2026 例行賽 37 萬顆主審判決的球回答：**什麼情境下挑戰最划算？球隊實際的做法離最佳有多遠？**

做法是比較「挑戰成功的收益 G」和「挑戰失敗的代價 C」，再納入球員肉眼判斷的誤差 p：

```
p × G ＞ (1 − p) × C   ⇔   p ＞ p* ＝ C ÷ (G ＋ C)
```

- **G**（模型一）：含球數的勝率模型，算出每顆球改判前後的勝率差，並用模糊斷點迴歸在真實比賽中驗證。
- **p**（模型二）：從球員實際挑戰紀錄，以最大概似法估計判斷誤差 σ、挑戰上限 λ 和心中門檻 t。
- **C**（模型三）：近似動態規劃，算出少一次挑戰讓之後的勝率損失多少。

主要結果：照最佳策略挑戰，每隊每季約可多 **1.05 勝**（95% 區間 0.94～1.17），球隊普遍太保守，後段最明顯。

## 快速開始：重現簡報上的數字

```bash
git clone https://github.com/Rayin118/abs-challenge-timing.git
cd abs-challenge-timing
pip install -r requirements.txt jupyter
jupyter notebook notebooks/reproduce_slides.ipynb
```

**`notebooks/reproduce_slides.ipynb` 依簡報頁碼（第 4–30 頁）逐頁重算數字**，每一項都印出「簡報寫的值」和「重算值」，最後彙整成對照表。

- 第一次執行：repo 不含原始資料，請把設定格的 `RUN_FETCH`、`RUN_PIPELINE` 改成 `True`。notebook 會自動下載資料（約 40–60 分鐘）並跑完整分析（約 3–5 分鐘）。
- 之後再執行：維持 `False`，直接讀 `data/processed/` 的中間檔核對。
- `RUN_BOOT=True` 會重跑 bootstrap 信賴區間（約 1 小時），預設使用 repo 內附的結果表。

也可以不開 notebook，直接用指令列跑完整流程：

```bash
bash run_all.sh --with-fetch     # 下載資料＋全部分析
bash run_all.sh                  # 已有資料時，只跑分析（02–14）
bash run_all.sh --with-boot      # 另外重跑 bootstrap
```

## 資料夾結構

```
abs-challenge-timing/
├── README.md
├── requirements.txt          Python 套件（鎖定版本）
├── pyproject.toml            ruff／black 設定（只用於程式風格）
├── run_all.sh                完整流程（00–14）
├── LICENSE                   MIT 授權
├── notebooks/
│   └── reproduce_slides.ipynb  逐頁重現簡報數字
├── scripts/                  步驟腳本 00–14，依編號執行
├── src/                      共用模組
│   ├── config.py             路徑、研究常數、隨機種子（常數旁註明來源）
│   ├── data_io.py            Statcast 載入、pickle 讀寫
│   ├── zone.py               ABS 好球帶重判
│   ├── we.py                 含球數的勝率／得分期望模型（逆向歸納）
│   ├── perception.py         判斷誤差模型
│   ├── option_value.py       動態規劃、策略模擬
│   └── rd.py                 斷點迴歸
├── data/
│   ├── ext/                  外部小檔（Savant 打者挑戰排行榜、球員姓名），已附在 repo
│   ├── raw/                  （不上傳）00、01 下載的原始資料
│   └── processed/            （不上傳）各步驟產生的中間檔
└── outputs/
    ├── tables/               結果表（CSV），已附在 repo
    └── figures/              報告與簡報用圖（PNG），已附在 repo
```

`data/raw/` 和 `data/processed/` 合計約 1.5 GB，不放進 repo（見 `.gitignore`），由 notebook 或 `run_all.sh` 重新產生。

## 步驟與輸出

| 步驟 | 內容 | 主要輸出 | 簡報頁 |
|---|---|---|---|
| 00 | 從 Baseball Savant 逐日下載 Statcast（716,792 球） | `data/raw/statcast/statcast_2026_MM.parquet` | 9、10 |
| 01 | 從 MLB Stats API 下載 2,429 場 playByPlay（含挑戰紀錄） | `data/raw/feed/2026/*.json.gz` | 9、10 |
| 02 | 攤平成每一球、找出 10,558 次挑戰 | `processed/pitches_2026.parquet` | 12 |
| 03 | 合併兩個來源、ABS 規則重判、推算剩餘挑戰次數 | `processed/calls_2026.parquet` | 11–14 |
| 04 | 勝率模型、驗證、每顆球的 G | `processed/calls_2026_G.parquet`、`we_model.pkl`、`tables/we_calibration.csv` | 6、8、15、16 |
| 05 | 判斷誤差模型（σ、λ、t） | `processed/opps_2026.parquet`、`perception.pkl`、`tables/perc_*.csv` | 19、20 |
| 06 | 動態規劃算 C、最佳門檻、各策略勝場 | `processed/nodes_opt.parquet`、`tables/dp_*.csv` | 21、25 |
| 07 | 門檻卡、實際 vs 最佳、各隊差異 | `tables/card_*.csv`、`h2_*.csv`、`h3_*.csv` | 22、24–26、29 |
| 08 | 報告用圖 | `figures/abs_f*.png` | |
| 09 | 模糊斷點迴歸 | `tables/rd_*.csv` | 17、18 |
| 10 | 球數路徑、壘況分組的穩健性檢查 | `tables/cc_*.csv` | |
| 11 | 得失分目標 vs 勝率目標 | `tables/ro_*.csv` | |
| 12 | 和 Savant 排行榜核對 | `tables/savant_compare_*.csv` | 6 |
| 13 | 簡報補充表（例子、壘況門檻、因素重要性、口訣吻合度、貝氏換算） | `tables/slide_examples.csv` 等 8 個表 | 16、21–23、27、28 |
| 14 | 簡報用圖 | `figures/slide_p*.png` | 20、21、23、27 |

每支腳本開頭的說明寫了做法、輸入輸出和執行方式，例如 `python scripts/05_perception.py`。

## 資料來源

- **MLB Stats API**（`statsapi.mlb.com/api/v1/game/{game_pk}/playByPlay`）：每一球的判決、球數、ABS 挑戰紀錄（`reviewDetails`）。
- **Baseball Savant Statcast**：每一球的進壘座標、好球帶上下緣、比賽狀態。
- **Baseball Savant ABS 挑戰排行榜**（`data/ext/savant_batter.csv`）：只用於 12 的核對。

如果已經在別的資料夾下載過 Statcast 月檔，設定環境變數即可跳過 00：

```bash
export ABS_STATCAST_DIR=/path/to/statcast   # 資料夾內為 statcast_2026_*.parquet
```

## 環境

- Python 3.10 以上，套件見 `requirements.txt`（另需 `jupyter` 才能開 notebook）。
- 記憶體約 3 GB。
- 圖的中文字型會依序找 PingFang TC、Heiti TC、Noto Sans CJK、Microsoft JhengHei。都沒有時中文會顯示成方塊。

## 已知限制

- **資料會變動**：Savant 與 MLB Stats API 偶爾會修正歷史資料，重新下載後少數數字可能和簡報有末位差異。
- **浮點環境**：不同平台的數值函式庫有 1e-16 等級差異，05 的最佳化可能放大到第 3–4 位小數。主要結論的兩位小數不受影響。
- **bootstrap**：沒有 `RUN_BOOT`／`--with-boot` 時，各策略勝場的 95% 區間使用 repo 內附的 `dp_boot_*.csv`。上游資料改變後請重跑 bootstrap。
- **好球帶近似**：以矩形外擴一個球半徑近似 ABS 好球帶，被挑戰球的分類正確率 99.7%。
- **判斷誤差模型**：假設誤差為常態、λ 與情境無關，|m| > 6 英吋的球不進估計。
- **動態規劃**：C 以同類情境的平均估計，狀態不含跑者與球數，並假設改判之後比賽發展路徑不變。

## 程式風格

```bash
pip install ruff==0.14.0 black==25.9.0
ruff check . && black --check .
```
