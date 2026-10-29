#!/usr/bin/env bash
# ABS 挑戰研究：完整流程（00–14）
#
# 用法：
#   bash run_all.sh                 # 已有 data/raw 時，只跑分析 02–14（約 3–5 分鐘）
#   bash run_all.sh --with-fetch    # 先下載 Statcast（00）與 playByPlay（01），約 40–60 分鐘
#   bash run_all.sh --with-boot     # 另外重跑 bootstrap（06：200 次、11：100 次），約 1 小時
#
# 已在別處下載過 Statcast：export ABS_STATCAST_DIR=/path/to/statcast 後再執行，可省略 00。
set -euo pipefail
cd "$(dirname "$0")"
CAF=""; command -v caffeinate >/dev/null 2>&1 && CAF="caffeinate -i"   # Mac：執行期間不休眠
PY="$CAF python"

if [[ " $* " == *" --with-fetch "* ]]; then
  $PY scripts/00_fetch_statcast.py
  $PY scripts/01_fetch_feed.py
fi
$PY scripts/02_build_challenges.py
$PY scripts/03_build_opportunities.py
for p in fit valid gain; do $PY scripts/04_win_expectancy.py --part $p; done
$PY scripts/05_perception.py
$PY scripts/06_option_value.py --iters 60
$PY scripts/06_option_value.py --sens
if [[ " $* " == *" --with-boot "* ]]; then
  for s in 0 25 50 75 100 125 150 175; do $PY scripts/06_option_value.py --boot $s 25; done
fi
for p in card h2 forward teams; do $PY scripts/07_analysis.py --part $p; done
$PY scripts/09_rd.py
for p in path state rd; do $PY scripts/10_count_checks.py --part $p; done
$PY scripts/11_runs_objective.py
if [[ " $* " == *" --with-boot "* ]]; then
  for s in 0 50; do $PY scripts/11_runs_objective.py --boot $s 50; done
  $PY scripts/11_runs_objective.py --summary
fi
$PY scripts/12_savant_check.py
$PY scripts/13_slide_extras.py
$PY scripts/08_figures.py
$PY scripts/14_slide_figures.py
echo "完成。打開 notebooks/reproduce_slides.ipynb 逐頁核對簡報數字。"
