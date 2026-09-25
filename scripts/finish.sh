#!/bin/bash
# Unattended completion: CPSC2018 cache once downloaded -> wait for all scheduled runs
# -> CPSC voting ensemble -> CAM -> tables and figures.
#   nohup scripts/finish.sh > logs/finish.log 2>&1 &
cd "$(dirname "$0")/.."
source .venv/bin/activate
until grep -q CPSC_DOWNLOAD_DONE logs/download_cpsc2018.log 2>/dev/null; do sleep 120; done
[ -f data/cache/cpsc2018_X.npy ] || { echo "$(date) building CPSC2018 cache"; python -m src.data cpsc2018; }
until grep -q ALL_JOBS_DONE logs/scheduler.log 2>/dev/null; do sleep 300; done
echo "$(date) CPSC2018 ensemble"; python -m src.cpsc
echo "$(date) CAM"; CUDA_VISIBLE_DEVICES="" python -m src.cam --tag msdnn_pw_aug_s0
echo "$(date) report"; CUDA_VISIBLE_DEVICES="" python -m src.report > logs/report.log 2>&1
echo "$(date) FINISH_DONE"
