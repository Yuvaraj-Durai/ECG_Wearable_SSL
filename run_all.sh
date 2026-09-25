#!/bin/bash
# One command for the whole project (Lai et al., Nat. Commun. 2023, reproduced on public data).
#   ./run_all.sh          -> rebuild tables + figures from saved results (minutes)
#   ./run_all.sh full     -> build data caches, pre-train, run every experiment, then report (many hours)
set -e
cd "$(dirname "$0")"
source .venv/bin/activate
mkdir -p logs results figures

if [ "$1" = "full" ]; then
  echo "== 1/6 data caches (500 Hz, 0.5 Hz high-pass)"
  [ -f data/cache/chapman_X.npy ] || python -m src.data chapman
  [ -f data/cache/code15_X.npy ]  || python -m src.data code15
  [ -f data/cpsc2018/files.txt ] && grep -q CPSC_DOWNLOAD_DONE logs/download_cpsc2018.log 2>/dev/null \
    || scripts/download_cpsc2018.sh > logs/download_cpsc2018.log 2>&1
  [ -f data/cache/cpsc2018_X.npy ] || python -m src.data cpsc2018
  [ -f data/cache/chapman_test_noisy_X.npy ] || python -m src.noisy_test
  echo "== 2/6 self-supervised pre-training (MoCo + distributional divergence, GPU 1)"
  [ -f checkpoints/pretrain/moco.DONE ] || CUDA_VISIBLE_DEVICES=1 nohup python -m src.pretrain --epochs 50 >> logs/pretrain.log 2>&1 &
  echo "== 3/6 all fine-tuning runs (both GPUs, load-aware; finished runs are skipped)"
  python scripts/scheduler.py
  echo "== 4/6 CPSC2018 voting ensemble"
  python -m src.cpsc
  echo "== 5/6 class activation maps"
  python -m src.cam --tag msdnn_pw_aug_s0
fi

echo "== tables and figures"
python -m src.report > logs/report.log 2>&1
echo "Done. Results: results/RESULTS.md, figures/"
