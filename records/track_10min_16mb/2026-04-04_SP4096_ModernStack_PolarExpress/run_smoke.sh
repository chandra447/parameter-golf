#!/usr/bin/env bash
set -euo pipefail
echo "=== SP4096 Modern Stack + Polar Express NS — Smoke Test ==="
echo "GPU count: $(python3 -c 'import torch; print(torch.cuda.device_count())')"
pip install trackio brotli sentencepiece --quiet 2>&1 | tail -2

MATCHED_FINEWEB_REPO_ID=kevclark/parameter-golf \
python3 data/cached_challenge_fineweb.py --variant sp4096 --skip-manifest

RECUR_LAYERS=4,5 RECUR_START_STEP=3000 PARALLEL_START_LAYER=7 \
QK_GAIN_INIT=5.0 MUON_WD=0.090 EMBED_WD=0.090 MATRIX_LR=0.022 SEED=314 \
TRACKIO_RUN_NAME="sp4096_polarexpress_smoke_seed314" \
torchrun --standalone --nproc_per_node=1 \
  records/track_10min_16mb/2026-04-04_SP4096_ModernStack_PolarExpress/train_gpt.py \
  2>&1 | tee smoke_seed314.log

echo "=== Results ==="
grep -E "val_bpb|step_avg|model_params|final|stopping_early" smoke_seed314.log | tail -20
