#!/usr/bin/env bash
# Full 3-seed run on 8×H100
# Usage: bash run_3seed.sh
set -euo pipefail
echo "=== SP4096 Modern Stack + Polar Express NS — 3-Seed Full Run (8×H100) ==="
echo "GPU count: $(python3.12 -c 'import torch; print(torch.cuda.device_count())')"

MATCHED_FINEWEB_REPO_ID=kevclark/parameter-golf \
python3.12 data/cached_challenge_fineweb.py --variant sp4096

for SEED in 42 314 999; do
  echo "=== Seed ${SEED} ==="
  RECUR_LAYERS=4,5 RECUR_START_STEP=3000 PARALLEL_START_LAYER=7 \
  QK_GAIN_INIT=5.0 MUON_WD=0.090 EMBED_WD=0.090 MATRIX_LR=0.022 \
  SEED=$SEED \
  TRACKIO_RUN_NAME="sp4096_polarexpress_seed${SEED}" \
  torchrun --standalone --nproc_per_node=8 \
    records/track_10min_16mb/2026-04-04_SP4096_ModernStack_PolarExpress/train_gpt.py \
    2>&1 | tee "run_seed${SEED}.log"

  echo "=== Seed ${SEED} Results ==="
  grep -E "val_bpb|model_params|final|stopping_early|artifact" "run_seed${SEED}.log" | tail -10
done

echo "=== 3-Seed Summary ==="
for SEED in 42 314 999; do
  echo -n "Seed ${SEED}: "
  grep "val_bpb" "run_seed${SEED}.log" | tail -1
done
