# Hypothesis: SP4096 Modern Stack + Polar Express NS

**Status**: PENDING SMOKE TEST
**Date**: 2026-04-04
**Based on**: PR #1334 (SP4096 + Depth Recurrence + Parallel Residuals + MuonEq-R + QK-Gain 5.0)
**Expected delta**: ~−0.001 to −0.002 BPB from Polar Express NS on top of PR #1334 (1.0897)

---

## Motivation

PR #1334 achieves 1.0897 BPB with the full modern stack. This experiment adds one clean, independent improvement:

**Polar Express Newton-Schulz** — replaces the standard 5-step single-coefficient NS polynomial with 4-step minimax-optimal coefficients from arXiv:2505.16932. The minimax coefficients minimize worst-case singular value deviation, yielding a better-conditioned gradient update with one fewer iteration.

PR #1332 tested Polar Express on an older base and got 1.0959 BPB. That run predates depth recurrence and parallel residuals. Since Polar Express is a pure optimizer improvement independent of architecture, it should stack additively on the stronger #1334 foundation.

**Why not NoPE?** Critique analysis ruled it out: layer 0 NoPE removes the only positional signal entering the first computation (catastrophic at 11-layer scale); layer 4 NoPE conflicts with depth recurrence (shared MLP between layers 4,5 would have inconsistent positional encoding across passes); layer 8 NoPE stacks with parallel residuals adding uncontrolled interaction. Zero prior art in this competition.

---

## What We Took from PR #1334 (unchanged)

| Component | Value | Source |
|-----------|-------|--------|
| SP4096 vocabulary | SentencePiece 4096-token BPE | PR #1218 @clarkkev |
| MLP multiplier | 4× (hidden=2048) | PR #1218 @clarkkev |
| Weight decay | muon_wd=0.090, embed_wd=0.090 | PR #1285, #1334 @dexhunter |
| Matrix LR | 0.022 (recovers quality at WD=0.090) | PR #1331, #1344 |
| Depth Recurrence | layers 4,5 share MLP weights from step 3000 | PR #1260 @dexhunter |
| Parallel Residuals | layers 7–10: attn+mlp from same pre-norm x | PR #1289 @MatoTeziTanka |
| MuonEq-R | row-normalize gradient before Newton-Schulz | PR #1260 @dexhunter |
| QK-Gain | init=5.0 per-head query scaling | PR #1217 @bigbag |
| Brotli compression | COMPRESSOR=brotli | PR #1218 @clarkkev |

## What We Added (this experiment)

| Component | Value | Status |
|-----------|-------|--------|
| **Polar Express NS** | 4-step minimax: (3.4445,−4.7750,2.0315), (0.8976,0.1135,0), (1.0,0,0), (1.0,0,0) | NEW |

---

## Architecture Table

| Component | Setting |
|-----------|---------|
| Layers | 11 (512d, 8 GQA heads, 4 KV heads) |
| MLP | 4× (2048d) with LeakyReLU(0.5)² |
| Attention | XSA on all 11 layers |
| Vocabulary | SP4096 SentencePiece BPE |
| RoPE | Partial (16/64 dims) on all layers |
| QK-Gain | init=5.0 learnable per-head |
| VE128 | Layers 9–10 |
| U-Net skips | Encoder-decoder connections (resid_mix) |
| Depth Recurrence | blocks[5].mlp = blocks[4].mlp (step 3000+) |
| Parallel Residuals | Layers 7–10: attn+mlp read from same x_in |
| MuonEq-R | row-normalize gradient before NS |
| Newton-Schulz | **Polar Express 4-step minimax** |
| Quantization | Full Hessian GPTQ int6 + Brotli |
| Evaluation | Sliding window stride=64 |

---

## Expected BPB

| Step | BPB |
|------|-----|
| Merged SOTA (PR #1019) | 1.1147 |
| PR #1334 (our base) | 1.0897 |
| + Polar Express NS | ~1.088 (est. −0.002) |
| **Target** | **~1.088** |

---

## Risk / Unknowns

1. **Polar Express step 3+4 are identity** `(a=1.0, b=0, c=0)` — these are no-ops mathematically but add two matrix multiplications. Benchmarked step time vs baseline to confirm no regression.
2. **bfloat16 numerics**: Polar Express coefficients derived for float32. Verify no NaN/inf in early training steps.
3. **Additive stacking**: PR #1332 showed +0.001 BPB improvement on older stack. Not guaranteed to stack identically on #1334.

---

## Reproduction

```bash
pip install trackio brotli sentencepiece --quiet

# Download SP4096 tokenized data (first time only)
MATCHED_FINEWEB_REPO_ID=kevclark/parameter-golf \
python3 data/cached_challenge_fineweb.py --variant sp4096 --skip-manifest

# Smoke test (1×H100)
RECUR_LAYERS=4,5 RECUR_START_STEP=3000 PARALLEL_START_LAYER=7 \
QK_GAIN_INIT=5.0 MUON_WD=0.090 EMBED_WD=0.090 MATRIX_LR=0.022 SEED=314 \
TRACKIO_RUN_NAME="sp4096_polarexpress_smoke_seed314" \
torchrun --standalone --nproc_per_node=1 \
  records/track_10min_16mb/2026-04-04_SP4096_ModernStack_PolarExpress/train_gpt.py

# Full run (8×H100, 3 seeds)
for SEED in 42 314 999; do
  RECUR_LAYERS=4,5 RECUR_START_STEP=3000 PARALLEL_START_LAYER=7 \
  QK_GAIN_INIT=5.0 MUON_WD=0.090 EMBED_WD=0.090 MATRIX_LR=0.022 SEED=$SEED \
  TRACKIO_RUN_NAME="sp4096_polarexpress_seed${SEED}" \
  torchrun --standalone --nproc_per_node=8 \
    records/track_10min_16mb/2026-04-04_SP4096_ModernStack_PolarExpress/train_gpt.py
done
```

---

## Results

| Seed | Sliding BPB | Artifact | Steps | ms/step |
|------|-------------|----------|-------|---------|
| 42 | PENDING | — | — | — |
| 314 | PENDING | — | — | — |
| 999 | PENDING | — | — | — |
| **Mean** | **PENDING** | | | |

Delta vs SOTA (1.1147): PENDING
Delta vs PR #1334 (1.0897): PENDING

---

## Credits

| Component | Source |
|-----------|--------|
| SP4096 base | PR #1218 @clarkkev |
| WD synergy | PR #1285 @dexhunter |
| Depth Recurrence | PR #1204 @msisovic, PR #1260 @dexhunter |
| Parallel Residuals | PR #1204 @msisovic, PR #1289 @MatoTeziTanka |
| MuonEq-R | PR #1260 @dexhunter (arXiv:2603.28254) |
| QK-Gain 5.0 | PR #1217 @bigbag |
| **Polar Express NS** | This submission (arXiv:2505.16932) |
