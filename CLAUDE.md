# Parameter Golf — Claude Code Workspace

## Challenge Overview

**OpenAI Model Craft Challenge: Parameter Golf**
- **Goal**: Train the best language model that fits in a **16MB artifact** and trains in under **10 minutes on 8×H100s**
- **Metric**: Bits per byte (BPB) on the FineWeb validation set — lower is better
- **Current SOTA**: 1.1147 BPB (AR Self-Gen GPTQ + XSA-all + BigramHash 3072×112)
- **Challenge deadline**: April 30, 2026
- **Baseline**: 1.2244 BPB (9L, 512d, 1024 vocab, TiedEmbeddings)

The challenge optimizes **L(N)** — lowest loss given fixed number of parameters — unconstrained by data, compute, steps, or architecture.

---

## Repository Structure

```
parameter-golf/
├── train_gpt.py          # Main training script (H100/CUDA)
├── train_gpt_mlx.py      # MLX training script (Apple Silicon)
├── data/                 # Dataset scripts and tokenizer specs
│   ├── cached_challenge_fineweb.py
│   ├── download_hf_docs_and_tokenize.py
│   └── tokenizer_specs.json
├── records/
│   ├── track_10min_16mb/          # Official leaderboard submissions
│   └── track_non_record_16mb/    # Unlimited compute submissions
└── requirements.txt
```

---

## Git Strategy — Hypothesis-Driven Development

Every experiment is a branch. Treat the repo like a lab notebook.

### Branch Naming Convention

```
exp/<date>-<short-hypothesis>
# Examples:
exp/2026-04-04-gptq-wider-bigramhash
exp/2026-04-05-ternary-quant-unet
exp/2026-04-06-mamba-depth-recurrence
```

### Experiment Workflow

1. **Hypothesize** — Write a brief hypothesis (see Research Workflow below)
2. **Branch** — `git checkout -b exp/<date>-<hypothesis>`
3. **Implement** — Make targeted changes to `train_gpt.py`
4. **Train** — Run on RunPod (see Training section)
5. **Record** — Log results in `records/` with a README following the existing format
6. **Commit** — Detailed commit with BPB result and delta vs SOTA
7. **Merge or discard** — Merge into `main` if it improves; tag and archive if not

### Commit Message Format

```
exp: <short description> — <BPB result>

Hypothesis: <what we expected>
Result: <what happened>
Delta vs SOTA: <+/- BPB>
Key changes:
- <change 1>
- <change 2>
```

### Record Directory Format

Follow the existing pattern in `records/track_10min_16mb/`:
```
records/track_10min_16mb/<date>_<ShortName>/
└── README.md   # Architecture table, results table, main changes, ablations
```

---

## Research Workflow — Hypothesis Generation

Before implementing anything new:

### 1. Study Existing Submissions
- Read ALL READMEs in `records/track_10min_16mb/` to understand the lineage
- Track which PRs introduced which innovations (PR numbers reference the upstream openai/parameter-golf repo)
- Map the improvement stack: Baseline → each delta

### 2. Identify Gaps & Opportunities
Cross-check the "Requests for PRs" list in the main README:
- [ ] JEPA
- [ ] Text diffusion
- [ ] H-net tokenization
- [ ] Universal transformer / depth recurrence
- [ ] Megakernels
- [ ] State-space models / E2E TTT
- [ ] Learning adapters on random linear maps

### 3. Write a Hypothesis Document

For each experiment, create a brief wiki entry before coding:

```markdown
## Hypothesis: <name>

**Date**: <date>
**Based on**: <PR / paper / idea source>
**Expected delta**: ~X BPB improvement

### Motivation
<Why do we think this will work?>

### Mechanism
<How does it work mechanically?>

### Risk / Unknowns
<What could go wrong? What are we uncertain about?>

### Implementation Plan
1. <step 1>
2. <step 2>

### Success Criteria
- BPB < current SOTA (<X.XXXX)
- Artifact size ≤ 16MB
- Training time ≤ 600s on 8×H100
```

Store hypothesis documents in `experiments/hypotheses/<date>-<name>.md`.

### 4. Research Resources
- Study upstream PR discussion on the openai/parameter-golf GitHub for failed attempts and ablations
- Key techniques already validated: XSA, BigramHash, GPTQ (Full Hessian), Partial RoPE, LN Scale, SmearGate, EMA, Int6 QAT, Sliding Window Eval, U-Net skips, MLP 3×

---

## Training Infrastructure

### RunPod (Primary — H100s)

Training is done on RunPod with H100 instances. The HF token must be passed as an environment variable.

```bash
# Get HF token from environment
export HF_TOKEN=$(printenv HF_TOKEN)

# When launching RunPod pod, pass env vars:
# HF_TOKEN=$HF_TOKEN
```

**RunPod setup for each experiment:**
```bash
# Install dependencies
pip install -r requirements.txt

# Download data (uses HF token)
HF_TOKEN=$HF_TOKEN python data/download_hf_docs_and_tokenize.py

# Run training
python train_gpt.py
```

**Target hardware**: 8×H100 SXM (for official leaderboard), single H100 acceptable for ablation runs.

**For ablation runs** (proving concept before full 8×H100 run):
- Run fewer steps to check convergence direction
- Compare pre-quant BPB trend, not final artifact BPB

### Local (Apple Silicon — Ablations)

```bash
python train_gpt_mlx.py
```

Use for quick architecture sanity checks only. Not representative of H100 performance.

---

## Trackio — Experiment Tracking

All training runs must log to Trackio.

- **Space**: https://huggingface.co/spaces/Chandra447/trackio
- **HF Token**: Read from `$HF_TOKEN` environment variable — never hardcode

### Logging Integration

Add Trackio logging to `train_gpt.py` for each experiment:

```python
import trackio

# Initialize at start of training
trackio.init(
    project="parameter-golf",
    run_name=f"{experiment_name}_{seed}",
    config={
        "layers": num_layers,
        "dim": dim,
        "mlp_mult": mlp_mult,
        "quantization": quant_scheme,
        "seed": seed,
    }
)

# Log at each step
trackio.log({
    "train/loss": loss.item(),
    "train/bpb": loss.item() / math.log(2),
    "train/step_ms": step_time_ms,
    "train/lr": current_lr,
})

# Log final results
trackio.log({
    "eval/bpb": val_bpb,
    "eval/artifact_size_mb": artifact_size / 1e6,
})

trackio.finish()
```

### Run Naming Convention

```
<experiment_name>_seed<N>
# Example: gptq_wider_bigramhash_seed42
```

---

## SOTA Architecture Reference

### Merged SOTA (1.1147 BPB — PR #1019, our previous baseline)

| Component | Setting | Source |
|-----------|---------|--------|
| Layers | 11 (512d, 8 GQA heads, 4 KV heads) | Baseline |
| MLP | 3× (1536) with LeakyReLU(0.5)² | PR #493 |
| Attention | XSA on all 11 layers | PR #478 |
| BigramHash | 3072 × dim=112 | PR #1019 |
| RoPE | Partial (16/64 dims) | PR #315 |
| LN Scale | 1/√(layer+1) | PR #315 |
| VE128 | Layers 9-10 | PR #374 |
| SmearGate | Position-mixing gate | PR #65 |
| U-Net skips | Encoder-decoder connections | PR #289 |
| Quantization | Full Hessian GPTQ (AR self-gen calib) | PR #1019 |
| Compression | lzma preset=9 | PR #1019 |
| Evaluation | Sliding window stride=64 | PR #549-era |
| Training | Muon optimizer, WD=0.04, warmdown 3500 | Various |

### Pending Best Non-SLOT (1.0897 BPB — PR #1334, our next target)

| Component | Setting | Source |
|-----------|---------|--------|
| Vocabulary | SP4096 SentencePiece (replaces 1024-BPE) | PR #1218 |
| Layers | 11 (512d, 8 GQA heads, 4 KV heads) | Baseline |
| MLP | 4× (2048) with LeakyReLU(0.5)² | PR #1218 (enabled by high WD) |
| Attention | XSA on all 11 layers | PR #478 |
| Depth recurrence | Layers 4,5 share MLP weights (virtual 13L) | PR #1204, #1260 |
| Parallel residuals | From layer 7+ (attn + MLP in parallel) | PR #1289, #1334 |
| QK-Gain | 5.0 (per-head learnable query scaling) | PR #1125 |
| RoPE | Partial (16/64 dims) | PR #315 |
| LN Scale | 1/√(layer+1) | PR #315 |
| Optimizer | MuonEq-R (row-normalized) + Polar Express NS | PR #1260, #1332 |
| Weight Decay | 0.085-0.095 (compression lever) | PR #1285 |
| Quantization | Full Hessian GPTQ int6 | PR #1019 |
| Compression | LZMA/Brotli | Various |
| Evaluation | Sliding window stride=64 | PR #549-era |
| **Removed** | BigramHash, SmearGate (subsumed by SP4096) | PR #1218 |

### Pending Best with Causal SLOT (1.0766 BPB — PR #1333)

Same as above + Causal SLOT-16 eval-time hidden-state delta optimization (context-only, provably legal)

---

## Constraints (Hard Rules)

1. **Artifact ≤ 16MB** — total saved model file after quantization + compression
2. **Training ≤ 600s on 8×H100 SXM** — wall-clock time for leaderboard submissions
3. **No val data during training** — calibration data for quantization must be self-generated or from training split
4. **Tokenizer-agnostic evaluation** — BPB metric, not token-level perplexity

---

## Experiment Ideas Backlog

Ordered by **current priority** based on accumulated findings. See `experiments/next-experiment-recommendation.md` for full reasoning.

### What Stride-16 Taught Us (2026-04-04)

Stride-16 eval gave +0.00004 BPB — noise. The SOTA model is **not context-starved**: XSA + stride=64 already gives 1,984 tokens of left context per token; stride=16 adds 48 more. This rules out eval-time tricks as a productive direction and updates the bottleneck hypothesis:

- **Bottleneck is NOT**: context access (stride-16 proved this)
- **Bottleneck IS**: representation richness per parameter, training signal quality, and capacity per byte

This elevates MTP (forces richer representations) and MiniPLM (better training signal) to the top of the queue, and de-prioritizes any eval-time or context-coverage experiments.

### What BigramHash 4096 Taught Us (2026-04-04)

BigramHash 4096×112 was neutral on 3-seed 8×H100 (+0.00009 BPB vs SOTA). The 1×H100 smoke test (−0.0021) was misleading — full GPTQ + proper eval washed out the gain. Lessons:

- **1×H100 smoke tests are directional only** — always validate with 3-seed 8×H100
- **Embedding table widening has diminishing returns** at 3072→4096 — collision reduction doesn't translate to BPB at this scale
- **Artifact size is tight** — seed 42 busted 16MB limit. Use `TARGET_MB=15.7` for larger embeddings

### What PR #1222 (Prime MLP TTT) Taught Us (2026-04-04)

abaybektursun's TTT study (PR #1222) conclusively showed:
- **Naive TTT is an artifact of short-context eval** — with proper sliding window, TTT provides zero benefit (+0.0001 BPB at best LR)
- **Prime MLPs (separate bf16 adapters)** bypass GPTQ incompatibility but still don't help on strong models
- **FOMAML meta-learning** improves base model (−0.111 BPB) but costs 44% training budget and TTT adds only −0.001 on top
- **This kills all TTT approaches** for the SOTA model — deprioritize TTT entirely

### Priority Queue

| Priority | Idea | Expected Delta | Status | Notes |
|----------|------|---------------|--------|-------|
| **▶ NEXT** | **SP4096 Modern Stack Rebase** | **−0.025 BPB (→ ~1.090)** | **🔲 Implement** | Fork PR #1334. SP4096 vocab, MLP 4x, WD=0.09, MuonEq-R, QK-Gain 5.0, depth recurrence L4-5, parallel residuals L7+, Polar Express NS. Proven by multiple teams. |
| 2 | Causal SLOT-16 eval | −0.01 to −0.02 BPB | 🔲 After rebase | Hidden-state delta optimized on context-only positions at eval time. Provably legal. PR #1333. |
| 3 | 3-Layer Recurrence (L3,4,5) + WD=0.095 | −0.001 BPB | 🔲 Incremental | Extend depth recurrence. PR #1331. |
| 4 | Full SLOT (if legal) | −0.3 to −0.4 BPB | 🔲 Legality unclear | SLOT-48 on all positions. Legality debated (Issue #1336). If legal, single biggest improvement. |
| 5 | RWKV-7 hybrid (SSM + sparse attn) | Unknown | 🔲 To try | Requested by OpenAI. Linear RNN = O(1)/token. Needs Triton kernel work. |
| 6 | BLT / H-Net byte tokenization | Unknown | 🔲 To try | Requested by OpenAI. Eliminates vocab table, native bpb output. Fundamental architecture change. |
| 7 | Mamba SSM (PR #1342) | Unknown | 🔲 Watch | Just submitted. Byte-level 260-vocab. Results unknown. |

### What the Competition Landscape Taught Us (2026-04-04, late session)

Deep research revealed the competition moved far beyond our 1.1147 BPB SOTA:
- **SP4096 vocabulary** is the single biggest win (~-0.017 BPB), making BigramHash/SmearGate obsolete
- **Higher WD (0.09)** is a compression lever: smaller weights → better LZMA → more param budget → MLP 4x
- **Simple depth recurrence (layers 4,5)** works where full recursive failed — 2-3 layer sharing is the sweet spot
- **SLOT eval-time optimization** is the game-changer (Causal SLOT is provably legal)
- **JEPA confirmed dead** — two independent teams (PR #1312, #1330) showed fundamental incompatibility with causal LM
- **Trigram Hash killed by critique** — SOTA author already built and disabled it; 326K collisions/bucket is noise
- **MuToR abandoned** — sequence doubling and custom 4D masks incompatible with FlashAttention-3

### Completed / Negative Results

| Idea | Result | Notes |
|------|--------|-------|
| **Stride-16 eval on SOTA** | ❌ NEGATIVE +0.00004 BPB (2026-04-04) | stride=64 already saturates context (1,984 tokens). |
| 4k+ sequence length | ❌ NEGATIVE +BPB (1.2014) | Too slow per step under 600s budget. |
| Plain depth recurrence | ❌ NEGATIVE +0.025 BPB (PR #363) | Quantization compounding + step overhead. Simple 2-3 layer recurrence (L4,5) works (PR #1260). |
| TTT (full weights) on GPTQ stack | ❌ NEUTRAL (25 runs, PR #756) | GPTQ minima incompatible with SGD. |
| **MTP v1 (4 heads, w=0.3)** | ❌ NEGATIVE +0.006 BPB (2026-04-04, 1×H100 smoke) | Step penalty, gradient crowding, eval contamination. |
| **BigramHash 4096×112 full** | ❌ NEUTRAL +0.00009 BPB (2026-04-04, 3-seed 8×H100) | Diminishing returns on hash embedding widening. Seed 42 over 16MB. |
| **Prime MLP TTT (PR #1222)** | ❌ NEUTRAL (abaybektursun, 2026-04-03) | TTT is artifact of short-context eval. Kills all TTT approaches. |
| **MuToR Register Tokens** | ❌ ABANDONED (2026-04-04) | FA3 can't express custom 4D masks. Sequence doubling fatal under 600s. |
| **Trigram Hash Embedding** | ❌ KILLED BY CRITIQUE (2026-04-04) | 326K collisions/bucket = noise. SOTA author built & disabled it. SP4096 subsumes hash embeddings. |
| **MiniPLM Data Reweighting** | ❌ RULED OUT (2026-04-04) | Requires teacher model — violates competition spirit. |
| **JEPA** | ❌ NEGATIVE 1.46 BPB (PR #1312, #1330) | Fundamental incompatibility with causal LM. Task collapses. |

---

## Key Principles

- **Signs of life first**: A new algorithm doesn't need to immediately beat SOTA. Prove it can beat baseline in N steps, then optimize for speed.
- **Ablate one thing at a time**: Stack changes make it impossible to know what worked.
- **3-seed mean for records**: Always report mean ± std across seeds 42, 314, 999.
- **Significance testing**: Use Welch's t-test for comparing to SOTA (see existing record READMEs for format).
- **Document failures**: Failed experiments are as valuable as successes. Record what didn't work and why.
