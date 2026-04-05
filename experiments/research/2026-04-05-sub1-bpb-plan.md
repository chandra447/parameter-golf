# Path to Sub-1.0 BPB — Research Report (2026-04-05)

## Bug Found: EMA + Depth Recurrence Interaction

**Root cause of +0.87 BPB GPTQ degradation:**

1. `state_dict()` deduplicates shared tensors after `blocks[5].mlp = blocks[4].mlp` at step 3000
2. EMA update loop using `state_dict()` stops updating `blocks.5.mlp.*` EMA slots (frozen at step 3000)
3. `load_state_dict(avg_state)` breaks sharing — block 5 gets stale step-3000 weights
4. GPTQ quantizes broken model → +0.87 BPB

**Fix applied:**
- EMA loop now uses `named_parameters()` with data-pointer dedup + shared-weight sync
- After `load_state_dict`, re-establish `blocks[tgt].mlp = blocks[src].mlp`
- Fixed `for h in hooks` variable shadowing in `collect_hessians`

## Competition Landscape (April 2026)

### Leaderboard (merged SOTA: 1.1147 BPB, PR #1019)

| BPB | PR | Technique | Status |
|-----|----|-----------|--------|
| **0.9300** | #1229 | Scored-Position Full SLOT | Open, legality OK |
| **0.9462** | #1303 | Full SLOT-16 + QK-Gain | Open |
| **1.0766** | #1333 | SP4096 + Depth Recurrence + **Causal SLOT-16** | Open |
| **1.0897** | #1334 | SP4096 + Depth Recurrence + Parallel Residuals + MuonEq-R + QK-Gain | Open |
| **1.0923** | #1344 | SP4096 + Polar Express + 3-layer recurrence | Open |
| **1.1147** | #1019 | AR Self-Gen GPTQ + XSA + BigramHash 3072 | **Merged (official SOTA)** |

### Key Findings

1. **Causal SLOT is legal** (optimizes delta on context-only positions, single left-to-right pass)
2. **Full SLOT is legal** with scored-position masking (PR #1229, #1303)
3. **L-BFGS Causal SLOT** (PR #1350, #1372) reaches ~1.005 BPB on old stack
4. **Gated DeltaNet** (PR #875, #1370): pure architecture, 1.003-1.023 BPB
5. Sub-1.0 BPB is achievable: PR #1229 gets 0.93 BPB with scored-position SLOT

## Roadmap to Sub-1.0 BPB

### Phase 1: Fix and Validate (~1.09 BPB)
- [x] Fix EMA + depth recurrence bug
- [ ] Re-run 3-seed on 8×H100 with fix
- [ ] Validate GPTQ roundtrip delta < 0.05 BPB
- Expected: ~1.09 BPB (matching PR #1334)

### Phase 2: Add SLOT (~1.07 BPB)
- [ ] Implement Causal SLOT-16 (PR #1333 reference)
  - Separate `forward_hidden()` and `compute_logits()` in GPT class
  - Add delta parameter `[bsz, 1, d_model]`
  - AdamW optimizer, lr=0.008, 16 steps on context-only positions
- Expected: ~1.077 BPB (Causal SLOT adds −0.013 BPB)

### Phase 3: Stronger SLOT (~1.00-1.05 BPB)
- [ ] Replace AdamW with L-BFGS for SLOT optimization (PR #1350)
- Expected: ~1.005 BPB (L-BFGS is significantly stronger for small delta optimization)

### Phase 4: Scored-Position SLOT (~0.93 BPB)
- [ ] Implement scored-position masking (PR #1229)
- [ ] Add logit bias parameter `[bsz, 1, vocab]`
- Expected: ~0.93 BPB

## Key Technical Details for SLOT Implementation

### Causal SLOT-16 (Legal, PR #1333)
```python
# Per sliding window (stride=64):
# 1. Run forward pass → get frozen hidden states H
# 2. Split: context = 0..1983, new = 1984..2047
# 3. delta = zeros([bsz, 1, d_model])
# 4. 16 AdamW steps minimizing CE on context positions only
# 5. Score new positions with compute_logits(H + delta)
# 6. Reset delta for next window

SLOT_ENABLED=1 SLOT_LR=0.008 SLOT_STEPS=16
```

### L-BFGS Causal SLOT (PR #1350)
- Replace AdamW with L-BFGS (torch.optim.LBFGS)
- ~1.005 BPB even on old PR #1019 stack
- 18% faster than cascaded variant

### Scored-Position Full SLOT (PR #1229, Sub-1.0)
- Delta shape: `[bsz, 1, d_model]` + logit bias `[bsz, 1, vocab]`
- Scored-position masking (only last stride tokens per non-first window)
- 16 AdamW steps, cosine LR 0.008→0.0008
- Eval time: ~384s on 8×H100

## Risk Assessment

| Risk | Mitigation |
|------|-----------|
| GPTQ still degrades | Test roundtrip delta on 1×H100 smoke before full run |
| SLOT eval too slow | L-BFGS is faster than AdamW; budget ~300s for eval |
| SLOT legality challenged | Causal SLOT has strong theoretical backing; scored-position is accepted |
| Credit budget tight | Validate every step on cheap 1×H100 before scaling |
