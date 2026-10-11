# RAGGA Status Report

## Tests: 229 PASSED

All tests pass with device-parametric fixes. Two ordering issues were fixed in `evals.py` - the Stub classes (lines 309 and 407) had generator-pinned tensor creation without device placement. Fixed by adding `DEVICE` constant and moving tensors to device after CPU generator-pinned creation.

```
python3 -m pytest src/rag/tests -q
229 passed
```

## Build Pairs: WORKING

Fixed HuggingFace dataset namespace resolution. The old `glue`, `paws`, `snli`, `squad` identifiers no longer work with current `datasets` library. Updated `SOURCES` dict in `build_pairs.py`:

- `glue` → `nyu-mll/glue`
- `paws` → `google-research-datasets/paws`
- `snli` → `stanfordnlp/snli`
- `squad` → `rajpurkar/squad`

Fixed `rows_qqp` to use `r["label"]` instead of `r["is_duplicate"]` (API change).

Result: 33,595 pairs built (16,076 positive) from 7 sources.

## M1/M2 Gate Training: PROBLEMATIC

### Performance Issue

Training is **extremely slow**: ~22 seconds per step. At 300 steps, this is ~110 minutes total.

Per-step breakdown:
- Batch size: 8 pairs
- Each step: 8 pairs × 2 sides × 24 layers = 384 M1/M2 write operations
- The W18 replay recomputes deltas per-call for gradient exactness

### Convergence Issue - CRITICAL

After 50 steps (~18 minutes):

| Step | Loss | Cos(pos) | Cos(neg) | Gap |
|------|------|----------|----------|-----|
| 0 | 2.0794 | +0.000 | +0.000 | 0.000 |
| 25 | 2.0794 | +0.000 | +0.000 | 0.000 |
| 50 | 2.0794 | +0.000 | +0.000 | 0.000 |

**The gates are not learning.** Loss unchanged, cosine similarities flat at 0.000. This suggests:

1. **Gradient signal too weak** - The contrastive loss may not be propagating to the gates effectively
2. **Gate init issue** - Gates start at write=0.0, read=1.0 (P3 init). With write_gate near zero, M1/M2 updates are vanishingly small
3. **Learning rate too low for this architecture** - The lr=0.01 may need to be higher for gate weights specifically
4. **Potential bug in gradient flow** - The W18 `_GatedMemoryReplay` backward pass may not be accumulating correctly

### What the previous handover claimed vs reality

The W18 handover claimed:
> "gate gradients EXACTLY equal to the plain chain's; multi-call layers accumulate"

But after 50 steps, gates show zero differentiation. Either:
- The test suite passes but production training has a different code path
- The gradient exactness is mathematically correct but numerically vanishing
- The loss function doesn't actually train the gates (read_gate in particular is never touched by contrastive loss)

## Outstanding Issues

1. **Training not converging** - Root cause unknown. Need to:
   - Add gradient magnitude logging to the training loop
   - Check if `write_gate_k`, `write_gate_v`, `read_gate` actually have non-zero gradients after backward
   - Consider if the loss function design connects to the gates at all

2. **Read-gate not trainable** - The handover admits: "the contrastive loss never sees read_gate". This is a fundamental architecture gap, not a bug.

3. **Extremely slow training** - 22s/step is impractical for iterating on the model. Need to profile where time is spent:
   - Prefill?
   - Replay backward (the delta recomputation)?
   - Data loading?

## Files Modified This Session

1. `src/rag/evals.py` - Added `DEVICE` constant, fixed two Stub classes for device placement
2. `scripts/gpu/build_pairs.py` - Updated SOURCES dict with correct HF namespaces, fixed QQP field name

## Next Steps (NOT recommendations, just options)

1. Debug why gates aren't learning:
   - Add `print(model.m1m2.write_gate_k.grad.abs().max())` after backward
   - If zero, the gradient path is broken

2. Profile the slow training:
   - Is it the W18 replay backward?
   - Is it prefill per batch?

3. Reconsider the architecture:
   - Should write_gate_k/v be initialized higher?
   - Does the InfoNCE loss actually connect to M1/M2 state?
