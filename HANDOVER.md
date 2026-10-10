# RAGGA Handover Document

## Current State (post-W14 bisection)

| item | status |
|---|---|
| CPU suite | **179 tests green** (`python3 -m pytest src/rag/tests -q`) |
| frame check | **OK** — CUDA FHT kernel agrees with reference dequant |
| S-read / conv paths | **OK** — all variants pass quant/dequant roundtrip |
| true-doc (raw cache) | **OK** (conf 0.895) — model generates correctly from raw cache |
| full install | **GARBAGE** (conf 0.429, rep 0.64) — FAILS gen gate |
| FLA-off (torch decode) | **SAME GARBAGE** — Triton route NOT the issue |

---

## The W14 Finding: TQ Layer Itself Is the Issue

### What the bisection proved

1. **frame check: OK** — CUDA FHT kernel matches reference dequant (rel-MSE ~0.02)
2. **S-read: OK** — all variants (reseed, s-only, conv-only, full) pass read checks
3. **conv: OK** — all variants pass conv read/reconstruction checks
4. **true-doc: OK** — model generates `"2?2?2+2?2?2?"` with conf 0.895 from RAW cache
5. **full install: GARBAGE** — same installed state through TQ layer fails with `\n\n\n\n...` garbage
6. **FLA-off: SAME** — disabling Triton decode (`FLUTE_NO_FLA=1`) gives identical garbage

### What this rules out

- ~~CUDA FHT kernel disagrees with reference~~ — frame check OK
- ~~S-read device/frame bug~~ — all S-read checks pass
- ~~Conv path corruption~~ — conv reads match reference
- ~~Semantic continuation (H-SEM)~~ — true-doc generates correctly from the SAME cache state
- ~~Triton FLA decode route~~ — FLA-off gives identical garbage

### What remains

**The TQ quantization layer itself** — specifically the interaction between:

1. `install_snapshot` writing codes to `TQCache`
2. `TQLinearAttentionLayer` reading those codes during generation

The model generates correctly from the RAW cache (true-doc), but incorrectly from the TQ-wrapped cache (full). The issue is NOT in:
- The underlying model state (true-doc works)
- The quant/dequant math (frame/S-read/conv all OK)
- The FLA Triton kernel (FLA-off same result)

The issue IS in:
- How `TQLinearAttentionLayer`'s _StateView reads installed codes during forward
- Or how `install_snapshot` writes codes in a way readable by quant/dequant but not by the layer's compute path

---

## Required Investigation

### Hypothesis: _StateView device/shape mismatch

The installed codes are readable by the quantizer's `dequant` but `TQLinearAttentionLayer._StateView` may:
- Read from stale `_s_codes` / `_conv_codes` attributes
- Have device mismatch between codes and layer weights
- Have shape mismatch between dequantized output and expected state geometry

### Debug script needed

```python
# After install_snapshot, before generation:
# 1. Check _s_codes device/shape on each layer
# 2. Call layer._state_view.s directly, compare to dequant(s_codes)
# 3. Trace the read path through a single forward pass

# The true-doc control proves the state is correct.
# The TQ layer must be reading it incorrectly.
```

### Key files

- `src/rag/tq_cache.py` — `TQLinearAttentionLayer`, `_StateView`, s_codes/conv_codes setters
- `src/rag/install.py` — `install_snapshot`, code summing and attribution
- `src/rag/snapshot.py` — `Snapshot`, `load_snapshot`
- `src/rag/turboquant.py` — `TurboQuant.dequant`, code validation

---

## Bisection Matrix (W14)

```
variant      S-read   conv     gen        conf   rep
reseed       OK       OK       OK        0.331  0.00
s-only       OK       OK       OK        0.343  0.00
conv-only    OK       OK       OK        0.280  0.00
full         OK       OK       GARBAGE   0.429  0.64
true-doc     -        -        OK        0.895  0.00
```

Same matrix with `--fla-off` — identical results.

---

## Model Details

- Qwen3.5-9B palettized — `/home/ubuntu/qwen3_5_9B_palettized` (+ `_heads`)
- vocab_size 248,320; EOS `<|im_end|>` (id 248,046)
- 32 layers ([L,L,L,F]×8 — 24 linear + 8 full-attention)
- GDN geometry: k_heads 8 / v_heads 32, head dims 128
- conv window 24,576 = 6,144×4 (FHT segments 16,384 + 8,192)
- S = 524,288 per layer

## Test Command

```bash
cd /home/ubuntu/RAGGA && python3 -m pytest src/rag/tests -q --tb=short
```

179 tests pass on CPU.
