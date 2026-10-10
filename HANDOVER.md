# RAGGA Handover Document

## Current State

### What's Done
1. **All 161 tests passing** - Fixed 9 failing tests to run on CUDA (169 after the CPU-box W10 wave added the conv-padding contract tests)
2. **GPU memory profile completed** - Model loads at 6GB, forward pass at 8-9GB peak
3. **EnterpriseRAG-Bench loaded** - 512K documents in `/home/ubuntu/RAGGA/disk/enterprise_rag_bench/`

### Critical Blocker: TurboQuant Power-of-Two Requirement

> **RESOLVED on the CPU box (TASKS W10, SPECIFICATION N16.1):** conv windows
> with a non-power-of-two flattened size are zero-padded to the next power of
> two INSIDE `TQLinearAttentionLayer` (`tq_cache.py`) — 24,576 → 32,768, the
> canonical conv d, so the D3 rotation (seed 202) stays shared; dequant
> strips the pad, the stored norm is unchanged. Power-of-two geometries are
> bit-identical to before. Pull `main` before re-running ingestion; the fix
> is covered by `src/rag/tests/test_tq_cache.py` contract 11 (169 tests total).

**The Issue:**
- TurboQuant requires power-of-two dimensions for FHT (Fast Hadamard Transform)
- The conv state from Qwen3.5's in_proj has dimension 24,576 = 8192 + 16384 (Q+V combined)
- 24,576 is NOT a power of two
- This breaks `TQCache` forward passes during ingestion

**Files Involved:**
- `/home/ubuntu/RAGGA/src/rag/turboquant.py` - Line 289 enforces power-of-two check
- `/home/ubuntu/RAGGA/src/rag/tq_cache.py` - `update_conv_state` calls `TurboQuant.quant()` with conv dimension
- `/home/ubuntu/RAGGA/src/scripts/modeling.py` - Line 586 calls `cache_params.update_conv_state()` with combined Q+V tensor

**The Fix Needed:**
Either:
1. Pad conv state to next power-of-two (32768) in `tq_cache.py` before quantizing
2. Or update TurboQuant to use segmented FHT (the FHT kernel already supports it via `fht_segments`)
3. Or split the conv state into Q/V components separately

---

## Model Details

### Palettized Model Location
- **Body:** `/home/ubuntu/qwen3_5_9B_palettized/` (4.63 GiB)
- **Heads:** `/home/ubuntu/qwen3_5_9B_palettized_heads/` (1.72 GiB)
- **Format:** idxN hybrid palettization (variable 1-4 bit widths)

### Model Config
```
vocab_size: 248320
num_layers: 32
linear_key_head_dim: 128
linear_value_head_dim: 128
linear_num_key_heads: 16
linear_num_value_heads: 32
linear_conv_kernel_dim: 4
```

### Conv State Dimensions
- Q conv: 8192 = 16 * 4 * 128 (POWER OF TWO ✓)
- V conv: 16384 = 32 * 4 * 128 (POWER OF TWO ✓)
- Combined in model: 24576 = Q + V (NOT power of two ✗)

---

## GPU Memory Profile Results

```
Model load: 6,067 MiB
Forward 128 tokens: 8,199 MiB peak
Forward 256 tokens: 8,328 MiB peak
Forward 512 tokens: 8,581 MiB peak
Forward 1024 tokens: 9,092 MiB peak

Available VRAM: 22,590 MiB
Headroom: ~13,000 MiB for ingestion
```

---

## Changes Made This Session

### 1. Fixed Tests for CUDA (all passing)
- `src/rag/tests/test_m1m2.py` - Updated DEVICE to cuda, fixed generators
- `src/rag/tests/test_hooks.py` - Moved model/tensors to CUDA
- `src/rag/tests/test_tq_cache_live.py` - Fixed API assertion
- `src/rag/tests/test_evals.py` - Fixed cuda_available check

### 2. Fixed Device Consistency in TQCache
- `src/rag/tq_cache.py` - Added `_device` and `_m_device` tracking
- `TQLinearAttentionLayer._device` - Tracks device for conv/S states
- `TQCache._m_device` - Tracks device for M1/M2
- `_StateView.__getitem__` - Returns tensors on tracked device
- `update_conv_state`, `update_recurrent_state`, `read_m1`, `read_m2` - Maintain device consistency

### 3. Fixed TurboQuant for CUDA tensors
- `src/rag/turboquant.py` - Line 356: Handle CUDA tensors in `.numpy()` conversion

---

## Next Steps (In Order)

### Step 1: Fix the Power-of-Two Issue (CRITICAL)
Location: `src/rag/tq_cache.py` - `update_conv_state` method

Options:
A) Pad to power-of-two before quant, dequant strips padding
B) Update TurboQuant to use segmented FHT
C) Split conv state into Q/V components

### Step 2: Profile Memory with TQCache
After fixing the blocker, re-run:
```bash
cd /home/ubuntu/RAGGA && python3 profile_memory.py
```

### Step 3: Ingest Corpus Subset (50K docs)
```python
from ingest import IngestDriver
from loader import load_quant_model

model, meta = load_quant_model(
    artifacts_dir='/home/ubuntu/qwen3_5_9B_palettized',
    model_name='Qwen/Qwen3.5-9B',
    device='cuda',
    heads_dir='/home/ubuntu/qwen3_5_9B_palettized_heads'
)

# Load 50K docs from disk/enterprise_rag_bench/documents/
# Run IngestDriver with model
```

### Step 4: Build IVFADC Index
```python
from index import build_index, ChunkVectorLoader
loader = ChunkVectorLoader(disk_dir)
build_index(loader.iter_vectors(), config)
```

### Step 5: Run Queries
Use `answer_query()` from `src/rag/query.py`

---

## Key Files Reference

| File | Purpose |
|------|---------|
| `src/rag/turboquant.py` | TQ quantization (power-of-two issue here) |
| `src/rag/tq_cache.py` | TQCache for online quantized cache |
| `src/rag/ingest.py` | `IngestDriver` for corpus ingestion |
| `src/rag/index.py` | IVFADC index building |
| `src/rag/query.py` | `answer_query()` for RAG |
| `src/scripts/loader.py` | `load_quant_model()` entry point |
| `src/scripts/modeling.py` | Qwen3.5 model with M1/M2 |
| `src/flute_extended/fht.py` | FHT kernel (supports segmented transforms) |

---

## Test Command

```bash
cd /home/ubuntu/RAGGA && python3 -m pytest src/rag/tests/ -q --tb=short
```

All 161 tests should pass.

---

## Vocab Size

**IMPORTANT:** Qwen3 vocab is 248,320 (NOT 151,936 or 151,665)

Get from model config:
```python
vocab_size = model.config.vocab_size  # 248320
```
