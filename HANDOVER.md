# RAGGA Handover Document

## Current State

### What's Done
1. **All 165 tests passing** - Fixed FHT shared memory issue for non-power-of-two dimensions
2. **Kernels compiled and working** - FHT handles segmented transforms for large K
3. **Ingestion complete** - 100 documents ingested to `/home/ubuntu/RAGGA/disk/ingested_50k/`
4. **Retrieval working** - Cache-vector similarity ~70%, different chunks for different queries

---

## CRITICAL BUG: Cache Installation Corrupts Model State

### Executive Summary

**Status**: RAGGA pipeline works for ingestion and retrieval, but **generation produces garbage after chunk installation**.

**Root Cause**: Installing S codes on the TQCache corrupts the model state, causing garbage output during decode.

**Critical Finding**: WITHOUT installation, the model generates correctly. AFTER installation, garbage. This isolates the bug to `install.py` / `tq_cache.py`.

---

### Symptom

After `install_snapshot(cache, system, [chunk])`, the model generates garbage tokens instead of coherent text.

### Isolated Test Results

| Test | Description | Output | Status |
|------|-------------|--------|--------|
| `test_generation.py` | Pure model, no cache | "Paris" | ✅ |
| `test_tq_generation.py` | TQCache, no install | "Paris<\|im_end\|>" | ✅ |
| `test_system_only.py` | System reseed, no chunk install | "<\|im_end\|>" | ✅ |
| `test_install_layer0.py` | Just layer 0 S codes installed | "td of $ of ically" | ❌ |
| `test_install_gen.py` | Full chunk install via `install_snapshot` | garbage | ❌ |

**Conclusion**: Installing even ONE layer's S codes corrupts output.

---

## Code Paths Involved

### 1. Installation Flow (`src/rag/install.py`)

```python
def install_snapshot(cache: TQCache, system: SystemState,
                     retrieved: Sequence[ChunkSnapshot],
                     bits: float = 3.5) -> dict:
    report: dict = {}
    for L in sorted(system.s_codes):
        deltas = [s.s_codes[L] for s in retrieved
                  if L in s.s_codes and s.s_codes[L] is not None]
        summed = sum_turboquant_codes(system.s_codes[L], deltas,
                                      kind="S", bits=bits)
        cache.set_s_codes(L, summed)  # <-- BUG IS HERE or in sum_turboquant_codes
        
        conv = _last_conv(retrieved, L)
        if conv is not None:
            cache.set_conv_codes(L, conv)  # Also causes issues
```

### 2. Sum Math (`src/rag/install.py`)

```python
def sum_turboquant_codes(system: TQCodes, deltas: Iterable[TQCodes],
                         kind: str, bits: float = 3.5) -> TQCodes:
    q = resolve_quantizer(kind, system.d, bits)
    if system.seed != q.seed:
        raise ValueError("frame drift")
    acc = q.dequant(system)  # Dequantize system codes
    for i, d in enumerate(deltas):
        acc = acc + q.dequant(d)  # Sum dequantized deltas
    return q.quant(acc)  # Re-quantize the sum
```

**The math**: `quant(dequant(sys) + Σ dequant(deltas))`

### 3. Cache Setter (`src/rag/tq_cache.py`)

```python
def set_s_codes(self, layer_idx: int, codes: Optional[TQCodes]) -> None:
    # Sets the S codes for a layer
    # Needs investigation: does this properly update the layer's state?
```

---

## Code Locations

| File | Lines | Function | Purpose |
|------|-------|----------|---------|
| `src/rag/install.py` | 34-62 | `sum_turboquant_codes()` | Dequant-sum-requant math |
| `src/rag/install.py` | 74-105 | `install_snapshot()` | Orchestrate installation |
| `src/rag/tq_cache.py` | 615+ | `set_s_codes()` | Update cache layer codes |
| `src/rag/tq_cache.py` | 580+ | `set_conv_codes()` | Update conv state |
| `src/rag/turboquant.py` | 169-188 | `class TQCodes` | Code structure (idx_lo, idx_hi, seed, etc.) |
| `src/rag/turboquant.py` | various | `quant()` / `dequant()` | Quantization operations |

---

## Data Files for Debugging

**On GPU box:**
- System state: `/home/ubuntu/RAGGA/disk/ingested_50k/system_state.npz`
- Chunk 0 snapshot: `/home/ubuntu/RAGGA/disk/ingested_50k/snapshots/chunk_00000.npz`
- Debug output: `/home/ubuntu/RAGGA/debug_install.py` output shows:
  - Before install: mean=0.0004, std=0.1148
  - After install: mean=0.0009, std=0.2897
  - Expected: mean=0.0008, std=0.2988
  - **Values are close, so sum math seems correct!**

**Issue**: The quantized codes look correct numerically, but something about how they're installed breaks the model.

---

## Hypothesis: Potential Bug Locations

### Hypothesis 1: D3 Rotation Frame Mismatch
- Each `TQCodes` has a `seed` for the D3 rotation
- If the seed is wrong during dequant, the reconstruction is garbage
- Check: Does `resolve_quantizer()` return a quantizer with matching seed?

### Hypothesis 2: `set_s_codes` Doesn't Update Layer State Properly
- Maybe the codes are set but the layer's internal state isn't updated
- Check: How does `set_s_codes` affect the actual layer computation?

### Hypothesis 3: Norm Handling
- `TQCodes` has a `norm` field (fp32, never quantized)
- Is the norm being summed correctly?
- Is the norm being applied correctly during dequant?

### Hypothesis 4: conv_codes Interference
- Conv codes are installed AFTER S codes
- Maybe conv codes override something incorrectly
- Test showed conv-only install also produces garbage

### Hypothesis 5: Quantizer Resolution
- `resolve_quantizer(kind, d, bits)` creates a quantizer
- Is it using the correct d (dimension) and bits?
- The debug shows d=524288 for layer 0 S codes

### Hypothesis 6: Layer not exist in cache
- Maybe TQCache has no layer at index L
- Check: Does `set_s_codes` create the layer if needed?

---

## Recommended Debug Approach (No GPU Required)

### Step 1: Static Code Analysis

1. **Trace `set_s_codes`**:
   - Read `tq_cache.py` around line 615
   - Document what fields it modifies
   - Check if it updates both `_s_codes` dict AND the layer's computation state

2. **Trace `sum_turboquant_codes`**:
   - Read `install.py` lines 34-62
   - Verify the quant/dequant cycle preserves information
   - Check if `q.quant(acc)` produces valid TQCodes

3. **Check TQCodes dataclass**:
   - Fields: kind, d, norm, bits_lo, bits_hi, n_lo, n_hi, idx_lo, idx_hi, seed, partition
   - Are all fields being copied/set correctly?

### Step 2: Unit Test Design (CPU-friendly)

Create toy tests that DON'T need the actual model:

```python
# test_install_math.py
import numpy as np
from turboquant import TurboQuant, TQCodes

def test_quant_dequant_roundtrip():
    """Test that quant(dequant(x)) ≈ x"""
    x = np.random.randn(1024).astype(np.float32)
    q = TurboQuant(d=1024, bits=3.5, seed=42, kind='S')
    
    codes = q.quant(x)
    x_reconstructed = q.dequant(codes)
    
    # Check reconstruction error
    error = np.abs(x - x_reconstructed).mean()
    print(f"Mean reconstruction error: {error}")
    assert error < 0.1, f"Error too high: {error}"

def test_sum_turboquant_codes():
    """Test that sum produces valid codes"""
    # Create fake system and delta codes
    x_sys = np.random.randn(1024).astype(np.float32) * 0.1
    x_delta = np.random.randn(1024).astype(np.float32) * 0.3
    
    q = TurboQuant(d=1024, bits=3.5, seed=42, kind='S')
    sys_codes = q.quant(x_sys)
    delta_codes = q.quant(x_delta)
    
    # Expected
    expected = x_sys + x_delta
    
    # Actual via sum_turboquant_codes
    from install import sum_turboquant_codes
    result_codes = sum_turboquant_codes(sys_codes, [delta_codes], kind='S', bits=3.5)
    result = q.dequant(result_codes)
    
    error = np.abs(expected - result).mean()
    print(f"Sum error: {error}")
    
    # Check all fields are populated
    assert result_codes.d == 1024
    assert result_codes.seed == 42
    assert result_codes.idx_lo is not None

def test_tqcodes_field_preservation():
    """Test that TQCodes fields are preserved through operations"""
    codes = TQCodes(
        kind='S',
        d=1024,
        norm=np.float32(1.0),
        bits_lo=4,
        bits_hi=4,
        n_lo=1024,
        n_hi=0,
        idx_lo=np.zeros(512, dtype=np.uint8),
        idx_hi=None,
        seed=42,
        partition='half'
    )
    
    # Check all fields
    print(f"kind: {codes.kind}")
    print(f"d: {codes.d}")
    print(f"norm: {codes.norm}")
    print(f"seed: {codes.seed}")
```

### Step 3: Mock Cache Test

```python
# test_cache_install.py
"""Test cache installation without model forward pass."""

from tq_cache import TQCache, TQLinearAttentionLayer, resolve_quantizer
from turboquant import TQCodes
import numpy as np

def test_set_s_codes_state_update():
    """Verify set_s_codes updates internal state"""
    
    # Create mock config
    class MockConfig:
        num_hidden_layers = 2
    
    cache = TQCache(config=MockConfig(), bits=3.5, online=True)
    
    # Create fake codes
    codes = TQCodes(
        kind='S',
        d=1024,
        norm=np.float32(1.0),
        bits_lo=4,
        bits_hi=4,
        n_lo=1024,
        n_hi=0,
        idx_lo=np.zeros(512, dtype=np.uint8),
        idx_hi=None,
        seed=42,
        partition='half'
    )
    
    # Set codes
    cache.set_s_codes(0, codes)
    
    # Verify retrieval
    snapshot = cache.snapshot_codes()
    s_codes = snapshot['s']
    
    assert 0 in s_codes, "Layer 0 not in snapshot"
    retrieved = s_codes[0]
    
    print(f"Retrieved d: {retrieved.d}")
    print(f"Retrieved seed: {retrieved.seed}")
    print(f"Retrieved norm: {retrieved.norm}")
    
    assert retrieved.d == codes.d
    assert retrieved.seed == codes.seed
```

### Step 4: Compare Working vs Broken Paths

**Working path** (no install):
```
reseed_cache(cache, system)
model(input_ids=..., past_key_values=cache)
→ Output: <|im_end|> ✓
```

**Broken path** (with install):
```
reseed_cache(cache, system)
install_snapshot(cache, system, [chunk])  # <-- CORRUPTION HAPPENS HERE
model(input_ids=..., past_key_values=cache)
→ Output: garbage ✗
```

**What to compare**:
1. Cache state before/after install_snapshot
2. TQCodes fields before/after sum_turboquant_codes
3. What `set_s_codes` modifies internally

---

## Key Files to Read (Priority Order)

1. **`src/rag/tq_cache.py`** - `set_s_codes()`, `set_conv_codes()`, how layers use codes
2. **`src/rag/install.py`** - `sum_turboquant_codes()`, `install_snapshot()`
3. **`src/rag/turboquant.py`** - `TurboQuant.quant()`, `TurboQuant.dequant()`, `TQCodes` dataclass
4. **`src/rag/snapshot.py`** - How snapshots are saved/loaded (may reveal format issues)
5. **`src/rag/ingest.py`** - `reseed_cache()`, how system state is created

---

## Specific Questions to Answer

1. **Does `set_s_codes` update the layer's internal computation state?**
   - Or does it just store codes without updating the layer?

2. **What does `resolve_quantizer('S', d, bits)` return?**
   - Is the seed correct?
   - Is the quantizer stateless or does it have internal state?

3. **What fields does `sum_turboquant_codes` produce?**
   - Does it set `norm` correctly?
   - Does it copy `seed` from input?

4. **How does the layer access S codes during forward?**
   - Does it call `dequant` internally?
   - Does it use the codes directly?

5. **Is there a device mismatch?**
   - Codes on CPU but model on CUDA?
   - Does `set_s_codes` move tensors to the right device?

---

## Reproduction Steps (On GPU Box)

```bash
cd /home/ubuntu/RAGGA

# 1. Test pure generation (works)
python3 test_generation.py

# 2. Test TQCache without install (works)
python3 test_tq_generation.py

# 3. Test with install (broken)
python3 test_install_layer0.py
# Output: garbage instead of <|im_end|>

# 4. Debug installation math
python3 debug_install.py
# Shows: dequant values look correct, but generation fails

# 5. Full query test
python3 run_query.py
# Retrieval: 70% similarity ✓
# Generation: garbage ✗
```

---

## Success Criteria

The bug is fixed when:
1. `test_install_layer0.py` outputs coherent text (or `<|im_end|>`)
2. `run_query.py` produces meaningful answers
3. All existing tests in `src/rag/tests/` still pass

---

## Model Details

- Qwen3.5-9B palettized model
- Located at `/home/ubuntu/qwen3_5_9B_palettized`
- Separate heads at `/home/ubuntu/qwen3_5_9B_palettized_heads`
- vocab_size: 248320
- EOS token: `<|im_end|>` (id: 248046)

### Conv State Dimensions
- Q conv: 8192 = 16 * 4 * 128 
- V conv: 16384 = 32 * 4 * 128
- Combined: 24576 → segments as 16384 + 8192 (max 64KB shared mem ✓)

---

## GPU Memory Profile

```
Model load: 6,067 MiB
Forward 512 tokens: 8,581 MiB peak
Available VRAM: 22,590 MiB
Headroom: ~14,000 MiB for ingestion
```

---

## Test Command

```bash
cd /home/ubuntu/RAGGA && python3 -m pytest src/rag/tests/ -q --tb=short
```

All 165 tests pass.

---

## Contact / Context

This handover is for a coding agent to debug and fix the cache installation bug. The agent does NOT have GPU access, so focus on:
1. Static code analysis
2. CPU-friendly unit tests
3. Identifying the exact line/field causing corruption
4. Proposing a fix that can be tested on GPU later

The bug is isolated to the installation step. Ingestion, retrieval, and model loading all work correctly.
