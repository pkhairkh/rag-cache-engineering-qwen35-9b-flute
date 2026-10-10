# RAGGA Handover Document

## Current State (post-W12)

| item | status |
|---|---|
| CPU suite | 175 tests green (`python3 -m pytest src/rag/tests -q`) |
| conv geometry | FIXED: full-window policy, d = 24,576 (segments 16,384 + 8,192) |
| ingestion | ✅ 100 chunks re-ingested with correct 24,576 conv geometry |
| index | ✅ Flat IndexFlatIP built (100 < 256 threshold) |
| verification | 5/6 stages pass |
| **BLOCKER** | G5 (full install_snapshot) fails → garbage output |

---

## The W12 Finding: S-read DRIFT

### Bisection Results

```
variant      S-read   conv-tail  gen   
reseed       DRIFT    OK         OK    
s-only       DRIFT    OK         OK    
conv-only    DRIFT    OK         OK    
full         DRIFT    OK         GARBAGE
```

**All variants show S-read DRIFT** (rel-MSE 7.66e-01 for reseed/conv-only, 1.47e-02 for s-only/full).

### What This Means

> **S-read DRIFT => device/frame bug in the quant path**

The S codes are not reading back correctly. When `dequant` is called on installed/seeded codes, the reconstructed values drift from expected.

**Key observations:**
1. `reseed` (just system, no install) shows DRIFT - the baseline is already broken
2. `s-only` and `full` show LOWER drift (1.47e-02) vs `reseed` (7.66e-01) - installing S codes IMPROVES drift
3. Conv-tail is OK - the W11 conv fix worked
4. Only `full` produces garbage - the combination of S + conv triggers failure

### Hypothesis

The S quant path has a device mismatch or frame rotation bug:

1. **Device mismatch**: Codes stored on CPU, dequant expects CUDA tensors
2. **Frame rotation mismatch**: The D3 rotation seed differs between quant and dequant
3. **Layer state not updated**: `set_s_codes` stores codes but doesn't propagate to layer computation

The DRIFT in `reseed` suggests the system state itself has the bug - stored codes don't reconstruct correctly.

---

## Code Paths to Investigate

### 1. Quant Path (`src/rag/turboquant.py`)

```python
class TurboQuant:
    def quant(self, x: np.ndarray) -> TQCodes:
        # Applies D3 rotation, quantizes to idx_lo/idx_hi
        # Check: is rotation applied on CPU but expected on CUDA?
    
    def dequant(self, codes: TQCodes) -> torch.Tensor:
        # Reconstructs from idx_lo/idx_hi
        # Check: returns CPU or CUDA tensor?
        # Check: does it apply inverse rotation correctly?
```

### 2. TQCache S-code Path (`src/rag/tq_cache.py`)

```python
def set_s_codes(self, layer_idx: int, codes: TQCodes) -> None:
    # Stores codes for layer
    # Check: does this update the layer's internal state?
    # Check: is _device set correctly?

def _ensure_s_layer(self, layer_idx: int) -> TQLinearAttentionLayer:
    # Creates/gets layer
    # Check: does layer have correct device/context?
```

### 3. Install Math (`src/rag/install.py`)

```python
def sum_turboquant_codes(system, deltas, kind, bits):
    q = resolve_quantizer(kind, system.d, bits)
    # Check: does resolve_quantizer return correct seed?
    acc = q.dequant(system)
    for d in deltas:
        acc = acc + q.dequant(d)
    return q.quant(acc)
    # Check: is the returned TQCodes on correct device?
```

### 4. Resolve Quantizer (`src/rag/tq_cache.py`)

```python
def resolve_quantizer(kind: str, d: int, bits: float) -> TurboQuant:
    # Returns a TurboQuant instance
    # Check: is seed derived correctly for kind?
    # Check: is device correct?
```

---

## Specific Questions

1. **Where is `resolve_quantizer` defined and what seed does it use?**
   - The seed must match between quant and dequant
   - Different seeds = different rotations = garbage reconstruction

2. **Does `TurboQuant.dequant` return CPU or CUDA tensors?**
   - If CPU, the forward pass may operate on wrong device
   - Need to trace device through the entire path

3. **What does `set_s_codes` actually do?**
   - Does it just store codes in a dict?
   - Does it propagate to the layer's computation state?
   - Is there a `_device` field that needs setting?

4. **Why does `reseed` (no install) show DRIFT?**
   - The system state was created during fresh ingestion
   - If re-reading system codes produces drift, the save/load path is broken

---

## GPU Tests to Run

```bash
# 1. Check S-code device/content
python3 -c "
from tq_cache import TQCache, resolve_quantizer
from ingest import load_system_state
import torch

system = load_system_state('/home/ubuntu/RAGGA/disk/ingested_50k')
s0 = system.s_codes[0]
q = resolve_quantizer('S', s0.d, 3.5)
dequant = q.dequant(s0)
print(f'dequant device: {dequant.device}')
print(f'dequant mean: {dequant.mean()}, std: {dequant.std()}')
"

# 2. Check if set_s_codes updates layer state
python3 -c "
from loader import load_quant_model
from tq_cache import TQCache
import torch

model, _ = load_quant_model(
    '/home/ubuntu/qwen3_5_9B_palettized',
    'Qwen/Qwen3.5-9B',
    device='cuda',
    heads_dir='/home/ubuntu/qwen3_5_9B_palettized_heads'
)
cache = TQCache(config=model.config, bits=3.5, online=True)
# Check internal state before/after set_s_codes
"

# 3. Run with --fla-off to test kernel route
python3 scripts/gpu/bisect_install.py --fla-off
```

---

## Files to Read (Priority Order)

1. **`src/rag/tq_cache.py`** - `set_s_codes()`, `resolve_quantizer()`, layer management
2. **`src/rag/turboquant.py`** - `TurboQuant.quant()`, `TurboQuant.dequant()`, device handling
3. **`src/rag/install.py`** - `sum_turboquant_codes()`, seed validation
4. **`src/rag/snapshot.py`** - How codes are saved/loaded (may affect device)
5. **`src/rag/ingest.py`** - `reseed_cache()`, system state creation

---

## Verification Matrix

| Stage | Description | Status |
|-------|-------------|--------|
| G1 | Pure model generation | ✅ PASS |
| G2 | TQCache without install | ✅ PASS |
| G3 | reseed(system) only | ✅ PASS |
| G4 | S-only install | ✅ PASS |
| G5 | Full install (S + conv) | ❌ FAIL |
| G6 | answer_query e2e | ✅ PASS |

**G5 is the blocker** - full install_snapshot produces garbage.

---

## Success Criteria

The bug is fixed when:
1. `bisect_install.py` shows S-read OK (no DRIFT)
2. `verify_pipeline.py` shows 6/6 stages pass
3. G5 produces coherent text or valid `<|im_end|>`

---

## Model Details

- Qwen3.5-9B palettized
- Located at `/home/ubuntu/qwen3_5_9B_palettized`
- Heads at `/home/ubuntu/qwen3_5_9B_palettized_heads`
- vocab_size: 248320
- EOS token: `<|im_end|>` (id: 248046)

---

## Test Command

```bash
cd /home/ubuntu/RAGGA && python3 -m pytest src/rag/tests/ -q --tb=short
```

All 175 tests pass on CPU.
