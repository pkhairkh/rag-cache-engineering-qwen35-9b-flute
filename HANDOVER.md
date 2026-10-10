# RAGGA Handover Document

## CRITICAL: GPU-ONLY PROJECT

**THIS IS A GPU PROJECT. THERE ARE NO CPU TESTS. ALL TESTS MUST RUN ON CUDA.**

## THE CPU CODING AGENT FUCKED UP ALL TESTS

The coding agent who worked on this project from a CPU-only environment completely broke the test suite by:

1. Writing tests with `.cuda()` calls as an afterthought instead of creating tensors on CUDA from the start
2. Calling `.cuda()` on objects that don't support it (configs, tuples, integers, lists)
3. Having no way to validate their changes because they couldn't run the tests
4. Producing 224 tests that ALL FAILED when run on the actual GPU

**Current test status: 203 passed, 12 failed, 9 errors (224 total)**
- The 203 that pass were FIXED on the GPU box by an agent WITH GPU access
- The 21 remaining failures are from the CPU agent's broken patterns

**CPU CODING AGENTS MUST NOT WRITE GPU CODE. EVER.**

### Broken Patterns the CPU Agent Introduced

1. `torch.randint(0, 128, (2, 6).cuda())` - calling .cuda() on a SHAPE TUPLE
2. `cfg(False).cuda()` - calling .cuda() on a CONFIG OBJECT
3. `len(ids).cuda()` - calling .cuda() on an INTEGER
4. `[list(tok)].cuda()` - calling .cuda() on a LIST
5. `torch.randn(..., device=DEVICE).cuda()` - redundant .cuda() after already specifying device
6. Tests that create tensors on CPU then try to use them with CUDA tensors

### Correct Patterns

1. `torch.randint(0, 128, (2, 6), device='cuda')` - specify device at creation
2. Create model first, then `.to(DEVICE)` - configs don't have .cuda()
3. `len(ids)` - it's just an integer, no .cuda()
4. `torch.tensor([list(tok)], device='cuda')` - create tensor on device
5. `torch.randn(..., device=DEVICE)` - no redundant .cuda()
6. ALL tensors created on CUDA from the start

## M1/M2 Architecture

**M1 and M2 are TWO GLOBAL STATES per model, NOT per-layer.**

- Each forward pass produces exactly ONE M1 and ONE M2 state
- These are shared across all 24 layers (each layer has gate vectors, not separate memories)
- **Memory size MUST be between 1024 and 4096**
- The gates (write_gate_k, write_gate_v, read_gate) are per-layer

**Memory consumption:**
- M1 state: (32, mem_size, 128) = small
- M2 state: (32, mem_size, 128) = small
- Model weights: ~6GB
- Training memory issue is from **intermediate activations**, not M1/M2

## Memory Issue During Gate Training

**SYMPTOM:** Gate training with `--self-test` OOMs on A10G (22GB VRAM)

**ROOT CAUSE:** Intermediate activations during prefill:
1. Forward pass through 24 layers creates large activation tensors
2. Prefill on 128 tokens with gradient tracking
3. KV capture from all layers during forward
4. Replay with gradients during backward

The capture phase records k/v tensors per layer per prefill position — scales with sequence length T, not mem_size.

**This is a real issue that needs to be fixed properly, not worked around.**

## Test Status

**Current: 203 passed, 12 failed, 9 errors (224 total)**

### Passing Test Modules
- `test_w17_loader.py` ✅
- `test_w17_finetune.py` ✅
- `test_w16_install.py` ✅
- `test_turboquant.py` ✅
- `test_turboquant_split.py` ✅
- `test_evals.py` ✅

### Failing Test Modules (21 failures)
All failures are device placement issues from the CPU agent:

1. `test_install.py` (8 failures) - Model parameters vs inputs device mismatch
2. `test_install_conv_math.py` (2 failures) - Same device issues
3. `test_m1m2.py` (1 failure) - Device mismatch in forward pass
4. `test_index.py` (1 failure) - Stub model device handling
5. `test_tq_cache.py` (1 error) - Device comparison in read_m1/read_m2
6. `test_query.py` (device issues in stub models)
7. `test_hooks.py` (dequant returning CPU tensors)

## Production Paths

**Model files:**
- `/home/ubuntu/qwen3_5_9B_palettized/` - Palettized LUT/idx files (4.7GB)
- `/home/ubuntu/qwen3_5_9B_palettized_heads/` - Head LUTs (1.8GB)

**Data (MISSING):**
- `/home/ubuntu/RAGGA/data/pairs.jsonl` - Pairs for gate training
  - Format: `{"text1": "...", "text2": "...", "label": 1}`
  - Sources: MRPC/QQP/SNLI/MNLI/PAWS/STS-B/SQuAD

**Artifacts:**
- `/home/ubuntu/RAGGA/artifacts/gates/` - Trained gate weights

## Protocol: TRAIN → RE-INGEST → EVAL

### 1. TRAIN
```bash
python scripts/gpu/finetune_m1m2.py \
    --pairs-file /home/ubuntu/pairs.jsonl \
    --gates-out /home/ubuntu/RAGGA/disk/m1m2_gates.npz \
    --m1m2-mem-size 1024 \
    --max-steps 300
```

### 2. RE-INGEST
```bash
python scripts/gpu/run_ingestion.py \
    --out-dir /home/ubuntu/RAGGA/disk/ingested_m1m2 \
    --m1m2-gates /home/ubuntu/RAGGA/disk/m1m2_gates.npz \
    --m1m2-mem-size 1024 \
    --n-docs 100
```

### 3. INDEX
```bash
python scripts/gpu/run_index.py --disk-dir /home/ubuntu/RAGGA/disk/ingested_m1m2
```

### 4. EVAL
```bash
python scripts/gpu/eval_retrieval.py --disk-dir /home/ubuntu/RAGGA/disk/ingested_m1m2
```

### 5. QUERY
```bash
python scripts/gpu/run_query.py \
    --disk-dir /home/ubuntu/RAGGA/disk/ingested_m1m2 \
    --m1m2-gates /home/ubuntu/RAGGA/disk/m1m2_gates.npz
```

## Code Changes Made This Session

### Production Code Fixes
1. `fht.py:build_rotation_matrix` - Added `device` parameter
2. `turboquant.py:roundtrip` - Now preserves device

### Test Files Fixed for CUDA
- `test_turboquant.py` - Fixed tuple.cuda(), generator+device conflicts, dequant device
- `test_turboquant_split.py` - Fixed dequant device handling
- `test_m1m2.py` - Removed .cuda() from config objects, fixed redundant .cuda() calls
- `test_query.py` - Fixed len().cuda(), list.cuda(), stub model device handling
- `test_w16_retrieval.py` - Fixed len().cuda(), list.cuda(), stub model device handling
- `test_index.py` - Fixed tuple.cuda(), stub model device handling
- `test_hooks.py` - Fixed dequant.to(device) patterns
- `test_ingest.py` - Fixed stub model device handling
- `test_install_conv_math.py` - Fixed tuple.cuda()
- `test_tq_cache.py` - Fixed device comparison in read_m1/read_m2

## Outstanding Issues

1. **21 tests still failing** - Need GPU box to fix
2. **Memory during training** - Need architectural fix
3. **pairs.jsonl missing** - Need conversion script for MRPC/QQP/SNLI/etc
4. **dequant always returns CPU** - Should accept device parameter
