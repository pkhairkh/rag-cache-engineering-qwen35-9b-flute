# Deployment and Bring-Up Guide

Target: NVIDIA A10G (SM_86), CUDA 12.0+, PyTorch 2.x. The same steps work
on any SM_80+ GPU (A100 / L40 / H100) with the appropriate clocks and
peak expectations.

Section 3 (ptxas register audit) is a HARD GATE: do not proceed to any
model work until it shows zero spills for all three instantiations of the
streaming kernel.

NOTE — repo scope: the dense CUTLASS baseline kernel, the benchmark
harness, and the kernel test-suite (test_flute.py /
test_qwen_weights.py) were removed with the cleanup; the canonical
palettizer is intentionally absent (the palettized model + heads
artifacts are provided pre-built). Correctness spot-checks below use the
`debug_simple` differential backend against `cutlass_streaming`.

## 1. Prerequisites

- NVIDIA GPU with compute capability 8.0+ (primary target: AWS A10G /
  SM_86 — 80 SM, 300 W; see docs/HARDWARE.md)
- CUDA toolkit 12.0+
- PyTorch 2.x (requires a C++20 compiler for the extension headers)
- Python deps: `pip install -r requirements.txt` (known-good pins:
  requirements.lock.txt)

Verify the environment:

```bash
nvidia-smi
nvcc --version
python -c "import torch; print(torch.__version__, torch.version.cuda, \
    torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))"
```

## 2. Build

```bash
python setup.py build_ext --inplace
```

Expected: no warnings or errors; the extension lands at
`flute_extended/_C.cpython-*.so`. No external dependencies — the
production `cutlass_streaming` kernel is raw-PTX and dependency-free.

Build configuration notes:

- `-std=c++20` for nvcc AND the host compiler (PyTorch 2.x requirement).
- Absolute `-I<repo>/include` — nvcc runs from a scratch build directory,
  so relative include paths do not resolve.
- gencode set SM_80 / SM_86 (primary) / SM_89 / SM_90. SM_75 is excluded:
  the f16 mma.m16n8k16 operation requires SM_80+ (PTX ISA target notes).

A clean rebuild:

```bash
rm -rf build flute_extended/_C*.so
python setup.py build_ext --inplace
```

## 2b. Verify the REBUILT module is the LOADED one (the stale-.so trap)

`python setup.py build_ext --inplace` updates the .so in the SOURCE tree
only. If `flute_extended` was installed non-editably (`pip install .`
rather than `pip install -e .`), `import flute_extended` resolves to the
site-packages COPY — which stays stale after the rebuild. After
rebuilding:

```bash
python - <<'EOF'
import flute_extended as fx
print("loaded module:", fx.__file__)
EOF
```

If the printed path is under `site-packages`, either reinstall editable
(`pip install -e . --no-build-isolation`) or copy the fresh
`build/lib*/flute_extended/_C*.so` over the installed copy.

## 3. ptxas register/spill audit (HARD GATE)

The streaming kernel targets ~130-210 registers per thread against the
255 cap. A spill destroys throughput silently, 2-3x, and invalidates the
performance expectations.

```bash
nvcc -std=c++20 -O3 --use_fast_math --expt-relaxed-constexpr \
     -gencode=arch=compute_86,code=sm_86 \
     -Iinclude -I$(python -c "import torch; print(torch.utils.cpp_extension.include_paths()[0])") \
     -Xptxas -v -c src/kernel_cutlass_streaming.cu -o /tmp/streaming.o 2>&1 | \
     grep -A3 "flute_kernel_streaming"
```

Expected for ALL THREE instantiations - `TileConfig<128,128,32,32>`,
`TileConfig<64,128,64,32>`, `TileConfig<64,128,64,64>`:

```
Used NNN registers, 0 spill stores, 0 spill loads
```

- Zero spill stores / zero spill loads: proceed.
- Any spills: STOP. Likely causes: a newer nvcc allocating more
  aggressively (try `--ptxas-options=-O3` or `-maxrregcount=255`), or the
  compiler refusing `__launch_bounds__(128, 2)`. Last-resort remedy:
  relax `__launch_bounds__(128, 2)` to `(128, 1)` and accept 1 block/SM
  for the BK=64 configs. Do not run model work on a spilling build.

Also sanity-check the reported dynamic shared memory: 42,240 / 50,432 /
49,920 bytes for the three instantiations, matching the header of
`src/kernel_cutlass_streaming.cu`.

## 4. Correctness spot-check (differential)

The full test-suite was removed with the cleanup; the fastest
on-box correctness check is the built-in differential: run the same
palettized GEMM through the production streaming kernel and its
scalar-staging twin (`debug_simple`) and require bit-exact agreement,
then compare both against a pure-torch dequant reference on a real
layer shape:

```python
import torch, numpy as np
import flute_extended as fx
from flute_extended.idxN import pack_idxn

torch.manual_seed(0)
M, K, N, b, gs = 256, 4096, 4096, 4, 32
A  = torch.randn(M, K, dtype=torch.float16, device="cuda") * 0.1
W  = (torch.randn(N, K, dtype=torch.float16, device="cuda") * 0.1)
lut = torch.randn(N // gs, 1 << b, dtype=torch.float16, device="cuda") * 0.1
idx = torch.randint(0, 16, (N, K), device="cuda")
indices = pack_idxn(idx.cpu().numpy(), b)          # flat idxN blob
indices = torch.from_numpy(indices).to("cuda")

y_stream = fx.qgemm_per_group_lut(A, indices, lut, bitwidth=b,
                                  group_size=gs, backend="cutlass_streaming")
y_debug  = fx.qgemm_per_group_lut(A, indices, lut, bitwidth=b,
                                  group_size=gs, backend="debug_simple")
# pure-torch reference (dequant W per group, then GEMM)
Wq = lut.reshape(-1)[ (torch.arange(N, device="cuda")[:, None] // gs) * (1 << b) + idx ].half()
y_ref = (A.float() @ Wq.float().T).half()
print("streaming == debug_simple:", torch.equal(y_stream, y_debug))
print("max |streaming - ref|    :", (y_stream.float() - y_ref.float()).abs().max().item())
```

Expected: `streaming == debug_simple: True` (bit-exact — this gate
re-proves the BM repartition, the ldmatrix/swizzle staging and the
cp.async pipeline); the torch reference agrees to fp16 rounding
(dequant-order differences are expected, magnitude ~1e-2 at these
scales). Bisection: if debug_simple also misses the reference, the bug
is in the shared dequant contract (docs/DEQUANT_SPEC.md); if only
streaming misses, it is in the staging machinery.

## 5. Using the library

```python
import flute_extended
C = flute_extended.qgemm_per_group_lut(
    A,                 # [M, K] fp16 CUDA
    indices,           # [N, (K+1)//2] uint8, packed 4-bit LSB-first
    lut,               # [ceil(N/group_size), 16] fp16
    bitwidth=4,
    group_size=32,     # 32 (MLP gate/up) or 64 (attention + down)
    backend="cutlass_streaming",   # or "auto" / "debug_simple"
)   # -> [M, N] fp16, Y = X @ W^T

# Deep-tile experiment for gs=32 layers (read once per process):
import os; os.environ["FLUTE_GS32_BK"] = "64"   # before the first call
```

## 6. Weight format

The palettized weight format (packed 4-bit indices, per-group 16-entry
LUT, LSB-first nibble order) is specified in docs/DEQUANT_SPEC.md. The
kernels treat that document as the contract. The model artifacts
(metadata.json + per-tensor .idx/.lut files + norm_gain_edits.json) are
provided pre-built and loaded by `scripts/loader.py::
load_quant_model` / `scripts/palettized_modules.py`.

## 7. Troubleshooting

- **Build fails with "C++20 or later compatible compiler is required"**:
  the host compiler defaults to an older standard; ensure g++ >= 11 or
  pass `-std=c++20` explicitly.
- **Build fails with "Unexpected instruction types specified for 'mma'"**:
  an SM_75 target crept into the gencode list; remove it (the kernel
  requires SM_80+).
- **`cudaFuncSetAttribute ... failed` at runtime**: the device does not
  expose the requested dynamic shared memory (100 KB needs SM_86+ with
  the opt-in; the launch helper requests it once per process).
- **Correctness fails only on multi-N-block shapes (N > 128)**: inspect
  the epilogue's block N-offset (`n0`) in the address math — every
  N-block must add its own n0.
- **Correctness fails only with `FLUTE_GS32_BK=64`**: the deep-tile
  dispatch guard (`K % 64 == 0`) or the 5-group LUT staging of
  `<64,128,64,32>`.
- **Performance 2-3x below expectations**: re-run the ptxas audit
  (section 3); spills are the usual cause.
- **Bank-conflict-like stalls**: compare the swizzle functions in
  `mma.cuh` against the store-side indexing in the kernel's dequant and
  staging paths.
