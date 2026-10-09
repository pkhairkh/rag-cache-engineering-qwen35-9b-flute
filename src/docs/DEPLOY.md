# Deployment and Bring-Up Guide

Target: NVIDIA A10G (SM_86), CUDA 12.0+, PyTorch 2.x. The same steps work
on any SM_80+ GPU (A100 / L40 / H100) with the appropriate clocks and
peak expectations.

Section 4 (ptxas register audit) is a HARD GATE: do not proceed to
benchmarking until it shows zero spills for all three instantiations of
the streaming kernel. Section 5 is the first-silicon bisection ladder -
walk it in order; each step isolates a failure class before the next one
runs.

## 1. Prerequisites

- NVIDIA GPU with compute capability 8.0+ (primary target: AWS A10G /
  SM_86 — 80 SM, 300 W; see docs/HARDWARE.md)
- CUDA toolkit 12.0+
- PyTorch 2.x (requires a C++20 compiler for the extension headers)
- Optional: CUTLASS checkout (2.x through 4.8+ all work) for the
  `cutlass_dense` baseline at `/home/ubuntu/cutlass` (or
  `$FLUTE_CUTLASS_HOME`, `$CUTLASS_HOME`, `/opt/cutlass`). NOT required -
  the production `cutlass_streaming` kernel is raw-PTX and dependency-free.

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
`flute_extended/_C.cpython-*.so`. The console prints
`[flute_extended] CUTLASS: <path or NOT FOUND ...>` plus, when found,
`cutlass_dense backend: ENABLED` - the build succeeds either way; only the
`cutlass_dense` baseline needs CUTLASS.

The dense TU uses the classic `cutlass::gemm::device::GemmUniversal` API
with EVERY template argument explicit (ArchTag = `arch::Sm80` policy tag -
the canonical choice for SM_86; the SASS target still comes from the
`-gencode ... sm_86` flag). Nothing instantiates
`DefaultGemmConfiguration`, which has no Sm86 specialization in ANY
CUTLASS release (2.11 through 4.8 - verified by grep) and was the root
cause of the original `device::Gemm<..., Sm86, ...>` build break.
`FLUTE_DENSE_STREAMK=1` switches the baseline to the Stream-K swizzle at
runtime (helps wave quantization at small M).

Build configuration notes:

- `-std=c++20` for nvcc AND the host compiler (PyTorch 2.x requirement).
- Absolute `-I<repo>/include` - nvcc runs from a scratch build directory,
  so relative include paths do not resolve.
- gencode set SM_80 / SM_86 (primary) / SM_89 / SM_90. SM_75 is excluded:
  the f16 mma.m16n8k16 operation requires SM_80+ (PTX ISA target notes).

A clean rebuild:

```bash
rm -rf build flute_extended/_C*.so
python setup.py build_ext --inplace
```

## 2b. W17: verify the REBUILT module is the LOADED one (the stale-.so trap)

`python setup.py build_ext --inplace` updates the .so in the SOURCE tree
only. If `flute_extended` was installed non-editably (`pip install .`
rather than `pip install -e .`), `import flute_extended` resolves to the
site-packages COPY — which stays stale after the rebuild, and the
palettizer's runtime GS probe will keep reporting "stale-kernel" and
capping the sweep at GS<=512 (W17: the run now aborts BEFORE the model
load, naming the loaded module's path). After rebuilding:

```bash
python - <<'EOF'
import flute_extended as fx
print("loaded module:", fx.__file__)
# the probe the palettizer runs (tiny GS=2048 qgemm):
import torch
from flute_extended import qgemm_per_group_lut
import numpy as np
# ... or simply rerun the palettizer: it probes at startup and prints
# the module path + verdict before any expensive work.
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
  for the BK=64 configs. Do not benchmark a spilling build.

Also sanity-check the reported dynamic shared memory: 42,240 / 50,432 /
49,920 bytes for the three instantiations, matching the header of
`src/kernel_cutlass_streaming.cu`.

## 4. Correctness - first-silicon bisection ladder

Walk these steps in order. Each one isolates a failure class before the
next runs:

```bash
# 4.1 scalar-staging twin (validates PTX fragment mapping + mma on real HW)
python test_flute.py --backend debug_simple

# 4.2 production kernel vs reference (the multi-backend suite)
python test_flute.py

# 4.3 real-weights oracle (streaming kernel vs pure-torch dequant)
python test_qwen_weights.py
```

Expected: every line `[PASS]`, final line

```
RESULT: n/n checks passed; differential gates: nibble=PASS,
streaming==debug_simple=PASS, guards=PASS
```

(exit code 0). The suite covers both group sizes, unaligned N, thin M,
single-K-tile cases, the tile tails (BM=128 and BM=64), the K=4128
dispatch-fallback shape, and the real Qwen3.5-9B layer shapes.

The three differential gates and what a failure means:

- **nibble gate** - bit-exact LSB-first unpacking (LUT=[0..15],
  A=identity means C equals the raw nibble values). A failure here means
  the packed-index format is misunderstood somewhere; no other result
  matters until it passes.
- **streaming == debug_simple** - bit-exact equality between the
  production kernel and its scalar-staging twin, which uses a different
  BM tiling, so the gate re-proves the BM repartition too. If this fails
  while both loosely match the torch reference, the bug is isolated to
  the production staging machinery (ldmatrix / swizzle / cp.async
  pipeline).
- **guards** - degenerate shapes return zeros/empty; misaligned
  storage-offset views are staged via .clone() and stay correct.

Bisection map: naive fails -> data plumbing or test harness;
debug_simple fails -> PTX fragment mapping or mma semantics; streaming
fails the reference check -> run the streaming==debug_simple gate to
separate staging bugs from dispatch bugs; only the deep-tile A/B fails ->
the `<64,128,64,32>` path (cp.async tile shape or the 5-group LUT).

The original quick suite also works:

```bash
python test_flute.py                    # quick multi-backend check
python test_flute.py --layer gate_proj  # real shape, single backend
```

## 5. Performance

```bash
python benchmark_kernel.py                       # default sweep (BK=32)
python benchmark_kernel.py --gs32-bk 64          # deep-tile A/B - run BOTH
python benchmark_kernel.py --compare-cublas      # dense FP16 upper bound
python benchmark_kernel.py --no-flush-l2         # optimistic upper bound
python benchmark_kernel.py --output results.json # machine-readable
```

TFLOPS is computed as `2*M*K*N / time`; the report prints `%peak` against
the 125 TFLOPS A10G FP16 Tensor Core dense peak (use `--peak` for other
GPUs or locked clocks). Expected bands, acceptance criteria, and the
miss-diagnosis guide: docs/PERFORMANCE.md.

### The BK/GS A/B experiment

`--gs32-bk 64` (or `FLUTE_GS32_BK=64` in the environment) opts gs=32
layers into `<64,128,64,32>` deep K-tiles where `K % 64 == 0`: half the
K-iterations, same one-barrier double-buffered pipeline, shared memory
50,432 B (2 blocks/SM still fits). Measure gate/up_proj at M >= 4096
both ways and keep the winner. Shapes with `K % 64 != 0` (e.g. K=4128)
automatically stay on the BK=32 path.

## 6. Profiling

```bash
bash tools/ncu_profile.sh 4096 gate_proj
```

Collects the four success gates (tensor-pipe >= 70%, bank conflicts
<= 1%, registers <= 255 + zero local-memory traffic, occupancy >= 50%)
plus `lts__t_sector_hit_rate` for the Q re-read analysis of the BM=64
configs. Gate thresholds and interpretation: docs/PERFORMANCE.md.

## 7. Using the library

```python
import flute_extended
C = flute_extended.qgemm_per_group_lut(
    A,                 # [M, K] fp16 CUDA
    indices,           # [N, (K+1)//2] uint8, packed 4-bit LSB-first
    lut,               # [ceil(N/group_size), 16] fp16
    bitwidth=4,
    group_size=32,     # 32 (MLP gate/up) or 64 (attention + down)
    backend="cutlass_streaming",   # or "auto" / "naive" / "optimized"
)   # -> [M, N] fp16, Y = X @ W^T

# Deep-tile experiment for gs=32 layers (read once per process):
import os; os.environ["FLUTE_GS32_BK"] = "64"   # before the first call
```

See `example.py` for a complete run with a built-in correctness check.

## 8. Weight format

The palettized weight format (packed 4-bit indices, per-group 16-entry
LUT, LSB-first nibble order) is specified in docs/DEQUANT_SPEC.md. The
kernels and the test suite treat that document as the contract.

## 9. Troubleshooting

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
  the epilogue's block N-offset (`n0`) in the address math - every
  N-block must add its own n0.
- **Correctness fails only for `--gs32-bk 64`**: the deep-tile dispatch
  guard (`K % 64 == 0`) or the 5-group LUT staging of
  `<64,128,64,32>`.
- **Performance 2-3x below the bands**: re-run the ptxas audit (section
  3); spills are the usual cause.
- **Bank-conflict gate fails in ncu**: compare the swizzle functions in
  `mma.cuh` against the store-side indexing in the kernel's dequant and
  staging paths.
