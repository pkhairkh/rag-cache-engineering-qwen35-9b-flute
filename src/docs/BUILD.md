# Build and bring-up

Building the two CUDA extensions, verifying the build is the one that
runs, and the correctness ladder to walk on first contact with new
hardware. Target: NVIDIA A10G (SM_86), CUDA 12.0+, PyTorch 2.x, C++20
host compiler. The same steps work on any SM_80+ GPU with adjusted clock
and peak expectations.

## 1. Prerequisites

```bash
nvidia-smi
nvcc --version
python -c "import torch; print(torch.__version__, torch.version.cuda, \
    torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))"
```

- CUDA toolkit 12.0+ (a 13.x toolkit against a 13.0-built torch warns
  about a minor mismatch and is fine in practice);
- `ninja` (`pip install ninja`) — without it the build falls back to a
  serial compile and a full rebuild costs tens of minutes instead of
  ~2;
- Optional: a CUTLASS checkout (2.x-4.8+) for the `cutlass_dense`
  baseline at `/home/ubuntu/cutlass` (or `$FLUTE_CUTLASS_HOME`,
  `$CUTLASS_HOME`, `/opt/cutlass`). Not required — the production
  kernels are raw-PTX and dependency-free.

## 2. Build

```bash
cd flute_extended
python setup.py build_ext --inplace     # extension .so lands in-tree
cd ../flute_train_kernels
python setup.py build_ext --inplace
```

What the build does, and the knobs:

- **Translation-unit split**: the kernel families live in one `.cu` per
  family, so ninja compiles them as independent jobs — an edit to one
  family recompiles only that family. `MAX_JOBS` defaults to
  `min(16, cpu_count)` and can be raised (`MAX_JOBS=8` is a safe floor
  on a busy box).
- **`FLUTE_CUDA_ARCHES`** (default `86` for the A10G): comma/space/
  semicolon list of SM arches for the `-gencode` set. SM_75 is rejected
  — the `mma.m16n8k16` f16 path requires SM_80+. For a portable build:
  `FLUTE_CUDA_ARCHES="80;86;89;90"`.
- `-std=c++20` for nvcc AND the host compiler; absolute `-I` include
  paths (nvcc runs from a scratch build dir). `-diag-suppress 177`
  keeps the unused-member noise out of the streaming kernel's template
  instantiations.
- CUTLASS detection prints `[flute_extended] CUTLASS: <path or NOT
  FOUND>`; the build succeeds either way (only the dense baseline
  needs it).

Expected result: a clean `[n/N]` ninja run with zero warnings, the
extension at `flute_extended/_C.cpython-*.so`.

Clean rebuild:

```bash
rm -rf build flute_extended/_C*.so
python setup.py build_ext --inplace
```

## 3. Verify the rebuilt module is the loaded one

`build_ext --inplace` updates the `.so` in the **source tree only**. If
`flute_extended` was installed non-editably, `import flute_extended`
resolves to the site-packages copy — which stays stale after a rebuild.
The palettizer probes the kernel's GS support at startup and aborts
before the model load, naming the loaded module's path, so a stale copy
is loud but the symptom looks like a kernel bug. After rebuilding:

```bash
python - <<'EOF'
import flute_extended as fx
print("loaded module:", fx.__file__)
EOF
```

If the path is under `site-packages`: `pip install -e .
--no-build-isolation`, or copy `build/lib*/flute_extended/_C*.so` over
the installed copy.

## 4. ptxas register audit (hard gate before benchmarking)

The streaming kernel targets ~130-210 registers per thread against the
255 cap. A spill destroys throughput silently (2-3x) and invalidates
every performance expectation:

```bash
nvcc -std=c++20 -O3 --use_fast_math --expt-relaxed-constexpr \
     -gencode=arch=compute_86,code=sm_86 \
     -Iinclude -I$(python -c "import torch; print(torch.utils.cpp_extension.include_paths()[0])") \
     -Xptxas -v -c src/kernel_streaming.cu -o /tmp/streaming.o 2>&1 | \
     grep -A3 "flute_kernel_streaming"
```

Expected for all three instantiations (`TileConfig<128,128,32,32>`,
`<64,128,64,32>`, `<64,128,64,64>`): `Used NNN registers, 0 spill
stores, 0 spill loads`, and dynamic shared memory 42,240 / 50,432 /
49,920 bytes. Any spill: STOP — try `--ptxas-options=-O3` or
`-maxrregcount=255`; the last resort is relaxing
`__launch_bounds__(128, 2)` to `(128, 1)` for the BK=64 configs. Do not
benchmark a spilling build.

## 5. Correctness ladder (walk in order)

Each step isolates a failure class before the next one runs:

```bash
# 5.1 scalar-staging twin (validates PTX fragment mapping + mma)
python test_flute.py --backend debug_simple

# 5.2 production kernel vs reference (multi-backend suite)
python test_flute.py

# 5.3 real-weights oracle (streaming kernel vs pure-torch dequant)
python test_qwen_weights.py
```

Expected: every line `[PASS]`, exit code 0, differential gates green
(nibble order, `streaming == debug_simple`, `idxN == kernel-internal ==
debug_simple`, guards). The suite covers both group sizes, unaligned N,
thin M, tile tails, the K=4128 dispatch fallback, and the real
Qwen3.5-9B layer shapes.

Bisection map: `naive` fails → data plumbing or harness; `debug_simple`
fails → PTX fragment mapping or mma semantics; `streaming` fails the
reference check → run the `streaming == debug_simple` gate to separate
staging from dispatch bugs; only the deep-tile A/B fails → the
`<64,128,64,32>` path (cp.async tile shape or the 5-group LUT).

For the decode GEMV family the box-side gate is
`scripts/verify_gemv.py` (the deployed kernel vs the fp32 reference on
every deployed residual rank and split regime).

## 6. Troubleshooting

| Symptom | Cause / fix |
|---|---|
| "C++20 or later compatible compiler is required" | host g++ < 11; upgrade or pass `-std=c++20` explicitly |
| "Unexpected instruction types specified for 'mma'" | an SM_75 target crept into the gencode list; remove it |
| `cudaFuncSetAttribute ... failed` at runtime | device does not expose the requested dynamic shared memory (100 KB needs SM_86+ with the opt-in; requested once per process) |
| Correctness fails only on multi-N-block shapes (N > 128) | inspect the epilogue's block N-offset (`n0`) — every N-block must add its own |
| Correctness fails only for `--gs32-bk 64` | the deep-tile dispatch guard (`K % 64 == 0`) or the 5-group LUT staging |
| Performance 2-3x below the bands | re-run the ptxas audit (§4); spills are the usual cause |
| Bank-conflict gate fails in ncu | compare the swizzle functions in `mma.cuh` against the store-side indexing in the kernel's dequant/staging paths |
| Build takes tens of minutes | `ninja` missing (install it) or `MAX_JOBS` unset — see §2 |
| Palettizer aborts with a "stale kernel" GS verdict | the loaded `_C` is not the rebuilt one — see §3 |
