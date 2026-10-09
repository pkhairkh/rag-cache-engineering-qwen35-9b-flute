#!/usr/bin/env python3
"""
benchmark_kernel.py — performance benchmark for the production kernel.

Measures TFLOPS = 2*M*K*N / time for cutlass_streaming (and optionally the
other backends) across the Qwen3.5-9B layer shapes and a batch-size sweep,
reporting % of the A10G FP16 Tensor Core dense peak.

Peak reference (see docs/HARDWARE.md — verified against AWS/NVIDIA sources):
the AWS A10G is an 80-SM GA102 part (320 Tensor Cores, 300 W board power,
600 GB/s, 6 MB L2). Its dense FP16 Tensor Core rates are ~125 TFLOPS with
FP16 ACCUMULATE and ~62.5 TFLOPS sustained (~70 at full 1710 MHz boost)
with FP32 ACCUMULATE. This kernel — and torch.matmul, and every cuBLAS
fp16 GEMM with default settings — accumulates in FP32, so 62.5 is the
honest default peak. Use --peak 125 only when comparing FP16-accumulate
kernels.

Timing methodology:
  - CUDA events around each iteration, torch.cuda.synchronize() between,
    median of the sorted times (robust to scheduling noise)
  - optional L2 flush between iterations (default ON) so the weight tensor
    (indices + LUT) is re-fetched from HBM as in real serving passes
    instead of being served from the 6 MB L2; the flush runs OUTSIDE the
    timed region (--no-flush-l2 disables it for the optimistic mode)
  - an inline cosine correctness check per data point: performance numbers
    are only trusted when the kernel is actually right

Decode sweep (--decode-sweep): M = 1..128 single-batch steps — the
weight-bandwidth-bound regime where a 4-bit palettized model must beat the
dense FP16 model. Reports tok/s and (with --energy) J/token for the
palettized kernel (idx4 production layout + the kernel-internal legacy
byte order as reference) vs a dense FP16 (torch.matmul) baseline,
optionally with CUDA Graphs (--graphs) to strip per-launch overhead.

Usage:
    python benchmark_kernel.py                          # streaming, real shapes
    python benchmark_kernel.py --backends cutlass_streaming cutlass_streaming_legacy
    python benchmark_kernel.py --compare-cublas         # add cuBLAS reference
    python benchmark_kernel.py --decode-sweep           # M=1..128 regime
    python benchmark_kernel.py --decode-sweep --graphs --energy
    python benchmark_kernel.py --peak 62.5              # override peak
    python benchmark_kernel.py --no-flush-l2            # optimistic mode
    python benchmark_kernel.py --gs32-bk 64             # deep-tile A/B (gs=32)
    python benchmark_kernel.py --output results.json
"""

import argparse
import json
import os
import sys
import threading
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import flute_extended  # noqa: E402
from flute_extended import qgemm_per_group_lut  # noqa: E402
from flute_extended.idxN import pack_idxn_from_packed as pack_idx4_from_packed  # noqa: E402

# Qwen3.5-9B layer shapes (metadata.json):
#   MLP gate/up : [N=12288, K=4096],  group_size=32
#   MLP down    : [N=4096,  K=12288], group_size=64
#   Attn QKV    : [N=8192,  K=4096],  group_size=64
#   Attn out    : [N=4096,  K=4096],  group_size=64
LAYERS = [
    ("gate_proj", 12288, 4096, 32),
    ("up_proj",   12288, 4096, 32),
    ("down_proj",  4096, 12288, 64),
    ("attn_qkv",   8192, 4096, 64),
    ("attn_out",   4096, 4096, 64),
]

DEFAULT_BATCH_SIZES = [128, 512, 1024, 2048, 4096]
DECODE_BATCH_SIZES = [1, 2, 4, 8, 16, 32, 64, 128]
# A10G (AWS variant): 80-SM GA102, 300 W, 600 GB/s. FP32-accumulate dense
# FP16 TC ceiling ~62.5 TFLOPS sustained / ~70 at 1710 MHz boost; the 125
# TFLOPS datasheet number is the FP16-ACCUMULATE rate (docs/HARDWARE.md).
PEAK_TFLOPS_DEFAULT = 62.5


def make_problem(M, K, N, group_size, seed=0):
    gen = torch.Generator(device="cuda")
    gen.manual_seed(seed)
    A = torch.randn(M, K, dtype=torch.float16, device="cuda", generator=gen)
    indices = torch.randint(0, 256, (N, (K + 1) // 2), dtype=torch.uint8,
                            device="cuda", generator=gen)
    lut = torch.randn((N + group_size - 1) // group_size, 16,
                      dtype=torch.float16, device="cuda", generator=gen)
    return A, indices, lut


def quick_correctness(A, indices, lut, group_size, C):
    """Fast cosine check vs an fp32 dequant reference: performance numbers
    are only trusted when the kernel is actually right."""
    N, Kp = indices.shape
    K = Kp * 2
    lo = (indices & 0x0F).long()
    hi = ((indices >> 4) & 0x0F).long()
    idx = torch.stack([lo, hi], dim=-1).reshape(N, K)
    grp = torch.arange(N, device=indices.device) // group_size
    W = lut[grp.unsqueeze(1).expand(N, K), idx]
    ref = (A.float() @ W.float().T).half()
    return torch.nn.functional.cosine_similarity(
        C.float().flatten(), ref.float().flatten(), dim=0).item()


def bench(fn, warmup=10, iters=50, flush_l2=True):
    """CUDA-event timing with optional L2 flush between iterations."""
    start_ev = torch.cuda.Event(enable_timing=True)
    end_ev = torch.cuda.Event(enable_timing=True)

    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    # 256 MB buffer to wipe L2 between iterations (A10G L2 = 6 MB)
    flush_buf = (torch.empty(64 * 1024 * 1024, dtype=torch.int8,
                             device="cuda") if flush_l2 else None)

    times = []
    for _ in range(iters):
        if flush_buf is not None:
            flush_buf.fill_(0)
        torch.cuda.synchronize()
        start_ev.record()
        fn()
        end_ev.record()
        torch.cuda.synchronize()
        times.append(start_ev.elapsed_time(end_ev))   # ms
    times.sort()
    # median is robust against scheduling noise
    return times[len(times) // 2]


def bench_one(backend, M, K, N, group_size, warmup=10, iters=50,
              flush_l2=True):
    A, indices, lut = make_problem(M, K, N, group_size)
    # Production path: idx4 blob through the public wrapper. The
    # "_legacy" suffix benches the kernel-internal q_layout=0 byte order
    # through the _C extension — a differential reference, not a
    # production path (identical shapes and data).
    use_legacy = backend.endswith("_legacy")
    real = backend[:-7] if use_legacy else backend
    eligible = N % 128 == 0 and K % 64 == 0

    if use_legacy or not eligible or real != "cutlass_streaming":
        W_empty = torch.empty(0, dtype=torch.float16, device=A.device)
        call = lambda: flute_extended._C.qgemm_per_group_lut(  # noqa: E731
            A, indices, lut, W_empty, 4, group_size, real, 0)
    else:
        q = torch.from_numpy(
            pack_idx4_from_packed(indices.cpu().numpy())).to(A.device)
        call = lambda: qgemm_per_group_lut(   # noqa: E731
            A, q, lut, bitwidth=4, group_size=group_size, backend=real,
            indices_layout="idx4")
    try:
        # correctness FIRST — one evaluation, cosine vs fp32 reference
        C0 = call()
        cos = quick_correctness(A, indices, lut, group_size, C0)
        del C0

        ms = bench(call, warmup, iters, flush_l2=flush_l2)
        tflops = 2.0 * M * K * N / (ms * 1e-3) / 1e12
        return {"tflops": tflops, "ms": ms, "cosine": cos, "error": None}
    except Exception as e:  # noqa: BLE001
        return {"tflops": 0.0, "ms": 0.0, "cosine": 0.0, "error": str(e)}


def bench_cublas(M, K, N, warmup=10, iters=50):
    gen = torch.Generator(device="cuda"); gen.manual_seed(0)
    A = torch.randn(M, K, dtype=torch.float16, device="cuda", generator=gen)
    W = torch.randn(N, K, dtype=torch.float16, device="cuda", generator=gen)
    fn = lambda: A @ W.T                    # noqa: E731
    ms = bench(fn, warmup, iters)
    return {"tflops": 2.0 * M * K * N / (ms * 1e-3) / 1e12, "ms": ms}


# ---------------------------------------------------------------------------
# Energy sampling (pynvml) — mean board power over a synchronized region
# ---------------------------------------------------------------------------
class PowerSampler:
    """Background thread sampling nvmlDeviceGetPower every ~5 ms.

    Usage:  s = PowerSampler(); s.start(); ...work...; watts = s.stop()
    Returns the mean power draw in W over the sampled region (NaN if pynvml
    is unavailable — pass --energy only on the box, `pip install nvidia-ml-py`).
    """

    def __init__(self, device_index=0):
        self._samples = []
        self._stop = threading.Event()
        self._thread = None
        self._handle = None
        try:
            import pynvml  # noqa: PLC0415
            pynvml.nvmlInit()
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
            self._pynvml = pynvml
        except Exception as e:  # noqa: BLE001
            print(f"[energy] pynvml unavailable ({e}); power columns disabled")

    @property
    def available(self):
        return self._handle is not None

    def start(self):
        if self._handle is None:
            return
        self._samples.clear()
        self._stop.clear()

        def loop():
            while not self._stop.is_set():
                try:
                    self._samples.append(
                        self._pynvml.nvmlDeviceGetPower(self._handle) / 1000.0)
                except Exception:  # noqa: BLE001
                    pass
                time.sleep(0.005)

        self._thread = threading.Thread(target=loop, daemon=True)
        self._thread.start()

    def stop(self):
        if self._handle is None:
            return float("nan")
        self._stop.set()
        self._thread.join()
        return (sum(self._samples) / len(self._samples)
                if self._samples else float("nan"))


def bench_energy(fn, warmup=10, iters=500, sampler=None):
    """Energy measurement: (ms/iter unflushed, mean W, J/iter).

    Runs `iters` back-to-back inside one CUDA-event region so the power
    sampler sees a steady state. Deliberately NOT interleaved with the L2
    flush — a 64 MB fill per iteration would add its own DRAM energy and
    dominate the J/iter number; timing honesty comes from the separate
    flushed bench() call, energy honesty from this clean loop.
    """
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    if sampler is not None:
        sampler.start()
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    watts = sampler.stop() if sampler is not None else float("nan")

    total_ms = start.elapsed_time(end)
    kernel_ms = max(total_ms, 1e-6) / iters
    joules = watts * (total_ms / 1e3) / iters
    return kernel_ms, watts, joules


def make_graphed_fn(fn):
    """Capture `fn` into a CUDA graph; returns a replay closure.

    The ~2-5 us per-launch CPU overhead dominates decode-sized GEMMs;
    graph replay removes it (P5). Allocation of the output inside capture
    goes to the graph private mempool — safe to replay repeatedly.
    """
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(s)

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    return lambda: g.replay()


# ---------------------------------------------------------------------------
# Decode sweep: the weight-bandwidth-bound regime (M = 1..128)
# ---------------------------------------------------------------------------
def run_decode_sweep(args, sampler):
    """tok/s + J/token for palettized (idx4 + kernel-internal legacy) vs dense.

    At these M the GEMM streams the full weight matrix once per step, so
    throughput is bounded by weight bytes / 600 GB/s: W4 = 0.5 B/param vs
    dense = 2 B/param -> the palettized path should win by up to ~4x as M
    shrinks (docs/PERFORMANCE.md, decode section).
    """
    results = {}
    for layer_name, N, K, gs in LAYERS:
        print(f"=== {layer_name}  (N={N}, K={K}, group_size={gs}) — decode ===")
        hdr = (f"{'M':>4} | {'W4 ms':>8} {'W4leg ms':>8} {'dense ms':>8} | "
               f"{'tok/s W4':>10} {'tok/s W4leg':>10} {'tok/s dens':>10} | "
               f"{'W4/dens':>9}")
        if sampler is not None and sampler.available:
            hdr += f" | {'J/tok W4':>10} {'J/tok dens':>10}"
        print(hdr)
        print("-" * len(hdr))
        layer_runs = []
        for M in args.decode_sizes:
            A, indices, lut = make_problem(M, K, N, gs)
            q_blob = torch.from_numpy(
                pack_idx4_from_packed(indices.cpu().numpy())).to(A.device)
            W = torch.randn(N, K, dtype=torch.float16, device="cuda")
            W_empty = torch.empty(0, dtype=torch.float16, device=A.device)

            fns = {
                "w4_idx4": lambda: qgemm_per_group_lut(   # noqa: E731
                    A, q_blob, lut, bitwidth=4, group_size=gs,
                    backend="cutlass_streaming",
                    indices_layout="idx4"),
                "w4_legacy": lambda: flute_extended._C.qgemm_per_group_lut(  # noqa: E731
                    A, indices, lut, W_empty, 4, gs, "cutlass_streaming", 0),
                "dense": lambda: A @ W.T,                  # noqa: E731
            }
            if args.graphs:
                fns = {k: make_graphed_fn(f) for k, f in fns.items()}

            row = {"M": M}
            ms = {}
            for k, f in fns.items():
                # timing: flushed, robust median (see bench())
                m = bench(f, warmup=20, iters=args.iters,
                          flush_l2=not args.no_flush)
                ms[k] = m
                row[f"{k}_ms"] = m
                # energy: unflushed steady-state loop
                if sampler is not None and sampler.available:
                    _, watts, joules = bench_energy(
                        f, warmup=20, iters=max(args.iters, 200),
                        sampler=sampler)
                    row[f"{k}_watts"] = watts
                    row[f"{k}_j_per_iter"] = joules

            for k in fns:
                row[f"{k}_tok_s"] = M / (ms[k] * 1e-3)
            row["idx4_dense_ratio"] = (M / ms["w4_idx4"]) / (M / ms["dense"])

            line = (f"{M:>4} | {ms['w4_idx4']:>8.4f} {ms['w4_legacy']:>8.4f} "
                    f"{ms['dense']:>8.4f} | "
                    f"{row['w4_idx4_tok_s']:>10.1f} "
                    f"{row['w4_legacy_tok_s']:>10.1f} "
                    f"{row['dense_tok_s']:>10.1f} | "
                    f"{row['idx4_dense_ratio']:>8.2f}x")
            if sampler is not None and sampler.available:
                jf = row["w4_idx4_j_per_iter"] / M
                jd = row["dense_j_per_iter"] / M
                line += f" | {jf:>10.3e} {jd:>10.3e}"
            print(line)
            layer_runs.append(row)
            del A, indices, lut, q_blob, W
        results[layer_name] = {"N": N, "K": K, "group_size": gs,
                               "mode": "decode", "graphs": args.graphs,
                               "runs": layer_runs}
        print()
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backends", nargs="+",
                    default=["cutlass_streaming"])
    ap.add_argument("--compare-cublas", action="store_true")
    ap.add_argument("--batch-sizes", type=int, nargs="+",
                    default=DEFAULT_BATCH_SIZES)
    ap.add_argument("--decode-sweep", action="store_true",
                    help="decode regime instead: M=1..128, tok/s + ratios "
                         "vs a dense fp16 baseline (the regime where W4 "
                         "wins on latency, throughput AND energy)")
    ap.add_argument("--decode-sizes", type=int, nargs="+",
                    default=DECODE_BATCH_SIZES,
                    help="M values for --decode-sweep")
    ap.add_argument("--graphs", action="store_true",
                    help="capture the decode step into a CUDA graph before "
                         "timing (strips per-launch overhead; P5)")
    ap.add_argument("--energy", action="store_true",
                    help="sample board power with pynvml during the loops; "
                         "adds W and J/token columns (needs nvidia-ml-py)")
    ap.add_argument("--peak", type=float, default=PEAK_TFLOPS_DEFAULT,
                    help="FP16 tensor core peak TFLOPS of the GPU (default: "
                         "62.5 = A10G FP32-ACCUMULATE sustained ceiling; "
                         "125 only for FP16-accumulate kernels)")
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--no-flush-l2", dest="no_flush", action="store_true",
                    help="do not flush L2 between iterations (optimistic "
                         "mode — the weight tensor then stays cached)")
    ap.add_argument("--gs32-bk", type=int, default=None, choices=[32, 64],
                    help="K-tile depth for gs=32 layers (sets FLUTE_GS32_BK; "
                         "A/B experiment: 64 = deep tiles where K % 64 == 0)")
    ap.add_argument("--output", default=None, help="write JSON results here")
    args = ap.parse_args()

    if args.gs32_bk is not None:
        # Must be set before the first streaming-kernel call (read once per
        # process by the C++ dispatch).
        os.environ["FLUTE_GS32_BK"] = str(args.gs32_bk)
        print(f"[config] FLUTE_GS32_BK={args.gs32_bk} "
              "(gs=32 layers use deep BK=64 tiles where K % 64 == 0)")

    if not torch.cuda.is_available():
        print("CUDA is not available — this benchmark requires a GPU.")
        sys.exit(2)

    print(f"device : {torch.cuda.get_device_name(0)} "
          f"(sm_{torch.cuda.get_device_capability(0)[0]}"
          f"{torch.cuda.get_device_capability(0)[1]})")
    print(f"torch  : {torch.__version__}, cuda {torch.version.cuda}")
    print(f"peak   : {args.peak:.1f} TFLOPS (FP16 TC dense; FP32-acc "
          f"ceiling on A10G — see docs/HARDWARE.md)")
    print(f"L2 flush between iterations: {not args.no_flush}")
    print()

    sampler = PowerSampler() if args.energy else None

    if args.decode_sweep:
        results = run_decode_sweep(args, sampler)
        if args.output:
            out = Path(args.output)
            out.parent.mkdir(parents=True, exist_ok=True)
            with open(out, "w") as f:
                json.dump({"mode": "decode", "graphs": args.graphs,
                           "layers": results}, f, indent=2)
            print(f"\nWrote: {out}")
        return

    results = {}
    best = 0.0
    for layer_name, N, K, gs in LAYERS:
        print(f"=== {layer_name}  (N={N}, K={K}, group_size={gs}) ===")
        print(f"{'M':>6}  {'Backend':<20} {'Time (ms)':>10} {'TFLOPS':>9} "
              f"{'%peak':>7} {'cosine':>8}")
        print("-" * 70)
        layer_runs = []
        for M in args.batch_sizes:
            for b in args.backends:
                r = bench_one(b, M, K, N, gs, args.warmup, args.iters,
                              flush_l2=not args.no_flush)
                eff = r["tflops"] / args.peak * 100
                err = f"  ERR: {r['error']}" if r["error"] else ""
                print(f"{M:>6}  {b:<20} {r['ms']:>10.3f} "
                      f"{r['tflops']:>9.2f} {eff:>6.1f}% {r['cosine']:>8.5f}{err}")
                layer_runs.append({"M": M, "backend": b, **r,
                                   "efficiency_pct": eff})
                if r["cosine"] < 0.999 and not r["error"]:
                    print(f"       WARNING: cosine {r['cosine']:.5f} < 0.999 — "
                          f"perf numbers for this point are NOT trustworthy")
                if r["tflops"] > best:
                    best = r["tflops"]
            if args.compare_cublas:
                r = bench_cublas(M, K, N, args.warmup, args.iters)
                eff = r["tflops"] / args.peak * 100
                print(f"{M:>6}  {'cublas (dense fp16)':<20} {r['ms']:>10.3f} "
                      f"{r['tflops']:>9.2f} {eff:>6.1f}%")
                layer_runs.append({"M": M, "backend": "cublas_dense_fp16", **r,
                                   "efficiency_pct": eff})
        results[layer_name] = {"N": N, "K": K, "group_size": gs,
                               "runs": layer_runs}
        print()

    print("=" * 60)
    print(f"Best kernel throughput : {best:.2f} TFLOPS "
          f"({best / args.peak * 100:.1f}% of {args.peak:.1f} TFLOPS peak)")
    print(f"FP32-acc ceiling band : 55-62 TFLOPS on MLP shapes (M>=1024) "
          f"vs 62.5 sustained peak; the 125 TFLOPS figure is the")
    print(f"FP16-ACCUMULATE rate (docs/HARDWARE.md). See docs/PERFORMANCE.md "
          f"before drawing conclusions from a single run.")

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            json.dump({"peak_tflops": args.peak, "best_tflops": best,
                       "layers": results}, f, indent=2)
        print(f"\nWrote: {out}")


if __name__ == "__main__":
    main()
