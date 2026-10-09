#!/usr/bin/env python3
"""
FLUTE-Extended multi-backend test suite.

Usage:
    python test_flute.py                       # Quick correctness on all backends
    python test_flute.py --benchmark           # Full benchmark sweep
    python test_flute.py --backend cutlass_streaming
    python test_flute.py --layer gate_proj     # Use real Qwen3.5-9B shape
    python test_flute.py --compare-dense       # Compare against CUTLASS dense
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch

# Make the local package importable regardless of checkout location.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import flute_extended
from flute_extended import qgemm_per_group_lut, qgemm_dense
from flute_extended.idxN import pack_idxn as pack_idx4
from flute_extended.idxN import pack_idxn, check_bits

# pack_idx4 = pack_idxn with bits=4
def pack_idx4(indices):
    return pack_idxn(indices, 4)


# ---------------------------------------------------------------------------
# Reference dequantization (slow but obviously correct)
# ---------------------------------------------------------------------------
def dequantize_reference(logical: torch.Tensor, lut: torch.Tensor, group_size: int,
    bitwidth: int = 4,
) -> torch.Tensor:
    """Materialize W [N, K] from logical b-bit indices + LUT. Slow but
    obviously correct."""
    palette = 1 << int(bitwidth)
    N, K = logical.shape
    n_groups = lut.shape[0]
    lut_expanded = lut.view(n_groups, 1, palette).expand(n_groups, group_size, palette).reshape(-1, palette)
    # Map (n, k) → group
    n_idx = torch.arange(N, device=logical.device) // group_size
    n_idx_exp = n_idx.view(N, 1).expand(N, K)
    # Gather from LUT
    W = torch.gather(lut[n_idx_exp],  # [N, K, palette]
        dim=-1,
        index=logical.long().unsqueeze(-1),
    ).squeeze(-1)
    return W.to(torch.float16)


def pack_logical_np(idx: np.ndarray, bits: int) -> np.ndarray:
    """Logical LSB-first packed rows [N, K*bits/8] (the q_layout=0 byte
    order the legacy kernel and debug_simple consume). K is a multiple
    of 64 in every tested shape, so K*bits/8 is exact (b=3 included)."""
    b = check_bits(bits)
    v = np.ascontiguousarray(idx, dtype=np.uint8)
    N, K = v.shape
    if b == 4:
        return (v[:, 0::2] | (v[:, 1::2] << 4)).copy()
    if b == 2:
        q = v.reshape(N, K // 4, 4)
        return (q[..., 0] | (q[..., 1] << 2) | (q[..., 2] << 4)
                | (q[..., 3] << 6)).copy()
    if b == 1:
        q = v.reshape(N, K // 8, 8)
        out = np.zeros((N, K // 8), dtype=np.uint8)
        for j in range(8):
            out |= q[..., j] << j
        return out
    # b == 3: 8 values per 3 bytes (24-bit little-endian groups)
    q = v.astype(np.uint32).reshape(N, K // 8, 8)
    acc = np.zeros((N, K // 8), dtype=np.uint32)
    for j in range(8):
        acc |= q[..., j] << (3 * j)
    out = np.empty((N, 3 * (K // 8)), dtype=np.uint8)
    out[:, 0::3] = (acc & 0xFF).astype(np.uint8)
    out[:, 1::3] = ((acc >> 8) & 0xFF).astype(np.uint8)
    out[:, 2::3] = ((acc >> 16) & 0xFF).astype(np.uint8)
    return out


# ---------------------------------------------------------------------------
# Qwen3.5-9B layer shapes
# ---------------------------------------------------------------------------
QWEN_LAYERS = {
    "gate_proj":  dict(N=12288, K=4096, group_size=32),
    "up_proj":    dict(N=12288, K=4096, group_size=32),
    "down_proj":  dict(N=4096,  K=12288, group_size=64),
    "attn_qkv":   dict(N=8192,  K=4096, group_size=64),
    "attn_out":   dict(N=4096,  K=4096, group_size=64),
}

# the kernel's full GS support set (scripts/HANDOVER.md issue 1).
GS_ALL = (16, 32, 64, 128, 256, 512)


def _run_backend(backend: str, A, logical, indices, lut, group_size: int):
    """Route a backend correctly: cutlass_streaming takes the production
    idx4 blob through the public wrapper; debug_simple reads the
    kernel-internal legacy byte order and is called through _C."""
    if backend == "cutlass_streaming" or backend == "debug_simple":
        return qgemm_per_group_lut(A, indices, lut, bitwidth=4,
                                   group_size=group_size, backend=backend,
                                   indices_layout="idx4")
    raise ValueError(f"Unknown backend: {backend}")


# ---------------------------------------------------------------------------
# Correctness test
# ---------------------------------------------------------------------------
def test_correctness(backend: str, M: int = 256, K: int = 512, N: int = 256,
    group_size: int = 32, tol_cosine: float = 0.99
) -> bool:
    print(f"\n[correctness] backend={backend}  M={M} K={K} N={N}  gs={group_size}")
    torch.manual_seed(42)

    A = torch.randn(M, K, dtype=torch.float16, device="cuda")
    logical = torch.randint(0, 16, (N, K), dtype=torch.uint8, device="cuda")
    indices = torch.from_numpy(pack_idx4(logical.cpu().numpy())).to("cuda")   # idx4 blob
    lut = torch.randn(((N + group_size - 1) // group_size, 16),
                    dtype=torch.float16, device="cuda")

    # Run kernel
    C = _run_backend(backend, A, logical, indices, lut, group_size)

    # Reference: dequantize W then matmul
    W = dequantize_reference(logical, lut, group_size)
    C_ref = A @ W.T

    # Metrics
    max_diff = (C - C_ref).abs().max().item()
    mean_diff = (C - C_ref).abs().mean().item()
    cosine = torch.nn.functional.cosine_similarity(
        C.flatten().to(torch.float32), C_ref.flatten().to(torch.float32), dim=0
    ).item()

    print(f"  max_diff  : {max_diff:.6f}")
    print(f"  mean_diff : {mean_diff:.6f}")
    print(f"  cosine    : {cosine:.6f}")
    ok = cosine > tol_cosine and not torch.isnan(C).any() and not torch.isinf(C).any()
    print(f"  status    : {'PASS' if ok else 'FAIL'}")
    return ok


# ---------------------------------------------------------------------------
# idxN family gates (bitwidths 1/2/3): the SAME differential chain as the
# 4-bit gates — streaming(fd blob) == debug_simple(legacy layout, _C) ==
# reference dequant matmul — plus the blob/size/layout refusal guards.
# ---------------------------------------------------------------------------
def test_correctness_idxn(bits: int, M: int = 256, K: int = 512, N: int = 256,
    group_size: int = 32,
) -> bool:
    """One width of the idxN family: pack a random b-bit index matrix,
    run the fragment-direct kernel through the public wrapper and the
    legacy-layout kernel through _C, and require BOTH to equal the
    reference dequant matmul BIT-EXACTLY (torch.equal)."""
    b = check_bits(bits)
    pal = 1 << b
    print(f"\n[correctness idxN] bits={b}  M={M} K={K} N={N}  gs={group_size}")
    torch.manual_seed(43 + b)

    A = torch.randn(M, K, dtype=torch.float16, device="cuda")
    logical = torch.randint(0, pal, (N, K), dtype=torch.uint8, device="cuda")
    blob = torch.from_numpy(pack_idxn(logical.cpu().numpy(), b)).to("cuda")        # idxN blob
    legacy = torch.from_numpy(pack_logical_np(logical.cpu().numpy(), b)).to("cuda")  # q_layout=0
    lut = torch.randn(((N + group_size - 1) // group_size, pal),
                      dtype=torch.float16, device="cuda")

    # fragment-direct path (the production idxN layout)
    C_fd = qgemm_per_group_lut(
        A, blob, lut, bitwidth=b, group_size=group_size,
        backend="cutlass_streaming", indices_layout=f"idx{b}")

    # legacy layout through the raw entrypoint (q_layout=0)
    legacy_2d = legacy.view(N, (K * b) // 8)
    C_leg = flute_extended._C.qgemm_cutlass_streaming(
        A, legacy_2d, lut, b, group_size, 0)

    # debug_simple differential twin (legacy layout, scalar loads)
    C_dbg = flute_extended._C.qgemm_debug_simple(
        A, legacy_2d, lut, b, group_size)

    # reference
    W = dequantize_reference(logical, lut, group_size, bitwidth=b)
    C_ref = A @ W.T

    ok_fd = torch.equal(C_fd, C_ref)
    ok_leg = torch.equal(C_leg, C_ref)
    ok_dbg = torch.equal(C_dbg, C_ref)
    cross_fd_leg = torch.equal(C_fd, C_leg)
    print(f"  blob bytes: {blob.numel()} (N*K*{b}/8 = {N * K * b // 8})")
    print(f"  fd == ref          : {'PASS' if ok_fd else 'FAIL'}")
    print(f"  legacy == ref      : {'PASS' if ok_leg else 'FAIL'}")
    print(f"  debug_simple == ref: {'PASS' if ok_dbg else 'FAIL'}")
    print(f"  fd == legacy       : {'PASS' if cross_fd_leg else 'FAIL'}")
    if not (ok_fd and ok_leg and ok_dbg and cross_fd_leg):
        md = (C_fd.float() - C_ref.float()).abs().max().item()
        print(f"  max|fd-ref| = {md:.6f}")
    ok = ok_fd and ok_leg and ok_dbg and cross_fd_leg
    print(f"  status    : {'PASS' if ok else 'FAIL'}")
    return ok


def test_idxn_guards() -> bool:
    """Refusal gates: wrong-width layout strings, blob-size mismatch,
    layout/bitwidth disagreement (through the public wrapper)."""
    print("\n[correctness idxN] guards")
    N, K, gs = 256, 512, 32
    logical = torch.randint(0, 4, (N, K), dtype=torch.uint8, device="cuda")
    blob = torch.from_numpy(pack_idxn(logical.cpu().numpy(), 2)).to("cuda")
    lut2 = torch.randn(((N + gs - 1) // gs, 4),
                       dtype=torch.float16, device="cuda")
    A = torch.randn(64, K, dtype=torch.float16, device="cuda")
    ok = True

    def _must_raise(fn, label):
        nonlocal ok
        try:
            fn()
            print(f"  {label}: FAIL (accepted)")
            ok = False
        except (ValueError, RuntimeError):
            print(f"  {label}: PASS (refused)")

    _must_raise(lambda: qgemm_per_group_lut(A, blob, lut2, bitwidth=4,
                                    group_size=gs, indices_layout="idx4"),
        "idx4 layout with an idx2 blob")
    _must_raise(lambda: qgemm_per_group_lut(A, blob, lut2, bitwidth=2,
                                    group_size=gs, indices_layout="idx4"),
        "bitwidth/layout disagreement")
    _must_raise(lambda: qgemm_per_group_lut(A, blob[:-1], lut2, bitwidth=2,
                                    group_size=gs, indices_layout="idx2"),
        "truncated blob")
    lut16 = torch.randn(((N + gs - 1) // gs, 16),
                        dtype=torch.float16, device="cuda")
    _must_raise(lambda: qgemm_per_group_lut(A, blob, lut16, bitwidth=2,
                                    group_size=gs, indices_layout="idx2"),
        "16-entry LUT at bitwidth 2")
    print(f"  status    : {'PASS' if ok else 'FAIL'}")
    return ok


def test_correctness_all_backends(M: int = 256, K: int = 512, N: int = 256,
                                  group_size: int = 32) -> dict[str, bool]:
    backends = ["cutlass_streaming"]
    # cutlass_dense requires dense W, tested separately
    results = {}
    for b in backends:
        try:
            results[b] = test_correctness(b, M=M, K=K, N=N, group_size=group_size)
        except Exception as e:
            print(f"  ERROR: {e}")
            results[b] = False
    return results


# ---------------------------------------------------------------------------
# Performance benchmark
# ---------------------------------------------------------------------------
def benchmark(backend: str, M: int, K: int, N: int, group_size: int,
    warmup: int = 5, iters: int = 20
) -> tuple[float, float]:
    torch.manual_seed(42)
    A = torch.randn(M, K, dtype=torch.float16, device="cuda")
    logical = torch.randint(0, 16, (N, K), dtype=torch.uint8, device="cuda")
    indices = torch.from_numpy(pack_idx4(logical.cpu().numpy())).to("cuda")   # idx4 blob
    lut = torch.randn(((N + group_size - 1) // group_size, 16),
                    dtype=torch.float16, device="cuda")

    # Warmup
    for _ in range(warmup):
        _ = _run_backend(backend, A, logical, indices, lut, group_size)
    torch.cuda.synchronize()

    # Timed
    start = time.time()
    for _ in range(iters):
        _ = _run_backend(backend, A, logical, indices, lut, group_size)
    torch.cuda.synchronize()
    elapsed = (time.time() - start) / iters

    flops = 2 * M * K * N
    tflops = flops / elapsed / 1e12
    return tflops, elapsed * 1000  # TFLOPS, ms


def benchmark_dense(
    M: int, K: int, N: int, warmup: int = 5, iters: int = 20
) -> tuple[float, float]:
    """Benchmark CUTLASS dense FP16 GEMM (Phase 1 baseline)."""
    torch.manual_seed(42)
    A = torch.randn(M, K, dtype=torch.float16, device="cuda")
    W = torch.randn(N, K, dtype=torch.float16, device="cuda")

    for _ in range(warmup):
        _ = qgemm_dense(A, W, group_size=32, backend="cutlass_dense")
    torch.cuda.synchronize()

    start = time.time()
    for _ in range(iters):
        _ = qgemm_dense(A, W, group_size=32, backend="cutlass_dense")
    torch.cuda.synchronize()
    elapsed = (time.time() - start) / iters

    flops = 2 * M * K * N
    return flops / elapsed / 1e12, elapsed * 1000


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="FLUTE-Extended test suite")
    parser.add_argument("--benchmark", action="store_true",
                        help="Run performance benchmark")
    parser.add_argument("--backend", default="all",
                        choices=["all", "debug_simple",
                                 "cutlass_streaming", "cutlass_dense"])
    parser.add_argument("--layer", default=None,
                        choices=list(QWEN_LAYERS.keys()),
                        help="Use a real Qwen3.5-9B layer shape")
    parser.add_argument("--M", type=int, default=4096)
    parser.add_argument("--K", type=int, default=4096)
    parser.add_argument("--N", type=int, default=12288)
    parser.add_argument("--group-size", type=int, default=32,
                        choices=[16, 32, 64, 128, 256, 512])
    parser.add_argument("--gs", type=int, default=None,
                        help="explicit GS override (16/32/64/128/256/512) "
                             "on top of --layer or the manual N/K")
    parser.add_argument("--gs-sweep", action="store_true",
                        help="run the differential chain at EVERY "
                             "supported GS (16/32/64/128/256/512)")
    parser.add_argument("--compare-dense", action="store_true",
                        help="Also benchmark CUTLASS dense FP16 for reference")
    parser.add_argument("--bits", default="all",
                        choices=["all", "1", "2", "3", "4"],
                        help="idxN family widths to gate: 'all' runs the "
                             "1/2/3-bit differential chain plus the 4-bit "
                             "gates; a specific width runs just that one")
    args = parser.parse_args()

    # Layer override
    if args.layer:
        cfg = QWEN_LAYERS[args.layer]
        args.N = cfg["N"]
        args.K = cfg["K"]
        args.group_size = cfg["group_size"]
        print(f"Using Qwen3.5-9B layer '{args.layer}': N={args.N} K={args.K} gs={args.group_size}")

    # explicit GS override (any supported value on any layer shape)
    if args.gs is not None:
        if args.gs not in GS_ALL:
            parser.error(f"--gs must be one of {GS_ALL}, got {args.gs}")
        args.group_size = args.gs
        print(f"GS override: gs={args.group_size}")

    # the full-GS differential sweep — the correctness chain at every
    # supported GS (the kernels claim {16..512}; this gate proves it).
    if args.gs_sweep:
        print("=" * 80)
        print(f"GS SWEEP : differential chain at every GS in {GS_ALL}")
        print("=" * 80)
        sweep_ok = True
        for gs in GS_ALL:
            print(f"\n===== GS = {gs} =====")
            for b in (1, 2, 3, 4):
                try:
                    if b == 4:
                        ok = test_correctness(
                            "cutlass_streaming", M=256, K=512, N=256,
                            group_size=gs)
                    else:
                        ok = test_correctness_idxn(b, M=256, K=512, N=256, group_size=gs)
                    sweep_ok = sweep_ok and ok
                except Exception as e:
                    print(f"  ERROR (idx{b}, gs={gs}): {e}")
                    sweep_ok = False
        print("\n" + "=" * 80)
        print(f"GS SWEEP: {'ALL PASS' if sweep_ok else 'FAILURES PRESENT'}"
              f" (GS {GS_ALL}, widths 1-4)")
        print("=" * 80)
        return

    # --- Correctness ---
    print("=" * 80)
    print("CORRECTNESS")
    print("=" * 80)
    widths = ([1, 2, 3, 4] if args.bits == "all" else [int(args.bits)])
    all_ok = {}
    for b in widths:
        if b == 4:
            if args.backend == "all":
                results = test_correctness_all_backends(
                    M=256, K=512, N=256, group_size=args.group_size
                )
                # Also test group_size=64 if requested
                if args.group_size == 64:
                    print("\n--- group_size=64 sanity ---")
                    for bkd in ["cutlass_streaming"]:
                        try:
                            results[bkd] = test_correctness(bkd, M=128, K=256, N=128, group_size=64)
                        except Exception as e:
                            print(f"  ERROR: {e}")
                            results[bkd] = False
                all_ok.update(results)
            else:
                all_ok[args.backend] = test_correctness(args.backend, M=256, K=512, N=256,
                    group_size=args.group_size)
        else:
            try:
                all_ok[f"idx{b}"] = test_correctness_idxn(b, M=256, K=512, N=256, group_size=args.group_size)
            except Exception as e:
                print(f"  ERROR (idx{b}): {e}")
                all_ok[f"idx{b}"] = False
            if b == 2:
                # guards once (idx2-flavored)
                try:
                    all_ok["idxn_guards"] = test_idxn_guards()
                except Exception as e:
                    print(f"  ERROR (guards): {e}")
                    all_ok["idxn_guards"] = False

    if widths != [4]:
        print("\n" + "=" * 80)
        n_pass = sum(1 for v in all_ok.values() if v)
        print(f"idxN GATES: {n_pass}/{len(all_ok)} PASS"
              + ("" if n_pass == len(all_ok) else "  <-- FAILURES PRESENT"))
        print("=" * 80)

    if not args.benchmark:
        print("\n" + "=" * 80)
        print("Run with --benchmark for performance numbers")
        print("=" * 80)
        return

    # --- Performance ---
    print("\n" + "=" * 80)
    print(f"PERFORMANCE  (M={args.M} K={args.K} N={args.N} gs={args.group_size})")
    print("=" * 80)
    print(f"{'Backend':<22} {'Time (ms)':>12} {'TFLOPS':>10} {'%peak':>8}")
    print("-" * 56)

    peak_tflops = 62.5   # A10G FP32-accumulate sustained ceiling (docs/HARDWARE.md)

    backends_to_run = (["cutlass_streaming"]
                       if args.backend == "all" else [args.backend])

    for b in backends_to_run:
        try:
            tflops, ms = benchmark(b, args.M, args.K, args.N, args.group_size)
            eff = tflops / peak_tflops * 100
            print(f"{b:<22} {ms:>12.2f} {tflops:>10.2f} {eff:>7.1f}%")
        except Exception as e:
            print(f"{b:<22} ERROR: {e}")

    if args.compare_dense or args.backend == "cutlass_dense":
        try:
            tflops, ms = benchmark_dense(args.M, args.K, args.N)
            eff = tflops / peak_tflops * 100
            print(f"{'cutlass_dense':<22} {ms:>12.2f} {tflops:>10.2f} {eff:>7.1f}%")
        except Exception as e:
            print(f"{'cutlass_dense':<22} ERROR: {e}")

    print("-" * 56)
    print(f"{'peak (A10G FP32-acc sustained)':<32} {peak_tflops:>10.1f} {100.0:>7.1f}%")


if __name__ == "__main__":
    main()
