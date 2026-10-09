/**
 * include/flute/entrypoints.h
 *
 * The PUBLIC compile surface of the kernel TUs - ONE declaration per
 * entrypoint, included by EVERY defining TU AND by src/bindings.cpp.
 * the monolith's `flute::GemvSegTab` namespace error (the box
 * had to hand-patch commit 749eef2 to build) was exactly this bug class
 * - a symbol referenced with a namespace qualifier it does not live in.
 * With this header, a mismatched signature or qualifier is a compile
 * error in EVERY TU at once, on any box, before the push.
 *
 * The heterogeneous multi/MLP entries (per-segment bitwidths, group
 * sizes, rotation signs, AWQ scales - see flute/gemv.cuh, GemvSegTab):
 *   qgemm_cutlass_gemv_multi  - the grouped QKV merge, ONE launch over
 *                                the concatenated tile space; the seg
 *                                table carries the per-segment specs.
 *   qgemm_cutlass_gemv_mlp    - the gate+up merge with the SiLU*mul
 *                                epilogue; TWO seg-table rows (gate, up),
 *                                the specs may differ.
 * The older plain/FHT entry pairs collapse: signs presence per segment
 * (table column) selects the FHT-fused path inside the impl.
 */

#pragma once

#include <torch/extension.h>

#include <cstdint>

// ---- streaming / prefill family (kernel_streaming.cu) --------------------
torch::Tensor qgemm_cutlass_streaming(torch::Tensor A,
    torch::Tensor indices,
    torch::Tensor lut,
    int64_t bitwidth,
    int64_t group_size,
    int64_t q_layout);

// ----  dual-stream fused decode family (kernel_streaming.cu) ----------
torch::Tensor qgemm_cutlass_dual_stream(torch::Tensor A,
    torch::Tensor indices,
    torch::Tensor lut,
    int64_t bitwidth,
    torch::Tensor indices2,
    torch::Tensor lut2,
    int64_t bitwidth2,
    torch::Tensor resB,
    torch::Tensor resA,
    torch::Tensor bias,
    int64_t group_size);

// ----  /  GEMV family (kernel_gemv.cu) --------------------------
torch::Tensor qgemm_cutlass_gemv_stream(torch::Tensor A,
    torch::Tensor indices,
    torch::Tensor lut,
    int64_t bitwidth,
    torch::Tensor indices2,
    torch::Tensor lut2,
    int64_t bitwidth2,
    torch::Tensor resB,
    torch::Tensor resA,
    torch::Tensor bias,
    int64_t group_size);

torch::Tensor qgemm_cutlass_gemv_fht_stream(torch::Tensor A,
    torch::Tensor indices,
    torch::Tensor lut,
    int64_t bitwidth,
    torch::Tensor indices2,
    torch::Tensor lut2,
    int64_t bitwidth2,
    torch::Tensor resB,
    torch::Tensor resA,
    torch::Tensor bias,
    int64_t group_size,
    torch::Tensor signs,
    torch::Tensor s);

// ----  split-K GEMV family (kernel_gemv_splitk.cu) --------------------------------
torch::Tensor qgemm_cutlass_gemv_splitk_stream(torch::Tensor A,
    torch::Tensor indices,
    torch::Tensor lut,
    int64_t bitwidth,
    torch::Tensor indices2,
    torch::Tensor lut2,
    int64_t bitwidth2,
    torch::Tensor resB,
    torch::Tensor resA,
    torch::Tensor bias,
    int64_t group_size);

torch::Tensor qgemm_cutlass_gemv_splitk_fht_stream(torch::Tensor A,
    torch::Tensor indices,
    torch::Tensor lut,
    int64_t bitwidth,
    torch::Tensor indices2,
    torch::Tensor lut2,
    int64_t bitwidth2,
    torch::Tensor resB,
    torch::Tensor resA,
    torch::Tensor bias,
    int64_t group_size,
    torch::Tensor signs,
    torch::Tensor s);

// ----  grouped multi-blob split-K GEMV (kernel_gemv_multi.cu) ----------
// seg_table: CPU int64 [count, 15] - per segment
//   [q1, lut1, q2, lut2, resB, resA, bias, C, N_seg, R_seg,
//    b1, b2, gs, signs, s]
// (address 0 = not supplied). Per-segment (b1, b2, gs, signs, s) may
// DIFFER across segments - the heterogeneous contract. Any signs
// nonzero selects the FHT-fused path; all-or-none rotation presence.
void qgemm_cutlass_gemv_multi(torch::Tensor A,
    torch::Tensor seg_table);

// ----  merged gate+up split-K GEMV + SiLU*mul (kernel_gemv_mlp.cu) ----
// seg_table: CPU int64 [2, 15] - row 0 = gate, row 1 = up (the same
// column contract as the multi table; N must be identical across rows).
// C: the persistent [1, N] fp16 output buffer (silu(gate(x)) * up(x),
// ONE fp16 round); the gate row's C column is ignored.
torch::Tensor qgemm_cutlass_gemv_mlp(torch::Tensor A,
    torch::Tensor seg_table,
    torch::Tensor C);
