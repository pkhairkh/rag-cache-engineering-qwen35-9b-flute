/**
 * flute/mma_bwd.cuh — backward-GEMM MMA helpers (flute_train_kernels).
 *
 * Extends the copied-in flute/mma.cuh with the pieces the backward
 * contraction needs:
 *
 *   * ldmatrix_x4_trans  — the .x4 variant of the .trans load (mma.cuh
 *     ships only .x2.trans). With .trans, thread t receives
 *     {M[2(t%4)][t/4], M[2(t%4)+1][t/4]} per 8x8 matrix, which lands on
 *     the PTX Fig. 81 B-fragment slots B[2(t%4)][t/4], B[2(t%4)+1][t/4].
 *
 *   * ldmatrix_bT_addr   — address math for the transposed B-fragment
 *     load from row-major sW[n][k]. The formulas are ldmatrix_a_addr's
 *     (A-style), NOT ldmatrix_b_addr's: .trans combined with the b_addr
 *     formulas is the K3 trap (executed at cos 0.507). The four 8x8
 *     matrices handed to the mma are
 *       M0 = sW[n0..+8  ][k0..+8  ] -> r0 = rb0 of output k-tile k0
 *       M1 = sW[n0+8..+16][k0..+8  ] -> r1 = rb1 of output k-tile k0
 *       M2 = sW[n0..+8  ][k0+8..+16] -> r2 = rb0 of output k-tile k0+8
 *       M3 = sW[n0+8..+16][k0+8..+16] -> r3 = rb1 of output k-tile k0+8
 *     Register wiring is unchanged versus the forward: bfrag[nt] = {r0,
 *     r1}; bfrag[nt+1] = {r2, r3}.
 *
 *   * cp_async_16_zfill  — 16 B cp.async with a source-size predicate:
 *     a zero size zero-fills the destination, which is the M-guard for
 *     ragged-M sA staging (spec CS-1.5).
 *
 *   * mma_m16n8k16_f32acc_bf16 — the bf16-operand twin of the production
 *     fp16 mma wrapper (spec CS-1.8): PTX form
 *     mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32, legal on sm_86.
 *
 *   * scalar fragment assembly (a_frag_scalar / b_frag_scalar) — the
 *     G-B2 differential twin path: reads the documented per-thread
 *     fragment positions directly from shared memory instead of through
 *     ldmatrix, so a bit-exact mismatch isolates the ldmatrix+swizzle
 *     path from the staging path.
 *
 *   * swz_half — scalar (non-block-aligned) swizzled half-offset within
 *     a row, consistent with swz_col16 for block-aligned columns.
 */
#pragma once

#include "flute/mma.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cstdint>

namespace flute {

// ---------------------------------------------------------------------------
// ldmatrix.x4.trans
// ---------------------------------------------------------------------------
__device__ __forceinline__ void ldmatrix_x4_trans(
    uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3,
    const __half* smem
) {
    const uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(smem));
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 "
        "{%0, %1, %2, %3}, [%4];\n"
        : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
        : "r"(addr)
    );
}

// ---------------------------------------------------------------------------
// Address math for the transposed B-fragment load (A-style formulas).
// ---------------------------------------------------------------------------
__device__ __forceinline__ const __half* ldmatrix_bT_addr(
    const __half* tile, int ld, int n_base, int k_base,
    int lane, int xor_mask
) {
    // A-style row/column selection — see the file header; this is the K3
    // trap's negative pole: substituting ldmatrix_b_addr here produces the
    // cos-0.507 failure the G-B3b gate pins.
    const int r = n_base + ((lane >> 3) & 1) * 8 + (lane & 7);
    const int c = k_base + (lane >> 4) * 8;
    return tile + (size_t)r * ld + swz_col16(r, c, xor_mask);
}

// ---------------------------------------------------------------------------
// cp.async with zero-fill predicate (PTX ISA 9.7.8.4: source-size operand
// form — a size of 0 fills the destination with zeros).
// ---------------------------------------------------------------------------
__device__ __forceinline__ void cp_async_16_zfill(
    void* dst_smem, const void* src_gmem, bool in_bounds
) {
    const uint32_t dst = static_cast<uint32_t>(__cvta_generic_to_shared(dst_smem));
    asm volatile(
        "cp.async.cg.shared.global [%0], [%1], 16, %2;\n"
        ::
        "r"(dst),
        "l"(src_gmem),
        "r"(in_bounds ? 16 : 0)
    );
}

// ---------------------------------------------------------------------------
// bf16 MMA (spec CS-1.8) — same fragment layout contract as the fp16 form.
// ---------------------------------------------------------------------------
__device__ __forceinline__ void mma_m16n8k16_f32acc_bf16(
    const uint32_t (&a)[4],
    const uint32_t (&b)[2],
          float    (&c)[4]
) {
    float c0 = c[0], c1 = c[1], c2 = c[2], c3 = c[3];
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0, %1, %2, %3}, "
        "{%4, %5, %6, %7}, "
        "{%8, %9}, "
        "{%10, %11, %12, %13};\n"
        : "=f"(c0), "=f"(c1), "=f"(c2), "=f"(c3)
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
          "r"(b[0]), "r"(b[1]),
          "f"(c0), "f"(c1), "f"(c2), "f"(c3)
    );
    c[0] = c0; c[1] = c1; c[2] = c2; c[3] = c3;
}

// ---------------------------------------------------------------------------
// Scalar fragment assembly (G-B2 twin path)
// ---------------------------------------------------------------------------
// Swizzled half-offset of logical column `col` in row `r` — consistent with
// swz_col16 for 16B-aligned columns; valid for arbitrary columns.
__device__ __forceinline__ int swz_half(int row, int col, int xor_mask) {
    return (((col >> 3) ^ (row & (xor_mask - 1))) << 3) | (col & 7);
}

// A-fragment (16x16 tile at logical (row_base, col_base), row stride ld):
// thread `lane` packs ra0..ra3 per the mma.cuh A layout table.
template <typename T>
__device__ __forceinline__ void a_frag_scalar(
    uint32_t (&a)[4],
    const T* tile, int ld, int row_base, int col_base,
    int lane, int xor_mask
) {
    const int g = lane >> 2;
    const int c = 2 * (lane & 3);
    auto load2 = [&](int r, int cc) -> uint32_t {
        // {tile[r][cc], tile[r][cc+1]} as one u32 (low element first).
        const T h0 = tile[(size_t)r * ld + swz_half(r, cc, xor_mask)];
        const T h1 = tile[(size_t)r * ld + swz_half(r, cc + 1, xor_mask)];
        return (uint32_t)*reinterpret_cast<const uint16_t*>(&h0)
             | ((uint32_t)*reinterpret_cast<const uint16_t*>(&h1) << 16);
    };
    a[0] = load2(row_base + g,     col_base + c);
    a[1] = load2(row_base + g + 8, col_base + c);
    a[2] = load2(row_base + g,     col_base + c + 8);
    a[3] = load2(row_base + g + 8, col_base + c + 8);
}

// B-fragment for the BACKWARD contraction (16x8 tile of the mma's B operand:
// contraction rows n_base..+16, output cols k_base..+8 of row-major sW[n][k]):
// thread `lane` needs rb0 = {W[2(l%4)][l/4], W[2(l%4)+1][l/4]},
// rb1 = {W[2(l%4)+8][l/4], W[2(l%4)+9][l/4]} (PTX Fig. 81).
template <typename T>
__device__ __forceinline__ void b_frag_scalar(
    uint32_t (&b)[2],
    const T* tile, int ld, int n_base, int k_base,
    int lane, int xor_mask
) {
    const int g = lane >> 2;
    const int c = 2 * (lane & 3);
    auto load2 = [&](int n) -> uint32_t {
        const T h0 = tile[(size_t)n * ld + swz_half(n, k_base + g, xor_mask)];
        const T h1 = tile[(size_t)(n + 1) * ld + swz_half(n + 1, k_base + g, xor_mask)];
        return (uint32_t)*reinterpret_cast<const uint16_t*>(&h0)
             | ((uint32_t)*reinterpret_cast<const uint16_t*>(&h1) << 16);
    };
    b[0] = load2(n_base + c);
    b[1] = load2(n_base + c + 8);
}

}  // namespace flute
