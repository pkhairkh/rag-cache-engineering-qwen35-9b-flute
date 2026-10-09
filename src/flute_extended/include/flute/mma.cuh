/**
 * flute/mma.cuh
 *
 * Minimal Ampere MMA helpers for FP16 GEMM: mma.sync.m16n8k16 wrappers,
 * ldmatrix (with 16B-block XOR swizzle address math), cp.async staging,
 * and the prmt.b32 merge used by the dequantizer.
 *
 * Instruction forms and fragment layouts below follow the PTX ISA
 * (release 9.4; sections 9.7.16.5.8 / .14 / .15, 9.7.8.4, 9.7.9.4).
 *
 * One MMA instruction computes:  D[16x8] = A[16x16] * B[16x8] + C[16x8]
 *
 * VALID INSTRUCTION FORMS (the four type fields are ordered  d.a.b.c ,
 * destination first — the only f16-operand forms for m16n8k16 are):
 *
 *   mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16   (f16 accumulators)
 *   mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32   (f32 accumulators)  <<< we use this
 *
 *   * ".row.col" is the ONLY matrix layout m16n8k16 supports (there is no
 *     .row.row form for f16 operands on any architecture).
 *   * The old spelling "...f16.f16.f32.f32" is INVALID (it would mean
 *     d=f16, a=f16, b=f32, c=f32) and is rejected by ptxas with
 *     "Unexpected instruction types specified for 'mma'".
 *   * Requires sm_80+ (sm_86 = A10G target is fine).
 *
 * Per-thread fragment sizes (PTX ISA 9.4, Fig. 82/84/86):
 *
 *   Variant            | A regs  | B regs  | C/D regs | A elems | B elems | C elems
 *   -------------------|---------|---------|----------|---------|---------|--------
 *   f32.f16.f16.f32    | 4 x b32 | 2 x b32 | 4 x f32  | 8 fp16  | 4 fp16  | 4 fp32
 *   f16.f16.f16.f16    | 4 x b32 | 2 x b32 | 2 x b32  | 8 fp16  | 4 fp16  | 4 fp16
 *
 * Fragment element layout (per thread t = 0..31), g = t/4, c = 2*(t%4):
 *
 *   A (16x16, row-major fragment) — registers in (T00, T10, T01, T11) order:
 *     ra0 = {A[g  ][c  ], A[g  ][c+1]}   rows  0-7, cols  0-7
 *     ra1 = {A[g+8][c  ], A[g+8][c+1]}   rows  8-15, cols 0-7
 *     ra2 = {A[g  ][c+8], A[g  ][c+9]}   rows  0-7, cols 8-15
 *     ra3 = {A[g+8][c+8], A[g+8][c+9]}   rows  8-15, cols 8-15
 *
 *   B (16x8 = K x N, "col-major" fragment) — *** K-PAIRS, n = groupID ***
 *     rb0 = {B[c  ][g], B[c+1  ][g]}     k 0-7,   n = g
 *     rb1 = {B[c+8][g], B[c+9][g]}       k 8-15,  n = g
 *   Each register packs two CONSECUTIVE K values for ONE column n = t/4.
 *
 *   C/D (16x8, row-major fragment), f32 variant:
 *     c0 = C[g  ][c  ], c1 = C[g  ][c+1], c2 = C[g+8][c  ], c3 = C[g+8][c+1]
 *
 * B storage correspondence used by our kernels:
 *   The GEMM is Y[M,N] = X[M,K] @ W[N,K]^T, i.e. B[k][n] = W[n][k]. With W
 *   staged in shared memory as sW[n_local][k_local] (row-major, K contiguous):
 *     B[k][n] = sW[n0+n][k0+k], and the required fragment for thread t is
 *       rb0 = {sW[n0+g][k0+c], sW[n0+g][k0+c+1]}   <- SAME (row=g, col=c)
 *       rb1 = {sW[n0+g][k0+c+8], sW[n0+g][k0+c+9]}     pattern as the A
 *   fragment — 4 elements instead of 8. Equivalently: a NON-transposed
 *   ldmatrix of the 8x8 tile sW[n0..n0+7][k0..k0+7] hands out exactly the
 *   required rb0 pairs (ldmatrix thread t receives row t/4, cols 2(t%4)..+1
 *   of the loaded tile). ldmatrix.trans must NOT be used for this layout.
 *
 * ldmatrix (PTX ISA 9.4, 9.7.16.5.15): threads 0-7 provide the row
 * addresses of matrix 0 (-> r0), threads 8-15 matrix 1 (-> r1), threads
 * 16-23 matrix 2 (-> r2), threads 24-31 matrix 3 (-> r3). Within each 8x8
 * matrix, thread t receives elements {M[t/4][2(t%4)], M[t/4][2(t%4)+1]}.
 * Addresses must be 16-byte aligned and point at 8 contiguous halves;
 * rows may have arbitrary stride ("Consecutive instances of row need not
 * be stored contiguously in memory") — 16B-block XOR swizzles are legal.
 */

#pragma once

#include <cuda_fp16.h>
#include <cstdint>

namespace flute {

// ---------------------------------------------------------------------------
// MMA wrappers
// ---------------------------------------------------------------------------

// ---- f32 accumulator variant (production path) -----------------------------
// a: 4 x b32 (packed f16x2 fragments, e.g. straight from ldmatrix.x4)
// b: 2 x b32 (packed f16x2 fragments)
// c: 4 x f32 in/out accumulator
__device__ __forceinline__ void mma_m16n8k16_f32acc(const uint32_t (&a)[4],
    const uint32_t (&b)[2],
          float    (&c)[4]
) {
    float c0 = c[0], c1 = c[1], c2 = c[2], c3 = c[3];
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
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

// ---- f16 accumulator variant (kept for completeness/API parity) -------------
// c: 4 x f16 in/out (2 packed b32)
__device__ __forceinline__ void mma_m16n8k16_f16acc(const __half a[8],
    const __half b[4],
          __half c[4]
) {
    const uint32_t a0 = reinterpret_cast<const uint32_t&>(a[0]);
    const uint32_t a1 = reinterpret_cast<const uint32_t&>(a[2]);
    const uint32_t a2 = reinterpret_cast<const uint32_t&>(a[4]);
    const uint32_t a3 = reinterpret_cast<const uint32_t&>(a[6]);
    const uint32_t b0 = reinterpret_cast<const uint32_t&>(b[0]);
    const uint32_t b1 = reinterpret_cast<const uint32_t&>(b[2]);

    uint32_t c0 = reinterpret_cast<const uint32_t&>(c[0]);  // {c[0], c[1]}
    uint32_t c1 = reinterpret_cast<const uint32_t&>(c[2]);  // {c[2], c[3]}

    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 "
        "{%0, %1}, "
        "{%2, %3, %4, %5}, "
        "{%6, %7}, "
        "{%8, %9};\n"
        : "=r"(c0), "=r"(c1)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
          "r"(b0), "r"(b1),
          "r"(c0), "r"(c1)
    );

    reinterpret_cast<uint32_t&>(c[0]) = c0;
    reinterpret_cast<uint32_t&>(c[2]) = c1;
}

// ---- legacy __half-array f32-accumulator wrapper ---------------------------
// Same instruction as mma_m16n8k16_f32acc above; accepts __half arrays.
// The accumulator binds as a reference to array (float (&c)[4]) — a plain
// pointer cannot alias the fixed-size register quadruple.
// B fragment convention (PTX-correct): b[0..3] must be
// {B[c][g], B[c+1][g], B[c+8][g], B[c+9][g]} with g = lane/4, c = 2*(lane%4),
// i.e. B[k][n] = sW[n_base+n][k_base+k] gives
//   b[0] = sW[n_base + lane/4][k_base + 2*(lane%4)    ]
//   b[1] = sW[n_base + lane/4][k_base + 2*(lane%4) + 1]
//   b[2] = sW[n_base + lane/4][k_base + 2*(lane%4) + 8]
//   b[3] = sW[n_base + lane/4][k_base + 2*(lane%4) + 9]
__device__ __forceinline__ void mma_m16n8k16_f32acc_half(const __half a[8],
    const __half b[4],
          float  (&c)[4]
) {
    uint32_t ar[4], br[2];
    ar[0] = reinterpret_cast<const uint32_t&>(a[0]);
    ar[1] = reinterpret_cast<const uint32_t&>(a[2]);
    ar[2] = reinterpret_cast<const uint32_t&>(a[4]);
    ar[3] = reinterpret_cast<const uint32_t&>(a[6]);
    br[0] = reinterpret_cast<const uint32_t&>(b[0]);
    br[1] = reinterpret_cast<const uint32_t&>(b[2]);
    mma_m16n8k16_f32acc(ar, br, c);
}

// ---------------------------------------------------------------------------
// ldmatrix primitives
// ---------------------------------------------------------------------------

// ldmatrix.x4: load 4 x (8x8 f16) tiles -> 4 b32 per thread.
// `smem` must be this thread's row address (16B aligned, 8 contiguous halves).
__device__ __forceinline__ void ldmatrix_x4(uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3,
    const __half* smem
) {
    uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(smem));
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 "
        "{%0, %1, %2, %3}, [%4];\n"
        : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
        : "r"(addr)
    );
}

// ldmatrix.x2 (non-trans): 2 tiles -> 2 b32 per thread.
__device__ __forceinline__ void ldmatrix_x2(uint32_t& r0, uint32_t& r1,
    const __half* smem
) {
    uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(smem));
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x2.shared.b16 "
        "{%0, %1}, [%2];\n"
        : "=r"(r0), "=r"(r1)
        : "r"(addr)
    );
}

// ldmatrix.x2.trans: 2 tiles loaded COLUMN-major (transposed distribution).
// NOT used for the B operand of this project (sW is [N,K] row-major, whose
// non-trans distribution already matches the B fragment). Kept as a utility.
__device__ __forceinline__ void ldmatrix_x2_trans(uint32_t& r0, uint32_t& r1,
    const __half* smem
) {
    uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(smem));
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 "
        "{%0, %1}, [%2];\n"
        : "=r"(r0), "=r"(r1)
        : "r"(addr)
    );
}

// ---------------------------------------------------------------------------
// ldmatrix address computation (swizzle-aware)
// ---------------------------------------------------------------------------
// Shared tiles use a 16-byte-block XOR swizzle: the 16B block index of row
// `r` is  block ^ (r & (xor_mask - 1)).  xor_mask = 1 disables the swizzle
// (plain padded layout). All returned addresses are 16B aligned and start 8
// contiguous halves — the ldmatrix contract.

// 16B-block swizzle for a block-aligned logical column (multiple of 8).
__device__ __forceinline__ int swz_col16(int row, int col, int xor_mask) {
    return ((col >> 3) ^ (row & (xor_mask - 1))) << 3;
}

// Word (uint32 = 2 halves) index within a row of `ldw` words, row `r`,
// logical word `w` (covers logical columns [2w, 2w+1)).
__device__ __forceinline__ int swz_word(int row, int w, int xor_mask) {
    return (((w >> 2) ^ (row & (xor_mask - 1))) << 2) | (w & 3);
}

// Address for ldmatrix.x4 loading the A-fragment 16x16 tile at logical
// (row_base, col_base) of a tile with row stride `ld` halves.
// Feeds the four 8x8 matrices in (T00, T10, T01, T11) order so that
// r0..r3 land exactly in mma A-fragment registers a0..a3:
//   threads  0-7 -> rows row_base+0..7  @ col_base      (matrix 0 -> r0)
//   threads  8-15 -> rows row_base+8..15 @ col_base      (matrix 1 -> r1)
//   threads 16-23 -> rows row_base+0..7  @ col_base+8    (matrix 2 -> r2)
//   threads 24-31 -> rows row_base+8..15 @ col_base+8    (matrix 3 -> r3)
__device__ __forceinline__ const __half* ldmatrix_a_addr(const __half* tile, int ld, int row_base, int col_base,
    int lane, int xor_mask
) {
    const int r = row_base + ((lane >> 3) & 1) * 8 + (lane & 7);
    const int c = col_base + (lane >> 4) * 8;
    return tile + (size_t)r * ld + swz_col16(r, c, xor_mask);
}

// Address for ldmatrix.x4 (NON-trans) loading B fragments of TWO adjacent
// 8-wide n-tiles whose top-left logical element is (row_base=n_base,
// col_base=k_base): r0,r1 = b-regs of tile n_base; r2,r3 = tile n_base+8.
//   threads  0-7 -> rows row_base+0..7 @ col_base      (matrix 0 -> r0 = b0)
//   threads  8-15 -> rows row_base+0..7 @ col_base+8    (matrix 1 -> r1 = b1)
//   threads 16-23 -> rows row_base+8..15 @ col_base     (matrix 2 -> r2 = b0')
//   threads 24-31 -> rows row_base+8..15 @ col_base+8   (matrix 3 -> r3 = b1')
// Because B fragments pack K-pairs and sW is K-contiguous per row, the
// NON-trans distribution is exactly the mma B-fragment layout.
__device__ __forceinline__ const __half* ldmatrix_b_addr(const __half* tile, int ld, int row_base, int col_base,
    int lane, int xor_mask
) {
    const int r = row_base + (lane >> 4) * 8 + (lane & 7);
    const int c = col_base + ((lane >> 3) & 1) * 8;
    return tile + (size_t)r * ld + swz_col16(r, c, xor_mask);
}

// ---------------------------------------------------------------------------
// cp.async — asynchronous global->shared 16 B copies (PTX ISA 9.4 9.7.8.4)
// ---------------------------------------------------------------------------
// Requires sm_80+ (the same floor the mma instructions already impose).
// .cg = cache-global: allocate in L2, do NOT pollute L1 — the A tile streams
// through once per block while cross-block reuse is served from L2.
// 16 B is the ONLY cp-size .cg supports, and both addresses must be 16 B
// aligned (guaranteed by the tile layouts: row strides are multiples of 8
// halves and the 16 B-block swizzle preserves alignment).
//
// Completion model (PTX ISA 9.7.8.4.2): the issuing thread tracks copies in
// per-thread groups; cp.async.commit_group closes a group and
// cp.async.wait_group N blocks until at most N groups are outstanding.
// Visibility to OTHER threads of the CTA is ordered by __syncthreads()
// AFTER the wait — the CUTLASS two-stage pattern.

// dst: 16 B-aligned SHARED address (generic pointer is fine for the
//      compiler's address-space inference; we pass the cvta'd u32).
// src: 16 B-aligned GLOBAL address.
__device__ __forceinline__ void cp_async_16(void* dst_smem, const void* src_gmem) {
    const uint32_t dst = static_cast<uint32_t>(__cvta_generic_to_shared(dst_smem));
    asm volatile(
        "cp.async.cg.shared.global [%0], [%1], 16;\n"
        ::
        "r"(dst),
        "l"(src_gmem)
    );
}

__device__ __forceinline__ void cp_async_commit() {
    asm volatile("cp.async.commit_group;\n");
}

// wait_group 0: block until EVERY cp.async group this thread issued (and
// has not already been waited for) has landed in shared memory.
__device__ __forceinline__ void cp_async_wait_all() {
    asm volatile("cp.async.wait_group 0;\n");
}

// wait_group 1: block until at most ONE cp.async group is outstanding —
// the steady-state wait of a 3-stage pipeline (issue two ahead, consume
// one). PTX ISA 9.7.8.4.2.
__device__ __forceinline__ void cp_async_wait_1() {
    asm volatile("cp.async.wait_group 1;\n");
}

// ---------------------------------------------------------------------------
// ld.global.nc — non-coherent load through texture cache
// ---------------------------------------------------------------------------
// .nc routes through the read-only / texture path (requires data not written
// by this kernel — true for Q). On Hopper (SM_90+), an L2::evict_first hint
// can be added to mark the line for early eviction, but this is ILLEGAL on
// Ampere (SM_80-89), so we use the plain .nc load for compatibility.
__device__ __forceinline__ void ldg_nc_evict_first_v4(uint4& r, const void* gmem
) {
    // Plain nc load — works on all architectures SM_70+
    asm volatile(
        "ld.global.nc.v4.u32 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w)
        : "l"(gmem)
    );
}

// Scalar u32 variant for the sub-4-bit idxN Q prefetches, whose K-tile
// halves are 4 B- but not 16 B-aligned (e.g. b=1 BK=32: 8 B halves,
// b=3 BK=32: 24 B halves). Same .nc path, same compatibility note.
__device__ __forceinline__ void ldg_nc_evict_first_u32(uint32_t& r, const void* gmem
) {
    asm volatile(
        "ld.global.nc.u32 %0, [%1];\n"
        : "=r"(r)
        : "l"(gmem)
    );
}

// ---------------------------------------------------------------------------
// prmt — byte permute (PTX ISA 9.4 9.7.9.4)
// ---------------------------------------------------------------------------
// prmt.b32 d, a, b, c treats {a, b} as 8 source bytes (a = bytes 0..3,
// b = bytes 4..7); each of the four selector nibbles of c picks one source
// byte (default mode, replicate/sign bit unused).

// Merge the LOW 16 bits of `lo` with the LOW 16 bits of `hi` into
// { hi[15:0], lo[15:0] }  ==  lo | (hi << 16):
//   out.b0 <- a.b0 (selector 0), out.b1 <- a.b1 (selector 1),
//   out.b2 <- b.b0 (selector 4), out.b3 <- b.b1 (selector 5)
//   => c = 0x5410
// One prmt replaces the SHF+OR pair in the dequant merge (XU pipe, not LSU).
__device__ __forceinline__ uint32_t prmt_merge_lo_hi(uint32_t lo, uint32_t hi) {
    uint32_t d;
    asm(
        "prmt.b32 %0, %1, %2, %3;\n"
        : "=r"(d)
        : "r"(lo), "r"(hi), "n"(0x5410)
    );
    return d;
}

}  // namespace flute
