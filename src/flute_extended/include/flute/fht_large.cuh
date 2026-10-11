/**
 * include/flute/fht_large.cuh
 *
 * FLUTE Extension -- the LARGE-K Fast Hadamard Transform: the math
 * contract header (W20.5.1). This file is a CONTRACT, not an
 * implementation: the constants, the stage-split arithmetic, the sign
 * conventions, and the ownership / pairing disciplines -- prose plus
 * host-inline functions only. The kernels are authored by the W20.5.2
 * unit (a sibling src/ file); this header deliberately contains no
 * kernel bodies and no device code.
 *
 * WHY A SECOND HEADER -- the domain split. The small-K path
 * (include/flute/fht.cuh + src/kernel_fht.cu) stages a whole segment in
 * one block's shared memory and decomposes K into at most 8 descending
 * power-of-two blocks (fht.cuh:184-187; kernel_fht.cu:21-24 caps K at
 * 2^16 - 32). Once a segment exceeds one 64 KiB tile the butterfly must
 * cross block boundaries, and fht_large owns exactly that regime:
 *
 *     K > 16384     hard: a row no longer fits one 16384-fp32 tile
 *                  (64 KiB); smaller K stays on the fht.cuh path
 *     K % 32 == 0   hard: every trailing block keeps >= 32 elements
 *                  (kernel_fht.cu:21)
 *
 * Target shapes: the powers of two 2^15 .. 2^22 (2 tiles per row at
 * 2^15, 32 at 2^19, 256 at 2^22).
 *
 * THE STAGE SPLIT (1) --------------------------------------------------
 *
 * The butterfly is the fht.cuh:29-34 stage loop: stages are indexed by
 * the pair distance len_ = 1, 2, 4, ..., b/2 (stage sh carries
 * len_ = 2^sh, sh = 0 .. log2(b)-1); stage len_ visits every pair
 * (i, i + len_) with i = 2*g*len_ + o, o < len_ -- each pair lives in
 * a group of width 2*len_.
 *
 * fht_large stages a row in tiles of FHT_LARGE_TILE = 16384 fp32
 * (65536 bytes = 64 KiB) and splits the stage list at HALF-TILE
 * granularity:
 *
 *   TILE-LOCAL: 2*len_ <= 8192 (len_ <= 4096). The pair window fits one
 *               8192-element half of a tile: both pair elements are in
 *               the same half, the same tile, the same CTA. Stages
 *               sh = 0..12 -- exactly FHT_LARGE_LOCAL_STAGES = 13.
 *   CROSS-TILE: 2*len_ >= 16384 (len_ >= 8192). The pair window spans a
 *               full tile or more: the remaining log2(K) - 13 stages,
 *               len_ = 8192 .. K/2. (len_ is a power of two, so no
 *               stage falls strictly between the two classes -- the
 *               partition is exhaustive.)
 *
 * For any K >= 2^14 the split is exactly 13 local stages -- a constant
 * of the tile, not of K. Checks: K = 2^15 -> 13 local + 2 cross = 15
 * stages; K = 2^19 -> 13 + 6 = 19; K = 2^22 -> 13 + 9 = 22.
 *
 * Boundary note -- the degenerate first cross stage: at len_ = 8192 the
 * window 2*len_ equals the FULL tile, so the pair (i, i + 8192) still
 * lies inside ONE tile (it joins the tile's two 8192-element halves);
 * every cross stage with len_ >= 16384 puts the two elements in
 * DIFFERENT tiles. This contract classifies len_ = 8192 as cross
 * because the split is drawn at the half-tile line (a full-tile
 * criterion, 2*len_ <= 16384, would instead give 14 local stages and
 * cross = log2(K) - 14). W20.5.2 may still LOWER the len_ = 8192 stage
 * to an intra-tile implementation (no partner CTA); the accounting
 * constant stays 13.
 *
 * THE SIGN CONVENTIONS (3) ---------------------------------------------
 *
 * T is COLUMN-scaled (fht.cuh:10-16): T = blockdiag_b( H_b diag(s_b) /
 * sqrt(b) ) with T[i,j] = H[i,j] * s[j] / sqrt(b) -- exactly
 * build_rotation_matrix's `hadamard(b) * s.view(1, b) / sqrt(b)`
 * (fht.py:166). The sign vector s is (K,) fp32 with entries +-1
 * (rotation_signs, fht.py:105; the kernel-side signature is
 * kernel_fht.cu:20-21); s[j] multiplies COLUMN j of T.
 *
 *   FORWARD   x @ T   = ((x @ H) * s) / sqrt(b)  -- signs AFTER the
 *              butterfly, on the output columns (fht.py:23; the
 *              reference is fht_reference, fht.py:206-222: butterfly at
 *              line 219, then out = (seg * s) * inv with
 *              inv = 1/sqrt(b) at lines 220-221).
 *   ADJOINT   y @ T^T = ((y * s) @ H) / sqrt(b)  -- signs BEFORE the
 *              butterfly, on the input columns (fht.py:33; the
 *              reference is fht_reference_adjoint, fht.py:225-238: the
 *              input is multiplied by s at line 236, the butterfly runs
 *              on (g * s), the output takes * 1/sqrt(b) at line 237).
 *
 * The butterfly itself is UNSIGNED -- the same H both directions
 * (fht.cuh:25-26): forward and adjoint differ ONLY in which epilogue
 * carries s. In the multi-tile regime the epilogues sit at the row's
 * first load and final store ONLY (the kernel_fht.cu:28-29 discipline:
 * "the sign multiply and the 1/sqrt(b) scaling are folded into the
 * load/store epilogues"): intermediate cross-stage global round-trips
 * are unscaled and unsigned, and s[j] / 1/sqrt(b) are applied exactly
 * once per element, at the same place fht_reference applies them --
 * forward out[j] = (bfly[j] * s[j]) * (1/sqrt(b)) (s first: the +-1
 * multiply is EXACT, a sign flip; the scale is the epilogue's one fp32
 * rounding), adjoint in[j] = v[j] * s[j] at the load and
 * out[j] = bfly[j] * (1/sqrt(b)) at the store. b is the SEGMENT size
 * (segments(), fht.py:118-134: descending powers of two); for the
 * power-of-two targets b = K.
 *
 * THE OWNERSHIP DISCIPLINE (4) -- the determinism contracts ----------
 *
 * CONTRACT: no atomics anywhere -- deterministic ownership (run-to-run
 * torch.equal). No atomicAdd / atomicExch / atomicCAS and no global
 * flag spin-waits in any fht_large kernel: every element write has
 * exactly one owner, fixed by construction, so the accumulation and
 * rounding order is a pure function of the launch geometry -- two runs
 * on the same inputs are bit-identical (torch.equal). Cross-stage
 * ordering is carried by kernel-launch boundaries driven from the
 * host, never by atomics.
 *
 * CONTRACT: fixed thread->pair map. Within a stage every pair
 * (i, i + len_) has exactly ONE owning thread, and pair -> thread is a
 * fixed function of (stage, threadIdx). The reference map is
 * fht.cuh:90's `for (int p = threadIdx.x; p < pairs; p += THREADS)`;
 * at FHT_LARGE_THREADS = 1024 a tile-local stage owns TILE/2 = 8192
 * pairs per tile -- 8 per thread.
 *
 * CONTRACT: fixed tile/warp/lane ownership. CTA (row r, tile t) owns
 * row r's tile t (grid = M * K/FHT_LARGE_TILE CTAs, one 64 KiB tile
 * per CTA); the lane -> element map is a fixed function of
 * (stage, warp, lane, tile), identical on every run.
 *
 * CONTRACT: unique writers by construction. The u+w / u-w pair writes
 * go to the same indices the reads came from: the owning thread reads
 * (i, i + len_) and is the ONLY writer of both i and i + len_ at that
 * stage (at a two-CTA cross stage, each of the two indices has exactly
 * one designated writer CTA).
 *
 * THE PAIRING DISCIPLINE (5) -- the rounding-order contract ---------
 *
 * CONTRACT: pairing -- low = u + w, high = u - w, ascending index
 * order. For each pair (u, w) = (v[i], v[i + len_]) with u at the
 * LOWER index: the lower index receives u + w, the higher receives
 * u - w -- exactly the fht.cuh:33-34 butterfly
 * `(v[2*g*len + o], v[2*g*len + o + len]) -> (a + b, a - b)` and
 * fht.cuh:94-95 `s[i] = a + bv; s[i + len] = a - bv`, accumulated in
 * fp32 shared memory regardless of the io dtype (kernel_fht.cu:26-28).
 * Each pair is exactly one fp32 add and one fp32 subtract, u on the
 * left -- no reassociation, no multi-term accumulation, no fma
 * contraction feeding the writes: the per-element rounding is fully
 * determined. Stages run in ASCENDING len_ order (fht.cuh:29); within
 * a stage the pairs are disjoint (each element belongs to exactly one
 * pair), so intra-stage visit order cannot change any value -- the
 * fixed stage order plus the fixed pair arithmetic is what makes the
 * output run-to-run bit-stable and differentially equal to
 * fht_reference.
 *
 * RESOURCES (6) -- the sm_86 accounting, kernel_fht.cu:33-47 discipline
 *
 *   FHT_LARGE_TILE         16384 fp32 = 65536 bytes (64 KiB) dynamic
 *                          smem per CTA -- ABOVE the 48 KiB default, so
 *                          every launcher opts in via
 *                          cudaFuncSetAttribute, checked and loud on
 *                          failure (kernel_fht.cu:38-39).
 *   FHT_LARGE_THREADS      1024 -- the pinned CTA width, the top of the
 *                          {128..1024} range (kernel_fht.cu:35); 16
 *                          elements per thread; every kernel
 *                          instantiation is __launch_bounds__(1024)
 *                          (kernel_fht.cu:40). At 1024 threads the CTA
 *                          is thread-limited to 1 block/SM
 *                          (kernel_fht.cu:42-43).
 *   FHT_LARGE_LOCAL_STAGES 13 (the stage split above).
 *   FHT_LARGE_SMEM_BYTES   4 * 16384 = 65536 (one fp32 tile).
 */

#pragma once  // house style: fht.cuh

#ifndef FLUTE_FHT_LARGE_CUH  // belt-and-braces alongside #pragma once
#define FLUTE_FHT_LARGE_CUH

#include <cstdint>  // the only include: <cstdint>-grade header

// Host-inline linkage: __host__ __forceinline__ under nvcc (the fht.cuh
// house style -- fht_segments, fht.cuh:195), plain inline in non-CUDA
// translation units (this header has no device code, so host-side .cpp
// files may include it too).
#ifdef __CUDACC__
#define FLUTE_FHT_LARGE_HD __host__ __forceinline__
#else
#define FLUTE_FHT_LARGE_HD inline
#endif

namespace flute {

// ---------------------------------------------------------------------------
// Constants -- the W20.5.2 kernels and the launcher bind to these; no
// magic numbers downstream. Each is part of the contract above.
// ---------------------------------------------------------------------------

// The tile: 16384 fp32 elements staged in dynamic shared memory per
// CTA -- 16384 * 4 = 65536 bytes = 64 KiB (past the 48 KiB default; the
// launcher's cudaFuncSetAttribute opt-in is part of the contract). A
// row of K columns is K / FHT_LARGE_TILE tiles (2 at 2^15, 32 at 2^19,
// 256 at 2^22).
constexpr int FHT_LARGE_TILE = 16384;

// Tile-local stage count: the stages whose pair window 2*len_ fits one
// 8192-element half of the tile (len_ = 1..4096, stages sh = 0..12).
// See THE STAGE SPLIT for the len_ = 8192 boundary note.
constexpr int FHT_LARGE_LOCAL_STAGES = 13;

// The pinned CTA width: 1024 threads -- 16 elements per thread per
// tile, 8 pairs per thread per tile-local stage. Every kernel
// instantiation carries __launch_bounds__(FHT_LARGE_THREADS)
// (kernel_fht.cu:40); occupancy is thread-limited to 1 CTA/SM
// (kernel_fht.cu:42-43).
constexpr int FHT_LARGE_THREADS = 1024;

// The dynamic shared-memory request per CTA: 4 * 16384 = 65536 bytes
// (64 KiB) -- one FHT_LARGE_TILE of fp32.
constexpr int FHT_LARGE_SMEM_BYTES = 4 * 16384;

static_assert(FHT_LARGE_TILE == 16384, "16384 fp32 = 64 KiB -- the tile");
static_assert(FHT_LARGE_SMEM_BYTES == 4 * FHT_LARGE_TILE, "one fp32 tile");
static_assert(FHT_LARGE_THREADS == 1024, "the pinned CTA width");
static_assert(FHT_LARGE_TILE % FHT_LARGE_THREADS == 0,
              "16 elements per thread -- the ownership map divides");
static_assert((1 << FHT_LARGE_LOCAL_STAGES) * 2 == FHT_LARGE_TILE,
              "the first cross len_ = 8192 spans the full tile window");

// ---------------------------------------------------------------------------
// fht_large_stages(K) -- the stage-split arithmetic, host-side (2).
//
// Returns {local, cross} for the row's head power-of-two segment plus a
// validation status. This header never throws: the C++-side refusal is
// the status enum below, and the launcher turns a non-Ok status into a
// loud TORCH_CHECK (the kernel_fht.cu house discipline).
//
//   FhtLargeStatus::Ok              K > 16384 and K % 32 == 0
//   FhtLargeStatus::KTooSmall       K <= 16384 -- one tile or less; the
//                                   small-K fht.cuh path owns it
//   FhtLargeStatus::KNotMultiple32  violates K % 32 == 0
//                                   (kernel_fht.cu:21)
//
// For status Ok: local == FHT_LARGE_LOCAL_STAGES (13) and
// cross == floor(log2(K)) - 13. Exact row totals when K is a power of
// two (the 2^15..2^22 targets; larger valid K still computes, but is
// outside the tested envelope):
//
//     K = 2^15 = 32768      local 13   cross 2    tiles/row   2
//     K = 2^19 = 524288     local 13   cross 6    tiles/row  32
//     K = 2^22 = 4194304    local 13   cross 9    tiles/row 256
//
// For non-power-of-two valid K the counts describe the head segment
// 2^floor(log2(K)); trailing segments repeat the same shape at their
// own scale (a 2^14 segment is a single tile: 13 + 1).
// ---------------------------------------------------------------------------

enum class FhtLargeStatus : int32_t {
    Ok = 0,             // the split above applies
    KTooSmall = 1,      // K <= 16384 -- refused (the fht.cuh path owns it)
    KNotMultiple32 = 2, // K % 32 != 0 -- refused
};

struct FhtLargeStages {
    FhtLargeStatus status; // Ok, or the refusal (the counts below are 0)
    int32_t local;         // tile-local stages (13 when Ok)
    int32_t cross;         // cross-tile stages (log2_floor(K) - 13 when Ok)
};

// floor(log2(K)) for K >= 1. A shift loop, deliberately: the header
// stays <cstdint>-grade (no <cmath>, no compiler builtins).
FLUTE_FHT_LARGE_HD int fht_large_log2_floor(int K) {
    int L = 0;
    while (L < 30 && (1 << (L + 1)) <= K) ++L;
    return L;
}

FLUTE_FHT_LARGE_HD FhtLargeStages fht_large_stages(int K) {
    FhtLargeStages r;
    r.local = 0;
    r.cross = 0;
    if (K <= FHT_LARGE_TILE) { // K <= 16384: one tile or less
        r.status = FhtLargeStatus::KTooSmall;
        return r;
    }
    if (K % 32 != 0) { // the kernel_fht.cu:21 discipline
        r.status = FhtLargeStatus::KNotMultiple32;
        return r;
    }
    r.status = FhtLargeStatus::Ok;
    r.local = FHT_LARGE_LOCAL_STAGES; // 13
    r.cross = fht_large_log2_floor(K) // log2(K) - 13
            - FHT_LARGE_LOCAL_STAGES; // (cross = 6 at K = 2^19, 9 at 2^22)
    return r;
}

}  // namespace flute

#endif  // FLUTE_FHT_LARGE_CUH
