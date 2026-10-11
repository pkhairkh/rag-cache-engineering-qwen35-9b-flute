/**
 * src/kernel_tq.cu
 *
 * FLUTE Extension -- the TQ kernel family's home. Born in W20.5.2 with
 * fht_forward_large, the LARGE-K Fast Hadamard Transform: the
 * multi-block butterfly for rows that no longer fit one CTA's shared
 * memory (the production targets K = 2^19..2^22; every power of two
 * 2^15..2^22 qualifies). The family grows in place: fht_backward_large
 * (the adjoint -- W20.5.3) and the batched quantize/dequant kernels
 * tq_quant / tq_dequant (W20.7 / W20.8) join this TU later. The small-K
 * path (kernel_fht.cu, K <= 2^16-32) stays byte-identical and keeps
 * everything it accepts.
 *
 * THE LAW (PROMPT.md §1): this file is authored STATICALLY on the CPU
 * dev box -- no test run, no test authored, no repo module imported,
 * and no compiler (nvcc is ABSENT here, env/W20_ENV.md). Compilation,
 * the differential suite, and the determinism gates all execute on the
 * A10G box (W20.5.4 registers the TU; W20.6 runs the tests). Every
 * defect left here is a box-side discovery -- the file is written to
 * be re-read, not trusted.
 *
 * THE CONTRACT: include/flute/fht_large.cuh (W20.5.1) -- the stage
 * split, the sign conventions, the ownership and pairing disciplines.
 * This TU is that contract's first implementation, in two passes:
 *
 *   PASS 1 (tile-local): grid = M rows x K/16384 tiles. CTA (row,
 *   tile) stages its 16384-fp32 (64 KiB) tile into dynamic shared
 *   memory -- cp.async 16 B raw copies on the fp32 path (the
 *   mma.cuh:277-302 idiom; bit-identical to a converting loop, it
 *   moves bytes), kernel_fht.cu:85-88's converting loop on fp16/bf16
 *   (cp.async cannot cast) -- then runs the butterfly with
 *   kernel_fht.cu's EXACT primitive: flute::fht_block<1024>(smem,
 *   16384). Same shift/mask stage loop, same pairing
 *       low  = u + w    at the LOWER index i
 *       high = u - w    at the HIGHER index i + len_
 *   in ascending stage order (fht.cuh:84-97, kernel_fht.cu:90),
 *   accumulated in fp32 shared memory regardless of the I/O dtype
 *   (kernel_fht.cu:26-28). fht_block runs sh = 0..13: the contract's
 *   13 tile-local stages (len_ = 1..4096) PLUS the boundary stage
 *   len_ = 8192, LOWERED to an intra-tile execution -- the header's
 *   Boundary note (fht_large.cuh:53-62) explicitly permits this: the
 *   pair window 2*len_ equals the full tile, so (i, i+8192) joins the
 *   tile's two halves inside one CTA, needing no partner CTA and no
 *   global round-trip. The accounting constant stays 13; the header's
 *   cross count (log2 K - 13: 6 at 2^19, 9 at 2^22) still counts
 *   len_ = 8192, so the GLOBAL cross launches number log2 K - 14
 *   (5 at 2^19, 8 at 2^22 -- one fewer launch, identical arithmetic).
 *
 *   PASS 2 (cross stages, len_ = 16384..K/2): an elementwise global
 *   pair op over the fp32 butterfly state, ONE LAUNCH PER STAGE --
 *   kernel-launch boundaries are the ONLY cross-CTA synchronization
 *   (the contract: "cross-stage ordering is carried by kernel-launch
 *   boundaries driven from the host", fht_large.cuh:106-108). Within a
 *   launch, pair p -> thread is the fixed grid-stride map over the
 *   whole (M, K/2) pair space -- the fht.cuh:90 reference discipline
 *   the header cites (fht_large.cuh:110-115) -- with the same
 *   shift/mask pair index i = ((p >> sh) << (sh+1)) + (p & (len_-1))
 *   as fht_stage_sh. The owning thread reads (v[i], v[i+len_]) and is
 *   the ONLY writer of both i and i+len_ (pairs are disjoint): unique
 *   writers by construction, no cross-CTA races, no reduction
 *   primitives of any kind in this file (the ownership contract,
 *   fht_large.cuh:99-127).
 *
 *   EPILOGUE (forward): signs AFTER the butterfly, on the output
 *   columns (fht.py:206-222 -- fht_reference: butterfly at line 219,
 *   out = (seg * s) * inv with inv = 1/sqrt(b) at lines 220-221).
 *   Applied exactly once per element, in pass 2's FINAL stage
 *   (len_ = K/2 -- every element's last write), folded EXACTLY as
 *   kernel_fht.cu folds it: the store-side multiply chain
 *   t * signs[j] * rsqrtf(b) (kernel_fht.cu:83,93-95), b = K for the
 *   single-segment powers of two (fht_large.cuh:95-97). Intermediate
 *   cross-stage round-trips are unscaled and unsigned
 *   (fht_large.cuh:87-96). fp16/bf16 I/O keeps every intermediate in
 *   a distinct fp32 state tensor and casts ONCE at the final store --
 *   kernel_fht.cu's fp32-accumulation contract stretched across the
 *   passes (the multi-pass state-buffer discipline of
 *   kernel_gemv_splitk.cu:62-76's workspace, minus the reduction
 *   machinery this transform does not need).
 *
 * DETERMINISM CONTRACT: run-to-run torch.equal assertable. The launch
 * geometry is a pure function of (M, K, dtype) -- no device queries
 * feed it -- every element write has exactly one owner fixed by
 * construction, every pair is one fp32 add and one fp32 subtract with
 * u on the left (no reassociation, no multi-term accumulation), and
 * the stage order is ascending and fixed. Two runs on the same inputs
 * produce identical bits. Nothing in this file coordinates CTAs by
 * shared mutable flags or cross-launch reduction orders -- the only
 * inter-CTA ordering mechanism is the stream's kernel-launch
 * sequence.
 *
 * W20.6's negative controls (compile-time, both default OFF -- see
 * the blocks below):
 *   FLUTE_TQ_CANARY  poison-fill every output byte with NaN before
 *                    the real writes; zero poison must survive (the
 *                    write-coverage control).
 *   FLUTE_TQ_TRAP    flip the pair arithmetic on a timing bit; the
 *                    output becomes run-dependent (the determinism
 *                    control -- an assert that passes with the trap
 *                    armed is a broken assert).
 *
 * Resource accounting (sm_86 / A10G, the fht_large.cuh:147-162 list):
 *   * pass 1: 1024 threads (FHT_LARGE_THREADS, the pinned CTA width),
 *     65536 B dynamic shared memory per CTA (FHT_LARGE_SMEM_BYTES) --
 *     above the 48 KiB default, so every launch opts in via
 *     cudaFuncSetAttribute (checked, loud on failure --
 *     kernel_fht.cu:38-39's precedent). At 1024 threads + 64 KiB the
 *     CTA is 1-per-SM (thread-limited: 1536 threads/SM on sm_86; the
 *     64 KiB smem agrees). 16 elements per thread, 8 pairs per thread
 *     per tile-local stage.
 *   * pass 2: 1024 threads, no shared memory, grid = ceil(M*K/2 /
 *     1024) CTAs (>= 16 at the K = 2^15 floor), >= 1 pair per thread,
 *     grid-stride over the flattened pair space -- pure streaming.
 *   * every __global__ instantiation carries
 *     __launch_bounds__(FHT_LARGE_THREADS == 1024).
 *   * global traffic per row: pass 1 reads K and writes K (fp32
 *     state), each cross stage reads + writes K fp32 (the final stage
 *     writes the I/O dtype instead), so the transform costs
 *     (log2 K - 12) fp32 round-trips of the row -- the price of the
 *     multi-block regime, against O(K^2) for the explicit rotation
 *     matmul it replaces.
 *
 * Numerics scope: K must be a power of two > 16384 (the 2^15..2^22
 * envelope). Non-power-of-two K <= 2^16-32 stays on kernel_fht.cu's
 * segmented path; non-power-of-two K above that is REFUSED loudly
 * here (a single-segment butterfly would silently mis-transform it) --
 * see fht_large_check_args and the report's open question for 5.5.
 */

#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cstdint>
#include <type_traits>

#include "flute/fht.cuh"        // fht_block -- kernel_fht.cu's butterfly
#include "flute/fht_large.cuh"  // the W20.5.1 contract (constants, checks)

namespace {

// ---------------------------------------------------------------------------
// cp.async -- asynchronous global->shared 16 B copies. Identical
// instruction forms to flute/mma.cuh:277-302 (cp.async.cg, commit_group,
// wait_group 0 -- PTX ISA 9.4 9.7.8.4), kept LOCAL so the TQ family stays
// independent of the GEMM headers. Requires sm_80+ (the build's arch floor
// is 86). Completion model: wait_group 0 drains the issuing thread's
// groups; visibility to the CTA's OTHER threads is ordered by the
// __syncthreads() AFTER the wait (mma.cuh:268-272).
// ---------------------------------------------------------------------------

// dst: 16 B-aligned shared address; src: 16 B-aligned global address.
__device__ __forceinline__ void tq_cp_async_16(void* dst_smem, const void* src_gmem) {
    const uint32_t dst = static_cast<uint32_t>(__cvta_generic_to_shared(dst_smem));
    asm volatile(
        "cp.async.cg.shared.global [%0], [%1], 16;\n"
        ::
        "r"(dst),
        "l"(src_gmem)
    );
}

__device__ __forceinline__ void tq_cp_async_commit() {
    asm volatile("cp.async.commit_group;\n");
}

// wait_group 0: block until EVERY group this thread issued has landed.
__device__ __forceinline__ void tq_cp_async_wait_all() {
    asm volatile("cp.async.wait_group 0;\n");
}

// The first GLOBAL cross stage: len_ = 16384 (sh = 14) = the 13 local
// stages plus the boundary stage len_ = 8192 that pass 1 lowers
// intra-tile. This is the ONLY place the stage arithmetic is pinned --
// everything else derives from the contract header's constants.
constexpr int kFirstCrossSh = flute::FHT_LARGE_LOCAL_STAGES + 1;
static_assert((1 << kFirstCrossSh) == flute::FHT_LARGE_TILE,
              "the first global cross stage sits one full tile wide");
static_assert(flute::FHT_LARGE_SMEM_BYTES == 4 * flute::FHT_LARGE_TILE,
              "one fp32 tile of dynamic shared memory");

// ---------------------------------------------------------------------------
// Pass 1 -- the tile-local butterfly.
//
// CTA (row r, tile t) owns row r's tile t: stages the 16384-fp32 tile
// into 64 KiB dynamic shared memory, butterflies it with
// kernel_fht.cu's exact device primitive, and writes the result back
// to the fp32 state UNSIGNED and UNSCALED (the sign and 1/sqrt(K)
// epilogue lives in pass 2's final stage only -- fht_large.cuh:87-96:
// intermediate cross-stage round-trips carry neither).
//
// Staging: kernel_fht.cu:85-88's load-barrier-compute discipline
// (plain strided loop, one __syncthreads, compute, store). The fp32
// path issues the loads as cp.async 16 B raw copies (the mma.cuh
// idiom above -- same bytes, no registers spent); the fp16/bf16 path
// keeps the converting loop verbatim (cp.async cannot cast 2-byte
// input to the fp32 staging). There is no double buffer to manage:
// each CTA stages exactly ONE tile for its whole lifetime, so the
// pipeline degenerates to the single-tile form of kernel_fht.cu's
// segment loop (load -> barrier -> butterfly -> store).
//
// grid = (M rows, K/FHT_LARGE_TILE tiles): grid.x = rows (the 2^31-1
// axis; M is TORCH_CHECKed to fit), grid.y = tiles (the 65535 axis;
// tiles <= 256 at the K = 2^22 target, checked anyway).
// ---------------------------------------------------------------------------
template <typename scalar_t>
__global__ void __launch_bounds__(flute::FHT_LARGE_THREADS)
fht_large_local_kernel(const scalar_t* __restrict__ x,  // (M, K) input, read-only
                       float* __restrict__ ws,          // (M, K) fp32 butterfly state
                       int K) {
    // 16 B-aligned dynamic shared base (kernel_streaming.cu:865's
    // declaration idiom) -- the cp.async copies below require it.
    extern __shared__ __align__(16) unsigned char smem_raw[];
    float* smem = reinterpret_cast<float*>(smem_raw);

    const int row = blockIdx.x;   // grid.x = M
    const int tile = blockIdx.y;  // grid.y = K / FHT_LARGE_TILE
    const int64_t base = static_cast<int64_t>(row) * K
                       + static_cast<int64_t>(tile) * flute::FHT_LARGE_TILE;

    // ---- staging: x tile -> smem (fp32) --------------------------------
    if constexpr (std::is_same_v<scalar_t, float>) {
        // 4096 x 16 B chunks, 4 per thread, block-strided (the
        // fht.cuh:90 map shape). Both addresses are 16 B aligned:
        // base is a multiple of 4 floats (K and TILE are powers of two
        // >= 4) and chunk c lands on smem[4c] = smem + 16c bytes.
        for (int c = threadIdx.x; c < flute::FHT_LARGE_TILE / 4;
             c += flute::FHT_LARGE_THREADS) {
            tq_cp_async_16(smem + 4 * c, x + base + 4 * c);
        }
        tq_cp_async_commit();
        tq_cp_async_wait_all();
    } else {
        // kernel_fht.cu:85-87 verbatim: the converting fp32 stage load.
        for (int j = threadIdx.x; j < flute::FHT_LARGE_TILE;
             j += flute::FHT_LARGE_THREADS) {
            smem[j] = static_cast<float>(x[base + j]);
        }
    }
    __syncthreads();  // cp.async visibility / converting loads published

    // ---- the tile-local butterfly --------------------------------------
    // fht_block = kernel_fht.cu:90's call, with b = 16384: the same
    // shift/mask stage loop, same fixed pair->thread map, same
    // low = u+w / high = u-w pairing (fht.cuh:84-97), fp32 shared
    // memory regardless of I/O dtype. It runs sh = 0..13 -- the 13
    // local stages PLUS the boundary stage len_ = 8192 executed
    // INTRA-TILE (the header's Boundary note: 2*len_ = the full tile,
    // the pair (i, i+8192) joins this CTA's two 8192-halves; the
    // accounting constant stays FHT_LARGE_LOCAL_STAGES = 13).
    flute::fht_block<flute::FHT_LARGE_THREADS>(smem, flute::FHT_LARGE_TILE);

    // ---- write-back: fp32 state, unsigned and unscaled -----------------
    // (every cross-stage round-trip until the final epilogue carries
    // no sign and no scale -- fht_large.cuh:87-96)
    for (int j = threadIdx.x; j < flute::FHT_LARGE_TILE;
         j += flute::FHT_LARGE_THREADS) {
        ws[base + j] = smem[j];
    }
    // No trailing barrier: the CTA exits; the next kernel's reads of
    // ws are ordered by the launch boundary (the only cross-CTA
    // synchronization this family has).
}

// ---------------------------------------------------------------------------
// Pass 2 -- the cross stages (len_ = 16384..K/2), one launch per stage.
//
// Elementwise global pair op over the fp32 state. Pair p of the whole
// (M, K/2) space maps to one thread by the fixed grid-stride map
// (fht.cuh:90's `for (p = tid; p < pairs; p += THREADS)` discipline,
// grid-level); the pair index arithmetic is fht_stage_sh's
// (fht.cuh:90-95) verbatim, so the pair identity matches the local
// stages' exactly:
//
//     row = p >> row_shift          (row_shift = log2(K) - 1)
//     q   = p & (K/2 - 1)           (the row-local pair index)
//     i   = ((q >> sh) << (sh+1)) + (q & (len_ - 1))   -- the LOWER index
//     u = ws[i], w = ws[i + len_]   -- u at the LOWER index
//     low  = u + w   -> index i         (kernel_fht.cu:94 / fht.cuh:94)
//     high = u - w   -> index i + len_  (kernel_fht.cu:95 / fht.cuh:95)
//
// Each pair is ONE fp32 add and ONE fp32 subtract, u on the left -- no
// reassociation, no multi-term accumulation: the per-element rounding
// is fully determined (the pairing contract, fht_large.cuh:128-145).
// The owning thread is the ONLY writer of both i and i+len_ at this
// stage; no other thread loads or stores those addresses in this
// launch (pairs partition the row), and the next stage's launch waits
// for this one. No cross-CTA coordination exists by construction.
//
// In-place discipline: the non-final stages read and write ws at the
// same indices, per pair, reads-then-writes -- legal for the same
// reason fht_inplace_ is (kernel_fht.cu:68-69), with the sharper
// guarantee that the two same-address stores (i, i+len_) are each
// data-dependent on the same-thread same-address load, so no
// instruction reordering can break them. The FINAL stage (kFinal)
// reads the fp32 state and writes the OUTPUT tensor (the I/O dtype
// for fp16/bf16 -- a distinct buffer, cast once at the last write,
// exactly kernel_fht.cu:93-95's single-cast epilogue; the SAME buffer
// as the state for fp32 -- in-place, safe as above).
//
// The forward epilogue (kFinal only): signs AFTER the butterfly on
// the output columns, then 1/sqrt(K) -- the multiply chain
// t * signs[j] * inv_sqrt_b, folded exactly as kernel_fht.cu:83+94
// fold it (rsqrtf computed in the kernel; b = K, the segment size for
// the single-segment powers of two). The +-1 sign multiply is exact
// (a flip); inv_sqrt_b is the epilogue's one fp32 rounding
// (fht_large.cuh:92-96).
// ---------------------------------------------------------------------------
template <typename scalar_t, bool kFinal>
__global__ void __launch_bounds__(flute::FHT_LARGE_THREADS)
fht_large_cross_kernel(float* __restrict__ ws,           // (M, K) fp32 butterfly state
                       const float* __restrict__ signs,  // (K,) +-1 (kFinal only)
                       scalar_t* __restrict__ out,       // (M, K) output (kFinal only;
                                                             // == ws allowed for fp32)
                       int M, int K, int sh, int row_shift) {
    if constexpr (!kFinal) { (void)signs; (void)out; }  // unused in transit stages
    const int len = 1 << sh;          // the pair distance len_
    const int lmask = len - 1;
    const int64_t total_pairs = static_cast<int64_t>(M) << row_shift;  // M * K/2

    // kernel_fht.cu:83's fold, verbatim: b = K for the single-segment
    // envelope, computed in the kernel at the epilogue (not the host).
    float inv_sqrt_b = 1.0f;
    if constexpr (kFinal) {
        inv_sqrt_b = rsqrtf(static_cast<float>(K));
    }

    const int64_t stride =
        static_cast<int64_t>(gridDim.x) * flute::FHT_LARGE_THREADS;
    const int64_t tid =
        static_cast<int64_t>(blockIdx.x) * flute::FHT_LARGE_THREADS + threadIdx.x;

    for (int64_t p = tid; p < total_pairs; p += stride) {
        const int row = static_cast<int>(p >> row_shift);
        const int q = static_cast<int>(p & ((static_cast<int64_t>(1) << row_shift) - 1));
        // fht_stage_sh's pair index (fht.cuh:91): shift/mask only, no
        // runtime division -- the fixed pair->element map.
        const int i = ((q >> sh) << (sh + 1)) + (q & lmask);
        const int64_t rbase = static_cast<int64_t>(row) * K;

        const float u = ws[rbase + i];        // the LOWER index's value
        const float w = ws[rbase + i + len];  // the HIGHER index's value
        float lo = u + w;  // low  = u + w  (the pairing contract)
        float hi = u - w;  // high = u - w

#ifdef FLUTE_TQ_TRAP
        // W20.6's negative control (FLUTE_TQ_TRAP, default OFF): a
        // deliberate determinism break. The pair's (u+w)/(u-w)
        // assignment flips on a timing bit (clock64 of the executing
        // SM), so two runs of the same launch produce different bits
        // -- with random data the flip is visible essentially
        // everywhere. A run-to-run torch.equal assert that still
        // PASSES with the trap armed is a broken assert. OFF, this
        // block compiles to nothing.
        if ((clock64() & 1) != 0) {
            const float t = lo; lo = hi; hi = t;
        }
#endif

        if constexpr (kFinal) {
            // the forward epilogue: butterfly first (u+w / u-w above),
            // then the per-element sign, then the scale -- exactly
            // fht_reference's out = (seg * s) * inv (fht.py:219-221)
            // and kernel_fht.cu:94's multiply chain. One dtype cast,
            // at the final store only.
            out[rbase + i] =
                static_cast<scalar_t>(lo * signs[i] * inv_sqrt_b);
            out[rbase + i + len] =
                static_cast<scalar_t>(hi * signs[i + len] * inv_sqrt_b);
        } else {
            ws[rbase + i] = lo;
            ws[rbase + i + len] = hi;
        }
    }
}

#ifdef FLUTE_TQ_CANARY
// ---------------------------------------------------------------------------
// W20.6's write-coverage control (FLUTE_TQ_CANARY, default OFF): fill
// every element of the output tensor (and of the fp32 state when it
// is a distinct buffer) with a NaN poison BEFORE the real kernels
// write. Every element of both buffers is written exactly once by the
// real path (pass 1 covers the state; the final cross stage covers
// the output), so ZERO poison must survive: any NaN in the result of
// a finite input marks an unwritten element (a coverage hole). OFF,
// this kernel and its launches compile away entirely.
// ---------------------------------------------------------------------------
template <typename scalar_t>
__global__ void __launch_bounds__(flute::FHT_LARGE_THREADS)
fht_large_poison_kernel(scalar_t* __restrict__ buf, int64_t n) {
    // 0x7fc0d1e1: a distinctive quiet-NaN payload. The cast to
    // fp16/bf16 maps NaN to NaN, so the poison survives dtype
    // narrowing identifiably.
    const float poison = __int_as_float(0x7fc0d1e1);
    const int64_t stride =
        static_cast<int64_t>(gridDim.x) * flute::FHT_LARGE_THREADS;
    for (int64_t j = static_cast<int64_t>(blockIdx.x) * flute::FHT_LARGE_THREADS
                     + threadIdx.x;
         j < n; j += stride) {
        buf[j] = static_cast<scalar_t>(poison);
    }
}
#endif  // FLUTE_TQ_CANARY

// ---------------------------------------------------------------------------
// Host-side launcher -- the pass structure and the launch-boundary
// synchronization (one launch per cross stage; the stream orders them).
// ---------------------------------------------------------------------------

// Pass 2's grid: ceil(total_pairs / 1024) CTAs, a pure function of
// (M, K) -- no device query feeds the geometry, so the pair->thread
// map is fixed across runs (the determinism contract). >= 16 CTAs at
// the K = 2^15 / M = 1 floor; 2048 at K = 2^22 / M = 1.
__forceinline__ int fht_large_cross_blocks(int64_t total_pairs) {
    return static_cast<int>(
        (total_pairs + flute::FHT_LARGE_THREADS - 1) / flute::FHT_LARGE_THREADS);
}

template <typename scalar_t>
void fht_large_run(const scalar_t* x, const float* signs, scalar_t* out,
                   float* ws, int M, int K, cudaStream_t stream) {
    const int smem_bytes = flute::FHT_LARGE_SMEM_BYTES;  // 65536

    // The 64 KiB opt-in (kernel_fht.cu:202-220's discipline, checked
    // and loud). Unlike kernel_fht.cu -- whose production shapes
    // stayed under the 48 KiB default -- the 64 KiB tile is needed on
    // EVERY launch, so the opt-in is unconditional here.
    int device = 0;
    C10_CUDA_CHECK(cudaGetDevice(&device));
    int max_smem = 0;
    C10_CUDA_CHECK(cudaDeviceGetAttribute(
        &max_smem, cudaDevAttrMaxSharedMemoryPerBlockOptin, device));
    TORCH_CHECK(smem_bytes <= max_smem,
                "fht_large: the 64 KiB tile needs ", smem_bytes,
                " bytes of shared memory per block, but this device's "
                "opt-in maximum is ", max_smem);
    const void* kfn = reinterpret_cast<const void*>(&fht_large_local_kernel<scalar_t>);
    C10_CUDA_CHECK(cudaFuncSetAttribute(
        kfn, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes));

#ifdef FLUTE_TQ_CANARY
    // Poison the output and (when distinct) the fp32 state before any
    // real write -- W20.6's coverage control (see the kernel above).
    const int64_t n = static_cast<int64_t>(M) * K;
    const int pblocks = fht_large_cross_blocks(n);
    fht_large_poison_kernel<scalar_t>
        <<<pblocks, flute::FHT_LARGE_THREADS, 0, stream>>>(out, n);
    if (static_cast<const void*>(ws) != static_cast<const void*>(out)) {
        fht_large_poison_kernel<float>
            <<<pblocks, flute::FHT_LARGE_THREADS, 0, stream>>>(ws, n);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
#endif

    // ---- pass 1: the tile-local butterflies (all rows, all tiles) ----
    const int tiles = K / flute::FHT_LARGE_TILE;
    dim3 grid(M, tiles);  // grid.x = rows (2^31-1 axis), grid.y = tiles
    fht_large_local_kernel<scalar_t>
        <<<grid, flute::FHT_LARGE_THREADS, smem_bytes, stream>>>(x, ws, K);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    // ---- pass 2: the cross stages, ONE LAUNCH PER STAGE ---------------
    // len_ = 16384 .. K/2 (sh = 14 .. log2(K)-1): log2(K) - 14 launches,
    // each synchronized against the next by its launch boundary. The
    // final launch (len_ = K/2) carries the forward epilogue. K is a
    // power of two > 16384 (checked), so this loop ALWAYS runs at least
    // once -- the signs always have a carrier stage.
    const int row_shift = flute::fht_large_log2_floor(K) - 1;  // log2(K/2)
    const int blocks =
        fht_large_cross_blocks(static_cast<int64_t>(M) << row_shift);
    for (int sh = kFirstCrossSh; (1 << sh) < K; ++sh) {
        const bool is_final = ((1 << sh) << 1) == K;  // len_ == K/2
        if (is_final) {
            fht_large_cross_kernel<scalar_t, true>
                <<<blocks, flute::FHT_LARGE_THREADS, 0, stream>>>(
                    ws, signs, out, M, K, sh, row_shift);
        } else {
            fht_large_cross_kernel<scalar_t, false>
                <<<blocks, flute::FHT_LARGE_THREADS, 0, stream>>>(
                    ws, signs, out, M, K, sh, row_shift);
        }
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
}

// Argument validation -- kernel_fht.cu:304-327's check discipline
// (loud TORCH_CHECKs, the family's message prefix), plus the contract
// header's own refusal statuses bound in as the K gate.
void fht_large_check_args(const torch::Tensor& x, const torch::Tensor& signs) {
    TORCH_CHECK(x.is_cuda(), "fht_large: x must be a CUDA tensor");
    TORCH_CHECK(x.dim() == 2,
                "fht_large: x must be 2-D (M, K) - the Python wrapper "
                "flattens (batch, seq, K)");
    TORCH_CHECK(x.is_contiguous(), "fht_large: x must be contiguous");
    TORCH_CHECK(x.scalar_type() == at::kFloat ||
                x.scalar_type() == at::kHalf ||
                x.scalar_type() == at::kBFloat16,
                "fht_large: x dtype must be float32/float16/bfloat16, got ",
                x.scalar_type());
    TORCH_CHECK(signs.is_cuda() && signs.dim() == 1 &&
                signs.scalar_type() == at::kFloat &&
                signs.numel() == x.size(1) && signs.is_contiguous(),
                "fht_large: signs must be a contiguous (K,) float32 CUDA "
                "tensor matching x's last dim (got numel=", signs.numel(),
                ", K=", x.size(1), ")");
    TORCH_CHECK(static_cast<int64_t>(x.size(0)) <= INT32_MAX,
                "fht_large: M must fit a grid.x launch (got M=", x.size(0), ")");
    TORCH_CHECK(x.size(1) <= static_cast<int64_t>(INT32_MAX),
                "fht_large: K must fit an int (got K=", x.size(1), ")");

    const int K = static_cast<int>(x.size(1));

    // Launch-grid hygiene: pass 2 and the canary fill derive their block
    // counts as ints from M*K/2 and M*K; a tensor large enough to
    // overflow them cannot exist (2^41 elements = 8 TiB of fp32), but
    // the launch must be constructible by arithmetic, not by faith.
    TORCH_CHECK(static_cast<int64_t>(x.size(0)) * K <= (int64_t{1} << 41),
                "fht_large: M*K = ", static_cast<int64_t>(x.size(0)) * K,
                " exceeds the launch grid's addressable pairs");

    // The contract's own gate (flute::fht_large_stages,
    // fht_large.cuh:268-285) -- the refusal statuses, loudly.
    const flute::FhtLargeStages st = flute::fht_large_stages(K);
    TORCH_CHECK(st.status != flute::FhtLargeStatus::KTooSmall,
                "fht_large: K=", K, " <= 16384 - one 64 KiB tile or less; "
                "the small-K fht path owns it");
    TORCH_CHECK(st.status != flute::FhtLargeStatus::KNotMultiple32,
                "fht_large: K must be divisible by 32 (the "
                "kernel_fht.cu:21 discipline), got K=", K);

    // The W20.5.2 envelope: the SINGLE-SEGMENT butterfly. The cross
    // stages below transform the whole row as one power-of-two
    // segment, which is only the block-diagonal T when K IS a power
    // of two. Non-power-of-two K <= 2^16-32 stays on kernel_fht.cu's
    // segmented path; anything larger is refused here -- loudly, not
    // silently mis-transformed (segmented large-K is the family's
    // later work; see the file docstring's scope note).
    TORCH_CHECK((K & (K - 1)) == 0,
                "fht_large: K must be a power of two in the W20.5.2 "
                "envelope (2^15..2^22; got K=", K, ") - segmented "
                "non-power-of-two K is not implemented");

    // grid.y bound: tiles/row <= 65535 (K <= 2^30 -- far above the
    // tested envelope, but the launch must be constructible).
    const int tiles = K / flute::FHT_LARGE_TILE;
    TORCH_CHECK(tiles >= 1 && tiles <= 65535,
                "fht_large: K=", K, " needs ", tiles,
                " tiles per row; grid.y caps at 65535");
}

}  // namespace

// ---------------------------------------------------------------------------
// Public entrypoint (bound by src/bindings.cpp in W20.5.4; plain
// torch::Tensor f(torch::Tensor, torch::Tensor) signature).
//
//   fht_forward_large(x, signs) -> out = x @ T   (out-of-place, fresh
//   output; x is never written)
//
// out = ((x @ H) * s) / sqrt(K) over the whole row -- signs AFTER the
// butterfly (the forward convention). Dtype surface matches
// kernel_fht.cu: fp32 / fp16 / bf16 in and out (identical dtype), a
// (K,) fp32 +-1 sign vector, 2-D contiguous (M, K) CUDA tensors. The
// butterfly state is fp32 throughout: for fp32 I/O the output tensor
// IS the state (the final stage finishes in place); for fp16/bf16 a
// distinct fp32 state tensor carries the cross stages and the dtype
// cast happens once, at the final store -- kernel_fht.cu's
// fp32-accumulation contract preserved across the pass boundary.
// ---------------------------------------------------------------------------
torch::Tensor fht_forward_large(torch::Tensor x, torch::Tensor signs) {
    fht_large_check_args(x, signs);
    const int M = static_cast<int>(x.size(0));
    const int K = static_cast<int>(x.size(1));

    torch::Tensor out = torch::empty_like(x);  // fresh, out-of-place
    if (M == 0) {
        return out;  // nothing to transform; a zero-size grid is not
                     // a legal launch
    }

    // The fp32 butterfly state: `out` itself for fp32 I/O (the final
    // cross stage reads and writes the same indices per pair -- the
    // fht_inplace_ precedent, kernel_fht.cu:68-69); a distinct fp32
    // workspace for fp16/bf16 so every cross-stage round-trip stays
    // fp32 and the I/O dtype is cast exactly once. (The multi-pass
    // state-buffer discipline of kernel_gemv_splitk.cu:62-76, reduced
    // to its benign form: one producer pass, in-place consumers, a
    // final cast -- no per-split reduction machinery to own.)
    torch::Tensor ws = (x.scalar_type() == at::kFloat)
        ? out
        : torch::empty(x.sizes(), x.options().dtype(at::kFloat));

    auto stream = at::cuda::getCurrentCUDAStream();  // the house kernels' stream
    AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, x.scalar_type(),
                                    "fht_forward_large", [&] {
        fht_large_run<scalar_t>(x.data_ptr<scalar_t>(),
                                signs.data_ptr<float>(),
                                out.data_ptr<scalar_t>(),
                                ws.data_ptr<float>(), M, K, stream);
    });
    return out;
}

// ===========================================================================
// W20.5.3 -- fht_backward_large: the adjoint, gx = g @ T^T.
//
// THE ORIENTATION (the one decision this unit owns): the sign moves to
// the INPUT side; the 1/sqrt(K) fold stays on the OUTPUT side --
//
//     gx = g @ T^T = ((g * s) @ H) / sqrt(K)
//
// exactly fht_reference_adjoint (fht.py:225-238: the input is
// multiplied by s at line 236, the butterfly runs on (g * s), the
// output takes * 1/sqrt(b) at line 237) and exactly kernel_fht.cu's
// single-block adjoint (fht_backward_kernel, kernel_fht.cu:100-131):
//
//   * PROLOGUE, at FIRST LOAD (pass 1's staging):
//         smem[j] = float(g[col + j]) * signs[col + j]
//     -- kernel_fht.cu:118-121's converting stage load with the
//     input-side sign, verbatim. The +-1 multiply is exact (a flip),
//     so no rounding rides the prologue.
//   * BUTTERFLY: the SAME one. H is symmetric and H @ H = b * I
//     (fht.cuh:21-26), so the adjoint of the FWHT butterfly is the
//     butterfly itself: kernel_fht.cu:124 calls the same fht_block
//     the forward calls, in the same ascending stage order -- so does
//     this file. Same stage split (13 local + the lowered len_ = 8192
//     boundary stage, then len_ = 16384..K/2 cross), same pairing
//     (low = u + w at the lower index, high = u - w at the higher),
//     same ownership, same launch-boundary sync.
//   * EPILOGUE, at the FINAL STORE (pass 2's last cross stage, len_ =
//     K/2 -- every element's last write):
//         out = bfly * rsqrtf(K)
//     -- kernel_fht.cu:116+126-128's fold, verbatim placement AND
//     instruction: an rsqrtf (NOT a reciprocal divide), computed in
//     the kernel, applied ONCE per element after the butterfly, with
//     NO sign (the signs already rode the load). The contract header
//     pins this placement: "adjoint in[j] = v[j] * s[j] at the load
//     and out[j] = bfly[j] * (1/sqrt(b)) at the store"
//     (fht_large.cuh:94-95; the transit round-trips carry neither,
//     fht_large.cuh:87-96).
//
// REUSE DECISIONS (evaluated, then chosen):
//   * pass 1 local kernel: CLONED, not reused. fht_large_local_kernel
//     is epilogue-free (its stores are unsigned and unscaled) but has
//     no PROLOGUE slot: the adjoint's sign-at-load cannot be injected
//     without editing the committed forward kernel, and the W20.5.3
//     contract is additions-only (the forward path stays
//     byte-identical). The clone, fht_large_local_backward_kernel,
//     differs in one substantive line: the staging multiply.
//   * cp.async is DROPPED in the adjoint's pass 1, for every dtype:
//     the forward's 16 B raw copies cannot carry a multiply, and the
//     sign prologue needs one between the global read and the shared
//     write. Applying the signs as a second smem pass after an async
//     copy would cost an extra shared round-trip and a barrier to
//     save nothing; kernel_fht.cu's adjoint itself stages with the
//     plain converting loop (kernel_fht.cu:119-121). Determinism is
//     unaffected: the staging map stays the same fixed
//     thread->element function, and the values are identical bits.
//   * pass 2 transit stages: REUSED VERBATIM. The unsigned, unscaled
//     butterfly is orientation-free by construction -- the transit
//     epilogue carries neither s nor 1/sqrt(K)
//     (fht_large.cuh:84-86: "the butterfly itself is UNSIGNED -- the
//     same H both directions") -- so the adjoint's cross stages ARE
//     fht_large_cross_kernel<scalar_t, false>. This also inherits
//     W20.6's FLUTE_TQ_TRAP on those stages for free.
//   * pass 2 final stage: a kBackward-style template flag on the
//     forward's cross kernel was the 5.2 report's suggestion; it is
//     REJECTED here -- retrofitting the flag edits the committed
//     kernel (violating additions-only), and on a fresh clone the
//     flag is dead weight (the adjoint's final kernel only ever runs
//     final). One new kernel, fht_large_cross_backward_kernel,
//     carries the adjoint epilogue: the scale, no signs.
//
// The CANARY/TRAP scaffolding stays forward-only in this unit: W20.6
// extends both controls to the backward (the transit stages already
// share the forward's trapped cross kernel through the reuse above;
// the local and final stages are W20.6's to arm).
//
// Everything else is the forward's discipline, mirrored: out-of-place
// (g is never written), dtype-templated (fp16/bf16 in and out with a
// distinct fp32 state, cast once at the final store), the 64 KiB smem
// opt-in on the local pass, launch-boundary sync, unique writers by
// construction, __launch_bounds__(1024) on both new __global__s.
// ===========================================================================

namespace {

// ---------------------------------------------------------------------------
// Adjoint pass 1 -- the tile-local butterfly with the sign prologue.
//
// CTA (row r, tile t) stages row r's tile t into 64 KiB dynamic shared
// memory WITH the input-side sign (the adjoint prologue), butterflies
// it with the same fht_block the forward's pass 1 runs, and writes the
// fp32 state back UNSIGNED and UNSCALED -- the 1/sqrt(K) fold is pass
// 2's final-store epilogue (kernel_fht.cu:116+126-128's placement; the
// transit round-trips carry neither sign nor scale, fht_large.cuh:87-96).
//
// Staging: kernel_fht.cu:118-121's prologue verbatim -- the converting
// stage load with the sign multiply folded in, for EVERY dtype (the
// forward's cp.async path is dropped; see the W20.5.3 banner). The
// sign index is the COLUMN index (signs is (K,)): col0 + j with
// col0 = tile * FHT_LARGE_TILE -- NOT the flat base, which carries the
// row offset kernel_fht.cu's per-row pointers never had.
//
// grid = (M rows, K/FHT_LARGE_TILE tiles) -- the forward pass 1's map.
// ---------------------------------------------------------------------------
template <typename scalar_t>
__global__ void __launch_bounds__(flute::FHT_LARGE_THREADS)
fht_large_local_backward_kernel(const scalar_t* __restrict__ g,  // (M, K) gradient, read-only
                                const float* __restrict__ signs, // (K,) +-1
                                float* __restrict__ ws,          // (M, K) fp32 butterfly state
                                int K) {
    // The forward's aligned-declaration idiom (kernel_streaming.cu:865),
    // kept for family consistency even though the adjoint's staging is
    // compute-carrying (no cp.async here to require the alignment).
    extern __shared__ __align__(16) unsigned char smem_raw[];
    float* smem = reinterpret_cast<float*>(smem_raw);

    const int row = blockIdx.x;   // grid.x = M
    const int tile = blockIdx.y;  // grid.y = K / FHT_LARGE_TILE
    const int col0 = tile * flute::FHT_LARGE_TILE;  // this tile's column offset
    const int64_t base = static_cast<int64_t>(row) * K + col0;

    // ---- staging + prologue: the signs at FIRST LOAD --------------------
    // kernel_fht.cu:118-121 verbatim ("input-side sign, no output
    // sign"): the fp32 cast, then the exact +-1 multiply. Every dtype
    // takes this loop -- the W20.5.3 banner's cp.async rationale.
    for (int j = threadIdx.x; j < flute::FHT_LARGE_TILE;
         j += flute::FHT_LARGE_THREADS) {
        smem[j] = static_cast<float>(g[base + j]) * signs[col0 + j];
    }
    __syncthreads();  // the staged (and signed) tile is published

    // ---- the tile-local butterfly: the SAME one -------------------------
    // The forward pass 1's exact call: the self-adjoint butterfly
    // (kernel_fht.cu runs the same fht_block in both directions). sh =
    // 0..13 -- the 13 local stages PLUS the boundary stage len_ = 8192
    // lowered intra-tile (the header's Boundary note, fht_large.cuh:53-62).
    flute::fht_block<flute::FHT_LARGE_THREADS>(smem, flute::FHT_LARGE_TILE);

    // ---- write-back: fp32 state, unsigned and unscaled ------------------
    for (int j = threadIdx.x; j < flute::FHT_LARGE_TILE;
         j += flute::FHT_LARGE_THREADS) {
        ws[base + j] = smem[j];
    }

#ifdef FLUTE_TQ_TRAP
    // W20.6.2: the adjoint pass 1's share of the negative control
    // (FLUTE_TQ_TRAP, default OFF; the forward's trap lives in
    // fht_large_cross_kernel above, and the adjoint's transit stages
    // inherit it through the verbatim reuse). fht_block itself is
    // shared with kernel_fht.cu and stays untouched (additions-only),
    // so the flip rides this kernel's own write-back: on a timing bit
    // (clock64 of the executing SM) the element stored at base + j
    // becomes its butterfly-pair partner's value -- j ^ (TILE >> 1) is
    // exactly the boundary stage's (len_ = 8192, fht_large.cuh's
    // Boundary note) pair mate, and both members of that pair live in
    // this CTA's tile. smem is butterfly-final and published (fht_block's
    // trailing __syncthreads, fht.cuh:174-177), and the re-store hits an
    // address this thread just wrote (program order), so no barrier is
    // needed. Two runs of the same backward launch produce different
    // bits; a run-to-run torch.equal assert that PASSES with the trap
    // armed is a broken assert. OFF, this loop compiles to nothing and
    // the write-back above stands alone.
    for (int j = threadIdx.x; j < flute::FHT_LARGE_TILE;
         j += flute::FHT_LARGE_THREADS) {
        if ((clock64() & 1) != 0) {
            ws[base + j] = smem[j ^ (flute::FHT_LARGE_TILE >> 1)];
        }
    }
#endif  // FLUTE_TQ_TRAP

    // No trailing barrier: the CTA exits; the launch boundary orders the
    // next kernel's reads of ws (the only cross-CTA sync this family has).
}

// ---------------------------------------------------------------------------
// Adjoint pass 2, FINAL cross stage only (len_ = K/2): the same
// elementwise global pair op as the forward's cross kernel -- the fixed
// grid-stride pair->thread map over the whole (M, K/2) pair space, the
// shift/mask pair index i = ((q >> sh) << (sh+1)) + (q & (len_-1)), u
// at the LOWER index, one fp32 add and one fp32 subtract, the owning
// thread the ONLY writer of both i and i+len_ (the ownership contract,
// fht_large.cuh:99-127) -- with the ADJOINT epilogue at the store:
//
//     gx[i]      = lo * rsqrtf(K)     (NO sign -- it rode the load)
//     gx[i+len_] = hi * rsqrtf(K)
//
// kernel_fht.cu:116+126-128's fold verbatim: rsqrtf of the segment
// size (b = K for the single-segment powers of two), computed in the
// kernel, multiplied at the final store -- the epilogue's one fp32
// rounding per element. For fp32 I/O gx IS ws (in-place per pair: the
// same-thread same-address load->store dependency the forward's kFinal
// stage already carries, fht_large_cross_kernel's in-place note); for
// fp16/bf16 gx is the distinct output tensor and this store is the
// transform's ONE dtype cast. The non-final (transit) cross stages
// are the forward's kernel, reused verbatim (see the banner):
// unsigned, unscaled, orientation-free.
// ---------------------------------------------------------------------------
template <typename scalar_t>
__global__ void __launch_bounds__(flute::FHT_LARGE_THREADS)
fht_large_cross_backward_kernel(float* __restrict__ ws,     // (M, K) fp32 butterfly state
                                scalar_t* __restrict__ gx,  // (M, K) output; == ws allowed for fp32
                                int M, int K, int sh, int row_shift) {
    const int len = 1 << sh;          // the pair distance len_ (= K/2 here)
    const int lmask = len - 1;
    const int64_t total_pairs = static_cast<int64_t>(M) << row_shift;  // M * K/2

    // kernel_fht.cu:116's convention, verbatim: rsqrtf (not a
    // reciprocal divide), b = K, computed in the kernel at the
    // epilogue -- matching the forward's kFinal fold exactly, minus
    // the signs the adjoint moved to the load.
    const float inv_sqrt_b = rsqrtf(static_cast<float>(K));

    const int64_t stride =
        static_cast<int64_t>(gridDim.x) * flute::FHT_LARGE_THREADS;
    const int64_t tid =
        static_cast<int64_t>(blockIdx.x) * flute::FHT_LARGE_THREADS + threadIdx.x;

    for (int64_t p = tid; p < total_pairs; p += stride) {
        const int row = static_cast<int>(p >> row_shift);
        const int q = static_cast<int>(p & ((static_cast<int64_t>(1) << row_shift) - 1));
        // fht_stage_sh's pair index (fht.cuh:91): shift/mask only, no
        // runtime division -- the fixed pair->element map.
        const int i = ((q >> sh) << (sh + 1)) + (q & lmask);
        const int64_t rbase = static_cast<int64_t>(row) * K;

        const float u = ws[rbase + i];        // the LOWER index's value
        const float w = ws[rbase + i + len];  // the HIGHER index's value
        const float lo = u + w;  // low  = u + w  (the pairing contract)
        const float hi = u - w;  // high = u - w

        // the adjoint epilogue: butterfly first (u+w / u-w above), then
        // the scale, NO sign -- exactly fht_reference_adjoint's
        // out = seg * (1/sqrt(b)) (fht.py:237) and kernel_fht.cu:127's
        // store. One dtype cast, at the final store only.
        gx[rbase + i] = static_cast<scalar_t>(lo * inv_sqrt_b);
        gx[rbase + i + len] = static_cast<scalar_t>(hi * inv_sqrt_b);

#ifdef FLUTE_TQ_TRAP
        // W20.6.2: the adjoint FINAL stage's mirror of the forward cross
        // kernel's trap (fht_large_cross_kernel above): the pair's
        // (u+w)/(u-w) assignment flips on a timing bit (clock64 of the
        // executing SM). The flip re-stores the pair the owning thread
        // just wrote -- same thread, same two addresses, program order,
        // the ownership contract untouched -- so the armed build's
        // visible store is the swapped pair. lo/hi stay const above; the
        // trap never edits them, and with the macro OFF the stores
        // above stand exactly as written. Two runs of the same backward
        // launch produce different bits; a run-to-run torch.equal assert
        // that PASSES with the trap armed is a broken assert.
        if ((clock64() & 1) != 0) {
            gx[rbase + i] = static_cast<scalar_t>(hi * inv_sqrt_b);
            gx[rbase + i + len] = static_cast<scalar_t>(lo * inv_sqrt_b);
        }
#endif  // FLUTE_TQ_TRAP
    }
}

// ---------------------------------------------------------------------------
// Host-side launcher -- the adjoint's pass structure. Identical geometry
// to the forward's fht_large_run (same grids, same ascending stage loop,
// same launch-boundary synchronization); the differences are exactly the
// two kernels that carry the orientation (the sign-prologue local kernel,
// the scaled no-sign final cross stage).
// ---------------------------------------------------------------------------
template <typename scalar_t>
void fht_backward_large_run(const scalar_t* g, const float* signs,
                            scalar_t* gx, float* ws, int M, int K,
                            cudaStream_t stream) {
    const int smem_bytes = flute::FHT_LARGE_SMEM_BYTES;  // 65536

    // The 64 KiB opt-in for the adjoint's local kernel: the attribute is
    // per-function, so the clone needs its own call even though the
    // forward's runner already opted in fht_large_local_kernel
    // (kernel_fht.cu:202-220's discipline, checked and loud).
    int device = 0;
    C10_CUDA_CHECK(cudaGetDevice(&device));
    int max_smem = 0;
    C10_CUDA_CHECK(cudaDeviceGetAttribute(
        &max_smem, cudaDevAttrMaxSharedMemoryPerBlockOptin, device));
    TORCH_CHECK(smem_bytes <= max_smem,
                "fht_backward_large: the 64 KiB tile needs ", smem_bytes,
                " bytes of shared memory per block, but this device's "
                "opt-in maximum is ", max_smem);
    const void* kfn =
        reinterpret_cast<const void*>(&fht_large_local_backward_kernel<scalar_t>);
    C10_CUDA_CHECK(cudaFuncSetAttribute(
        kfn, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes));

    // (The CANARY poison-fill lives in the forward's runner, W20.5.2;
    // W20.6 extends the coverage control to the backward path.)

#ifdef FLUTE_TQ_CANARY
    // W20.6.2: the backward path's share of the write-coverage control
    // -- the extension the comment above assigned to W20.6 (the
    // banner's "the local and final stages are W20.6's to arm"). The
    // forward's runner poisons its out/ws identically (above). Here:
    // gx takes every element's LAST write from the final cross stage;
    // ws takes every element's write from the local kernel's write-back
    // (the transit stages rewrite it in place, pair by pair); g itself
    // is only ever READ (out-of-place). Zero poison must survive -- any
    // NaN in gx on a finite input marks an unwritten element (a
    // coverage hole). The poison kernel is the forward's (the unnamed
    // namespace blocks above are the same namespace), and the launches
    // mirror fht_large_run's, including the ws == gx aliasing guard
    // (fp32 I/O folds the state into the output). OFF, these launches
    // compile away entirely.
    const int64_t poison_n = static_cast<int64_t>(M) * K;
    const int poison_blocks = fht_large_cross_blocks(poison_n);
    fht_large_poison_kernel<scalar_t>
        <<<poison_blocks, flute::FHT_LARGE_THREADS, 0, stream>>>(gx, poison_n);
    if (static_cast<const void*>(ws) != static_cast<const void*>(gx)) {
        fht_large_poison_kernel<float>
            <<<poison_blocks, flute::FHT_LARGE_THREADS, 0, stream>>>(ws, poison_n);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
#endif  // FLUTE_TQ_CANARY

    // ---- pass 1: the tile-local butterflies with the sign prologue ----
    const int tiles = K / flute::FHT_LARGE_TILE;
    dim3 grid(M, tiles);  // grid.x = rows (2^31-1 axis), grid.y = tiles
    fht_large_local_backward_kernel<scalar_t>
        <<<grid, flute::FHT_LARGE_THREADS, smem_bytes, stream>>>(g, signs, ws, K);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    // ---- pass 2: the cross stages, ONE LAUNCH PER STAGE ---------------
    // The SAME ascending sequence as the forward's (sh = 14..log2(K)-1,
    // len_ = 16384..K/2): the butterfly is self-adjoint, so the stage
    // list is the forward's verbatim. Transit stages launch the
    // FORWARD's cross kernel (kFinal = false: unsigned, unscaled,
    // orientation-free); the final stage (len_ = K/2) launches the
    // adjoint's epilogue kernel. K is a power of two > 16384 (checked),
    // so the loop always runs at least once -- the 1/sqrt(K) fold
    // always has a carrier stage.
    const int row_shift = flute::fht_large_log2_floor(K) - 1;  // log2(K/2)
    const int blocks =
        fht_large_cross_blocks(static_cast<int64_t>(M) << row_shift);
    for (int sh = kFirstCrossSh; (1 << sh) < K; ++sh) {
        const bool is_final = ((1 << sh) << 1) == K;  // len_ == K/2
        if (is_final) {
            fht_large_cross_backward_kernel<scalar_t>
                <<<blocks, flute::FHT_LARGE_THREADS, 0, stream>>>(
                    ws, gx, M, K, sh, row_shift);
        } else {
            // The forward's transit kernel, reused verbatim: signs and
            // gx are the (void)-discarded parameters of this
            // instantiation (the pair loop touches ws only) -- the
            // orientation rides the prologue and the final stage,
            // never the transit.
            fht_large_cross_kernel<scalar_t, false>
                <<<blocks, flute::FHT_LARGE_THREADS, 0, stream>>>(
                    ws, signs, gx, M, K, sh, row_shift);
        }
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
}

}  // namespace

// ---------------------------------------------------------------------------
// Public entrypoint (bound by src/bindings.cpp in W20.5.4; plain
// torch::Tensor f(torch::Tensor, torch::Tensor) signature).
//
//   fht_backward_large(g, signs) -> gx = g @ T^T   (out-of-place, fresh
//   output; g is never written)
//
// gx = ((g * s) @ H) / sqrt(K) over the whole row -- signs BEFORE the
// butterfly, scale after it (the adjoint convention; the W20.5.3 banner
// above). Same surface as the forward: fp32 / fp16 / bf16 in and out
// (identical dtype), a (K,) fp32 +-1 sign vector, 2-D contiguous
// (M, K) CUDA tensors, K a power of two > 16384 -- the shared
// fht_large_check_args gate (kernel_fht.cu:304-327's precedent of one
// checker serving fht_forward AND fht_backward, so the "fht_large: "
// message prefix is the family's, both entrypoints). The butterfly
// state is fp32 throughout: for fp32 I/O the output IS the state (the
// final cross stage finishes in place); for fp16/bf16 a distinct fp32
// state tensor carries the cross stages and the dtype cast happens
// once, at the final store.
// ---------------------------------------------------------------------------
torch::Tensor fht_backward_large(torch::Tensor g, torch::Tensor signs) {
    fht_large_check_args(g, signs);
    const int M = static_cast<int>(g.size(0));
    const int K = static_cast<int>(g.size(1));

    torch::Tensor gx = torch::empty_like(g);  // fresh, out-of-place
    if (M == 0) {
        return gx;  // nothing to transform; a zero-size grid is not
                    // a legal launch
    }

    // The fp32 butterfly state: `gx` itself for fp32 I/O (the final
    // cross stage reads and writes the same indices per pair -- the
    // forward's in-place precedent); a distinct fp32 workspace for
    // fp16/bf16 so every cross-stage round-trip stays fp32 and the
    // I/O dtype is cast exactly once, at the final store.
    torch::Tensor ws = (g.scalar_type() == at::kFloat)
        ? gx
        : torch::empty(g.sizes(), g.options().dtype(at::kFloat));

    auto stream = at::cuda::getCurrentCUDAStream();  // the house kernels' stream
    AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, g.scalar_type(),
                                    "fht_backward_large", [&] {
        fht_backward_large_run<scalar_t>(g.data_ptr<scalar_t>(),
                                         signs.data_ptr<float>(),
                                         gx.data_ptr<scalar_t>(),
                                         ws.data_ptr<float>(), M, K, stream);
    });
    return gx;
}

// ===========================================================================
// W20.6.1 -- the tq_debug scalar twins: tq_debug_fht_large_forward /
// tq_debug_fht_large_backward.
//
// test-only scalar twins of fht_forward_large / fht_backward_large --
// the differential reference G20.P2 compares the real kernels against
// (never on the production path: fht.py routes only to the real pair;
// the W20.6.3 test file and the G20.P2 gate function are the only
// callers). Host-side plain C++/ATen assembly of the SAME math,
// fragment-wise (tile by tile, stage by stage, pair by pair), IN THE
// SAME ORDER the kernels above execute -- the twin is meaningless
// otherwise:
//
//   FORWARD (twins fht_large_run + fht_large_local_kernel +
//   fht_large_cross_kernel, kernel_tq.cu:420-485 / 207-264 / 310-376):
//     * pass 1, per (row, tile), row-major: the converting stage load
//       (kernel_tq.cu:222-240 -- the fp32 path's cp.async raw copies
//       move the same bits, so the one converting loop twins both
//       paths), unsigned and unscaled; then the 14 tile-local stage
//       visits of fht_block<1024>(smem, 16384) (kernel_tq.cu:252):
//       sh = 0..13 -- the 13 local stages (len_ = 1..4096) PLUS the
//       lowered len_ = 8192 boundary stage (the header's Boundary
//       note, fht_large.cuh:53-62).
//     * pass 2, one loop per cross stage, the launcher's exact
//       ascending sequence (kernel_tq.cu:472-484): sh = 14..log2(K)-1,
//       len_ = 16384..K/2. The forward epilogue rides the FINAL stage
//       only (len_ = K/2): the left-associated multiply chain
//       (t * signs[j]) * rsqrtf(K) at the store, ONE dtype cast
//       (kernel_tq.cu:361-370).
//   BACKWARD (twins fht_backward_large_run +
//   fht_large_local_backward_kernel + fht_large_cross_backward_kernel,
//   kernel_tq.cu:823-887 / 708-749 / 774-814): the signs at FIRST LOAD
//   (the prologue's cast-then-multiply, kernel_tq.cu:729-732, signs
//   indexed by COLUMN), the SAME 14-stage tile butterfly and the SAME
//   ascending cross stages (transit stages unsigned and unscaled --
//   the forward's cross kernel, kernel_tq.cu:876-883), and the adjoint
//   epilogue at the final store: (u+w / u-w) * rsqrtf(K), NO sign.
//
// THE PAIRING-ORDER GUARANTEE (the twin's reason to exist): stages run
// in the kernels' fixed ascending order (len_ = 1, 2, ..., K/2); every
// stage visits pairs by the kernels' pair identity p (row-major over
// the (M, K/2) space for the cross stages, fht_large_cross_kernel:
// 334-339; flat over the tile's 8192 pairs for the local stages,
// fht.cuh:90-95) with the kernels' shift/mask index i = ((p >> sh) <<
// (sh+1)) + (p & (len_-1)) and the kernels' arithmetic: u at the LOWER
// index, one fp32 add and one fp32 subtract, u on the left, low =
// u + w at i, high = u - w at i + len_ (fht_large.cuh:128-145). The
// kernels' grid-stride thread interleave and fht_block's register-
// local early stages (fht.cuh:141-170 -- the same pair set, since
// 2*len divides the 16-element run, with the same per-pair arithmetic)
// reorder only WHICH thread owns WHICH pair: pairs at one stage are
// disjoint, so the visit order cannot change a value
// (fht_large.cuh:141-145). The flat p loop IS the kernels'
// enumeration; fp32 add/sub/mul are IEEE round-to-nearest on host and
// device alike, so the butterfly bits agree with the kernels' exactly.
//
// THE PLACEMENT (documented choice): the twin computes on the HOST, in
// CPU-side clones -- the flat pair loop is the simplest formulation
// that preserves the order, and the differential comparison stays
// device-side (inputs CUDA, like the real pair; the twin migrates
// internally and returns on x's device). The ONE constant the host
// cannot reproduce faithfully, rsqrtf(K) (a device function the
// kernels fold in-kernel, kernel_tq.cu:324-327 / 783-787), is computed
// ON x's DEVICE through torch's CUDA rsqrt -- the same CUDA math
// library ::rsqrt the kernels' rsqrtf compiles to -- and read back
// (tq_debug_inv_sqrt_b below), so the epilogue constant matches
// bit-for-bit; any residual ulp divergence is one scalar's tail,
// absorbed by G20.P2's allclose (1e-5). fp16/bf16 I/O: the state stays
// fp32 end-to-end, the dtype cast happens ONCE at the final store (the
// c10 round-to-nearest-even conversions -- the same rounding as the
// kernels' __float2half_rn / __float2bfloat16 store casts).
//
// THE LAW (PROMPT.md §1): authored STATICALLY on the CPU dev box -- no
// test run, no test authored, no repo module imported; nvcc is ABSENT
// here, so compilation and the differential suite defer to the A10G
// box (recorded, legal). No reduction or synchronization primitives
// anywhere in the twin (host code: one thread, unique writers by
// construction). Verified by re-read + rg on this box.
// ===========================================================================

namespace {

// ---------------------------------------------------------------------------
// The twin's tile-local butterfly: the 14 stage visits of
// fht_block<1024>(smem, FHT_LARGE_TILE) (the call at kernel_tq.cu:252 /
// 740), FLAT over the tile's 8192 pairs per stage -- fht_stage_sh's
// enumeration and pairing, verbatim arithmetic (fht.cuh:90-95: pair p
// -> i = ((p >> sh) << (sh+1)) + (p & lmask); s[i] = a + bv;
// s[i + len] = a - bv). Covers the register-local early stages
// (fht.cuh:141-170: sh = 0..3 inside each thread's 16-element run --
// the same pair set, since 2*len divides 16, with the same per-pair
// arithmetic) and the shared-memory stages (sh = 4..13) alike; the
// lowered len_ = 8192 boundary stage is the sh = 13 visit. Disjoint
// pairs per stage -> the flat visit order is value-identical to every
// thread decomposition of the same stage (fht_large.cuh:141-145).
// ---------------------------------------------------------------------------
void tq_debug_tile_butterfly(float* v) {  // one 16384-fp32 tile, host memory
    static_assert(
        flute::FHT_LARGE_TILE == (1 << (flute::FHT_LARGE_LOCAL_STAGES + 1)),
        "sh = 0..LOCAL_STAGES is len_ = 1..8192: the full tile butterfly");
    constexpr int kStageMax = flute::FHT_LARGE_LOCAL_STAGES;  // 13
    constexpr int kPairs = flute::FHT_LARGE_TILE >> 1;        // 8192
    for (int sh = 0; sh <= kStageMax; ++sh) {
        const int len = 1 << sh;
        const int lmask = len - 1;
        for (int p = 0; p < kPairs; ++p) {
            // fht.cuh:91's pair index, verbatim.
            const int i = ((p >> sh) << (sh + 1)) + (p & lmask);
            const float u = v[i];        // the LOWER index's value
            const float w = v[i + len];  // the HIGHER index's value
            v[i] = u + w;                // low  = u + w  (fht.cuh:94)
            v[i + len] = u - w;          // high = u - w  (fht.cuh:95)
        }
    }
}

// ---------------------------------------------------------------------------
// The twin's cross stages: the launcher's ascending stage loop
// (kernel_tq.cu:472-484 -- the backward's identical loop, 869-884), one
// flat pass per stage over the whole (M, K/2) pair space in the
// kernels' row-major p order (p = row * (K/2) + q), the same
// shift/mask pair index (fht_large_cross_kernel:334-339 and the
// backward's 794-799), the same one-add-one-subtract pairing, u at the
// LOWER index. kBackward selects the epilogue exactly as the two final
// kernels differ:
//   forward  final store: (t * signs[j]) * inv_sqrt_b  (lines 361-370)
//   backward final store: t * inv_sqrt_b, NO sign      (lines 807-812)
// Transit stages carry neither sign nor scale (fht_large.cuh:87-96);
// the final stage's store is the transform's ONE dtype cast.
// ---------------------------------------------------------------------------
template <typename scalar_t, bool kBackward>
void tq_debug_cross_stages(float* ws, const float* signs, scalar_t* out,
                           int M, int K, float inv_sqrt_b) {
    if constexpr (kBackward) {
        (void)signs;  // the adjoint's signs rode the load (the (void)
                      // idiom of kernel_tq.cu:317)
    }
    const int row_shift = flute::fht_large_log2_floor(K) - 1;  // log2(K/2)
    const int row_mask = (1 << row_shift) - 1;                 // K/2 - 1
    const int64_t total_pairs = static_cast<int64_t>(M) << row_shift;

    for (int sh = kFirstCrossSh; (1 << sh) < K; ++sh) {  // the launcher's loop
        const bool is_final = ((1 << sh) << 1) == K;      // len_ == K/2
        const int len = 1 << sh;
        const int lmask = len - 1;
        for (int64_t p = 0; p < total_pairs; ++p) {
            // the kernel's pair decomposition, verbatim (lines 335-339):
            // row-major (M, K/2) flattening, shift/mask index, no division.
            const int row = static_cast<int>(p >> row_shift);
            const int q = static_cast<int>(p) & row_mask;
            const int i = ((q >> sh) << (sh + 1)) + (q & lmask);
            const int64_t rbase = static_cast<int64_t>(row) * K;

            const float u = ws[rbase + i];        // the LOWER index's value
            const float w = ws[rbase + i + len];  // the HIGHER index's value
            const float lo = u + w;  // low  = u + w  (the pairing contract)
            const float hi = u - w;  // high = u - w

            if (is_final) {
                if constexpr (kBackward) {
                    // the adjoint epilogue (kernel_tq.cu:811-812): the
                    // scale, NO sign -- it rode the load.
                    out[rbase + i] =
                        static_cast<scalar_t>(lo * inv_sqrt_b);
                    out[rbase + i + len] =
                        static_cast<scalar_t>(hi * inv_sqrt_b);
                } else {
                    // the forward epilogue (kernel_tq.cu:367-370): the
                    // sign, then the scale -- the kernels'
                    // left-associated multiply chain, ONE dtype cast at
                    // the final store.
                    out[rbase + i] = static_cast<scalar_t>(
                        (lo * signs[i]) * inv_sqrt_b);
                    out[rbase + i + len] = static_cast<scalar_t>(
                        (hi * signs[i + len]) * inv_sqrt_b);
                }
            } else {
                ws[rbase + i] = lo;        // the in-place transit
                ws[rbase + i + len] = hi;  // round-trip: unsigned, unscaled
            }
        }
    }
}

// ---------------------------------------------------------------------------
// The epilogue constant: rsqrtf((float)K), computed the way the kernels
// compute it -- ON THE DEVICE (rsqrtf is a device-only function; the
// kernels fold it in-kernel, kernel_tq.cu:324-327 / 783-787). The twin
// routes a one-element fp32 CUDA tensor through torch's CUDA rsqrt --
// the same CUDA math library ::rsqrt(float) the kernels' rsqrtf
// compiles to -- and reads the value back, so the epilogue constant
// matches the kernels' bit-for-bit (and even on a box where torch's
// rsqrt disagreed, the divergence is ~1 ulp of ONE scalar, absorbed by
// G20.P2's allclose 1e-5 -- a box-side discovery, loudly noted).
// ---------------------------------------------------------------------------
float tq_debug_inv_sqrt_b(int K, const torch::Tensor& like) {
    torch::Tensor kdev = torch::full(
        {1}, static_cast<double>(static_cast<float>(K)),
        like.options().dtype(at::kFloat));
    return torch::rsqrt(kdev).item<float>();
}

}  // namespace

// ---------------------------------------------------------------------------
// tq_debug_fht_large_forward -- test-only scalar twin of
// fht_forward_large: the differential reference for G20.P2 (never on
// the production path). Same argument surface and same gate
// (fht_large_check_args), same stage sequence and pairing order as the
// real kernel (the W20.6.1 banner above), host-side flat-loop assembly
// in fp32, the single dtype cast at the final store. Returns a fresh
// tensor on x's device; x is never written.
// ---------------------------------------------------------------------------
torch::Tensor tq_debug_fht_large_forward(torch::Tensor x, torch::Tensor signs) {
    fht_large_check_args(x, signs);  // the same surface as the real pair
    const int M = static_cast<int>(x.size(0));
    const int K = static_cast<int>(x.size(1));

    if (M == 0) {
        return torch::empty_like(x);  // nothing to transform (the real
                                      // pair's M == 0 guard)
    }

    // the epilogue constant, device-computed (the helper's banner)
    const float inv_sqrt_b = tq_debug_inv_sqrt_b(K, x);

    // host-side staging: exact device->host byte copies (the casts sit
    // where the kernels put them -- at the loads and the final store),
    // a fresh host fp32 state, and a host output in the I/O dtype that
    // IS the state for fp32 (the twin of fht_forward_large's ws/out
    // aliasing, kernel_tq.cu:582-592 -- the final stage finishes in
    // place).
    torch::Tensor x_cpu = x.cpu();
    torch::Tensor signs_cpu = signs.cpu();
    torch::Tensor state =
        torch::empty(x.sizes(), x_cpu.options().dtype(at::kFloat));
    torch::Tensor out_cpu = (x.scalar_type() == at::kFloat)
        ? state
        : torch::empty(x.sizes(), x_cpu.options());

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::kHalf, at::kBFloat16, x.scalar_type(),
        "tq_debug_fht_large_forward", [&] {
            const scalar_t* xp = x_cpu.data_ptr<scalar_t>();
            const float* sp = signs_cpu.data_ptr<float>();
            float* wsp = state.data_ptr<float>();
            scalar_t* op = out_cpu.data_ptr<scalar_t>();

            // ---- pass 1: the tile-local butterflies (twins
            // fht_large_run:456-461 + fht_large_local_kernel:222-260);
            // (row, tile) row-major -- tiles are disjoint.
            const int tiles = K / flute::FHT_LARGE_TILE;
            for (int row = 0; row < M; ++row) {
                for (int tile = 0; tile < tiles; ++tile) {
                    const int64_t base = static_cast<int64_t>(row) * K
                        + static_cast<int64_t>(tile) * flute::FHT_LARGE_TILE;
                    // the converting stage load (kernel_tq.cu:236-239;
                    // the fp32 cp.async path at 228-231 moves the same
                    // bits): cast at the load, no sign, no scale.
                    for (int j = 0; j < flute::FHT_LARGE_TILE; ++j) {
                        wsp[base + j] = static_cast<float>(xp[base + j]);
                    }
                    // fht_block<1024>(smem, 16384) (kernel_tq.cu:252):
                    // sh = 0..13 -- the 13 local stages plus the lowered
                    // len_ = 8192 boundary stage.
                    tq_debug_tile_butterfly(wsp + base);
                    // write-back (kernel_tq.cu:257-260) is the identity
                    // here: the twin's state IS the smem.
                }
            }

            // ---- pass 2: the cross stages + the forward epilogue
            // (twins fht_large_run:463-484 + fht_large_cross_kernel:
            // 310-376); for fp32 I/O op == wsp (the real pair's
            // in-place final stage).
            tq_debug_cross_stages<scalar_t, false>(wsp, sp, op, M, K,
                                                   inv_sqrt_b);
        });

    // the move home: a same-dtype device copy (no conversion rides it)
    return out_cpu.to(x.device());
}

// ---------------------------------------------------------------------------
// tq_debug_fht_large_backward -- test-only scalar twin of
// fht_backward_large: the differential reference for G20.P2 (never on
// the production path). The adjoint's orientation, twinned: signs at
// FIRST LOAD (the prologue), the SAME butterfly, rsqrtf(K) at the
// final store -- kernel_tq.cu:605-940's contract in flat-loop form,
// with the same stage sequence and pairing-order guarantee as the
// forward twin (the W20.6.1 banner above). Returns a fresh tensor on
// g's device; g is never written.
// ---------------------------------------------------------------------------
torch::Tensor tq_debug_fht_large_backward(torch::Tensor g, torch::Tensor signs) {
    fht_large_check_args(g, signs);  // the family checker (the real
                                     // backward's reuse, kernel_tq.cu:912)
    const int M = static_cast<int>(g.size(0));
    const int K = static_cast<int>(g.size(1));

    if (M == 0) {
        return torch::empty_like(g);
    }

    // the epilogue constant, device-computed (the helper's banner)
    const float inv_sqrt_b = tq_debug_inv_sqrt_b(K, g);

    // host-side staging: the same buffers as the forward twin, with the
    // gx/state aliasing for fp32 I/O (the twin of fht_backward_large's
    // ws/gx aliasing, kernel_tq.cu:922-929).
    torch::Tensor g_cpu = g.cpu();
    torch::Tensor signs_cpu = signs.cpu();
    torch::Tensor state =
        torch::empty(g.sizes(), g_cpu.options().dtype(at::kFloat));
    torch::Tensor gx_cpu = (g.scalar_type() == at::kFloat)
        ? state
        : torch::empty(g.sizes(), g_cpu.options());

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::kHalf, at::kBFloat16, g.scalar_type(),
        "tq_debug_fht_large_backward", [&] {
            const scalar_t* gp = g_cpu.data_ptr<scalar_t>();
            const float* sp = signs_cpu.data_ptr<float>();
            float* wsp = state.data_ptr<float>();
            scalar_t* op = gx_cpu.data_ptr<scalar_t>();

            // ---- pass 1: the tile-local butterflies WITH the sign
            // prologue (twins fht_backward_large_run:850-855 +
            // fht_large_local_backward_kernel:708-749): the signs at
            // FIRST LOAD, indexed by COLUMN (col0 + j) -- cast, then
            // the exact +-1 multiply (kernel_tq.cu:729-732's math).
            const int tiles = K / flute::FHT_LARGE_TILE;
            for (int row = 0; row < M; ++row) {
                for (int tile = 0; tile < tiles; ++tile) {
                    const int col0 = tile * flute::FHT_LARGE_TILE;
                    const int64_t base =
                        static_cast<int64_t>(row) * K + col0;
                    for (int j = 0; j < flute::FHT_LARGE_TILE; ++j) {
                        wsp[base + j] =
                            static_cast<float>(gp[base + j]) * sp[col0 + j];
                    }
                    // the SAME tile butterfly (the same fht_block call,
                    // kernel_tq.cu:740 -- H is self-adjoint)
                    tq_debug_tile_butterfly(wsp + base);
                }
            }

            // ---- pass 2: the transit stages (the forward's cross
            // kernel, unsigned/unscaled -- kernel_tq.cu:876-883) and the
            // adjoint's final epilogue (fht_large_cross_backward_kernel:
            // 774-814: the scale, no sign); for fp32 I/O op == wsp.
            tq_debug_cross_stages<scalar_t, true>(wsp, sp, op, M, K,
                                                  inv_sqrt_b);
        });

    // the move home: a same-dtype device copy (no conversion rides it)
    return gx_cpu.to(g.device());
}

// ===========================================================================
// W20.7.1 -- tq_quant_norm: the deterministic per-unit L2 norm reduction.
//
// The first stage of tq_quant_batched (WAVE W20.7, PROPOSAL §3.3's fused
// encode). For each of the M units -- the rows of an (M, K) matrix, K a
// power of two in the family envelope (the production d = 2^19..2^22) --
// compute ||x_m||_2 with turboquant.py:510's `x32.norm()` semantics: the
// sum of squares over the K elements in fp32, then the square root. The
// result lands as an (M,) fp32 CUDA tensor on x's device: each element
// is its unit's norm -- the "0-dim per unit" discipline downstream (the
// tensor math treats them as scalars; TQCodes rev-1 stores each unit's
// norm as a 0-dim fp32 device tensor, and the host NEVER reads the
// values back -- no .item(), no .cpu(), no D2H sync anywhere here; the
// degenerate zero-norm branch is W20.7.4's DEVICE-side concern, never
// the host's -- turboquant.py:511-523's known cost, eliminated by this
// kernel design).
//
// THE REDUCTION STRUCTURE -- the association order IS the contract:
//
//   stage 1   tq_quant_norm_partial_kernel: grid (M, 64), 256 threads.
//   CTA (m, p) owns the CONTIGUOUS element slice [p*chunk, (p+1)*chunk)
//   of unit m (chunk = K/64, exact -- both are powers of two):
//     * per THREAD: the block-stride element map (fht.cuh:90's
//       `for (j = tid; j < n; j += THREADS)` discipline, slice-local) --
//       a pure function of (M, K); no device query feeds the geometry.
//       The squares accumulate in ASCENDING j order, one fmaf per
//       element: the fma intrinsic is ALWAYS the single-rounding fused
//       op, fast-math flags included (unlike div/sqrt); std::fmaf in a
//       CPU twin reproduces the sequence bit-for-bit (the one edge:
//       --use_fast_math also sets ftz, so DENORMAL inputs flush to zero
//       here where torch's kernels keep them -- the N(0,1) production
//       rows never feed it; the differential gate's tolerance owns the
//       edge).
//     * per WARP: the shfl_xor butterfly at FIXED XOR widths
//       16 -> 8 -> 4 -> 2 -> 1. fp32 redux.sync.add does not exist
//       (PTX ISA 9.7.13.12 -- the reduction op is integer-only); the
//       shfl tree IS the fp32 warp path. The widths pin the
//       association: level 1 pairs (i, i^16), level 2 pairs those sums
//       across ^8, down to ^1. Every lane converges to the warp total
//       BIT-IDENTICALLY: at each exchange lane L computes v_L + v_{L^w}
//       while lane L^w computes the same two terms the other way
//       around, and IEEE addition is commutative (a+b and b+a round to
//       the same bits; only ASSOCIATION is order-sensitive -- and the
//       tree's levels are fixed).
//     * per CTA: the 8 warp sums red[0..7] fold in FIXED ASCENDING warp
//       order by thread 0 -- red[0] + red[1] + ... + red[7],
//       left-associated (kernel_gemv_splitk.cu:338-339's red[row][0..3]
//       fixed-order fold, the family precedent's shape).
//     * the CTA's partial lands in its EXCLUSIVELY-OWNED workspace slot
//       partials[p*M + m]: CTA (m, p) is the ONLY writer of that
//       element, and the grid covers every (m, p) exactly once -- the
//       wrapper's torch::empty needs no zeroing (kernel_gemv_splitk.cu
//       :120-126's P discipline).
//
//   stage 2   tq_quant_norm_fold_kernel -- the SECOND-LAUNCH finalizer.
//   One thread per unit folds the 64 partials in FIXED ASCENDING
//   partial order -- ((...((p0 + p1) + p2)...) + p63), left-associated
//   -- and stores __fsqrt_rn(total): the correctly-rounded IEEE root via
//   the _rn intrinsic, the epilogue's one rounding beyond the adds. The
//   intrinsic is REQUIRED: this build compiles with --use_fast_math
//   (setup.py:171), which maps a plain sqrtf to sqrt.approx.f32 (~1-2
//   ulp, deterministic but not correctly rounded); __fsqrt_rn lowers to
//   sqrt.rn.f32 regardless of the fast-math flag -- and torch's own
//   final sqrt is sqrt.rn, so given the same accumulated bits the
//   epilogue rounds IDENTICALLY to the torch path (the only divergence
//   from torch.norm is the sum's association, not the root).
//
// THE FINALIZER CHOICE -- route (b), the second launch, taken
// deliberately. kernel_gemv_splitk.cu:362-407 reduces its split
// partials through an ARRIVAL TICKET: each CTA __threadfence()s, then
// takes one fetch-and-add on a per-tile ticket counter, and the LAST
// arrival folds the workspace BY INDEX (kernel_gemv_splitk.cu:62-76
// documents the split: the ticket is CONTROL FLOW -- it selects WHICH
// CTA folds, never the order the data sums in -- so that reduction is
// deterministic in the data despite the run-dependent arrival order;
// the CUDA threadFenceReduction pattern). That precedent is sound, but
// this file's invariant is stricter: NO reduction primitive of ANY
// kind may appear in kernel_tq.cu (the module grep must stay at zero).
// The norm therefore avoids tickets entirely: the fold is a SECOND
// kernel launch, and the stream's launch boundary -- this family's
// ONLY cross-CTA synchronization (fht_large.cuh:106-108, the pass-2
// discipline) -- orders it after every partial write. Zero tickets,
// zero fences, zero cross-CTA shared state: the fold order is fixed by
// the PROGRAM, not by arrival -- deterministic unconditionally, not
// merely "deterministic in the data".
//
// DETERMINISM CONTRACT: run-to-run torch.equal assertable -- every
// workspace slot has exactly one writer, every fold is a fixed left
// association, the element->thread map is a pure function of (M, K).
// This is NOT a claim of bit-identity with torch.norm: torch's own
// reduction tree carries its own (different) association, so the
// differential gate compares with tolerance while the determinism gate
// compares kernel-to-kernel (two calls, equal bits -- the class of
// contract every decode kernel in this family carries,
// kernel_gemv_splitk.cu:107-114).
//
// FORWARD BUILD (W20.7.2-.4): the boundary ladder scales its 16
// codebook boundaries by norms[m] (the boundary-scaling identity), the
// multi-tile butterfly + bfi packing consume x UNSCALED, and the
// batched entry drives this two-launch chain as its first stage. The
// kernels below take raw pointers so 7.4 can aim the fold at a
// caller-provided out_norms buffer directly; this section's wrapper
// (tq_quant_norm_rows) allocates and returns a fresh (M,) tensor per
// the W20.7.1 brief. Nothing here is bound in bindings.cpp -- W20.7.4
// owns the public tq_quant_batched and its registration.
//
// Resource accounting (sm_86): both stages 256 threads/CTA; stage 1
// grid (M, 64) with 32 B static shared (red[8]); at M = 1 that is 64
// CTAs (80% of the 80 SMs, bandwidth-bound streaming); chunk = K/64 =
// 8192..65536 elements per CTA at K = 2^19..2^22 (32..256 per thread);
// the workspace is (64, M) fp32 = 256 B per unit; stage 2's grid is
// ceil(M/256) CTAs whose warps read one 128 B line per partial index
// (the (64, M) partial-major layout is the coalescing reason).
//
// COMPILE RECORD: nvcc is ABSENT on this authoring box (env/W20_ENV.md)
// -- compile and behavioral verification defer to the A10G box (the
// W20.5.4 TU registration carries this file; W20.6/W20.7.5 run the
// gates). Authored statically, re-read for review; every defect found
// here is a box-side discovery.
// ===========================================================================

namespace {

// The stage geometry: a compile-time constant pair (the launch geometry
// must never query the device -- the determinism contract). 64 partial
// CTAs per unit, 256 threads (8 whole warps) each.
constexpr int kTqNormThreads  = 256;
constexpr int kTqNormPartials = 64;  // partial CTAs per unit (grid.y)

static_assert(kTqNormThreads % 32 == 0,
              "the warp butterfly needs whole 32-lane warps");
static_assert((kTqNormPartials & (kTqNormPartials - 1)) == 0,
              "partials per unit is a power of two (chunk = K/64 exact)");
static_assert(kTqNormPartials <= 65535,
              "the partial axis rides grid.y (its 65535 cap)");

// ---------------------------------------------------------------------------
// Stage 1 -- the partial sum-of-squares (the banner's contract, stage 1).
// ---------------------------------------------------------------------------
template <typename scalar_t>
__global__ void __launch_bounds__(kTqNormThreads)
tq_quant_norm_partial_kernel(
    const scalar_t* __restrict__ x,  // (M, K) input, read-only
    float* __restrict__ partials,    // (kTqNormPartials, M) exclusive slots
    int M, int K, int chunk) {
    const int unit = blockIdx.x;  // grid.x = M units
    const int part = blockIdx.y;  // grid.y = 64 partial CTAs per unit
    const int tid  = threadIdx.x;
    const int64_t base = static_cast<int64_t>(unit) * K
                       + static_cast<int64_t>(part) * chunk;

    // ---- per-thread accumulation: the block-stride map, ascending j --
    // One fmaf per element (single rounding, exactly defined); the
    // sequence is a left fold over the thread's ascending elements. The
    // unroll overlaps LOADS only: every fmaf consumes acc, so the add
    // chain stays sequential in j (the data dependence pins the order
    // regardless of unrolling).
    float acc = 0.0f;
    #pragma unroll 4
    for (int j = tid; j < chunk; j += kTqNormThreads) {
        // the x32 cast (turboquant.py:505): exact widening for
        // fp16/bf16, identity for fp32 -- the square is always fp32.
        const float v = static_cast<float>(x[base + j]);
        acc = fmaf(v, v, acc);
    }

    // ---- the warp butterfly: shfl_xor at fixed widths 16 -> 1 --------
    // (fp32 redux.sync.add does not exist -- PTX ISA 9.7.13.12; the
    // shfl tree IS the fp32 warp path.) All 32 lanes execute this
    // uniform code, so the full mask is legal; every lane ends with
    // the warp total, bit-identical across lanes (commutativity).
    #pragma unroll
    for (int w = 16; w >= 1; w >>= 1) {
        acc += __shfl_xor_sync(0xffffffffu, acc, w);
    }

    // ---- the CTA fold: the warps in fixed ascending warp order ------
    __shared__ float red[kTqNormThreads / 32];
    if ((tid & 31) == 0) {
        red[tid >> 5] = acc;  // lane 0 publishes (every lane holds it)
    }
    __syncthreads();
    if (tid == 0) {
        // red[0] + red[1] + ... + red[7], left-associated ascending --
        // kernel_gemv_splitk.cu:338-339's red[row][0..3] fold's shape.
        float cta = red[0];
        #pragma unroll
        for (int w = 1; w < kTqNormThreads / 32; ++w) {
            cta += red[w];
        }
        // the EXCLUSIVE slot: CTA (unit, part) is the only writer of
        // partials[part*M + unit]; every slot is written exactly once
        // (the grid covers (M, 64) fully), so no zeroing is ever needed.
        partials[static_cast<int64_t>(part) * M + unit] = cta;
    }
    // No trailing barrier: the CTA exits, and the fold launch that
    // reads partials is ordered after it by the stream (this family's
    // only cross-CTA synchronization).
}

// ---------------------------------------------------------------------------
// Stage 2 -- the finalizer fold (the banner's contract, stage 2): one
// thread per unit, the fixed ascending partial order, the root. The
// workspace is partial-major ((64, M), slot = p*M + m) so a warp's 32
// consecutive units read one 128 B line per partial index -- coalesced.
// ---------------------------------------------------------------------------
__global__ void __launch_bounds__(kTqNormThreads)
tq_quant_norm_fold_kernel(
    const float* __restrict__ partials,  // (kTqNormPartials, M)
    float* __restrict__ norms,           // (M,) out
    int M, int num_partials) {
    const int unit = blockIdx.x * kTqNormThreads + threadIdx.x;
    if (unit >= M) {
        return;  // the tail CTA's overrun
    }
    // the fixed ascending partial fold: start at p = 0, left-associate.
    // The unroll again overlaps loads only -- each add consumes acc, so
    // the association order is pinned.
    float acc = partials[unit];
    #pragma unroll 8
    for (int p = 1; p < num_partials; ++p) {
        acc += partials[static_cast<int64_t>(p) * M + unit];
    }
    // __fsqrt_rn: the IEEE correctly-rounded root -- the flag-independent
    // form (a plain sqrtf would lower to sqrt.approx under the build's
    // --use_fast_math, setup.py:171). Deterministic either way; _rn also
    // matches torch's final sqrt round-for-round on the same bits.
    norms[unit] = __fsqrt_rn(acc);
}

// ---------------------------------------------------------------------------
// Argument validation -- the family's loud TORCH_CHECK discipline
// (fht_large_check_args's shape, the norm's own envelope).
// ---------------------------------------------------------------------------
void tq_quant_norm_check_args(const torch::Tensor& x) {
    TORCH_CHECK(x.is_cuda(), "tq_quant_norm: x must be a CUDA tensor");
    TORCH_CHECK(x.dim() == 2,
                "tq_quant_norm: x must be 2-D (M, K) - the batched unit "
                "matrix (the caller flattens (..., K) to rows)");
    TORCH_CHECK(x.is_contiguous(), "tq_quant_norm: x must be contiguous");
    TORCH_CHECK(x.scalar_type() == at::kFloat ||
                x.scalar_type() == at::kHalf ||
                x.scalar_type() == at::kBFloat16,
                "tq_quant_norm: x dtype must be "
                "float32/float16/bfloat16, got ", x.scalar_type());
    TORCH_CHECK(static_cast<int64_t>(x.size(0)) <= INT32_MAX,
                "tq_quant_norm: M must fit a grid.x launch (got M=",
                x.size(0), ")");

    const int K = static_cast<int>(x.size(1));
    TORCH_CHECK(K > 0 && (K & (K - 1)) == 0,
                "tq_quant_norm: K must be a power of two (the tq_quant "
                "family envelope - the FHT rotation the norm feeds "
                "requires it), got K=", K);
    TORCH_CHECK(K >= kTqNormPartials,
                "tq_quant_norm: K=", K, " is below the partial-CTA floor "
                "(", kTqNormPartials, ")");
}

// ---------------------------------------------------------------------------
// tq_quant_norm_rows -- the INTERNAL host wrapper (anonymous namespace;
// W20.7.4 builds the public tq_quant_batched on top of this chain).
// Allocates the (64, M) workspace, launches stage 1 then stage 2 (the
// second-launch finalizer), returns the (M,) fp32 norms on x's device.
// The host NEVER reads the device values: no .item(), no .cpu(), no
// D2H sync (turboquant.py:490-497's encode discipline).
// ---------------------------------------------------------------------------
torch::Tensor tq_quant_norm_rows(const torch::Tensor& x) {
    tq_quant_norm_check_args(x);
    const int M = static_cast<int>(x.size(0));
    const int K = static_cast<int>(x.size(1));
    const int chunk = K / kTqNormPartials;

    // (M,) fp32 on x's device: element m is unit m's norm.
    torch::Tensor norms =
        torch::empty({x.size(0)}, x.options().dtype(at::kFloat));
    if (M == 0) {
        return norms;  // nothing to reduce; a zero-size grid is not a
                       // legal launch (the family's discipline)
    }

    // the partial workspace: (64, M) fp32, wrapper-allocated, one
    // EXCLUSIVELY-OWNED slot per partial CTA (stage 1 writes each slot
    // exactly once; no zeroing). Allocation, both launches, and the
    // scope exit all ride the same stream -- the caching allocator's
    // stream-ordered discipline; a later launch on this stream cannot
    // observe the freed block.
    torch::Tensor partials = torch::empty(
        {static_cast<int64_t>(kTqNormPartials), x.size(0)},
        x.options().dtype(at::kFloat));

    auto stream = at::cuda::getCurrentCUDAStream();  // the house kernels' stream

    // ---- stage 1: the partial sum-of-squares --------------------------
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::kHalf, at::kBFloat16, x.scalar_type(),
        "tq_quant_norm_rows", [&] {
            dim3 grid(M, kTqNormPartials);  // grid.x = units, grid.y = partials
            tq_quant_norm_partial_kernel<scalar_t>
                <<<grid, kTqNormThreads, 0, stream>>>(
                    x.data_ptr<scalar_t>(),
                    partials.data_ptr<float>(), M, K, chunk);
            C10_CUDA_KERNEL_LAUNCH_CHECK();
        });

    // ---- stage 2: the second-launch finalizer (dtype-independent) ----
    const int fold_blocks = (M + kTqNormThreads - 1) / kTqNormThreads;
    tq_quant_norm_fold_kernel
        <<<fold_blocks, kTqNormThreads, 0, stream>>>(
            partials.data_ptr<float>(), norms.data_ptr<float>(),
            M, kTqNormPartials);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return norms;
}

}  // namespace

// ===========================================================================
// W20.7.2 -- the scaled-boundary compare ladder: the encode pass's
// bucketize primitive (PROPOSAL.md:53's "in-register bucketize fused in
// tq_quant (§3.3) -- a <=16-entry compare, no search"; §3.3 step 3's
// ladder -- "the vectorization the paper demands"). Two device
// primitives land here, both consumed by W20.7.3's multi-tile butterfly
// + bfi packing kernel; NOTHING is launched, wrapped, or bound in this
// section (the kernel is 7.3's, the registration 7.4's -- the hand-off
// 7.1's forward-build note already planned, kernel_tq.cu:1467-1475):
//
//   * tq_scale_boundaries_smem -- the boundary-scaling step: the CTA's
//     first B threads scale the codebook's (B,) fp32 boundary table by
//     the unit's norm into shared memory, ONE __fmul_rn multiply per
//     boundary (B <= 15 multiplies -- PROPOSAL.md:152-154's "16
//     multiplies, in smem" capacity at b = 4), ONCE per CTA: the scaled
//     table is shared by every coordinate the CTA encodes.
//   * tq_ladder_index -- the per-coordinate compare ladder: an
//     in-register linear scan of the <= 16 scaled boundaries, the FIRST
//     k with v <= b_scaled[k], else B -- no search structure, no
//     branch tree deeper than the table, uint8 out.
//
// THE IDENTITY -- boundary-scaling identity (PROPOSAL.md §3.3 step 2,
// lines 149-155; turboquant.py:533-536's comment, the W20.3.1 encode
// rewrite that first folded it in): the paper's encode is
// argmin_k|y_j/norm - c_k|; multiplying every comparison by norm > 0
// gives argmin_k|y_j - norm*c_k| -- IDENTICAL bucket decisions
// (monotone, exact). The rotated coordinate y_j therefore arrives
// UNSCALED (the FHT is linear, turboquant.py:525-529: fht(x) =
// norm*fht(x/norm), so 7.3's butterfly output carries the norm) and
// the boundaries scale instead: THE NORMALIZED VECTOR IS NEVER
// MATERIALIZED -- no x/norm pass, no y/norm pass, no division ANYWHERE
// in this section; the whole cost of the normalization is B multiplies
// in smem. n > 0 is the CALLER's contract (the identity's
// precondition -- the degenerate zero-norm unit is W20.7.4's
// DEVICE-side branch, turboquant.py:511-523's host branch eliminated
// by the kernel design; it never reaches the ladder).
//
// THE CARDINALITY -- codebooks.py:219-228, the Codebook contract:
// bits b in {3, 4} (the 3.5-bit recipe's fixed 50/50 coordinate
// split), centroids (2**b,) ascending symmetric +/-, boundaries
// (2**b - 1,) ascending Voronoi edges (boundaries[i] =
// (centroids[i] + centroids[i+1]) / 2). So B = 2^b - 1: 7 boundaries
// for the b=3 lo half, 15 for the b=4 hi half; the ladder's index
// range is 0..B -- B + 1 = 2^b cells, one per centroid: the uint8
// output's domain is exactly Codebook.encode's (codebooks.py:237-240).
// PROPOSAL.md:158's "2^b <= 16" counts the LEVELS; the boundary table
// that partitions them is B = 2^b - 1 -- the ladder scans the
// boundaries, the index names the cell. tq_boundary_count pins the
// formula in one place; kTqLadderMaxB is the 7.3 smem array capacity.
//
// THE INDEX MATH -- searchsorted parity, the tie direction pinned.
// The torch path is torch.searchsorted(cb.boundaries * norm, y,
// right=False) (turboquant.py:542-545); the numpy original is
// np.searchsorted(boundaries, x) -- side='left' (codebooks.py:239);
// turboquant.py:539-541 pins the equivalence ("leftmost insertion
// point; cell i = (boundaries[i-1], boundaries[i]]", the W16-verified
// CPU/GPU parity; the frozen fixtures pin the same behavior,
// turboquant.py:146-149). side='left' returns the count of boundaries
// STRICTLY LESS than v -- equivalently the FIRST index i with
// b[i] >= v. The ladder is that definition, linearized over the
// ascending table:
//     idx = B;  for k in 0..B-1:  if (v <= b_scaled[k]) { idx = k; break; }
// The comparison direction is v <= b_scaled[k] -- a value landing
// exactly ON a boundary takes that boundary's OWN index (v ==
// b_scaled[k] is true at k: leftmost insertion, the right-closed cell
// (b[k-1], b[k]]); v strictly above every boundary returns B, the top
// overflow cell -- searchsorted's len(a) return. ASCENDING b is the
// precondition (codebooks.py:445-446 validates strict ascent at
// load): the linear scan is exact because the table is sorted.
//
// TIE SEMANTICS -- the level-2 acceptance, documented where it lives.
// The COMPARISONS are exact (fp32 <= is exact; no rounding rides the
// ladder itself), and given the same norm bits and the same y bits the
// ladder's index is EXACTLY torch.searchsorted's: __fmul_rn(b[k], n)
// is the same single round-to-nearest fp32 multiply torch's
// `boundaries * norm` performs (mul.rn), and the compares are exact
// both sides. The divergence is entirely in the INPUT bits: the
// norm's own bits differ at the association level (7.1's banner: the
// shfl tree's fixed association vs torch.norm's tree -- tolerance-
// gated, never bit-equal, kernel_tq.cu:1457-1465), and y_j's bits
// ride the FHT passes' numerics (the same class). A coordinate within
// an ulp of a scaled boundary can therefore flip to the adjacent cell
// vs the torch path -- the boundary-exact flip class the G20.P3 gate
// counts (TASKS.md:1382: TQ kernel parity, G20.P1 thresholds on
// kernel outputs), level-2 accepted at ties <= 0.01% -- the same
// exposure the torch path itself carries across its own norm and
// rotation rounding. Production inputs are N(0,1) rows: the mass
// within an ulp of a boundary is vanishingly small, and the gate
// MEASURES it rather than assumes it. Finite y_j is the envelope: a
// NaN compares false at every rung and falls out the top (numpy's
// own searchsorted leaves NaN placement undefined -- out of contract
// on both paths, never a production input).
//
// Numerics scope: __fmul_rn is REQUIRED for the same reason 7.1 chose
// __fsqrt_rn (kernel_tq.cu:1594-1597): this build compiles with
// --use_fast_math (setup.py:171), the _rn intrinsics lower to the
// correctly-rounded IEEE op regardless of the flag, and a bare
// `a * b` could contract with a neighboring add under fast-math (fma
// contraction) -- _rn pins the ONE multiply the identity performs.
// The scaling step is a predicated store + __syncthreads: no
// reduction primitive of ANY kind appears in this section (the module
// grep stays at zero, the 7.1 invariant, kernel_tq.cu:1447-1448) --
// the ladder is per-thread control flow over shared read-only data.
//
// THE 7.3 CALL PATTERN (the contract these primitives pin):
//     __shared__ float b_lo[kTqLadderMaxB];  // 7 live slots (b=3)
//     __shared__ float b_hi[kTqLadderMaxB];  // 15 live slots (b=4)
//     const float n = norms[unit];  // 7.1's (M,) output: ONE broadcast
//                                   // read per CTA (every thread loads
//                                   // the same address; the sector is
//                                   // served once) -- the VALUE feeds
//                                   // the ladder, never re-reduced
//     tq_scale_boundaries_smem(cb_lo_boundaries, n, B_lo, b_lo);
//     tq_scale_boundaries_smem(cb_hi_boundaries, n, B_hi, b_hi);
//     ... 7.3's multi-tile butterfly leaves v in a register ...
//     const uint8_t idx = (j < n_lo) ? tq_ladder_index(v, b_lo, B_lo)
//                                    : tq_ladder_index(v, b_hi, B_hi);
// (a CTA whose tile lies wholly in one half scales only that half's
// table -- the 3.5-bit split never forces both).
//
// Resource accounting (sm_86): the ladder adds B fp32 of static
// shared memory per CTA (28 B at b=3, 60 B at b=4) beside 7.3's
// 64 KiB tile staging -- a rounding error. Every CTA of a unit
// re-scales its own copy: the <= 16-entry re-scale per (unit, tile)
// is the price of PROPOSAL.md:156-165's tile independence -- tiles
// are disjoint, no cross-tile dependency, therefore no cross-CTA
// sharing of the scaled table and no sync beyond the CTA's own
// barrier.
//
// COMPILE RECORD: nvcc is ABSENT on this authoring box (env/W20_ENV.md)
// -- compile and behavioral verification defer to the A10G box (the
// W20.5.4 TU registration carries this file; W20.6/W20.7.5 run the
// gates). Authored statically, re-read for review; every defect found
// here is a box-side discovery.
// ===========================================================================

namespace {

// ---------------------------------------------------------------------------
// The cardinality, pinned in ONE place: B = 2^bits - 1 (codebooks.py:220,
// boundaries shape (2**bits - 1,)). Host+device: the 7.3 kernel's runtime
// B and the 7.4 wrapper's TORCH_CHECKs both derive from this one line.
// ---------------------------------------------------------------------------
__host__ __device__ constexpr int tq_boundary_count(int bits) {
    return (1 << bits) - 1;
}

// The ladder's compile-time capacity: the b=4 table. 7.3 declares its
// shared scaled-boundary arrays [kTqLadderMaxB] regardless of b -- the
// b=3 half uses the first 7 slots.
constexpr int kTqLadderMaxB = tq_boundary_count(4);

static_assert(tq_boundary_count(3) == 7, "the b=3 (lo half) boundary count");
static_assert(kTqLadderMaxB == 15, "the b=4 (hi half) boundary count");
static_assert(kTqLadderMaxB < 32,
              "the scaling step rides the CTA's first warp (tid < B)");

// ---------------------------------------------------------------------------
// The boundary-scaling step -- PROPOSAL.md §3.3 step 2's "16 multiplies,
// in smem". The CTA's first B threads each scale ONE boundary by the
// unit's norm; the barrier publishes the table to the whole CTA.
//
// CONTRACT (the caller's, all of it):
//   * EVERY thread of the CTA calls this -- the __syncthreads() is
//     cooperative (the standard smem-staging discipline; a divergent
//     call is UB, exactly like every other barrier in this family).
//   * blockDim.x >= B (the family geometry is 256 threads, B <= 15).
//   * n > 0 -- the identity's precondition; the degenerate zero-norm
//     unit is 7.4's device-side branch and never reaches here.
//   * boundaries[0..B-1] ASCENDING fp32 in global memory (the
//     codebooks.py:445-446-validated table via the 2.2 device mirror,
//     turboquant.py:556-570).
//   * smem_scaled has capacity >= B (declare [kTqLadderMaxB]).
// One __fmul_rn per boundary: the correctly-rounded single multiply --
// the same round torch's `boundaries * norm` performs, so given the
// same norm bits the scaled table is BIT-IDENTICAL to the torch
// path's (the tie banner above owns the norm-bits divergence).
// ---------------------------------------------------------------------------
__device__ __forceinline__ void tq_scale_boundaries_smem(
    const float* __restrict__ boundaries,  // global: (B,) ascending, unscaled
    float n,                               // the unit's norm, n > 0
    int B,                                 // 2^bits - 1 (7 or 15)
    float* smem_scaled) {                  // shared: [>= B] out
    if (threadIdx.x < B) {
        smem_scaled[threadIdx.x] = __fmul_rn(boundaries[threadIdx.x], n);
    }
    __syncthreads();  // the scaled table is CTA-visible from here on
}

// ---------------------------------------------------------------------------
// The compare ladder -- the per-coordinate bucketize (PROPOSAL.md:53's
// "a <=16-entry compare, no search"; §3.3 step 3's "in-register linear
// scan of the <=16 scaled boundaries ... the vectorization the paper
// demands"). np.searchsorted(side='left') == torch.searchsorted(
// right=False), linearized over the ASCENDING scaled table:
//     idx = B;  for k in 0..B-1:  if (v <= b_scaled[k]) { idx = k; break; }
// -- the FIRST k with b_scaled[k] >= v, else B. The tie direction is
// v <= b_scaled[k]: a boundary-exact v takes that boundary's OWN index
// (leftmost insertion, cell i = (b[i-1], b[i]], turboquant.py:539-541);
// v above every boundary takes the top overflow cell B. v arrives
// UNSCALED -- the identity: the norm lives in the boundaries, never
// in the coordinate. The scan runs entirely in registers: v and the
// emitted idx never touch memory, and b_scaled is read-only shared
// (or a register array the caller built -- the pointer serves both).
// ---------------------------------------------------------------------------
__device__ __forceinline__ uint8_t tq_ladder_index(
    float v,                      // the UNSCALED rotated coordinate (finite)
    const float* b_scaled,        // ASCENDING, norm-scaled boundaries
    int B) {                      // 2^bits - 1 (7 or 15)
    int idx = B;  // the overflow default: searchsorted's len(a) return
                  // (v above every boundary -> the top cell)
    for (int k = 0; k < B; ++k) {
        if (v <= b_scaled[k]) {  // >= with the tie on the boundary's OWN
                                 // side: leftmost insertion
            idx = k;
            break;               // the FIRST hit only -- the table is
                                 // ascending, the later rungs are moot
        }
    }
    return static_cast<uint8_t>(idx);
}

}  // namespace

// ===========================================================================
// W20.7.3 -- tq_quant_ladder_pack: the ladder + packing pass, the encode
// chain's third and final kernel stage (PROPOSAL.md:156-165, §3.3 step 3;
// TASKS.md:1010-1028). Per (unit, tile): 7.2's smem boundary scaling, the
// per-coordinate compare ladder, and the b-bit index packing into the
// packed uint8 code streams -- the "syndrome: np.packbits" unit: the
// legacy array era's np.packbits call (and W20.3.2's torch-native
// pack_bits_t) become one fused device pass over the ROTATED coordinates.
//
// THE COMPOSITION DECISION -- batched butterfly reuse, not a re-inline.
// PROPOSAL.md:156-157 sketches step 3 as "the multi-tile butterfly (§3.2
// inline), then per coordinate the ladder, then pack" -- one fused
// per-unit encode kernel. It is realized here as an ORCHESTRATION, and
// the split is deliberate:
//   * 5.2's two-pass butterfly is ALREADY (M, K)-batched: pass 1's grid
//     is (M rows, K/16384 tiles) -- fht_large_local_kernel,
//     kernel_tq.cu:203-205 -- and pass 2's pair space is the whole
//     (M, K/2) -- fht_large_cross_kernel, kernel_tq.cu:269-276. The 7.4
//     entry drives those EXACT launches over the prefill's (M, K) unit
//     matrix and hands this kernel the rotated y; the "inlined
//     butterfly" of §3.3 step 3 is realized as that batched reuse plus
//     THIS fused ladder/pack pass -- the fusion that carries the paper's
//     vectorization claim (the ladder + pack run per coordinate in
//     registers; no intermediate index tensor is ever materialized).
//   * Why not literally re-inline the butterfly into this kernel: the
//     cross stages NEED kernel-launch boundaries -- they are this
//     family's ONLY cross-CTA synchronization (fht_large.cuh:106-108) --
//     so one fused kernel would have to re-invent that ordering as
//     persistent-CTA machinery (tickets, fences) that 7.1's banner
//     explicitly bans from this file (kernel_tq.cu:1447-1455); and
//     duplicating the stage arithmetic + cp.async staging into a second
//     copy would fork the fixture-pinned butterfly (W20.5.2's gates own
//     exactly ONE copy of it). The price of the reuse is ONE extra fp32
//     read pass of y (64 KiB per tile) -- a fraction of the butterfly's
//     own (log2 K - 12) fp32 round-trips of the row
//     (kernel_tq.cu:115-120).
//   * The chain count still lands where the proposal wants it: ONE host
//     entry, ONE stream-ordered chain per prefill (the norm launches +
//     the butterfly's launches + this pack launch) -- PROPOSAL.md:189-
//     193's "two kernel chains per prefill instead of ~4,000 host round
//     trips" (the encode chain here; the decode chain is W20.8's).
//   * The math ordering is preserved exactly: the FULL butterfly (local
//     + cross) completes BEFORE the ladder -- quant's bucketize consumes
//     the FINAL rotated coordinates (turboquant.py:530-545: y =
//     fht.fht_apply(...), then searchsorted over y), and the stream's
//     launch boundaries order that for free.
//
// THE LAYOUT PIN -- the byte layout is IDENTICAL to today's pack_bits
// (turboquant.py:212-245, pack_bits_t, the W20.3.2 canonical) --
// fixture-pinned: G20.P1's frozen fixtures are the pinning gate, G20.P3's
// kernel-parity gate counts divergences (the same pin as W20.3.2,
// restated here per the 7.3 invariant). Bit-precise, all four clauses:
//   * bit order: LITTLE-ENDIAN -- bit k of index i sits at flat stream
//     position i*bits + k (each index's LSB first);
//   * byte assembly: byte m holds flat bits 8m..8m+7 with flat bit 8m in
//     the LSB (bytes LSB-first); stream length = ceil(n*bits/8) =
//     tq_packed_len(n, bits) -- turboquant.py:998's _packed_len,
//     replicated bit-for-bit below;
//   * tail padding: the final partial byte's unused high flat bits are
//     zero;
//   * the b=4 nibble fast path (byte j = idx[2j] | (idx[2j+1] << 4),
//     the earlier index in the LOW nibble) and the b=3 general path
//     (turboquant.py:256-262 vs :263-273) COINCIDE under the word
//     formulation below -- the torch path's two paths are an
//     optimization, never a semantic fork.
//
// THE PACKING FORMULATION -- one uint64 word per thread. Each thread
// owns a fixed 16-coordinate slice of its tile (PROPOSAL.md:159; 16 =
// FHT_LARGE_TILE / FHT_LARGE_THREADS); the slice's 16 laddered indices
// fold into ONE uint64:
//     word = sum_c idx[c] << (c * bits)
// -- which IS the flat-stream segment [i0*bits, (i0+16)*bits) read as a
// little-endian integer: flat position i0*bits + c*bits + k lands at
// word bit c*bits + k, and byte w of the run is word >> (8*w).
// PROPOSAL.md:160-161 cites PTX's bfi (PTX ISA 9.7.1.20) for the field
// inserts; this implementation uses the plain C++ shift/mask
// OR-accumulation -- equally exact (pure integer ops, nothing to round),
// one statement per property to verify by eye, and the compiler is free
// to fuse the inserts; a hand-bfi.b32 version would split the 48-bit
// b=3 run across two 32-bit halves for zero determinism benefit (the
// layout is bits, not instructions). THE SPLIT: the lo/hi halves pack
// into SEPARATE streams with their own bits and their own lengths
// (TQCodes' idx_lo/idx_hi, turboquant.py:549-550) -- per half, the
// coordinate's half-LOCAL index i (lo: i = j; hi: i = j - n_lo) drives
// the flat arithmetic; the entry (7.4) passes the split geometry (n_lo,
// bits_lo, n_hi, bits_hi), exactly the signature sketch 7.4 binds.
//
// THE OWNERSHIP MAP -- no cross-tile dependencies (the 7.3 invariant, as
// a construction, not a hope): 16*bits is a whole number of bytes for
// both TQ widths (static_asserts below), so a slice's flat segment is
// BYTE-ALIGNED and spans EXACTLY 6 (b=3) or 8 (b=4) whole bytes; the
// slices partition each half's index space, and tiles are disjoint
// contiguous coordinate ranges -- therefore every output byte has
// EXACTLY ONE owning thread, no byte is shared across threads, tiles, or
// units, and no synchronization of ANY kind rides the packing (the
// single-writer discipline the whole file carries; the module grep for
// reduction primitives stays at zero). The grid-stride loop below
// (R20.6's overshoot guard, for 7.4's capped grids) keeps it: a byte's
// VALUE is a pure function of (y, norms, boundaries, geometry) -- which
// CTA computes it changes nothing, so run-to-run torch.equal holds for
// any grid the entry chooses.
//
// THE ALIGNMENT CONTRACT: the 16-slice ownership needs every slice
// wholly inside ONE half -- the hi half's slice bases anchor at n_lo --
// so n_lo must be a multiple of 16. The launcher REFUSES misaligned
// splits LOUDLY (a TORCH_CHECK, never a silently mis-packed stream); the
// realized production geometry always satisfies it (group=1, K =
// 2^15..2^22, frac in {0, .25, .5, .75} -> n_hi = round(frac*K) is a
// multiple of 16), and an exotic frac (e.g. bits=3.1) falls back to the
// torch twin -- the auto-backend discipline (fht.py's
// kernel-when-eligible, torch-otherwise, lifted to the TurboQuant level
// by 7.4's entry).
//
// THE CANARY NOTE: FLUTE_TQ_CANARY does not extend to the codes -- the
// fht_large control poisons fp outputs with a NaN no finite result can
// equal, but the packed byte space is FULLY reachable (any uint8 is a
// legal packing of b-bit indices), so no poison value exists; this
// pass's coverage control is the ownership map itself (every byte of the
// (M, L) batches written exactly once) plus W20.6's parity gates on the
// values.
//
// WHAT 7.4 OWES (the binding surface: raw pointers in, tensors out):
//   * y: the butterfly's fp32 (M, K) output (5.2's forward epilogue --
//     signs and 1/sqrt(K) INCLUDED; the norm NEVER divided out, 7.2's
//     identity), contiguous, row m = unit m;
//   * norms: 7.1's (M,) fp32, norms[m] > 0 (the zero-norm degenerate is
//     7.4's device-side branch, never this kernel's -- 7.2's banner);
//   * boundaries_{lo,hi}: the 2.2 device mirrors' fp32 tables, ASCENDING,
//     lengths 2^bits - 1 (tensor-level asserts are 7.4's; this launcher
//     takes raw pointers and checks the integer geometry only);
//   * codes_{lo,hi}: (M, tq_packed_len(n_half, bits_half)) contiguous
//     uint8 batches, row m = unit m's packed half-stream -- 7.4 slices
//     the rows into TQCodes' per-unit 1-D streams (zero-copy views); the
//     EMPTY half (uniform bits: n_hi = 0, a (M, 0) tensor whose
//     data_ptr may be null) is legal and never dereferenced. (The
//     sketch's single out_codes pointer widens to codes_lo + codes_hi:
//     the halves are separate tensors end-to-end with different lengths;
//     two passed pointers verify better than one base plus two derived
//     offsets.)
//
// Resource accounting (sm_86): 1024 threads (FHT_LARGE_THREADS), grid =
// M * K/16384 CTAs (6,144 at the 2^19 / 192-unit prefill -- PROPOSAL
// §3.6's ~6,100), 120 B of STATIC shared memory (the two scaled tables),
// no dynamic shared memory (no cudaFuncSetAttribute opt-in -- the 64 KiB
// request is 5.2's, not this pass's), ~64 KiB of y read + 6-8 KiB of
// codes written per tile, 16 fp32 + one uint64 in flight per thread.
//
// COMPILE RECORD: nvcc is ABSENT on this authoring box (env/W20_ENV.md)
// -- compile and behavioral verification defer to the A10G box (the
// W20.5.4 TU registration carries this file; W20.6/W20.7.5 run the
// gates). Authored statically, re-read for review; every defect found
// here is a box-side discovery.
// ===========================================================================

namespace {

// ---------------------------------------------------------------------------
// The packed-stream length, pinned in ONE place: turboquant.py:998's
// _packed_len(n, bits) = (n*bits + 7)//8, replicated bit-for-bit in C++
// (Python floor division on non-negative ints == C++ integer division).
// Host+device: the kernel derives its byte bounds and 7.4's wrapper
// derives its tensor shapes from this one line.
// ---------------------------------------------------------------------------
__host__ __device__ constexpr int tq_packed_len(int n, int bits) {
    return (n * bits + 7) / 8;
}

static_assert(tq_packed_len(16, 3) == 6,
              "the b=3 16-coordinate slice packs to 6 bytes (PROPOSAL 3.3)");
static_assert(tq_packed_len(16, 4) == 8,
              "the b=4 16-coordinate slice packs to 8 bytes (PROPOSAL 3.3)");

// The per-thread coordinate slice: PROPOSAL.md:159's "fixed 16-coordinate
// slice" = the tile / the CTA width (16384 / 1024). 16 * bits is a whole
// number of bytes for both TQ widths -- the property that makes every
// output byte single-owned (the ownership map in the banner).
constexpr int kTqPackSlice = flute::FHT_LARGE_TILE / flute::FHT_LARGE_THREADS;

static_assert(kTqPackSlice == 16, "PROPOSAL.md:159's 16-coordinate slice");
static_assert((kTqPackSlice * 3) % 8 == 0,
              "the b=3 slice is byte-aligned in the flat stream");
static_assert((kTqPackSlice * 4) % 8 == 0,
              "the b=4 slice is byte-aligned in the flat stream");
static_assert(kTqPackSlice * 4 <= 64,
              "the slice's indices fold into ONE uint64 word");

// ---------------------------------------------------------------------------
// tq_quant_ladder_pack_kernel -- the ladder + packing pass. One CTA per
// (unit, tile), grid-stride over the flattened M * (K/16384) space (the
// R20.6 guard); per CTA: the unit's norm (ONE broadcast read), 7.2's smem
// boundary scaling, then each thread's 16-coordinate slice: ladder ->
// indices -> the 6/8-byte packed word -> its EXCLUSIVE byte run.
// ---------------------------------------------------------------------------
__global__ void __launch_bounds__(flute::FHT_LARGE_THREADS)
tq_quant_ladder_pack_kernel(
    // (M, K) fp32: the units' ROTATED rows -- 5.2's forward epilogue
    // output (signs + 1/sqrt(K) INCLUDED; the norm NEVER divided out,
    // 7.2's identity), contiguous, row m = unit m
    const float* __restrict__ y,
    // (M,) fp32: 7.1's output; norms[unit] > 0 (7.4's contract)
    const float* __restrict__ norms,
    // (2^bits_lo - 1,) fp32 ASCENDING, UNSCALED (the 2.2 device mirror)
    const float* __restrict__ boundaries_lo,
    // (2^bits_hi - 1,) fp32 ASCENDING, UNSCALED
    const float* __restrict__ boundaries_hi,
    // (M, tq_packed_len(n_lo, bits_lo)) uint8: row m = unit m's lo stream
    uint8_t* __restrict__ codes_lo,
    // (M, tq_packed_len(n_hi, bits_hi)) uint8: row m = unit m's hi stream
    uint8_t* __restrict__ codes_hi,
    int M, int K,            // units; coords (a power of two > 16384)
    int n_lo, int bits_lo,   // the lo half: coords [0, n_lo) at bits_lo
    int n_hi, int bits_hi,   // the hi half: coords [n_lo, K) at bits_hi
    int tile_shift) {        // log2(K) - 14; tiles per row = 1 << tile_shift
    const int tiles = 1 << tile_shift;
    const int64_t total_ctas = static_cast<int64_t>(M) * tiles;
    const int64_t grid_stride = static_cast<int64_t>(gridDim.x);

    // 7.2's smem scaled-boundary tables (the call pattern 7.2's banner
    // pinned). The scaling conditions -- n_lo > 0 / n_hi > 0 -- are
    // GRID-UNIFORM kernel parameters, invariant across the grid-stride
    // loop, so the cooperative barrier inside the helper is uniform at
    // every call. (7.2's banner sketched per-tile conditions; the
    // grid-uniform form wastes <= 22 multiplies on a wholly-one-half
    // tile -- noise against the tile's 16384 loads -- and keeps the
    // barrier conditions iteration-invariant: one less per-iteration
    // property to re-verify.)
    __shared__ float b_lo[kTqLadderMaxB];  // 7 live slots at b=3
    __shared__ float b_hi[kTqLadderMaxB];  // 15 live slots at b=4
    const int B_lo = tq_boundary_count(bits_lo);
    const int B_hi = tq_boundary_count(bits_hi);
    const int lo_bytes = tq_packed_len(n_lo, bits_lo);
    const int hi_bytes = tq_packed_len(n_hi, bits_hi);

    for (int64_t cta = static_cast<int64_t>(blockIdx.x);
         cta < total_ctas; cta += grid_stride) {
        // unit-major flat map: consecutive CTAs walk one unit's tiles;
        // tiles is a power of two, so the split is shift/mask only.
        const int unit = static_cast<int>(cta >> tile_shift);
        const int tile = static_cast<int>(cta & (tiles - 1));
        const int tile_start = tile * flute::FHT_LARGE_TILE;

        // ONE broadcast read per CTA-visit: every thread loads the same
        // (M,) element (the sector is served once); the VALUE feeds the
        // ladder, never re-reduced (7.2's call pattern).
        const float nrm = norms[unit];
        if (n_lo > 0) {
            tq_scale_boundaries_smem(boundaries_lo, nrm, B_lo, b_lo);
        }
        if (n_hi > 0) {
            tq_scale_boundaries_smem(boundaries_hi, nrm, B_hi, b_hi);
        }

        // ---- this thread's slice: 16 contiguous coords, ONE half -----
        // j0 < K always (the tile lives inside the row); the alignment
        // contract (launcher-checked: n_lo % 16 == 0) guarantees no
        // slice straddles the split, so the WHOLE slice ladders against
        // ONE table and packs into ONE contiguous byte run.
        const int j0 = tile_start + threadIdx.x * kTqPackSlice;
        const bool hi = j0 >= n_lo;
        const int bits = hi ? bits_hi : bits_lo;
        const int n_half = hi ? n_hi : n_lo;       // the half's coord count
        const int B = hi ? B_hi : B_lo;            // its boundary count
        const float* tbl = hi ? b_hi : b_lo;       // its SCALED table (smem)
        const int i0 = hi ? (j0 - n_lo) : j0;  // slice base, half-LOCAL (a multiple of 16)
        if (i0 < n_half) {  // vacuous in the checked geometry (j0 <= K-16
                            // and the half selection pin i0 below its
                            // half's end) -- kept as the boundary belt
            const int64_t ybase = static_cast<int64_t>(unit) * K + j0;
            // coords this slice OWNS. Under the alignment contract (n_lo
            // a multiple of 16) every slice of BOTH halves is full -- each
            // half's index range is an exact 16-slice multiple, so the
            // packed streams come out whole-byte (no partial byte at
            // all). The rem/clip machinery is the general-layout belt,
            // kept so the pin's tail clauses (the zero pad, the clipped
            // overrun) are enforced by the code that owns them should
            // the contract ever be relaxed.
            const int rem = n_half - i0;
            const int cnt = (rem < kTqPackSlice) ? rem : kTqPackSlice;

            // ---- the ladder (7.2): 16 coords -> 16 uint8 cells --------
            // v arrives UNSCALED; the norm lives in the boundaries (the
            // identity). The tail slice's phantom coords never ladder --
            // their flat bits are the stream's zero pad (the pin's tail
            // clause), so they contribute nothing to the word (cnt < 16
            // keeps word's high fields at zero).
            uint64_t word = 0;
            #pragma unroll
            for (int c = 0; c < kTqPackSlice; ++c) {
                if (c < cnt) {
                    const uint8_t idx =
                        tq_ladder_index(y[ybase + c], tbl, B);
                    word |= static_cast<uint64_t>(idx) << (c * bits);
                }
            }

            // ---- the pack: the slice's 6 (b=3) / 8 (b=4) bytes -------
            // word IS the flat-stream segment [i0*bits, (i0+16)*bits)
            // as a little-endian integer; byte w of the run is
            // word >> (8*w), stored to this thread's EXCLUSIVE range
            // [byte0, byte0 + n_bytes). A full slice's range ends at or
            // below the stream length; a partial slice's overrun bytes
            // (a belt case -- whole-byte in the checked envelope) are
            // exactly the stream's end, clipped here, and their phantom
            // flat bits were never set: the zero tail.
            uint8_t* out = (hi ? codes_hi : codes_lo)
                + static_cast<int64_t>(unit) * (hi ? hi_bytes : lo_bytes);
            const int byte0 = (i0 * bits) >> 3;   // 16*bits % 8 == 0 -> integral
            const int n_bytes = (kTqPackSlice * bits) >> 3;  // 6 or 8
            for (int w = 0; w < n_bytes; ++w) {
                const int byte_idx = byte0 + w;
                if (byte_idx < (hi ? hi_bytes : lo_bytes)) {
                    out[byte_idx] =
                        static_cast<uint8_t>(word >> (8 * w));
                }
            }
        }

        // ---- the table-lifetime barrier --------------------------------
        // The next iteration may carry a DIFFERENT unit -> a different
        // norm -> different scaled tables, and the scaling helper
        // writes b_lo/b_hi BEFORE its own barrier: this trailing
        // __syncthreads() orders every thread's table READS of this
        // iteration before the next iteration's writes. The cta
        // sequence is CTA-uniform, so the trip count -- and every
        // barrier here -- is cooperative.
        __syncthreads();
    }
}

// ---------------------------------------------------------------------------
// tq_quant_ladder_pack_run -- the INTERNAL raw-pointer launcher (anonymous
// namespace; 7.4 binds the tensors and owns the tensor-level asserts -- y
// fp32 (M, K) contiguous CUDA, norms (M,) fp32, boundary tables
// (2^bits-1,) fp32 ASCENDING, codes (M, L) uint8 contiguous -- plus the
// output allocation; this launcher checks the INTEGER geometry only,
// loudly). The host never touches device memory content: no readback, no
// D2H sync anywhere (turboquant.py:490-497's encode discipline).
// ---------------------------------------------------------------------------
void tq_quant_ladder_pack_run(
    const float* y, const float* norms,
    const float* boundaries_lo, const float* boundaries_hi,
    uint8_t* codes_lo, uint8_t* codes_hi,
    int M, int K, int n_lo, int bits_lo, int n_hi, int bits_hi,
    cudaStream_t stream) {
    // the family envelope: the (M, K)-batched butterfly upstream (5.2)
    // owns exactly this K, and the tile arithmetic below rides it.
    TORCH_CHECK(K > flute::FHT_LARGE_TILE && (K & (K - 1)) == 0,
                "tq_quant_ladder_pack: K must be a power of two > 16384 "
                "(the fht_large envelope the encode chain rides), got K=",
                K);
    TORCH_CHECK(M > 0,
                "tq_quant_ladder_pack: M must be > 0 (a zero-size grid is "
                "not a legal launch -- the M == 0 branch is 7.4's)");
    TORCH_CHECK(bits_lo == 3 || bits_lo == 4,
                "tq_quant_ladder_pack: bits_lo must be 3 or 4 (the TQ "
                "index stream widths, codebooks.py:219-228), got ",
                bits_lo);
    TORCH_CHECK(bits_hi == 3 || bits_hi == 4,
                "tq_quant_ladder_pack: bits_hi must be 3 or 4 (the TQ "
                "index stream widths, codebooks.py:219-228), got ",
                bits_hi);
    TORCH_CHECK(n_lo >= 0 && n_hi >= 0 && n_lo + n_hi == K,
                "tq_quant_ladder_pack: the lo/hi split must partition K "
                "(n_lo=", n_lo, ", n_hi=", n_hi, ", K=", K, ")");
    // THE ALIGNMENT CONTRACT, checked loudly (the banner's reasoning):
    // misaligned splits are REFUSED, never silently mis-packed; the
    // caller (7.4) routes them to the torch twin.
    TORCH_CHECK(n_lo % kTqPackSlice == 0,
                "tq_quant_ladder_pack: n_lo must be a multiple of ",
                kTqPackSlice, " (the 16-coordinate slice ownership; got "
                "n_lo=", n_lo, ") -- misaligned splits stay on the torch "
                "twin");
    // raw-pointer hygiene: a null pointer where the half is live is a
    // binding bug, not a silent device fault.
    TORCH_CHECK(y != nullptr && norms != nullptr,
                "tq_quant_ladder_pack: y and norms must be live device "
                "pointers");
    TORCH_CHECK(n_lo == 0 ||
                (boundaries_lo != nullptr && codes_lo != nullptr),
                "tq_quant_ladder_pack: the live lo half needs its "
                "boundary table and code stream pointers");
    TORCH_CHECK(n_hi == 0 ||
                (boundaries_hi != nullptr && codes_hi != nullptr),
                "tq_quant_ladder_pack: the live hi half needs its "
                "boundary table and code stream pointers");

    // the launch geometry: a pure function of (M, K) -- no device query
    // feeds it (the determinism contract). tiles = K/16384 (a power of
    // two, K checked above); tile_shift = log2(K) - 14, so the kernel's
    // unit-major shift/mask map is exact -- checked, not assumed (the
    // "constructible by arithmetic, not faith" discipline,
    // fht_large_check_args's grid hygiene).
    const int tiles = K / flute::FHT_LARGE_TILE;
    const int tile_shift = flute::fht_large_log2_floor(K)
                         - flute::fht_large_log2_floor(flute::FHT_LARGE_TILE);
    TORCH_CHECK((1 << tile_shift) == tiles,
                "tq_quant_ladder_pack: the shift/mask (unit, tile) map "
                "must be exact (tile_shift=", tile_shift, ", tiles=",
                tiles, ")");
    const int64_t total_ctas = static_cast<int64_t>(M) * tiles;
    TORCH_CHECK(total_ctas <= INT32_MAX,
                "tq_quant_ladder_pack: M*tiles = ", total_ctas,
                " exceeds the grid.x launch bound");

    // the FULL grid by default (every CTA walks exactly one (unit,
    // tile)); the kernel's grid-stride loop is the R20.6 guard that lets
    // 7.4 cap this without touching the kernel -- the byte values are
    // grid-independent (the ownership map in the banner).
    const int blocks = static_cast<int>(total_ctas);
    tq_quant_ladder_pack_kernel
        <<<blocks, flute::FHT_LARGE_THREADS, 0, stream>>>(
            y, norms, boundaries_lo, boundaries_hi, codes_lo, codes_hi,
            M, K, n_lo, bits_lo, n_hi, bits_hi, tile_shift);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace

// ===========================================================================
// W20.7.4 -- tq_quant_batched: the batched encode entry + the zero-norm
// sweep. TASKS.md:1044-1061; PROPOSAL.md:187-193 (§3.6) and §3.3's step
// orchestration. The prefill's ONE host call.
//
// THE ORCHESTRATION -- two kernel chains per prefill, not ~4,000 host
// round trips (PROPOSAL.md:189-193: "one tq_quant_batched over all of a
// prefill's ~120-192 units ... two kernel chains per prefill instead of
// ~4,000 host round trips" -- the ENCODE chain is this entry; the decode
// chain is W20.8's). The entry composes the family's existing stages;
// NOTHING is re-implemented, every kernel stays the committed
// fixture-pinned copy (the additions-only law; 7.3's composition
// decision realized as the host launch sequence it planned,
// kernel_tq.cu:1922-1953):
//   (a) norms = tq_quant_norm_rows(units)  -- 7.1's chain: the (M, 64)
//       partial grid + the fold launch, the norm of the UNROTATED rows
//       (turboquant.py:510: norm = x32.norm() BEFORE the rotation -- the
//       FHT is norm-preserving, the order is the twin's). (M,) fp32,
//       computed once, RETURNED once -- the entry's third output IS
//       7.1's tensor (no copy, no re-reduce, no host read; the
//       forward-build note's caller-owned-buffer option exists,
//       kernel_tq.cu:1467-1475, but the wrapper's fresh (M,) IS the
//       natural output here).
//   (b) y = fht_forward_large(units, signs) -- 5.2's (M, K)-batched
//       two-pass butterfly: the quant rotation (turboquant.py:530-531),
//       signs AFTER the butterfly, 1/sqrt(K) in the epilogue, the norm
//       NEVER divided out (7.2's identity, kernel_tq.cu:1710-1724). y is
//       (M, K) fp32, the butterfly's own out-of-place state tensor.
//   (c) tq_quant_ladder_pack_run(...)      -- 7.3's ladder + pack over
//       the ROTATED rows, through its own launcher (grid = M x K/16384
//       tiles; the in-kernel grid-stride loop is R20.6's overshoot
//       guard -- the CTA-count guard, kernel_tq.cu:2009-2013, 2317-2321:
//       2^22's 256-tile-per-unit grids stay launch-legal and the byte
//       values grid-independent, "256-tile units still ~1 wave-pair at
//       80 SMs", PROPOSAL.md:318).
//   (d) the zero-norm sweep (below)        -- the degenerate-unit
//       branch, device-side.
// The chain's cross-stage ordering rides the stream's launch
// boundaries -- this family's ONLY cross-CTA synchronization
// (fht_large.cuh:106-108): the FULL butterfly completes before the
// ladder (quant's bucketize consumes the FINAL rotated coordinates,
// turboquant.py:530-545), and the pack pass completes before the sweep.
// The host NEVER touches device memory content anywhere: no .item(),
// no .cpu(), no D2H sync (turboquant.py:490-497's encode discipline) --
// the twin's single host-visible branch is exactly what this entry
// eliminates.
//
// THE SIGNS CONTRACT -- passed IN, never generated: the (d,) fp32 +-1
// rotation vector is the QUANTIZER's (fht.rotation_signs(d, seed),
// fht.py:105-115, (K,) float32 storage; turboquant.py:446). The caller
// (W20.8's dispatcher, TASKS.md:1097-1116) supplies it on the units'
// device; this C++ owns no RNG and no seed -- the seeds live in the
// Python frame (PROPOSAL D3), and generating signs here would fork the
// frame's determinism source.
//
// THE ZERO-NORM BRANCH -- the placement decision. 7.2/7.3 assigned it
// to this unit: "n > 0 is the CALLER's contract -- the degenerate
// zero-norm unit is W20.7.4's device-side branch ... it never reaches
// the ladder" (kernel_tq.cu:1721-1724; the 7.3 banner's hand-off,
// kernel_tq.cu:2038-2039: "the zero-norm degenerate is 7.4's
// device-side branch, never this kernel's"). Realized as a SEPARATE
// device-side sweep kernel, NOT a branch edited into 7.3's pack kernel:
//   * the additions-only law -- 7.3's kernel is committed and stays
//     byte-identical (the layout-pin gates own exactly one copy of it);
//   * the semantics are the torch twin's explicit branch
//     (turboquant.py:511-523): norm == 0.0 -> CODES OF ZERO -- the
//     unit's two packed rows become ZERO-FILLED streams (the zero-codes
//     layout, turboquant.py:519-522), norm 0. The branch condition is
//     the exact IEEE compare norms[m] == 0.0f (the twin's
//     `if norm == 0.0`, turboquant.py:511), evaluated ON DEVICE: the
//     branch costs ZERO host round trips ("the torch-twin's known cost,
//     eliminated by the W20.7 kernel design", turboquant.py:512-514).
//   * ORDER: the sweep is the chain's LAST launch -- its zeros are the
//     final bytes. For the TRUE all-zero row the degenerate ladder
//     ALREADY emits codes-of-zero (y = 0 exactly; every scaled boundary
//     is +-0; 0 <= 0 at rung 0 -> idx 0 for every coordinate -> the
//     zero word -- the layout pin's zero stream; the ladder's compare
//     direction, kernel_tq.cu:1894-1908), so the sweep's overwrite is a
//     parity no-op there; its real work is the measure-zero UNDERFLOW
//     degenerate (a row of values whose fp32 squares all flush to 0:
//     norm == 0 with y != 0, where the degenerate ladder's decisions
//     are NOT the twin's explicit zeros). The sweep pins the twin's
//     branch in both cases.
//   * DETERMINISM: the sweep's writes are a pure function of the
//     norms' zero bits; every byte single-owned (row byte b -> thread
//     b % width, iteration b / width); run-to-run torch.equal holds.
//     No barrier rides the sweep's grid-stride loop -- the kernel has
//     no shared memory, so the healthy unit's early exit is plain
//     divergence, legal exactly where the pack kernel's smem table
//     lifetime forbids it (kernel_tq.cu:2230-2238).
//   * the same ulp class as the ladder ties: the twin branches on ITS
//     torch.norm bits, this chain on 7.1's tree bits
//     (kernel_tq.cu:1457-1465) -- a norm straddling 0 within an ulp can
//     take the two paths' different branches; G20.P3's norm threshold
//     (<= 1 ulp) and tie counter own the exposure (TASKS.md:1122-1124).
//
// THE TORCH_CHECK FAMILY (loud, metadata-only):
//   * units: 2-D (M, d) contiguous CUDA fp32 -- fp32 ONLY (the encode
//     chain consumes the butterfly's fp32 output; fp16/bf16 rows stay
//     on the torch twin, which casts first, turboquant.py:505);
//   * d == units.size(1) (a mismatch is frame drift, PROPOSAL D3);
//     d a power of two > 16384, % 32 == 0 -- the butterfly's envelope
//     (fht_large_check_args's family, kernel_tq.cu:490-531; a power of
//     two above 16384 is necessarily 32-aligned -- the clause mirrors
//     the family's refusal order, checked loudly, never assumed);
//   * M <= INT32_MAX (grid.x) and M*d <= 2^41 (the butterfly's launch
//     arithmetic, kernel_tq.cu:514-520);
//   * signs: (d,) fp32 contiguous CUDA -- the contract above;
//   * bits_lo, bits_hi in {3, 4} (codebooks.py:219-228);
//   * n_lo >= 0, n_hi >= 0, n_lo + n_hi == d (the split partitions d);
//     n_lo % 16 == 0 -- THE ALIGNMENT CONTRACT (kTqPackSlice, the
//     16-coordinate slice ownership, kernel_tq.cu:2015-2024):
//     misaligned fractions are REFUSED loudly and route to the torch
//     twin (the auto-backend discipline -- never a silently mis-packed
//     stream);
//   * boundaries_{lo,hi}: 1-D fp32 contiguous CUDA, numel == 2^bits - 1
//     (tq_boundary_count; the 2.2 device mirrors, turboquant.py:556-570)
//     -- checked UNCONDITIONALLY: the length is a property of bits (the
//     codebook's mirror; TQCodes carries the split's bits even for an
//     empty half), and an empty half's table is never dereferenced (the
//     pack kernel's own n_half > 0 guards). ASCENDING is NOT re-checked
//     -- it would read the tables' CONTENT, a banned D2H;
//     codebooks.py:445-446's load-time strict-ascent validation is the
//     cited precondition (kernel_tq.cu:1755-1757).
//
// THE GRID ACCOUNTING (the PROPOSAL's budget line, realized as the
// launch sequence): the sketch's fused grid -- "grid = SUM(units x
// tiles) + norm CTAs ~ 6,100 CTAs at 2^19 -- ~76 waves of the 80 SMs,
// each wave microseconds" (PROPOSAL.md:190-191; :225's "6,100-CTA
// batched grids = ~76 waves"), 120-192 units x 2^19 in the brief's
// range -- lands on the ladder+pack pass: 192 x 32 tiles = 6,144 CTAs
// at the 192-unit 2^19 prefill (120 units -> 3,840), the 64 KiB-smem
// 1-CTA/SM pass the waves count. The realized chain's other grids: the
// norm chain's (M, 64) partial CTAs + ceil(M/256) fold CTAs (12,288 +
// 1 at M = 192, 256-thread bandwidth streaming), the butterfly's
// pass-1 (M, K/16384) = the same 6,144 and pass-2's ceil(M*K/2 / 1024)
// streaming CTAs per cross stage (5 stages at 2^19), and the sweep's M
// CTAs. Every grid is a pure function of (M, K) -- no device query
// feeds any launch (the determinism contract).
//
// THE M == 0 BRANCH: the entry's (the pack launcher refuses M == 0
// loudly, kernel_tq.cu:2263-2265 -- "the M == 0 branch is 7.4's"):
// zero units -> the three empty tensors, no launch (a zero-size grid is
// not a legal launch -- the family discipline, kernel_tq.cu:577-580).
//
// THE WORKSPACES: the entry allocates exactly the OUTPUTS -- codes_lo
// (M, L_lo) uint8, codes_hi (M, L_hi) uint8 (L = tq_packed_len(n_half,
// bits_half), the _packed_len pin, kernel_tq.cu:2070-2083) -- and
// returns 7.1's (M,) norms. y and the norm chain's partials are the
// stages' own wrapper-allocated workspaces, freed at scope exit on the
// same stream -- the caching allocator's stream-ordered discipline
// (kernel_tq.cu:1652-1657): a later launch on this stream cannot
// observe the freed blocks. y costs (M, K) fp32 -- 2 MiB per unit at
// 2^19, 384 MiB at the 192-unit prefill -- the butterfly's committed
// design, the price 7.3's banner already booked (ONE extra fp32 read
// pass of y, kernel_tq.cu:1946-1948).
//
// THE LAYOUT PIN (inherited, cited per the 7.3 invariant): the packed
// byte layout is IDENTICAL to today's pack_bits (turboquant.py:212-245,
// the W20.3.2 canonical) -- fixture-pinned by G20.P1's frozen fixtures;
// G20.P3's kernel-parity gate counts divergences (codes equal except
// ties <= 0.01%, norm <= 1 ulp, TASKS.md:1122-1124).
//
// Resource accounting (sm_86): the sweep is 256 threads/CTA (the norm
// chain's small-CTA width), grid.x = M (the chain's smallest grid), no
// shared memory, no dynamic-smem opt-in; ~4 B read (the broadcast
// norm) and 0..L bytes written per CTA.
//
// COMPILE RECORD: nvcc is ABSENT on this authoring box (env/W20_ENV.md)
// -- compile and behavioral verification defer to the A10G box (the
// W20.5.4 TU registration carries this file; W20.7.5 / W20.8.3 run the
// gates). Authored statically, re-read for review; every defect found
// here is a box-side discovery.
// ===========================================================================

namespace {

// The sweep's CTA width: 256 threads (the norm chain's small-CTA shape
// -- 8 whole warps; the sweep's work is one broadcast read plus, only
// for degenerate units, a byte fill).
constexpr int kTqZeroSweepThreads = 256;

static_assert(kTqZeroSweepThreads % 32 == 0,
              "whole warps (the family discipline, kTqNormThreads's)");

// ---------------------------------------------------------------------------
// tq_quant_zero_norm_sweep_kernel -- the degenerate-unit branch, on
// device. One CTA per unit, grid-stride over the M units (R20.6's guard
// pattern: the values are a pure function of the norms' zero bits, so
// any grid a future caller caps to computes the same bytes -- the pack
// kernel's own guard discipline, kernel_tq.cu:2009-2013). A healthy
// unit (norms[unit] != 0) exits without a single write -- its bytes are
// the pack pass's, already final. A degenerate unit (norms[unit] ==
// 0.0f) zero-fills its two packed rows: turboquant.py:511-523's codes
// of zero. NO shared memory, NO barrier: the per-unit early exit is
// plain divergence (nothing here is cooperative -- unlike the pack
// kernel's smem table lifetime, kernel_tq.cu:2230-2238, whose barriers
// must stay CTA-uniform).
// ---------------------------------------------------------------------------
__global__ void __launch_bounds__(kTqZeroSweepThreads)
tq_quant_zero_norm_sweep_kernel(
    const float* __restrict__ norms,  // (M,) 7.1's output, read-only
    uint8_t* __restrict__ codes_lo,   // (M, lo_bytes); null if lo_bytes == 0
    uint8_t* __restrict__ codes_hi,   // (M, hi_bytes); null if hi_bytes == 0
    int M, int lo_bytes, int hi_bytes) {
    // int64 unit -- the family's int-envelope belt (M <= INT32_MAX, but
    // the stride increment must not wrap a signed int at M > 2^30;
    // constructible by arithmetic, not faith).
    for (int64_t unit = static_cast<int64_t>(blockIdx.x);
         unit < M; unit += gridDim.x) {
        // the branch condition: the EXACT IEEE compare, on device -- the
        // twin's `if norm == 0.0` (turboquant.py:511) with the host read
        // eliminated. -0.0 == 0.0 fires too (a value the fp32
        // sum-of-squares cannot emit; the compare is the twin's
        // semantics, verbatim). NaN != 0 skips (out of contract -- the
        // ladder's own NaN banner, kernel_tq.cu:1777-1780).
        if (norms[unit] != 0.0f) {
            continue;  // healthy: the pack pass owns the unit's bytes
        }
        // codes of zero: both packed rows zero-filled (the twin's
        // torch.zeros(_packed_len(...)), turboquant.py:519-522). Row
        // byte b is written by exactly ONE thread (b % width, iteration
        // b / width): single-writer per byte, nothing cooperative. An
        // empty half (bytes == 0) never dereferences its pointer (a
        // null base plus a zero offset is the null base).
        uint8_t* row_lo =
            codes_lo + unit * static_cast<int64_t>(lo_bytes);
        uint8_t* row_hi =
            codes_hi + unit * static_cast<int64_t>(hi_bytes);
        for (int b = threadIdx.x; b < lo_bytes; b += kTqZeroSweepThreads) {
            row_lo[b] = uint8_t(0);
        }
        for (int b = threadIdx.x; b < hi_bytes; b += kTqZeroSweepThreads) {
            row_hi[b] = uint8_t(0);
        }
    }
}

// ---------------------------------------------------------------------------
// tq_quant_zero_norm_sweep_run -- the sweep's launcher (the family's
// raw-pointer pattern: integer geometry + pointer hygiene checked
// loudly, the host never touching device content). The FULL grid by
// default (one CTA per unit, grid.x = M <= INT32_MAX -- the entry's
// check); the kernel's grid-stride loop is the R20.6 guard that keeps
// any future cap value-correct.
// ---------------------------------------------------------------------------
void tq_quant_zero_norm_sweep_run(
    const float* norms,
    uint8_t* codes_lo, uint8_t* codes_hi,
    int M, int lo_bytes, int hi_bytes,
    cudaStream_t stream) {
    TORCH_CHECK(M > 0,
                "tq_quant_zero_sweep: M must be > 0 (a zero-size grid is "
                "not a legal launch -- the M == 0 branch is the entry's)");
    TORCH_CHECK(norms != nullptr,
                "tq_quant_zero_sweep: norms must be a live device pointer");
    TORCH_CHECK(lo_bytes == 0 || codes_lo != nullptr,
                "tq_quant_zero_sweep: a live lo stream needs its pointer");
    TORCH_CHECK(hi_bytes == 0 || codes_hi != nullptr,
                "tq_quant_zero_sweep: a live hi stream needs its pointer");
    tq_quant_zero_norm_sweep_kernel
        <<<M, kTqZeroSweepThreads, 0, stream>>>(
            norms, codes_lo, codes_hi, M, lo_bytes, hi_bytes);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// ---------------------------------------------------------------------------
// Argument validation -- the entry's loud TORCH_CHECK family (the
// banner's list; metadata only, never content -- the internal wrappers
// and launchers re-check their own geometry as the belt, so a failure
// surfaces with the entry's prefix first).
// ---------------------------------------------------------------------------
void tq_quant_batched_check_args(
    const torch::Tensor& units, int64_t d,
    const torch::Tensor& signs,
    const torch::Tensor& boundaries_lo, const torch::Tensor& boundaries_hi,
    int64_t n_lo, int64_t bits_lo, int64_t n_hi, int64_t bits_hi) {
    TORCH_CHECK(units.is_cuda(),
                "tq_quant_batched: units must be a CUDA tensor");
    TORCH_CHECK(units.dim() == 2,
                "tq_quant_batched: units must be 2-D (M, d) - the "
                "prefill's unit matrix (the dispatcher flattens "
                "(..., d) to rows)");
    TORCH_CHECK(units.is_contiguous(),
                "tq_quant_batched: units must be contiguous");
    TORCH_CHECK(units.scalar_type() == at::kFloat,
                "tq_quant_batched: units dtype must be float32 (the "
                "encode chain consumes the butterfly's fp32 output; "
                "fp16/bf16 rows stay on the torch twin, which casts "
                "first, turboquant.py:505), got ", units.scalar_type());
    TORCH_CHECK(d == units.size(1),
                "tq_quant_batched: d must match units' last dim (got d=",
                d, ", units.size(1)=", units.size(1), ") - frame drift? "
                "(PROPOSAL D3)");
    TORCH_CHECK(d > 0 && d <= static_cast<int64_t>(INT32_MAX),
                "tq_quant_batched: d must be a positive int-sized count "
                "(got d=", d, ")");
    const int K = static_cast<int>(d);
    // The butterfly's envelope, mirrored (fht_large_check_args's
    // family, kernel_tq.cu:490-531).
    TORCH_CHECK(K > flute::FHT_LARGE_TILE,
                "tq_quant_batched: d=", K, " <= 16384 - one 64 KiB tile "
                "or less; the small-K paths own it");
    TORCH_CHECK((K & (K - 1)) == 0,
                "tq_quant_batched: d must be a power of two (the FHT "
                "rotation the encode chain rides), got d=", K);
    TORCH_CHECK(K % 32 == 0,
                "tq_quant_batched: d must be divisible by 32 (the "
                "kernel_fht.cu:21 discipline), got d=", K);
    // M's launch bounds (grid.x everywhere in the chain).
    TORCH_CHECK(units.size(0) <= static_cast<int64_t>(INT32_MAX),
                "tq_quant_batched: M must fit a grid.x launch (got M=",
                units.size(0), ")");
    // The butterfly's launch arithmetic (kernel_tq.cu:514-520).
    TORCH_CHECK(units.size(0) * d <= (int64_t{1} << 41),
                "tq_quant_batched: M*d = ", units.size(0) * d,
                " exceeds the butterfly launch grid's addressable pairs");
    // The signs contract (the banner's): the quantizer's (d,) fp32
    // rotation vector, passed IN by the caller.
    TORCH_CHECK(signs.is_cuda() && signs.dim() == 1 &&
                signs.scalar_type() == at::kFloat &&
                signs.numel() == d && signs.is_contiguous(),
                "tq_quant_batched: signs must be a contiguous (d,) "
                "float32 CUDA tensor - the quantizer's rotation vector "
                "(fht.rotation_signs, supplied by the caller), got "
                "numel=", signs.numel(), ", d=", d);
    // The split's cardinals.
    TORCH_CHECK(bits_lo == 3 || bits_lo == 4,
                "tq_quant_batched: bits_lo must be 3 or 4 (the TQ index "
                "stream widths, codebooks.py:219-228), got ", bits_lo);
    TORCH_CHECK(bits_hi == 3 || bits_hi == 4,
                "tq_quant_batched: bits_hi must be 3 or 4 (the TQ index "
                "stream widths, codebooks.py:219-228), got ", bits_hi);
    TORCH_CHECK(n_lo >= 0 && n_hi >= 0 && n_lo + n_hi == d,
                "tq_quant_batched: the lo/hi split must partition d "
                "(n_lo=", n_lo, ", n_hi=", n_hi, ", d=", d, ")");
    // THE ALIGNMENT CONTRACT (the banner's): refuse loudly, route to
    // the torch twin.
    TORCH_CHECK(n_lo % kTqPackSlice == 0,
                "tq_quant_batched: n_lo must be a multiple of ",
                kTqPackSlice, " (the 16-coordinate slice ownership; "
                "got n_lo=", n_lo, ") - misaligned fractions stay on "
                "the torch twin");
    // The boundary mirrors: 1-D fp32 CUDA, length 2^bits - 1 --
    // UNCONDITIONALLY (bits validated above; the banner's reasoning).
    const int64_t want_lo = tq_boundary_count(static_cast<int>(bits_lo));
    const int64_t want_hi = tq_boundary_count(static_cast<int>(bits_hi));
    TORCH_CHECK(boundaries_lo.is_cuda() && boundaries_lo.dim() == 1 &&
                boundaries_lo.scalar_type() == at::kFloat &&
                boundaries_lo.is_contiguous() &&
                boundaries_lo.numel() == want_lo,
                "tq_quant_batched: boundaries_lo must be a contiguous (",
                want_lo, ",) float32 CUDA tensor (the 2^bits_lo - 1 "
                "codebook mirror), got numel=", boundaries_lo.numel());
    TORCH_CHECK(boundaries_hi.is_cuda() && boundaries_hi.dim() == 1 &&
                boundaries_hi.scalar_type() == at::kFloat &&
                boundaries_hi.is_contiguous() &&
                boundaries_hi.numel() == want_hi,
                "tq_quant_batched: boundaries_hi must be a contiguous (",
                want_hi, ",) float32 CUDA tensor (the 2^bits_hi - 1 "
                "codebook mirror), got numel=", boundaries_hi.numel());
    // ASCENDING is NOT re-checked: it would read the tables' CONTENT
    // (a banned D2H); codebooks.py:445-446's load-time strict-ascent
    // validation is the precondition (kernel_tq.cu:1755-1757).
}

}  // namespace

// ---------------------------------------------------------------------------
// tq_quant_batched -- the public entry (bound by src/bindings.cpp in
// W20.7.4). ONE host call over the prefill's whole unit batch: 7.1's
// norm chain, 5.2's batched butterfly, 7.3's ladder+pack, the zero-norm
// sweep -- two kernel chains per prefill, not ~4,000 host round trips
// (PROPOSAL.md:189-193). The wrapper allocates the outputs, launches
// the chain, and NEVER reads device memory content (no .item(), no
// .cpu(), no D2H sync anywhere).
// ---------------------------------------------------------------------------
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> tq_quant_batched(
    torch::Tensor units, int64_t d, torch::Tensor signs,
    torch::Tensor boundaries_lo, torch::Tensor boundaries_hi,
    int64_t n_lo, int64_t bits_lo, int64_t n_hi, int64_t bits_hi) {
    tq_quant_batched_check_args(units, d, signs, boundaries_lo,
                                boundaries_hi, n_lo, bits_lo, n_hi,
                                bits_hi);

    const int M = static_cast<int>(units.size(0));
    const int K = static_cast<int>(d);
    const int in_lo = static_cast<int>(n_lo);       // range-checked above
    const int in_hi = static_cast<int>(n_hi);
    const int ibits_lo = static_cast<int>(bits_lo);
    const int ibits_hi = static_cast<int>(bits_hi);

    // ---- the outputs (allocation only; sizes from the ONE pinned
    // length function, kernel_tq.cu:2070-2083) -------------------------
    const int64_t L_lo = tq_packed_len(in_lo, ibits_lo);
    const int64_t L_hi = tq_packed_len(in_hi, ibits_hi);
    TORCH_CHECK(L_lo <= static_cast<int64_t>(INT32_MAX) &&
                L_hi <= static_cast<int64_t>(INT32_MAX),
                "tq_quant_batched: a packed stream length exceeds int "
                "(L_lo=", L_lo, ", L_hi=", L_hi, ")");
    torch::Tensor codes_lo = torch::empty({units.size(0), L_lo},
                                          units.options().dtype(at::kByte));
    torch::Tensor codes_hi = torch::empty({units.size(0), L_hi},
                                          units.options().dtype(at::kByte));

    // The M == 0 branch (the pack launcher's TORCH_CHECK documents it
    // as this entry's): the three empty tensors, no launch (the family
    // discipline, kernel_tq.cu:577-580).
    if (M == 0) {
        torch::Tensor norms =
            torch::empty({0}, units.options().dtype(at::kFloat));
        return {codes_lo, codes_hi, norms};
    }

    // ---- (a) the norm chain: 7.1's wrapper (its own (64, M) workspace
    // + its two launches; the (M,) fp32 output is this entry's third
    // return -- computed once, returned once, never read on the host) --
    torch::Tensor norms = tq_quant_norm_rows(units);

    // ---- (b) the batched butterfly: 5.2's public forward (out-of-
    // place, (M, K) fp32 rotated rows -- signs and 1/sqrt(K) in the
    // epilogue, the norm never divided out) ----------------------------
    torch::Tensor y = fht_forward_large(units, signs);

    auto stream = at::cuda::getCurrentCUDAStream();  // the house kernels'
                                                     // stream

    // ---- (c) the ladder + pack pass: 7.3's launcher (its own grid
    // arithmetic + TORCH_CHECKs; grid = M x K/16384 tiles, the in-kernel
    // grid-stride loop the R20.6 overshoot guard) ----------------------
    tq_quant_ladder_pack_run(
        y.data_ptr<float>(), norms.data_ptr<float>(),
        boundaries_lo.data_ptr<float>(), boundaries_hi.data_ptr<float>(),
        codes_lo.data_ptr<uint8_t>(), codes_hi.data_ptr<uint8_t>(),
        M, K, in_lo, ibits_lo, in_hi, ibits_hi, stream);

    // ---- (d) the zero-norm sweep: the degenerate units' codes-of-zero,
    // device-side (the chain's LAST launch -- its zeros are the final
    // bytes) ------------------------------------------------------------
    tq_quant_zero_norm_sweep_run(
        norms.data_ptr<float>(),
        codes_lo.data_ptr<uint8_t>(), codes_hi.data_ptr<uint8_t>(),
        M, static_cast<int>(L_lo), static_cast<int>(L_hi), stream);

    // y and the norm chain's partials free at scope exit on the same
    // stream -- the caching allocator's stream-ordered discipline
    // (kernel_tq.cu:1652-1657): a later launch on this stream cannot
    // observe the freed blocks.
    return {codes_lo, codes_hi, norms};
}
