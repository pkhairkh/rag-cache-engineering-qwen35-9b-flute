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
