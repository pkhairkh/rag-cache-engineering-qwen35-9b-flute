/**
 * include/flute/gemv_host.h
 *
 * Shared HOST helpers of the decode-GEMV families (//) -
 * the split-K policy, the per-device split-K workspace cache and the
 * env-flag reader. Defined ONCE in src/gemv_host.cpp (a plain C++ TU)
 * so every kernel TU links against a SINGLE workspace instance (the
 * older static-in-anonymous-namespace copies would have become one
 * per translation unit).
 */

#pragma once

#include <torch/extension.h>

#include <cstdint>

namespace flute {

// FLUTE_NO_FD=1 disables every idxN-layout consumer (the GEMV families).
bool fd_disabled_by_env();

// The split-K policy (host, mirrored in Python: gemv_split_hint):
// target >= 160 CTAs (2 waves of the 80 SMs), SPLIT a power of two
// 1..16, G % (4*SPLIT) == 0 (every j4 warp keeps >= 1 g-tile).
int gemv_pick_split(int tiles, int G);

// The current CUDA device index (for the workspace cache).
int gemv_current_device();

// The split-K workspace: per-device, geometrically grown, zeroed ONCE at
// (re)allocation (the kernels' finalizer self-reset keeps the tickets at
// zero across calls; P never needs zeroing). Single-stream sequential
// use (the decode-graph replay order) - documented contract.
struct GemvWorkspace {
    torch::Tensor P;        // fp32, >= SPLIT*pstride elements
    torch::Tensor tickets;  // int32, >= tiles elements
};

GemvWorkspace& gemv_workspace_for(int device, int64_t want_floats,
                                    int64_t want_ints);

}  // namespace flute
