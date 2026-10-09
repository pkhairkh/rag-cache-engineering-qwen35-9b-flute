/**
 * src/gemv_host.cpp
 *
 * The single definition TU for the shared decode-GEMV host helpers
 * (include/flute/gemv_host.h). the monolith split moves these out
 * of kernel_streaming.cu's anonymous namespace - one linked
 * instance, shared by kernel_gemv_splitk.cu / kernel_gemv_multi.cu /
 * kernel_gemv_mlp.cu / kernel_gemv.cu.
 */

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>

#include <algorithm>
#include <cstdlib>
#include <mutex>
#include <unordered_map>

#include "flute/gemv_host.h"

namespace flute {

bool fd_disabled_by_env() {
    const char* e = std::getenv("FLUTE_NO_FD");
    return e != nullptr && std::string(e) == "1";
}

int gemv_pick_split(int tiles, int G) {
    if (tiles >= 160) return 1;
    int best = 1;
    for (int s = 2; s <= 16; s <<= 1) {
        if (G < 4 * s || (G % (4 * s)) != 0) break;
        best = s;
        if (tiles * s >= 160) break;
    }
    return best;
}

int gemv_current_device() {
    return (int)c10::cuda::current_device();
}

namespace {

std::mutex g_gemv_ws_mu;
std::unordered_map<int, GemvWorkspace> g_gemv_ws;

}  // namespace

GemvWorkspace& gemv_workspace_for(int device, int64_t want_floats,
                                    int64_t want_ints) {
    std::lock_guard<std::mutex> guard(g_gemv_ws_mu);
    auto it = g_gemv_ws.find(device);
    if (it != g_gemv_ws.end() && it->second.P.numel() >= want_floats
        && it->second.tickets.numel() >= want_ints) {
        return it->second;
    }
    const int64_t pf = std::max<int64_t>(
        {want_floats, 1024,
         (it == g_gemv_ws.end()) ? 0 : it->second.P.numel()});
    const int64_t ti = std::max<int64_t>(
        {want_ints, 256,
         (it == g_gemv_ws.end()) ? 0 : it->second.tickets.numel()});
    GemvWorkspace ws;
    ws.P = torch::zeros(
        {pf}, torch::TensorOptions().dtype(torch::kFloat32)
                  .device(torch::Device(torch::kCUDA, device)));
    ws.tickets = torch::zeros(
        {ti}, torch::TensorOptions().dtype(torch::kInt32)
                   .device(torch::Device(torch::kCUDA, device)));
    if (it == g_gemv_ws.end()) {
        it = g_gemv_ws.emplace(device, std::move(ws)).first;
    } else {
        it->second = std::move(ws);
    }
    return it->second;
}

}  // namespace flute
