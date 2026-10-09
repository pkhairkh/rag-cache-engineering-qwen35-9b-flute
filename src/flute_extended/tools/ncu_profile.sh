#!/usr/bin/env bash
# tools/ncu_profile.sh — Nsight Compute profiling of the production kernel
# with explicit success gates. Run AFTER `python setup.py build_ext --inplace`
# and AFTER the kernel test suite (test_flute.py) passes.
#
# Usage:
#  bash tools/ncu_profile.sh [M] [layer]
#  bash tools/ncu_profile.sh 4096 gate_proj      # default
#  bash tools/ncu_profile.sh 1024 down_proj
#
# What it measures and why (A10G / SM_86):
#  1. Tensor pipe utilization — the kernel is compute-bound at large M only
#     if the tensor pipe is the top limiter.
#  2. Shared-memory bank conflicts on ldmatrix (sA/sW) and the dequant LDS
#     (sLUT32) — the layouts are designed conflict-free (pad+8 / XOR-8 /
#     4-copy LUT); this gate proves it empirically.
#  3. Warp stall reasons on the memory barrier — a well-pipelined kernel
#     should not be dominated by barrier stalls (>30% flags a pipeline bug).
#  4. Achieved occupancy vs the 2-blocks-per-SM design target.
#  5. Register count / spill — target is <= 255 regs, ZERO spills.
#  6. L2 sector hit rate — the Q re-read behavior of the BM=64 configs
#     (see the deep-tile A/B note at the bottom).
#
# Success gates (docs/PERFORMANCE.md):
#  GATE 1  tensor-pipe throughput (smsp__pipe_tensor_cycles_active%) >= 70%
#          at M >= 4096 on gate_proj/up_proj
#  GATE 2  shared-memory bank conflicts (l1tex__data_bank_conflicts_pipe_lsu)
#          ~ 0 per kmain iteration (<= 1% of shared accesses)
#  GATE 3  launch__registers_per_thread <= 255 AND zero local-memory
#          traffic (l1tex__t_bytes_pipe_lsu_mem_local_op_ld == 0)
#  GATE 4  achieved occupancy >= 50% (2 CTAs x 4 warps on 2 SMSPs each)

set -euo pipefail

M="${1:-4096}"
LAYER="${2:-gate_proj}"

case "$LAYER" in
    gate_proj|up_proj) N=12288; K=4096;  GS=32 ;;
    down_proj)         N=4096;  K=12288; GS=64 ;;
    attn_qkv)          N=8192;  K=4096;  GS=64 ;;
    attn_out)          N=4096;  K=4096;  GS=64 ;;
    *) echo "unknown layer: $LAYER (gate_proj|up_proj|down_proj|attn_qkv|attn_out)"; exit 2 ;;
esac

echo "=== ncu profile: $LAYER (M=$M, N=$N, K=$K, gs=$GS) ==="

# Minimal repro: one kernel launch through the extension.
REPRO=$(mktemp --suffix=.py)
trap 'rm -f "$REPRO"' EXIT
cat > "$REPRO" <<EOF
import torch, sys
sys.path.insert(0, ".")
from flute_extended import qgemm_per_group_lut
M, N, K, GS = $M, $N, $K, $GS
A = torch.randn(M, K, dtype=torch.float16, device="cuda")
idx = torch.randint(0, 256, (N, (K + 1) // 2), dtype=torch.uint8, device="cuda")
lut = torch.randn((N + GS - 1) // GS, 16, dtype=torch.float16, device="cuda")
# warmup: lets clocks settle before the profiled launch
for _ in range(3):
    C = qgemm_per_group_lut(A, idx, lut, bitwidth=4, group_size=GS,
                            backend="cutlass_streaming")
torch.cuda.synchronize()
torch.cuda.profiler.start()
C = qgemm_per_group_lut(A, idx, lut, bitwidth=4, group_size=GS,
                        backend="cutlass_streaming")
torch.cuda.synchronize()
torch.cuda.profiler.stop()
EOF

ncu --profile-from-start off \
    --kernel-name "regex:flute_kernel_streaming" \
    --launch-count 1 \
    --metrics \
smsp__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed,\
smsp__inst_executed_pipe_hmma.sum,\
l1tex__data_bank_conflicts_pipe_lsu_mem_shared.sum,\
l1tex__data_pipe_lsu_wavefronts_mem_shared.sum,\
l1tex__t_bytes_pipe_lsu_mem_local_op_ld.sum,\
launch__registers_per_thread,\
lts__t_sector_hit_rate.pct,\
sm__warps_active.avg.pct_of_peak_sustained_active,\
smsp__average_warp_latency_per_inst_issued.ratio,\
smsp__warp_issue_stalled_barrier_per_warp_active.pct,\
smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct,\
smsp__warp_issue_stalled_short_scoreboard_per_warp_active.pct \
    python3 "$REPRO"

cat <<'EOF'

=== GATES ===
1. smsp__pipe_tensor_cycles_active ... >= 70%   (M>=4096, gs=32 layers)
2. bank conflicts / shared wavefronts ........ <= 1% (layout proof)
3. launch__registers_per_thread .............. <= 255, local ld bytes == 0
4. sm__warps_active ......................... >= ~50% (2 CTAs/SM design)

If GATE 1 fails with high long-scoreboard stalls -> DRAM-bound: raise M,
or inspect lts__t_sector_hit_rate. If GATE 2 fails -> the swizzle layout
assumption broke. If GATE 3 shows spills -> the remedies are, in order:
reduce the Q prefetch depth, or relax __launch_bounds__(128, 2) to
(128, 1) and accept 1 block/SM for the BK=64 configs (spills would crater
throughput 2-3x and invalidate the expected performance bands).

Deep-tile A/B: profile gate_proj with and without FLUTE_GS32_BK=64
(export it before running this script). The deep config <64,128,64,32>
re-reads Q twice as often per N-block (M/64 vs M/128 blocks); watch
lts__t_sector_hit_rate — the 6 MB L2 should absorb the ~786 KB per-block
Q working set. A materially lower hit rate on deep vs default means the
Q re-read is not absorbed: keep the default config.
EOF
