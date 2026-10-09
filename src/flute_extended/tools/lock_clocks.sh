#!/usr/bin/env bash
# tools/lock_clocks.sh — pin A10G clocks for reproducible benchmarking.
#
# Why: the AWS A10G (300 W board) boosts to ~1710 MHz momentarily and drops
# under sustained tensor-core load (the "250 TOPS INT8" AWS figure implies
# ~1.53 GHz sustained). Unlocked clocks make TFLOPS/tok-s numbers swing by
# 10-20% run to run, which is exactly how misleading A/B comparisons happen.
#
# Usage:
#  sudo tools/lock_clocks.sh 1530 645    # lock SM 1530 MHz, mem at max
#  sudo tools/lock_clocks.sh status      # show current clocks/limits
#  sudo tools/lock_clocks.sh reset       # restore defaults
#
# (memory clock argument: pass the max memory clock reported by `status`
# to keep memory at full speed while pinning the SM clock)
#
# Suggested lock points on the A10G (pick ONE and keep it for the whole A/B):
#  1410 : conservative sustained TC clock, ~52 TFLOPS FP32-acc ceiling
#  1530 : AWS "250 TOPS" implied sustained clock, ~62.5 TFLOPS FP32-acc
#  1710 : datasheet boost — will NOT sustain under GEMM load at 300 W
#
# Notes:
#  * requires root and a driver that accepts -lgc/-lmc (data-center driver
#    on the g5 box does; GeForce consumer drivers may not)
#  * memory clock: leave at default (maximum) unless studying bandwidth —
#    locking mem lower only distorts the decode-regime picture
set -euo pipefail

DEV=${FLUTE_DEVICE:-0}
SMI="nvidia-smi -i ${DEV}"

case "${1:-status}" in
    status)
        ${SMI} -q -d CLOCK | sed -n '/Clocks/,/^$/p'
        ${SMI} --query-gpu=name,power.draw,power.limit,clocks.sm,clocks.max.sm,temperature.gpu --format=csv
        ;;
    reset)
        ${SMI} -rgc                 # reset application clocks
        ${SMI} -rmc                 # reset memory clocks
        echo "clocks reset to defaults"
        ;;
    [0-9]*)
        if [ $# -lt 2 ]; then
            echo "usage: $0 <sm_mhz> <mem_mhz>   (or: status | reset)" >&2
            exit 1
        fi
        ${SMI} -lgc "$1"            # lock graphics/sm clock
        ${SMI} -lmc "$2"            # lock memory clock
        echo "locked: sm=$1 MHz, mem=$2 MHz on device ${DEV}"
        ${SMI} --query-gpu=clocks.sm,clocks.mem,power.limit --format=csv
        ;;
    *)
        echo "usage: $0 <sm_mhz> <mem_mhz> | status | reset" >&2
        exit 1
        ;;
esac
