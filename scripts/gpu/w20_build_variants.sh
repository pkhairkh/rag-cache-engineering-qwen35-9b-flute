#!/usr/bin/env bash
# =============================================================================
# scripts/gpu/w20_build_variants.sh -- W20.6.2's canary/trap build variants.
#
# A10G-BOX ONLY. This script BUILDS (nvcc + torch) and must NEVER run on the
# CPU dev box (PROMPT.md S1: nvcc is absent there). Authored statically,
# verified on the dev box by `bash -n` (parse-only) + re-read.
#
# WHAT IT BUILDS
#   The W20.6 negative controls of src/flute_extended/src/kernel_tq.cu are
#   #ifdef-gated compile-time macros, OFF in every default build (setup.py's
#   active nvcc flag list contains NEITHER -D; its 5.4 variant-guidance
#   comment block documents the one-off hand-append this script performs
#   mechanically):
#
#     canary : -DFLUTE_TQ_CANARY  NaN-poisons every output/state byte
#              BEFORE the real kernels write (write-coverage control:
#              zero poison must survive). BOTH orientations are armed --
#              the forward's runner and the backward's runner poison.
#     trap   : -DFLUTE_TQ_TRAP    flips the pair arithmetic on a clock64
#              timing bit (determinism negative control: a run-to-run
#              torch.equal assert that PASSES with the trap armed is a
#              broken assert). BOTH orientations are armed -- the forward
#              cross stages, the backward local + final stages, and the
#              backward transit stages (which reuse the forward's trapped
#              cross kernel).
#
# THE ARTIFACT CONTRACT (W20.6.3's test file depends on this)
#   `build_ext --inplace` ALWAYS writes the extension under its production
#   name (src/flute_extended/flute_extended/_C<EXT_SUFFIX>.so), i.e. an
#   in-place variant build WOULD overwrite the production .so. This script
#   therefore, in order:
#     1. backs up the existing production .so(s) (sha256 recorded);
#     2. removes it/them and builds the VARIANT in place, in SEPARATE build
#        directories (build_w20_<mode>/, build_w20_<mode>_lib/) so no
#        object file ever cross-contaminates a production build;
#     3. copies the fresh .so aside as <pkg>/_C_canary.so | _C_trap.so --
#        the exact tagless names, deterministic for the test to reference;
#     4. RESTORES the production .so from the bit-exact backup. Chosen
#        over the alternative (rebuild production clean at the end):
#        restore has zero compiler/toolchain drift and zero build
#        nondeterminism, and any failure path (set -e) hits the EXIT trap
#        and restores. If NO production .so existed before the run, one is
#        built CLEAN instead (same command, no macro) so the box is never
#        left without it.
#   The production .so at exit is therefore NEVER the variant binary.
#
#   W20.6.3 loads a variant WITHOUT importing the package: the .so exports
#   production's init symbols (it IS production plus one macro --
#   bindings.cpp:146's PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) bakes
#   "flute_extended._C"), so it is path-loaded as
#       importlib.machinery.ExtensionFileLoader(
#           "flute_extended._C",
#           "<repo>/src/flute_extended/flute_extended/_C_canary.so")
#   in a process that does NOT also import the production _C (the module
#   name would collide). The trap variant leaves no symbol difference --
#   its proof is behavioral (G20.6's determinism gate must FAIL); the
#   canary variant's _C_canary.so additionally carries the
#   fht_large_poison_kernel symbol (checked below as a build-armed proof).
#
# MODES (first arg, or env W20_VARIANT):  canary | trap | clean
#   clean: remove _C_canary.so / _C_trap.so, the build_w20_* dirs, and any
#          .w20bak backups; production is untouched.
#
# THE BUILD SHAPE (the brief's env-macro style; setup.py is NOT modified --
# the -D lands in nvcc's extra_compile_args exactly where setup.py's 5.4
# comment block prescribes the hand-append, injected by a build-time
# wrapper that monkeypatches torch's CUDAExtension constructor before
# runpy-executing setup.py):
#     FLUTE_TQ_CANARY=1 FLUTE_TQ_TRAP=0 MAX_JOBS=<nproc, capped 16> \
#     FLUTE_CUDA_ARCHES=86 python3 setup.py build_ext --inplace \
#         --build-temp build_w20_<mode> --build-lib build_w20_<mode>_lib
#
# USAGE
#   bash scripts/gpu/w20_build_variants.sh canary
#   bash scripts/gpu/w20_build_variants.sh trap
#   W20_VARIANT=canary bash scripts/gpu/w20_build_variants.sh
#   bash scripts/gpu/w20_build_variants.sh clean
#   then (env/W20_GPU_PROTOCOL.md): python3 scripts/gpu/w20_gates.py \
#       --gates G20.6
# =============================================================================
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PKG_DIR="$REPO_ROOT/src/flute_extended"        # setup.py's directory
MOD_DIR="$PKG_DIR/flute_extended"              # the python package (in-place target)

# ---------------------------------------------------------------------------
# Mode dispatch: first argument, else env W20_VARIANT.
# ---------------------------------------------------------------------------
MODE="${1:-${W20_VARIANT:-}}"
case "$MODE" in
    canary) MACRO=FLUTE_TQ_CANARY ;;
    trap)   MACRO=FLUTE_TQ_TRAP ;;
    clean)  MACRO="" ;;
    *)
        echo "usage: $0 {canary|trap|clean}   (or env W20_VARIANT=...)" >&2
        [ -n "$MODE" ] && echo "  unknown mode: '$MODE'" >&2
        exit 2
        ;;
esac

# ---------------------------------------------------------------------------
# clean : drop every variant artifact (idempotent, production untouched).
# ---------------------------------------------------------------------------
do_clean() {
    echo "[w20_build_variants] CLEAN: removing variant artifacts..."
    local removed=0 f
    for f in "$MOD_DIR"/_C_canary*.so "$MOD_DIR"/_C_trap*.so \
             "$MOD_DIR"/_C*.so.w20bak; do
        [ -f "$f" ] || continue
        rm -f "$f"
        echo "  removed: $f"
        removed=$((removed + 1))
    done
    for f in "$PKG_DIR"/build_w20_*; do
        [ -d "$f" ] || continue
        rm -rf "$f"
        echo "  removed: $f/"
        removed=$((removed + 1))
    done
    if [ "$removed" -eq 0 ]; then
        echo "  nothing to remove (already clean)."
    fi
    echo "[w20_build_variants] CLEAN done. Production .so (if any) untouched."
}

# ---------------------------------------------------------------------------
# Sanity + env assembly shared by the build modes.
# ---------------------------------------------------------------------------
preflight() {
    command -v python3 >/dev/null 2>&1 \
        || { echo "[w20_build_variants] ERROR: python3 not found" >&2; exit 1; }
    [ -f "$PKG_DIR/setup.py" ] \
        || { echo "[w20_build_variants] ERROR: $PKG_DIR/setup.py missing" >&2; exit 1; }
    [ -d "$MOD_DIR" ] \
        || { echo "[w20_build_variants] ERROR: $MOD_DIR missing" >&2; exit 1; }
}

build_env() {
    # MAX_JOBS: nproc, capped at 16 (setup.py also defaults this; explicit
    # here per the brief).
    local jobs
    jobs=$(nproc 2>/dev/null || true)
    case "$jobs" in ''|*[!0-9]*) jobs=8 ;; esac
    if [ "$jobs" -gt 16 ]; then jobs=16; fi
    export MAX_JOBS="$jobs"
    export FLUTE_CUDA_ARCHES="${FLUTE_CUDA_ARCHES:-86}"
    echo "[w20_build_variants] MAX_JOBS=$MAX_JOBS FLUTE_CUDA_ARCHES=$FLUTE_CUDA_ARCHES"
}

# ---------------------------------------------------------------------------
# The build itself (runs in a subshell: the cd does not leak). Arg 1:
# "canary" | "trap" | "" (empty = clean production rebuild, no macro).
# The macro rides the environment exactly as the brief sketches
# (FLUTE_TQ_CANARY=1 ... python3 setup.py build_ext --inplace); the -D is
# appended to nvcc's extra_compile_args by the wrapper below -- the repo's
# setup.py stays byte-identical.
# ---------------------------------------------------------------------------
run_build() (
    local variant="$1"
    cd "$PKG_DIR"
    if [ "$variant" = canary ]; then
        export FLUTE_TQ_CANARY=1 FLUTE_TQ_TRAP=0
    elif [ "$variant" = trap ]; then
        export FLUTE_TQ_TRAP=1 FLUTE_TQ_CANARY=0
    else
        export FLUTE_TQ_CANARY=0 FLUTE_TQ_TRAP=0
    fi
    export W20_VARIANT_BUILD="$variant"

    python3 - <<'PYEOF'
import os, runpy, sys

variant = os.environ.get("W20_VARIANT_BUILD", "")
armed = [n for n in ("FLUTE_TQ_CANARY", "FLUTE_TQ_TRAP")
         if os.environ.get(n, "0") == "1"]
if len(armed) > 1:
    sys.exit("[w20_build_variants] REFUSING: both FLUTE_TQ_CANARY and "
             "FLUTE_TQ_TRAP are 1 -- exactly one variant per build.")

import torch.utils.cpp_extension as _ce

_orig_cuda_ext = _ce.CUDAExtension

def _variant_cuda_ext(*args, **kwargs):
    if armed:
        # The 5.4 hand-append, done mechanically: the -D goes to the NVCC
        # flag list only (the macros are .cu-TU-local; kernel_tq.cu is the
        # only TU that reads them), exactly as setup.py's comment block
        # prescribes. setup.py itself is never modified.
        eca = kwargs.setdefault("extra_compile_args", {})
        nvcc = eca.setdefault("nvcc", [])
        for name in armed:
            nvcc.append("-D" + name)
        print("[w20_build_variants] {}: nvcc_flags += {}".format(
            variant, " ".join("-D" + n for n in armed)), flush=True)
    return _orig_cuda_ext(*args, **kwargs)

_ce.CUDAExtension = _variant_cuda_ext

sys.argv = ["setup.py", "build_ext", "--inplace"]
if variant:
    # separate build dirs per variant: no object cross-contamination
    sys.argv += ["--build-temp", "build_w20_" + variant,
                 "--build-lib",  "build_w20_" + variant + "_lib"]
runpy.run_path("setup.py", run_name="__main__")
PYEOF
)

# ---------------------------------------------------------------------------
# Symbol probe: does the .so mention a kernel-symbol substring anywhere?
# Concatenates every available tool's output (nm's symtab for the host
# stubs, readelf's tables, strings for the fatbin-embedded mangled kernel
# names), then matches in bash -- no pipelines, so pipefail cannot turn a
# grep -q early-exit into a false failure.
# Returns 0 = present, 1 = absent, 2 = indeterminate (no tool/no output).
# ---------------------------------------------------------------------------
so_has_symbol() {
    local file="$1" sym="$2" out=""
    if command -v nm >/dev/null 2>&1; then
        out+=$(nm "$file" 2>/dev/null || true)$'\n'
    fi
    if command -v readelf >/dev/null 2>&1; then
        out+=$(readelf --syms -W "$file" 2>/dev/null || true)$'\n'
    fi
    if command -v strings >/dev/null 2>&1; then
        out+=$(strings "$file" 2>/dev/null || true)$'\n'
    fi
    if [ -z "$out" ]; then
        return 2
    fi
    case "$out" in
        *"$sym"*) return 0 ;;
        *)        return 1 ;;
    esac
}

sha_of() { sha256sum "$1" | awk '{print $1}'; }

# ---------------------------------------------------------------------------
# Production backup/restore (bit-exact) with an EXIT-trap safety net.
# ---------------------------------------------------------------------------
BACKUP_NAMES=()   # original in-place filenames, e.g. _C.cpython-310-x86_64-linux-gnu.so
declare -A BACKUP_SHA=()
RESTORE_STATE=0   # 1 = originals removed, backups on disk

snapshot_production() {
    local f
    # _C*.so minus this script's own variant artifacts.
    while IFS= read -r f; do
        [ -n "$f" ] || continue
        BACKUP_NAMES+=("$f")
        BACKUP_SHA["$f"]="$(sha_of "$MOD_DIR/$f")"
    done < <(cd "$MOD_DIR" && ls _C*.so 2>/dev/null \
             | grep -v -E '^_C_(canary|trap)' || true)
}

restore_production() {
    local f s rc=0
    for f in "${BACKUP_NAMES[@]}"; do
        if ! cp "$MOD_DIR/$f.w20bak" "$MOD_DIR/$f"; then
            echo "[w20_build_variants] ERROR: restore FAILED for $f " \
                 "(backup kept: $MOD_DIR/$f.w20bak)" >&2
            rc=1
            continue
        fi
        s="$(sha_of "$MOD_DIR/$f")"
        if [ "$s" != "${BACKUP_SHA[$f]}" ]; then
            echo "[w20_build_variants] ERROR: sha mismatch after restore " \
                 "of $f (backup kept: $MOD_DIR/$f.w20bak)" >&2
            rc=1
            continue
        fi
        rm -f "$MOD_DIR/$f.w20bak"
    done
    return "$rc"
}

on_exit() {
    local rc=$?
    if [ "$RESTORE_STATE" -eq 1 ]; then
        if restore_production; then
            RESTORE_STATE=0
            echo "[w20_build_variants] EXIT trap: production .so restored " \
                 "from backup (rc=$rc)." >&2
        else
            rc=1
            echo "[w20_build_variants] EXIT trap: RESTORE FAILED -- recover " \
                 "manually from the .w20bak files in $MOD_DIR" >&2
        fi
    fi
    exit "$rc"
}

# ---------------------------------------------------------------------------
# main (build modes)
# ---------------------------------------------------------------------------
if [ "$MODE" = clean ]; then
    do_clean
    exit 0
fi

preflight
build_env
trap on_exit EXIT

echo "==============================================================="
echo " W20.6.2 variant build: MODE=$MODE  MACRO=-D$MACRO"
echo "   repo     : $REPO_ROOT"
echo "   package  : $MOD_DIR"
echo "==============================================================="

# 1. Back up the production .so (if any), then remove it so the variant
#    build's in-place output is unambiguous.
snapshot_production
if [ "${#BACKUP_NAMES[@]}" -gt 0 ]; then
    for prod_f in "${BACKUP_NAMES[@]}"; do
        cp "$MOD_DIR/$prod_f" "$MOD_DIR/$prod_f.w20bak"
        echo "[w20_build_variants] backed up: $MOD_DIR/$prod_f"
        echo "  sha256: ${BACKUP_SHA[$prod_f]}"
    done
    RESTORE_STATE=1
    rm -f "${BACKUP_NAMES[@]/#/$MOD_DIR/}"
    echo "[w20_build_variants] production .so removed for the variant build."
else
    echo "[w20_build_variants] NOTE: no pre-existing production .so found --"
    echo "  one will be built CLEAN at the end of this run."
fi

# 2. Build the variant in place (separate build dirs; the slow part).
echo "[w20_build_variants] building the $MODE variant (this is the slow part)..."
run_build "$MODE"

# 3. Identify the fresh in-place .so and copy it aside under the contract name.
mapfile -t FRESH < <(cd "$MOD_DIR" && ls _C*.so 2>/dev/null \
                     | grep -v -E '^_C_(canary|trap)' || true)
if [ "${#FRESH[@]}" -ne 1 ]; then
    echo "[w20_build_variants] ERROR: expected exactly ONE fresh in-place " \
         "_C*.so, found ${#FRESH[@]}: ${FRESH[*]:-<none>}" >&2
    exit 1
fi
VARIANT_SO="$MOD_DIR/_C_${MODE}.so"
cp "$MOD_DIR/${FRESH[0]}" "$VARIANT_SO"
rm -f "$MOD_DIR/${FRESH[0]}"
VARIANT_SHA="$(sha_of "$VARIANT_SO")"
echo "[w20_build_variants] variant artifact: $VARIANT_SO"

# 4. Restore production from the bit-exact backup (or build it clean).
if [ "${#BACKUP_NAMES[@]}" -gt 0 ]; then
    restore_production
    RESTORE_STATE=0
    echo "[w20_build_variants] production .so restored (bit-exact from backup)."
else
    echo "[w20_build_variants] no backup existed -- building production CLEAN..."
    run_build ""
fi

# 5. Verification: production sha + the canary symbol proofs.
echo "[w20_build_variants] verifying the production .so is variant-free..."
while IFS= read -r pf; do
    [ -n "$pf" ] || continue
    echo "[w20_build_variants] production $MOD_DIR/$pf sha256: $(sha_of "$MOD_DIR/$pf")"
    if [ "$MODE" = canary ]; then
        sym_rc=0
        so_has_symbol "$MOD_DIR/$pf" fht_large_poison_kernel || sym_rc=$?
        if [ "$sym_rc" -eq 0 ]; then
            echo "[w20_build_variants] *** ERROR: the PRODUCTION .so carries " \
                 "the canary poison kernel -- it is NOT clean. Rebuild it: " \
                 "cd $PKG_DIR && python3 setup.py build_ext --inplace" >&2
            exit 1
        elif [ "$sym_rc" -eq 2 ]; then
            echo "[w20_build_variants] WARNING: no nm/readelf/strings -- " \
                 "cannot verify the production .so is canary-free." >&2
        fi
    fi
done < <(cd "$MOD_DIR" && ls _C*.so 2>/dev/null | grep -v -E '^_C_(canary|trap)' || true)

if [ "$MODE" = canary ]; then
    sym_rc=0
    so_has_symbol "$VARIANT_SO" fht_large_poison_kernel || sym_rc=$?
    if [ "$sym_rc" -eq 0 ]; then
        echo "[w20_build_variants] canary-armed PROOF: $VARIANT_SO contains"
        echo "  the fht_large_poison_kernel symbol (forward + backward poison)."
    elif [ "$sym_rc" -eq 1 ]; then
        echo "[w20_build_variants] *** WARNING: $VARIANT_SO does NOT contain " \
             "the poison-kernel symbol -- the canary may not be armed." >&2
    else
        echo "[w20_build_variants] WARNING: could not check the poison-kernel " \
             "symbol (no binutils) -- the gate run is the real proof." >&2
    fi
fi

echo "==============================================================="
echo " W20.6.2 VARIANT BUILD COMPLETE: $MODE"
echo "   artifact : $VARIANT_SO"
echo "   sha256   : $VARIANT_SHA"
echo "   macro    : -D$MACRO (nvcc only; setup.py untouched)"
echo "   contract : W20.6.3 path-loads this .so via"
echo "                importlib.machinery.ExtensionFileLoader("
echo "                    \"flute_extended._C\", \"$VARIANT_SO\")"
echo "              in a process that does NOT import the production _C."
echo "   PRODUCTION .so: NOT the variant -- restored bit-exact from backup"
if [ "${#BACKUP_NAMES[@]}" -eq 0 ]; then
    echo "     (none existed before; built CLEAN this run)"
fi
echo "   next     : python3 scripts/gpu/w20_gates.py --gates G20.6"
echo "   cleanup  : bash $0 clean"
echo "==============================================================="
