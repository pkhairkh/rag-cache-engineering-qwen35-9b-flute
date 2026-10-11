#!/usr/bin/env bash
# W20.0.2 - A10G provisioning script (idempotent, latest-stable, sm_86-asserting)
#
# WHAT: the FIRST command the A10G operator runs (PROMPT.md section 6;
# TASKS.md W20.0.2). Asserts the Ampere device, provisions latest-stable
# deps UNPINNED (latest stable at run time - the lock is evidence, not a
# constraint), matches nvcc to torch's CUDA build, builds flute_extended
# in place for sm_86, runs the 237-test baseline suite as the pre-W20
# regression gate, and writes env/w20_gpu_env.lock.
#
# WHERE: A10G box ONLY. Never on the CPU dev box - STEP 1 asserts
# nvidia-smi and the Ampere family (compute capability major 8).
#
# USAGE:  bash scripts/gpu/w20_provision.sh
#         (from the repo root or anywhere inside the checkout; the repo
#          root is derived from this script's own location)
#
# IDEMPOTENT: every step re-echoes what it does and is safe to re-run;
# the lock file is rewritten on every run. A failing baseline suite is
# NEVER masked: the PASS/FAIL banner and the lock record the latest run.
#
# LAW (PROMPT.md section 6, latest-stable): no version pins anywhere -
# every install is an unpinned upgrade at run time. The cuXX suffix in
# STEP 4 is DERIVED at run time from torch's CUDA build (a match to
# torch, not a pin). STEPS 1 and 4 assert hard (exit 1); STEP 6 reports
# the suite's exit code and never aborts (the operator decides).

# bash guard (this script uses bash features; dash/sh cannot run it)
if [ -z "${BASH_VERSION:-}" ]; then
  echo "[provision][FATAL] this script requires bash - run: bash scripts/gpu/w20_provision.sh" >&2
  exit 1
fi

set -euo pipefail

# ---------------------------------------------------------------- helpers --
log()  { printf '[provision] %s\n' "$*"; }
warn() { printf '[provision][WARN] %s\n' "$*" >&2; }
die()  { printf '[provision][FATAL] %s\n' "$*" >&2; exit 1; }
step() { printf '\n[provision] ===== %s =====\n' "$*"; }

pkg_version() {
  # Version of an installed pip package; empty string if not installed.
  python3 -m pip show "$1" 2>/dev/null | sed -n 's/^Version: //p' || true
}

# Repo root derived from this script's own location: scripts/gpu/ -> root.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
if [ ! -f "$REPO_ROOT/src/flute_extended/setup.py" ]; then
  die "repo root mis-derived (${REPO_ROOT}): src/flute_extended/setup.py not found - run this script from a checkout of the W20 repo"
fi

command -v python3 >/dev/null 2>&1 \
  || die "python3 not found on PATH - the A10G box needs python3 with pip"

log "W20.0.2 A10G provisioning - start $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
log "host: $(hostname 2>/dev/null || echo unknown) | repo root: ${REPO_ROOT}"
log "GPU-box-only script: STEP 1 asserts nvidia-smi + Ampere (sm_86 expected)"

# --------------------------------------------------- STEP 1 - GPU assert ---
step "STEP 1/7 - GPU assert (nvidia-smi present, Ampere major 8 required, sm_86 expected)"

command -v nvidia-smi >/dev/null 2>&1 \
  || die "nvidia-smi not found on PATH - no NVIDIA driver detected. This script provisions the A10G box only (PROMPT.md section 6, TASKS.md W20.0.2). Install the NVIDIA datacenter driver, then re-run (idempotent)."

# First device: compute capability, name, driver version.
GPU_QUERY_ALL="$(nvidia-smi --query-gpu=compute_cap,name,driver_version --format=csv,noheader 2>/dev/null || true)"
GPU_QUERY="${GPU_QUERY_ALL%%$'\n'*}"
if [ -z "$GPU_QUERY" ]; then
  die "nvidia-smi is present but the compute_cap/name/driver_version query returned nothing (driver too old, or no GPU attached) - fix the driver/GPU state and re-run"
fi
COMPUTE_CAP="$(printf '%s' "$GPU_QUERY" | cut -d',' -f1 | tr -d '[:space:]')"
DEVICE_NAME="$(printf '%s' "$GPU_QUERY" | cut -d',' -f2 | sed 's/^[[:space:]]*//; s/[[:space:]]*$//')"
DRIVER_VERSION="$(printf '%s' "$GPU_QUERY" | cut -d',' -f3 | tr -d '[:space:]')"
CAP_MAJOR="${COMPUTE_CAP%%.*}"

# HARD assert: Ampere family (major 8). Never silently ignored.
if [ "$CAP_MAJOR" != "8" ]; then
  die "GPU compute capability ${COMPUTE_CAP} has major ${CAP_MAJOR} - NOT Ampere (8). W20 targets the A10G (sm_86) per PROPOSAL.md section 4 / PROMPT.md section 6. Aborting."
fi
# SOFT warn: exact A10G capability.
if [ "$COMPUTE_CAP" != "8.6" ]; then
  warn "compute capability is ${COMPUTE_CAP}, not the A10G's 8.6 - proceeding, but STEP 5 hardcodes FLUTE_CUDA_ARCHES=86 (kernels compile for sm_86)"
fi

log "device 0:          ${DEVICE_NAME}"
log "compute capability: ${COMPUTE_CAP} (sm_${COMPUTE_CAP})"
log "driver version:     ${DRIVER_VERSION}"
if command -v nvcc >/dev/null 2>&1; then
  log "nvcc at start: $(nvcc --version | tail -1)"
else
  log "nvcc at start: ABSENT (STEP 4 will provision it if torch needs it)"
fi

# --------------------------------------------- STEP 2 - record environment --
step "STEP 2/7 - record environment (driver, CUDA driver API, OS, python, CPU count)"

NVIDIA_SMI_OUT="$(nvidia-smi 2>/dev/null || true)"
log "nvidia-smi tail (footer, for the record):"
printf '%s\n' "$NVIDIA_SMI_OUT" | tail -2 | sed 's/^/[provision]   /'
CUDA_DRIVER_API="$(printf '%s\n' "$NVIDIA_SMI_OUT" | sed -n 's/.*CUDA Version: *\([0-9][0-9.]*\).*/\1/p' | head -1 || true)"
CUDA_DRIVER_API="${CUDA_DRIVER_API:-unknown}"
log "driver version:        ${DRIVER_VERSION}"
log "CUDA driver API version: ${CUDA_DRIVER_API}"
OS_PRETTY=""
if [ -r /etc/os-release ]; then
  OS_PRETTY="$(sed -n 's/^PRETTY_NAME="\(.*\)"/\1/p' /etc/os-release 2>/dev/null | head -1 || true)"
fi
OS_DESC="$(uname -sr)${OS_PRETTY:+ (${OS_PRETTY})}"
log "OS:                    ${OS_DESC}"
PYTHON_VERSION="$(python3 --version 2>&1)"
log "python3:               ${PYTHON_VERSION}"
NPROC="$(nproc 2>/dev/null || echo 4)"
log "cpu count (nproc):     ${NPROC} -> MAX_JOBS will be capped at 16 in STEP 5"

# ------------------------------------- STEP 3 - pip install -U (unpinned) --
step "STEP 3/7 - pip install -U, UNPINNED (latest stable at run time - PROMPT.md section 6)"

log "command: python3 -m pip install -U torch ninja transformers faiss-cpu numpy"
python3 -m pip install -U torch ninja transformers faiss-cpu numpy

log "installed versions (python3 -m pip show <pkg> | head -2):"
for PKG in torch ninja transformers faiss-cpu numpy; do
  python3 -m pip show "$PKG" | head -2 || true
done

log "latest-stable evidence, advisory (python3 -m pip index versions <pkg> | head -2):"
for PKG in torch ninja transformers faiss-cpu numpy; do
  python3 -m pip index versions "$PKG" 2>/dev/null | head -2 || true
done

# -------------------------- STEP 4 - CUDA toolkit (nvcc) matching torch ----
step "STEP 4/7 - CUDA toolkit (nvcc) matching torch's CUDA build"

TORCH_VERSION="$(python3 -c 'import torch; print(torch.__version__)' 2>/dev/null)" \
  || die "python3 -c 'import torch' failed - STEP 3 must succeed first"
TORCH_CUDA="$(python3 -c 'import torch; print(torch.version.cuda)' 2>/dev/null)" \
  || die "python3 -c 'import torch' failed - STEP 3 must succeed first"
if [ -z "$TORCH_CUDA" ] || [ "$TORCH_CUDA" = "None" ]; then
  die "torch ${TORCH_VERSION} reports no CUDA build (torch.version.cuda is '${TORCH_CUDA}') - the A10G box needs the CUDA torch wheel (python3 -m pip install -U torch, unpinned); re-run after"
fi
TORCH_CUDA_MAJOR="$(printf '%s' "$TORCH_CUDA" | cut -d'.' -f1)"
case "$TORCH_CUDA_MAJOR" in
  ''|*[!0-9]*) die "cannot parse torch CUDA major from torch.version.cuda='${TORCH_CUDA}'" ;;
esac
# Wheel/apt names are DERIVED at run time from torch's CUDA build - a
# match to torch, not a version pin (latest-stable law, PROMPT.md 6).
CU_FULL="cu$(printf '%s' "$TORCH_CUDA" | tr -d '[:space:].')"      # 13.0 -> cu130, 12.8 -> cu128
CU_MAJOR="cu${TORCH_CUDA_MAJOR}"                                  # 13    -> cu13
APT_NAME="cuda-toolkit-$(printf '%s' "$TORCH_CUDA" | tr '.' '-')"  # 13.0  -> cuda-toolkit-13-0
log "torch CUDA build: ${TORCH_CUDA} (major ${TORCH_CUDA_MAJOR}); derived toolkit names: ${CU_FULL} / ${CU_MAJOR} / ${APT_NAME}"

SITE_PKGS="$(python3 -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])' 2>/dev/null || true)"

NVCC_BIN=""
NVCC_DIR=""
NVCC_MAJOR_NOW=""
NVCC_FULL=""

refresh_nvcc() {
  # Find nvcc: PATH first, then the venv's nvidia-cuda-nvcc wheel layout.
  NVCC_BIN=""
  NVCC_DIR=""
  if command -v nvcc >/dev/null 2>&1; then
    NVCC_BIN="$(command -v nvcc)"
    return 0
  fi
  if [ -n "$SITE_PKGS" ] && [ -x "${SITE_PKGS}/nvidia/cuda_nvcc/bin/nvcc" ]; then
    NVCC_DIR="${SITE_PKGS}/nvidia/cuda_nvcc/bin"
    NVCC_BIN="${NVCC_DIR}/nvcc"
    export PATH="${NVCC_DIR}:${PATH}"
    export CUDA_HOME="${SITE_PKGS}/nvidia/cuda_nvcc"
  fi
}

nvcc_major() {
  # Major version of the nvcc at $1 (empty if unparsable).
  "$1" --version 2>/dev/null | sed -n 's/.*release \([0-9][0-9]*\).*/\1/p' | tail -1 || true
}

nvcc_state() {
  refresh_nvcc
  NVCC_MAJOR_NOW=""
  NVCC_FULL=""
  if [ -n "$NVCC_BIN" ]; then
    NVCC_FULL="$("$NVCC_BIN" --version 2>/dev/null | tail -1 || true)"
    NVCC_MAJOR_NOW="$(nvcc_major "$NVCC_BIN")"
  fi
}

nvcc_state
if [ -n "$NVCC_BIN" ]; then
  log "nvcc now: ${NVCC_BIN} (major ${NVCC_MAJOR_NOW:-unparsed})"
else
  log "nvcc now: ABSENT"
fi

NEED_TOOLKIT=1
if [ -n "$NVCC_BIN" ] && [ -n "$NVCC_MAJOR_NOW" ]; then
  if [ "$NVCC_MAJOR_NOW" -ge "$TORCH_CUDA_MAJOR" ]; then
    NEED_TOOLKIT=0
    log "nvcc major ${NVCC_MAJOR_NOW} >= torch CUDA major ${TORCH_CUDA_MAJOR} - toolkit already sufficient; skipping install"
  else
    log "nvcc major ${NVCC_MAJOR_NOW} is OLDER than torch CUDA major ${TORCH_CUDA_MAJOR} - provisioning a matching toolkit"
  fi
elif [ -n "$NVCC_BIN" ]; then
  warn "nvcc present (${NVCC_BIN}) but its version line is unparsable - attempting toolkit provisioning anyway (harmless; idempotent)"
fi

TOOLKIT_OK=0
if [ "$NEED_TOOLKIT" -eq 1 ]; then
  # (a) pip nvcc wheels into the venv, unpinned -U, cuXX derived at run time.
  for WHEEL in "nvidia-cuda-nvcc-${CU_FULL}" "nvidia-cuda-nvcc-${CU_MAJOR}"; do
    if [ "$TOOLKIT_OK" -eq 1 ]; then break; fi
    log "attempt (pip wheel): python3 -m pip install -U ${WHEEL}"
    if python3 -m pip install -U "$WHEEL" \
       && [ -n "$SITE_PKGS" ] \
       && [ -x "${SITE_PKGS}/nvidia/cuda_nvcc/bin/nvcc" ]; then
      NVCC_DIR="${SITE_PKGS}/nvidia/cuda_nvcc/bin"
      NVCC_BIN="${NVCC_DIR}/nvcc"
      export PATH="${NVCC_DIR}:${PATH}"
      export CUDA_HOME="${SITE_PKGS}/nvidia/cuda_nvcc"
      WHEEL_MAJOR="$(nvcc_major "$NVCC_BIN")"
      if [ -n "$WHEEL_MAJOR" ] && [ "$WHEEL_MAJOR" -ge "$TORCH_CUDA_MAJOR" ]; then
        TOOLKIT_OK=1
        log "pip wheel nvcc OK: ${NVCC_BIN} (major ${WHEEL_MAJOR})"
        log "PATH/CUDA_HOME export hints (persist for future shells, e.g. ~/.bashrc or the venv activate script):"
        log "  export PATH=\"${NVCC_DIR}:\$PATH\""
        log "  export CUDA_HOME=\"${CUDA_HOME}\""
      else
        warn "wheel ${WHEEL} provides nvcc major '${WHEEL_MAJOR:-unparsed}' < torch CUDA major ${TORCH_CUDA_MAJOR} - trying next candidate"
      fi
    else
      log "pip wheel ${WHEEL} not usable - trying next candidate"
    fi
  done
fi

if [ "$NEED_TOOLKIT" -eq 1 ] && [ "$TOOLKIT_OK" -eq 0 ]; then
  # (b) apt fallback - only if root.
  IS_ROOT=0
  if [ "$(id -u)" -eq 0 ]; then IS_ROOT=1; fi
  HAS_APT=0
  if command -v apt-get >/dev/null 2>&1; then HAS_APT=1; fi
  if [ "$IS_ROOT" -eq 1 ] && [ "$HAS_APT" -eq 1 ]; then
    log "attempt (apt, root): apt-get update && apt-get install -y ${APT_NAME} (fallback: nvidia-cuda-toolkit)"
    apt-get update || log "apt-get update failed - continuing (the toolkit package may still be installable)"
    if apt-get install -y "$APT_NAME" || apt-get install -y nvidia-cuda-toolkit; then
      nvcc_state
      if [ -n "$NVCC_MAJOR_NOW" ] && [ "$NVCC_MAJOR_NOW" -ge "$TORCH_CUDA_MAJOR" ]; then
        TOOLKIT_OK=1
        log "apt toolkit OK: ${NVCC_BIN} (major ${NVCC_MAJOR_NOW})"
      else
        warn "apt installed a toolkit but nvcc major '${NVCC_MAJOR_NOW:-absent}' < torch CUDA major ${TORCH_CUDA_MAJOR} - not sufficient"
      fi
    else
      warn "apt toolkit install failed (packages unavailable)"
    fi
  else
    if [ "$IS_ROOT" -ne 1 ]; then log "apt fallback skipped: not root (uid $(id -u))"; fi
    if [ "$HAS_APT" -ne 1 ]; then log "apt fallback skipped: apt-get not available"; fi
  fi
fi

if [ "$NEED_TOOLKIT" -eq 1 ] && [ "$TOOLKIT_OK" -eq 0 ]; then
  # (c) nothing worked - hard failure with remediation (never silent).
  printf '[provision][FATAL] no nvcc with CUDA major >= %s (torch CUDA %s) could be provisioned - pip wheels and apt both failed\n' "$TORCH_CUDA_MAJOR" "$TORCH_CUDA" >&2
  cat >&2 <<EOF
[provision][FATAL] remediation - do ONE of these, then re-run this script (idempotent; it re-probes and continues):
  1) pip wheel (no root needed):
       python3 -m pip install -U nvidia-cuda-nvcc-${CU_FULL}
       export PATH="${SITE_PKGS}/nvidia/cuda_nvcc/bin:\$PATH"
       export CUDA_HOME="${SITE_PKGS}/nvidia/cuda_nvcc"
  2) NVIDIA CUDA toolkit for CUDA ${TORCH_CUDA} (runfile or network repo):
       https://developer.nvidia.com/cuda-downloads
       then put its bin/ directory on PATH
  3) root with the NVIDIA CUDA apt repo configured:
       apt-get install -y ${APT_NAME}
EOF
  exit 1
fi

# Record the resulting nvcc and re-assert (STEP 4 failures are fatal).
nvcc_state
if [ -z "$NVCC_BIN" ]; then
  die "STEP 4 invariant violated: no nvcc found after toolkit provisioning"
fi
if [ -z "$NVCC_MAJOR_NOW" ] || [ "$NVCC_MAJOR_NOW" -lt "$TORCH_CUDA_MAJOR" ]; then
  die "STEP 4 failed: nvcc major '${NVCC_MAJOR_NOW:-none}' does not match torch CUDA major ${TORCH_CUDA_MAJOR} (torch ${TORCH_VERSION} + CUDA ${TORCH_CUDA})"
fi
log "resulting nvcc: ${NVCC_FULL:-UNKNOWN}"
log "nvcc binary:    ${NVCC_BIN}"

# --------------------------------- STEP 5 - build flute_extended in place --
step "STEP 5/7 - build flute_extended in place (single-arch sm_86; ninja-parallel, MAX_JOBS capped)"

MAX_JOBS=$(( $(nproc) > 16 ? 16 : $(nproc) ))
log "nproc=$(nproc) -> MAX_JOBS=${MAX_JOBS} (cap 16; setup.py header: single-arch + per-family TUs keep rebuilds to minutes)"
log 'exact command: MAX_JOBS=$(( $(nproc) > 16 ? 16 : $(nproc) )) FLUTE_CUDA_ARCHES=86 python3 setup.py build_ext --inplace'
log "cd ${REPO_ROOT}/src/flute_extended"
cd "$REPO_ROOT/src/flute_extended"
MAX_JOBS=$(( $(nproc) > 16 ? 16 : $(nproc) )) FLUTE_CUDA_ARCHES=86 python3 setup.py build_ext --inplace
log "flute_extended built in place (FLUTE_CUDA_ARCHES=86 -> sm_86 codegen; MAX_JOBS=${MAX_JOBS})"
cd "$REPO_ROOT"

# ---------------------- STEP 6 - pre-W20 regression gate (the 237 suite) --
step "STEP 6/7 - pre-W20 regression gate: the 237-test baseline suite (ON THE BOX)"

log "running from repo root: python3 -m pytest src/rag/tests -q"
log "this is the PRE-W20 regression gate (the 237-test baseline); the script reports the exit code and does NOT abort - the operator decides"
BASELINE_RC=0
python3 -m pytest src/rag/tests -q || BASELINE_RC=$?
if [ "$BASELINE_RC" -eq 0 ]; then
  echo "==================================================================================="
  echo "  BASELINE SUITE: PASS   (python3 -m pytest src/rag/tests -q -> exit 0)"
  echo "  pre-W20 regression gate GREEN - W20 work may proceed on this box"
  echo "==================================================================================="
else
  echo "==================================================================================="
  echo "  BASELINE SUITE: FAIL   (python3 -m pytest src/rag/tests -q -> exit ${BASELINE_RC})"
  echo "  pre-W20 regression gate RED - investigate BEFORE any W20 work on this box"
  echo "  (this script does not abort on test failure; re-runs do not mask this banner)"
  echo "==================================================================================="
  warn "baseline suite exited ${BASELINE_RC} - reported, not fatal (operator decides)"
fi

# ---------------------------------------- STEP 7 - write the environment lock
step "STEP 7/7 - write env/w20_gpu_env.lock (evidence, not a constraint)"

NINJA_VER="$(pkg_version ninja)"
TRANSFORMERS_VER="$(pkg_version transformers)"
FAISS_VER="$(pkg_version faiss-cpu)"
NUMPY_VER="$(pkg_version numpy)"
if [ -z "$NINJA_VER" ] || [ -z "$TRANSFORMERS_VER" ] || [ -z "$FAISS_VER" ] || [ -z "$NUMPY_VER" ]; then
  warn "one or more pip package versions unreadable via 'pip show' - recorded as UNKNOWN in the lock"
fi

GIT_SHA="unknown"
if command -v git >/dev/null 2>&1; then
  GIT_SHA="$(git -C "$REPO_ROOT" rev-parse --short HEAD 2>/dev/null || echo unknown)"
fi

mkdir -p "$REPO_ROOT/env"
LOCK="$REPO_ROOT/env/w20_gpu_env.lock"
cat > "$LOCK" <<EOF
# W20 GPU environment lock - written by scripts/gpu/w20_provision.sh (W20.0.2)
# evidence, not a constraint: re-provisioning re-queries and rewrites this file.
date:               $(date -u '+%Y-%m-%dT%H:%M:%SZ')
device:             ${DEVICE_NAME}
compute_capability: ${COMPUTE_CAP}
driver_version:     ${DRIVER_VERSION}
cuda_driver_api:    ${CUDA_DRIVER_API}
os:                 ${OS_DESC}
torch:              ${TORCH_VERSION}
torch_cuda:         ${TORCH_CUDA}
nvcc:               ${NVCC_FULL:-UNKNOWN}
nvcc_path:          ${NVCC_BIN:-UNKNOWN}
ninja:              ${NINJA_VER:-UNKNOWN}
transformers:       ${TRANSFORMERS_VER:-UNKNOWN}
faiss-cpu:          ${FAISS_VER:-UNKNOWN}
numpy:              ${NUMPY_VER:-UNKNOWN}
python:             ${PYTHON_VERSION}
max_jobs:           ${MAX_JOBS}
git_short_sha:      ${GIT_SHA}
baseline_suite_exit_code: ${BASELINE_RC}
note: latest-stable at run time; unpinned by design (PROMPT.md §6)
EOF
log "lock written (repo-root-relative: env/w20_gpu_env.lock): ${LOCK}"
log "lock contents:"
sed 's/^/[provision]   /' "$LOCK"

# ----------------------------------------------------------------- summary --
step "SUMMARY - W20.0.2 A10G provisioning"

log "device:      ${DEVICE_NAME} (sm_${COMPUTE_CAP}; driver ${DRIVER_VERSION}; CUDA driver API ${CUDA_DRIVER_API})"
log "torch:       ${TORCH_VERSION} + CUDA ${TORCH_CUDA} | nvcc: ${NVCC_FULL:-UNKNOWN}"
log "deps:        ninja ${NINJA_VER:-UNKNOWN} | transformers ${TRANSFORMERS_VER:-UNKNOWN} | faiss-cpu ${FAISS_VER:-UNKNOWN} | numpy ${NUMPY_VER:-UNKNOWN} (all unpinned)"
log "python:      ${PYTHON_VERSION} | nproc ${NPROC} | MAX_JOBS ${MAX_JOBS} (cap 16)"
log "flute:       built in place - FLUTE_CUDA_ARCHES=86, MAX_JOBS capped at 16"
log "baseline:    237-suite exit code ${BASELINE_RC} (banner above; reported, not fatal - operator decides)"
log "lock:        env/w20_gpu_env.lock (${LOCK})"
log "re-run safe: idempotent; every step re-echoes; the lock is rewritten each run"
log "provisioning complete - exit 0 (the baseline-suite verdict is recorded above and in the lock)"
exit 0
