#!/usr/bin/env bash
# scripts/provision_env.sh — GPU-box environment provisioning (sibling
# of provision_env_cpu.sh, which targets the CPU-only coding box).
#
# Installs the latest stable toolchain (CUDA toolkit + PyTorch wheel
# matching the driver's CUDA capability + Python dependencies), pins the
# resolved set into requirements.lock.txt, and records the environment in
# reports/environment.json. "Latest stable" is queried from the official
# channels at execution time — no cached version knowledge.
#
# Usage:
#   scripts/provision_env.sh            # install + pin + report
#   scripts/provision_env.sh --check    # verify only; exit 1 if stale
#
# Requirements: bash, curl, nvidia-smi (driver installed), python3 (>=3.10),
# pip. The CUDA toolkit install uses the apt path when the NVIDIA repo is
# configured; otherwise the runfile path (downloaded into ./downloads).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CHECK_ONLY=0
[[ "${1:-}" == "--check" ]] && CHECK_ONLY=1

log()  { echo "[provision] $*"; }
fail() { echo "[provision] ERROR: $*" >&2; exit 1; }

# ---------------------------------------------------------------------------
# 0. Prerequisites
# ---------------------------------------------------------------------------
command -v curl >/dev/null || fail "curl is required"
command -v nvidia-smi >/dev/null || fail "nvidia-smi not found — install the NVIDIA driver first"
command -v python3 >/dev/null || fail "python3 not found (>= 3.10 required)"
PYTHON=python3
[[ "$($PYTHON -c 'import sys; print(sys.version_info >= (3, 10))')" == "True" ]] \
    || fail "python3 >= 3.10 required"

DRIVER_CUDA="$(nvidia-smi | grep -oE 'CUDA Version: [0-9]+' | head -1 | grep -oE '[0-9]+')"
[[ -n "$DRIVER_CUDA" ]] || fail "could not read the driver's CUDA capability from nvidia-smi"
GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
log "driver CUDA capability: ${DRIVER_CUDA} (GPU: ${GPU_NAME})"

mkdir -p "$REPO_ROOT/reports" "$REPO_ROOT/downloads"

# ---------------------------------------------------------------------------
# 1. CUDA toolkit (nvcc, compute-sanitizer, Nsight): latest stable from the
#    NVIDIA apt repo index for this distribution.
# ---------------------------------------------------------------------------
detect_ubuntu_codename() {
    if [[ -r /etc/os-release ]]; then
        # shellcheck disable=SC1091
        . /etc/os-release
        case "${VERSION_ID:-}" in
            20.*) echo ubuntu2004 ;;
            22.*) echo ubuntu2204 ;;
            24.*) echo ubuntu2404 ;;
            *)    echo "" ;;
        esac
    fi
}

latest_repo_cuda_version() {
    # Parse the newest cuda-toolkit-* package version from the repo index.
    local repo_url="https://developer.download.nvidia.com/compute/cuda/repos/$1/x86_64/Packages.gz"
    curl -fsSL "$repo_url" 2>/dev/null | zcat 2>/dev/null \
        | grep -oE '^Package: cuda-toolkit-[0-9-]+$' \
        | grep -oE '[0-9-]+$' | tr '-' '.' \
        | sort -t. -k1,1n -k2,2n -k3,3n | tail -1 || true
}

ubuntu_codename="$(detect_ubuntu_codename)"
LATEST_CUDA=""
if [[ -n "$ubuntu_codename" ]]; then
    LATEST_CUDA="$(latest_repo_cuda_version "$ubuntu_codename")"
fi
[[ -n "$LATEST_CUDA" ]] || fail "could not query the latest stable CUDA version from the NVIDIA repo (distro: ${ubuntu_codename:-unknown})"
log "latest stable CUDA toolkit: ${LATEST_CUDA}"

installed_nvcc_version() {
    command -v nvcc >/dev/null && nvcc --version | grep -oE 'release [0-9.]+' | grep -oE '[0-9.]+' || true
}

NVCC_NOW="$(installed_nvcc_version)"
if [[ -z "$NVCC_NOW" || "$NVCC_NOW" != "$LATEST_CUDA" ]]; then
    if [[ "$CHECK_ONLY" -eq 1 ]]; then
        fail "nvcc ${NVCC_NOW:-absent} != latest stable ${LATEST_CUDA} (rerun without --check to install)"
    fi
    CUDA_MAJOR="$(echo "$LATEST_CUDA" | cut -d. -f1)"
    CUDA_MINOR="$(echo "$LATEST_CUDA" | cut -d. -f2)"
    APT_PKG="cuda-toolkit-${CUDA_MAJOR}-${CUDA_MINOR}"
    if apt-cache show "$APT_PKG" >/dev/null 2>&1; then
        log "installing ${APT_PKG} via apt (requires root)"
        sudo apt-get install -y "$APT_PKG"
    else
        RUNFILE="cuda_${LATEST_CUDA}_linux.run"
        URL="https://developer.download.nvidia.com/compute/cuda/${LATEST_CUDA}/local_installers/${RUNFILE}"
        log "apt package ${APT_PKG} unavailable; runfile path: ${URL}"
        [[ -f "$REPO_ROOT/downloads/$RUNFILE" ]] || curl -fSL -o "$REPO_ROOT/downloads/$RUNFILE" "$URL"
        sudo sh "$REPO_ROOT/downloads/$RUNFILE" --silent --toolkit
    fi
    NVCC_NOW="$(installed_nvcc_version)"
    [[ "$NVCC_NOW" == "$LATEST_CUDA" ]] || fail "toolkit install did not yield ${LATEST_CUDA} (got ${NVCC_NOW:-none})"
else
    log "nvcc ${NVCC_NOW} is already the latest stable"
fi

for tool in compute-sanitizer ncu nsys; do
    command -v "$tool" >/dev/null || fail "$tool not on PATH after toolkit install (expected in /usr/local/cuda/bin)"
done
command -v gcc >/dev/null || fail "gcc not found (CUDA toolkit dependency)"

# ---------------------------------------------------------------------------
# 2. PyTorch: latest stable wheel from the PyPI index (wheels bundle the
#    matching CUDA runtime; the driver capability must cover it).
# ---------------------------------------------------------------------------
TORCH_INDEX="https://download.pytorch.org/whl/cpu"
TORCH_TAG="cpu"
if [[ "$DRIVER_CUDA" -ge 12 ]]; then
    TORCH_INDEX="https://download.pytorch.org/whl/cu124"
    TORCH_TAG="cu124"
elif [[ "$DRIVER_CUDA" -ge 11 ]]; then
    TORCH_INDEX="https://download.pytorch.org/whl/cu118"
    TORCH_TAG="cu118"
else
    fail "driver CUDA capability ${DRIVER_CUDA} is below every supported torch wheel"
fi
log "torch wheel index: ${TORCH_INDEX} (driver capability ${DRIVER_CUDA})"

LATEST_TORCH="$($PYTHON -m pip index versions torch --index-url "$TORCH_INDEX" 2>/dev/null \
    | grep -oE 'torch \([0-9][^)]*\)' | head -1 | grep -oE '[0-9][a-zA-Z0-9+.]+' | head -1 || true)"
[[ -n "$LATEST_TORCH" ]] || LATEST_TORCH="$($PYTHON -m pip index versions torch 2>/dev/null \
    | grep -oE '\([0-9][^)]*\)' | head -1 | tr -d '()')"
[[ -n "$LATEST_TORCH" ]] || fail "could not query the latest stable torch version"
log "latest stable torch: ${LATEST_TORCH}"

TORCH_NOW="$($PYTHON -c 'import torch; print(torch.__version__)' 2>/dev/null || true)"
if [[ -z "$TORCH_NOW" || "$TORCH_NOW" != "$LATEST_TORCH"* ]]; then
    if [[ "$CHECK_ONLY" -eq 1 ]]; then
        fail "torch ${TORCH_NOW:-absent} != latest stable ${LATEST_TORCH} (rerun without --check to install)"
    fi
    $PYTHON -m pip install --upgrade "torch==${LATEST_TORCH}" --index-url "$TORCH_INDEX"
else
    log "torch ${TORCH_NOW} is already the latest stable"
fi

# ---------------------------------------------------------------------------
# 3. Python dependencies from requirements.txt, latest stable each.
# ---------------------------------------------------------------------------
if [[ "$CHECK_ONLY" -eq 0 ]]; then
    log "installing python dependencies (latest stable, from requirements.txt ranges)"
    $PYTHON -m pip install --upgrade -r "$REPO_ROOT/requirements.txt"
fi

# ---------------------------------------------------------------------------
# 4. Pin + report.
# ---------------------------------------------------------------------------
$PYTHON -m pip freeze > "$REPO_ROOT/requirements.lock.txt"
log "pinned: requirements.lock.txt"

TORCH_CUDA="$($PYTHON -c 'import torch; print(torch.version.cuda or "none")')"
NVCC_VER="$(nvcc --version | tail -1)"
DRIVER_VER="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)"
KERNEL_VER="$(uname -r)"

"$PYTHON" - "$REPO_ROOT/reports/environment.json" "$LATEST_CUDA" "$LATEST_TORCH" \
    "$TORCH_TAG" "$GPU_NAME" "$DRIVER_VER" "$TORCH_CUDA" "$NVCC_VER" "$KERNEL_VER" <<'PY'
import json, sys, subprocess, datetime, pathlib
out = pathlib.Path(sys.argv[1])
report = {
    "provisioned_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "cuda_toolkit_latest_stable": sys.argv[2],
    "torch_latest_stable": sys.argv[3],
    "torch_wheel_tag": sys.argv[4],
    "gpu": sys.argv[5],
    "driver_version": sys.argv[6],
    "torch_bundled_cuda": sys.argv[7],
    "nvcc": sys.argv[8],
    "kernel": sys.argv[9],
    "python": sys.version.split()[0],
}
report["nvcc_version"] = subprocess.run(["nvcc", "--version"], capture_output=True, text=True).stdout.strip().splitlines()[-1]
report["sanitizer_available"] = subprocess.run(["which", "compute-sanitizer"], capture_output=True).returncode == 0
out.write_text(json.dumps(report, indent=2) + "\n")
print(f"[provision] environment report: {out}")
PY

log "provision complete (toolkit ${LATEST_CUDA}, torch ${LATEST_TORCH})"
log "next: build both extensions (RUNBOOK 2 pre-flight) and rerun this script with --check to confirm freshness"
