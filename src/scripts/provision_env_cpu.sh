#!/usr/bin/env bash
# scripts/provision_env_cpu.sh — coding-box (CPU-only) environment
# provisioning, sibling of provision_env.sh (which targets the GPU box
# and requires nvidia-smi). TASKS.md W0.T1.
#
# Creates the dedicated venv ~/venv-coding and installs the latest
# stable CPU builds of every dependency this campaign needs:
#   torch (CPU wheel — never the CUDA build), numpy, pytest,
#   transformers, safetensors, tokenizers, datasets, accelerate, and
#   triton (latest stable — importable and interpreter-mode-runnable
#   without a GPU; the SM86 kernel compiles JIT on the GPU box).
# No version pins unless a real incompatibility forces one (any pin +
# reason is recorded in reports/env_coding_box.txt header).
#
# Records the environment: pip freeze > reports/env_coding_box.txt and
# verifies imports with a smoke check.
#
# Usage:
#   scripts/provision_env_cpu.sh            # provision + report
#   scripts/provision_env_cpu.sh --check    # verify only
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="${HOME}/venv-coding"
CHECK_ONLY=0
[[ "${1:-}" == "--check" ]] && CHECK_ONLY=1

log()  { echo "[provision-cpu] $*"; }
fail() { echo "[provision-cpu] ERROR: $*" >&2; exit 1; }

command -v python3 >/dev/null || fail "python3 not found (>= 3.10 required)"
PYTHON=python3
[[ "$($PYTHON -c 'import sys; print(sys.version_info >= (3, 10))')" == "True" ]] \
    || fail "python3 >= 3.10 required"
command -v git >/dev/null || fail "git not found"

# This box must never see CUDA: refuse to run if a GPU is present.
if command -v nvidia-smi >/dev/null && nvidia-smi >/dev/null 2>&1; then
    fail "nvidia-smi present — this script is for the CPU-only coding box"
fi

mkdir -p "$REPO_ROOT/reports"
PIP="$VENV/bin/python -m pip"

# ---------------------------------------------------------------------------
# 1. Venv + CPU packages (latest stable each).
# ---------------------------------------------------------------------------
if [[ "$CHECK_ONLY" -eq 0 ]]; then
    if [[ ! -x "$VENV/bin/python" ]]; then
        log "creating venv $VENV"
        $PYTHON -m venv "$VENV"
    fi
    log "upgrading pip"
    $PIP install --upgrade pip >/dev/null

    # torch: CPU wheel from the PyTorch CPU index — never the CUDA build.
    TORCH_INDEX="https://download.pytorch.org/whl/cpu"
    log "installing torch (latest stable, CPU wheel)"
    $PIP install --upgrade torch --index-url "$TORCH_INDEX"

    # everything else from PyPI, latest stable.
    log "installing python dependencies (latest stable)"
    $PIP install --upgrade numpy pytest transformers safetensors tokenizers
    $PIP install --upgrade datasets accelerate

    # triton: latest stable; importable + interpreter-mode-runnable with
    # no GPU present (the SM86 attention kernel JIT-compiles on the box).
    log "installing triton (latest stable)"
    $PIP install --upgrade triton
fi

[[ -x "$VENV/bin/python" ]] || fail "venv $VENV missing (rerun without --check)"

# ---------------------------------------------------------------------------
# 2. Record + smoke.
# ---------------------------------------------------------------------------
{
    echo "# coding-box environment (CPU-only) — pip freeze"
    echo "# provisioned by scripts/provision_env_cpu.sh at $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "# python $(${VENV}/bin/python --version 2>&1 | grep -oE '[0-9.]+')"
    echo "# pins: none (all latest stable at install time)"
    $PIP freeze
} > "$REPO_ROOT/reports/env_coding_box.txt"
log "recorded: reports/env_coding_box.txt"

log "smoke: importing torch, numpy, pytest, transformers, safetensors, tokenizers"
"$VENV/bin/python" - <<'PY'
import torch, numpy, pytest, transformers, safetensors, tokenizers
assert not torch.cuda.is_available(), "CPU wheel must not expose CUDA"
assert not torch.version.cuda, f"expected a CPU wheel, got cuda={torch.version.cuda}"
print(f"smoke ok: torch {torch.__version__} (cpu), numpy {numpy.__version__}, "
      f"pytest {pytest.__version__}, transformers {transformers.__version__}, "
      f"safetensors {safetensors.__version__}, tokenizers {tokenizers.__version__}")
PY

# triton smoke: plain import + interpreter mode in a SUBPROCESS (the env
# var must be set before triton is imported).
log "smoke: triton import + TRITON_INTERPRET=1 interpreter-mode smoke"
"$VENV/bin/python" - <<'PY'
import triton
print(f"smoke ok: triton {triton.__version__} (no GPU present, import-only)")
PY
TRITON_INTERPRET=1 "$VENV/bin/python" - <<'PY'
import os
assert os.environ.get("TRITON_INTERPRET") == "1"
import triton
import triton.language as tl
import torch

@triton.jit
def _add_kernel(x_ptr, y_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    y = tl.load(y_ptr + offs, mask=mask)
    tl.store(y_ptr + offs, x + 1.0, mask=mask)

x = torch.arange(32, dtype=torch.float32)
y = torch.zeros_like(x)
_add_kernel[(1,)](x, y, x.numel(), BLOCK=32)
assert torch.allclose(y, x + 1.0), "interpreter-mode add kernel mismatch"
print(f"smoke ok: TRITON_INTERPRET=1 tl kernel ran (triton {triton.__version__})")
PY

log "provision-cpu complete; next: python -m pytest tests/ -q from the venv"
