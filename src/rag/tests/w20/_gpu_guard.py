"""W20 GPU-ONLY: every test in this package requires a CUDA device (Ampere
A10G target); CPU boxes skip loudly at collection (PROMPT.md §1.3).

Helper module — NOT a test file (zero test functions). Importing it is the
canonical W20 collection-time guard: ``require_cuda()`` is invoked at the
bottom of this module, so any test file that imports it hard-skips loudly
on a CPU box — visible skips, never silent passes.

Canonical first lines of a W20 test file::

    from src.rag.tests.w20 import _gpu_guard

    _gpu_guard.require_cuda()  # module top: CPU boxes skip loudly here

On the A10G box, ``device_report()`` describes the gate device and
``gate_report_path(gate_id)`` pins where the gate's JSON report lands.
"""

import warnings
from pathlib import Path

import pytest
import torch

__all__ = ["require_cuda", "device_report", "gate_report_path"]

# A10G target per PROPOSAL.md §4: Ampere family (major 8), capability (8, 6).
_AMPERE_MAJOR = 8
_A10G_CAPABILITY = (8, 6)


def require_cuda() -> None:
    """Skip this module loudly unless a CUDA device is available.

    Called at import (collection) time — both at the bottom of this module
    and at the top of every W20 test file — so a CPU run shows a visible
    module-level skip and never a silent pass (PROMPT.md §1.3). On the
    A10G box this is a no-op and collection proceeds.
    """
    if not torch.cuda.is_available():
        pytest.skip("W20 GPU-ONLY: A10G required", allow_module_level=True)


def device_report() -> dict:
    """Return a dict describing the CUDA device W20 gates run on.

    Keys: ``device_name`` (device 0), ``capability`` ((major, minor) tuple),
    ``torch_version``, ``cuda_version``, ``device_count``.

    Raises:
        RuntimeError: no CUDA device is present ("A10G required"), or the
            device is outside the Ampere family — capability major != 8
            (PROPOSAL.md §4).

    Warns:
        UserWarning: the device is Ampere but not the A10G target, i.e.
            capability != (8, 6) — a warning, never a failure.
    """
    if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
        raise RuntimeError(
            "W20 GPU-ONLY: device_report() requires a CUDA device — "
            "A10G required (PROMPT.md §1.3)"
        )
    props = torch.cuda.get_device_properties(0)
    capability = (props.major, props.minor)
    if capability[0] != _AMPERE_MAJOR:
        raise RuntimeError(
            f"W20 GPU-ONLY: compute capability {capability} is outside the "
            f"Ampere family (major != {_AMPERE_MAJOR}); A10G (8, 6) "
            "required (PROPOSAL.md §4)"
        )
    if capability != _A10G_CAPABILITY:
        warnings.warn(
            f"W20 gate device is Ampere but not the A10G target: capability "
            f"{capability} != (8, 6); results are advisory for this device",
            stacklevel=2,
        )
    return {
        "device_name": props.name,
        "capability": capability,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "device_count": torch.cuda.device_count(),
    }


def gate_report_path(gate_id: str) -> Path:
    """Return the JSON gate-report path for ``gate_id``, directory created.

    The path is ``<repo_root>/env/w20_gate_reports/<gate_id>.json``. The
    repo root is derived from this file's own location —
    ``Path(__file__).resolve().parents[4]`` (w20 -> tests -> rag -> src ->
    repo root) — with no repo-module imports (src.rag._paths is never
    touched). ``mkdir(parents=True, exist_ok=True)`` runs on call so the
    A10G gate scripts can write their reports immediately.
    """
    repo_root = Path(__file__).resolve().parents[4]
    report_dir = repo_root / "env" / "w20_gate_reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    return report_dir / f"{gate_id}.json"


# Import-time guard — the only statement executed at import beyond imports
# and constants: a skip, not an assertion. On a CPU box the Skipped exception
# propagates through the importing test module to pytest's collector (loud
# module-level skip); on the A10G box this is a no-op.
require_cuda()
