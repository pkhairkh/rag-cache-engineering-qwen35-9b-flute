"""test_w27_regime_guards.py — W27: the gates-artifact and ingest-regime
guards that the box's hermes session forced into existence.

THE INCIDENT (the uploaded W26 box log, verbatim): the box's only gates
artifact is a pre-W21 (24,) per-layer file — the load_gates shape check
refused it — so the on-box agent HAND-BROADCAST it to (24, 32)
(`m1m2_gates_trained_fixed.npz`, np.repeat across heads) and that file
slipped past every check load_gates had: right shapes, right geometry
meta, no regime meta to disagree with. It then rode the E1-E3 ladder,
and E1's garbage answer was scored "above the bar" (conf 0.382) by a
gen gate that cannot see mixed-script soup. Three guards close exactly
that surface:

  A. THE REGIME META IS MANDATORY — an artifact without
     meta_write_addr cannot load into a content-addressing module
     (every save_gates file since W21 carries it; the driver passes it
     AND W27 bakes it in). The broadcast forgery has correct shapes and
     no regime meta -> REFUSED here, not at query time on the GPU box.
  B. THE BROADCAST SIGNATURE — a write_gate_k whose rows are constant
     across every head is measure-zero under real training and exactly
     what np.repeat((L,), H) produces -> REFUSED, with a message that
     names the box file that motivated it. (write_gate_v is NOT
     checked: fill_(c) is a legitimate calibration pattern the W17/W21
     suites themselves use.)
  C. THE CORPUS-REGIME INTROSPECTION — index.corpus_memory_state: the
     empty (zero-write ingest) / carries-content / absent trichotomy
     that run_query.py's W27 regime match consumes; a fake loader pins
     the slicing (the memory tail of the §4 delta) and the threshold
     semantics, and the real-loader leg runs in test_query.py.

Device-agnostic (CUDA when available, else CPU), no network, no
transformers import — module-level contracts only.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

import m1m2_finetune as ft
from index import corpus_memory_state
from m1m2 import M1M2

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

H, D, MEM, NLAY = 4, 16, 8, 3


def _module(**kw) -> M1M2:
    return M1M2(num_heads=H, head_dim=D, mem_size=MEM,
                num_linear_layers=NLAY, **kw).to(DEVICE)


def _gates_arrays(k_fill="linspace", v_fill=0.2):
    k = (torch.linspace(-0.3, 0.3, NLAY * H).view(NLAY, H)
         if k_fill == "linspace"
         else torch.full((NLAY, H), float(k_fill)))
    v = torch.full((NLAY, H), float(v_fill))
    r = torch.ones(NLAY)
    return {"write_gate_k": k.double().numpy(),
            "write_gate_v": v.double().numpy(),
            "read_gate": r.double().numpy()}


def _geometry_meta():
    return {f"meta_{k}": v for k, v in (
        ("num_heads", H), ("head_dim", D), ("mem_size", MEM),
        ("num_linear_layers", NLAY))}


# ================================= A. the mandatory regime meta ============
def test_w27a_regime_meta_is_mandatory(tmp_path):
    """The hermes forgery EXACTLY: (L,) values np.repeat-ed to (L, H),
    geometry meta intact, NO meta_write_addr -> the shape and geometry
    checks pass, the artifact is STILL refused (it cannot address a
    content-addressing kernel). The refusal message names the missing
    key so the box log is actionable."""
    mod = _module()
    # np.repeat of a per-layer vector: the literal construction the box
    # session used to build m1m2_gates_trained_fixed.npz
    per_layer = np.linspace(-0.3, 0.3, NLAY)
    arrays = {
        "write_gate_k": np.repeat(per_layer[:, None], H, axis=1),
        "write_gate_v": np.repeat((per_layer * 0.5)[:, None], H, axis=1),
        "read_gate": np.ones(NLAY),
    }
    path = str(tmp_path / "broadcast_no_meta.npz")
    np.savez(path, **arrays, **_geometry_meta())
    with pytest.raises(ValueError, match="meta_write_addr"):
        ft.load_gates(mod, path)
    # nothing mutated before the refusal (the guard runs pre-copy)
    assert torch.all(mod.write_gate_k == 0.0)


# ================================= B. the broadcast signature ==============
def test_w27b_broadcast_k_refused(tmp_path):
    """A file WITH the right regime meta and right shapes whose
    write_gate_k rows are constant across every head — the broadcast
    signature — is refused even though every earlier check passes.
    (write_gate_v is uniform here too and is deliberately NOT flagged:
    fill_ is a legitimate calibration pattern the W17/W21 suites use.)"""
    mod = _module()
    per_layer = np.linspace(-0.3, 0.3, NLAY)
    arrays = {
        "write_gate_k": np.repeat(per_layer[:, None], H, axis=1),
        "write_gate_v": np.full((NLAY, H), 0.2),
        "read_gate": np.ones(NLAY),
    }
    path = str(tmp_path / "broadcast_with_meta.npz")
    np.savez(path, **arrays, **_geometry_meta(),
             meta_write_addr="content", meta_center_write=True,
             meta_addr_blocks=4)
    with pytest.raises(ValueError, match="broadcast signature"):
        ft.load_gates(mod, path)
    assert torch.all(mod.write_gate_k == 0.0)


def test_w27c_genuine_artifact_loads_and_save_bakes_regime(tmp_path):
    """The honest round-trip post-W27: save_gates BAKES the regime meta
    (write_addr/center_write/addr_blocks) even when the caller passes
    none, and load_gates accepts that file; a regime-mismatched load
    (content artifact into a positional module) still refuses."""
    mod = _module()
    with torch.no_grad():
        mod.write_gate_k.copy_(
            torch.linspace(-0.2, 0.2, NLAY * H).view(NLAY, H).to(DEVICE))
        mod.write_gate_v.fill_(0.1)
    want = {k: getattr(mod, k).detach().clone()
            for k in ("write_gate_k", "write_gate_v", "read_gate")}
    path = ft.save_gates(mod, str(tmp_path / "g"))     # NO extra_meta
    # the regime meta is IN the file (baked, not caller-supplied)
    with np.load(path) as z:
        assert "meta_write_addr" in z.files
        assert str(z["meta_write_addr"].item()) == "content"
        assert "meta_center_write" in z.files
        assert "meta_addr_blocks" in z.files
    fresh = _module()
    meta = ft.load_gates(fresh, path)
    assert str(meta["write_addr"]) == "content"
    for k, w in want.items():
        assert torch.equal(getattr(fresh, k).detach(), w), k
    # the regime guard still bites: content artifact -> positional module
    pos = _module(write_addr="positional", center_write=False)
    with pytest.raises(ValueError, match="write_addr"):
        ft.load_gates(pos, path)


# ================================= C. the corpus-regime introspection =====
def _fake_loader(vectors, d_tail):
    """chunk_ids() + delta_vector() over fixed vectors; the system
    carries m1/m2 code stubs of total tail length d_tail (the helper
    reads only .d off them)."""
    m = SimpleNamespace(
        m1_codes=SimpleNamespace(d=d_tail // 2),
        m2_codes=SimpleNamespace(d=d_tail - d_tail // 2))
    return SimpleNamespace(
        chunk_ids=lambda: list(range(len(vectors))),
        delta_vector=lambda cid: vectors[int(cid)],
        system=m)


def test_w27d_memory_state_empty():
    """The zero-write corpus: every chunk's memory tail is exactly zero
    (the idempotent requant of a norm-0 unit) while the S head carries
    content -> True (EMPTY)."""
    d_s, d_m = 8, 4
    head = torch.tensor([3.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0])
    vecs = [torch.cat([head, torch.zeros(d_m)]) for _ in range(5)]
    assert corpus_memory_state(_fake_loader(vecs, d_m)) is True


def test_w27e_memory_state_carries_content():
    """One sampled chunk with real memory content -> False (the corpus
    needs the ingest-time gates artifact on the query side)."""
    d_s, d_m = 8, 4
    head = torch.tensor([3.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0])
    vecs = [torch.cat([head, torch.zeros(d_m)]),
            torch.cat([head, torch.tensor([0.5, 0.5, 0.5, 0.5])]),
            torch.cat([head, torch.zeros(d_m)])]
    assert corpus_memory_state(_fake_loader(vecs, d_m)) is False


def test_w27f_memory_state_absent():
    """No m1/m2 codes on the system (or no chunks) -> None: there is no
    memory regime to match at all."""
    head = torch.ones(8)
    loader = SimpleNamespace(
        chunk_ids=lambda: [0],
        delta_vector=lambda cid: head.clone(),
        system=SimpleNamespace(m1_codes=None, m2_codes=None))
    assert corpus_memory_state(loader) is None
    empty = SimpleNamespace(chunk_ids=lambda: [], system=None)
    assert corpus_memory_state(empty) is None


def test_w27g_memory_state_loud_on_layout_drift():
    """A delta vector SHORTER than the declared memory tail is a layout
    disagreement — loud ValueError, never a silent slice."""
    head = torch.ones(4)
    loader = _fake_loader([head], d_tail=8)
    with pytest.raises(ValueError, match="memory tail"):
        corpus_memory_state(loader)
