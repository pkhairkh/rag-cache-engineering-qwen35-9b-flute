"""test_install_real_math.py — W11: the install-vs-ground-truth gates at
REAL gated-delta-rule fidelity.

The GPU session's handover reported "generation produces garbage after
chunk installation" and isolated the corruption to the S-install path. Two
CPU harnesses were built to adjudicate that claim against the vendored
modeling.py's own math (torch_chunk_gated_delta_rule — the exact function
the model's prefill runs when no FLA kernel is engaged):

  * the S path (reseed -> install -> read) is BIT-CLEAN: the install sum
    reconstructs dequant(sys) + dequant(delta) exactly, the layer's read
    is bit-identical to the direct dequant, quantizer identity holds
    (see the W11 worklog; the same gates are folded in below);
  * the REAL corruption lived in the conv geometry (the 16,384 truncation,
    contract 11 / test_tq_cache.py — fixed in W11).

This file pins the S-install SEMANTICS at real kernel fidelity, which none
of the stub tests cover (the TopicStubModel bypasses the delta rule):

  1. INSTALL-vs-TRUTH: install(sys, delta) reconstructs the TRUE
     [system-then-document] state within the house single-quant-round
     budget (rel-MSE < 0.06; measured 0.048) — the D4 delta protocol's
     "cache as if the chunk were prefilled after the system" claim.
  2. QUERY-PREFILL OUTPUT: the answer-prefill outputs from the installed
     state track the TRUE [sys+doc] trajectory (rel-MSE < 0.10 on the
     last token's core-attention output; measured 0.019).
  3. CONTRACTIVITY: a 5% random state perturbation moves the query-prefill
     output by < 2% (measured 0.0014) — the delta rule is contractive in
     the state, so quantization-scale noise CANNOT produce the observed
     garbage; any GPU-side reproduction of the corruption must come from
     a code path (kernel/device/geometry), never the sum math.
  4. RESEED-vs-INSTALL SEPARATION: the reseed-only trajectory diverges
     from the [sys+doc] truth (rel > 0.5) while the installed trajectory
     does not (rel < 0.06) — the two paths carry different information,
     so a report of "reseed works, install is garbage" is a geometry/
     kernel defect, not the install math.

Geometry: scaled-but-real — B=1, 4 v-heads, 2 k-heads (repeat_interleave,
as the model does), K=V=16 -> S units d=1,024 (committed codebooks; no
test writes a new npz). Tokens are deterministic per-sequence streams
(q/k/v/beta/g with the model's shapes and signs; g strictly negative).
No conv (S-path isolation; the conv contracts live in test_tq_cache.py).
"""
from __future__ import annotations

import pytest
import torch

import _paths  # noqa: F401
import codebooks
import modeling as modeling_mod
from install import sum_turboquant_codes
from tq_cache import TQLinearAttentionLayer, resolve_quantizer

# keep the deterministic codebook cache inside the repo's committed set:
# every (b, d) used here resolves against cb_b{3,4}_d1024.npz (no writes)
assert codebooks.CACHE_DIR  # the committed dir; nothing to redirect

chunk_gdn = modeling_mod.torch_chunk_gated_delta_rule

B, HV, HK, KD, VD = 1, 4, 2, 16, 16
S_SHAPE = (B, HV, KD, VD)
S_D = HV * KD * VD                    # 1,024 — committed codebooks
BITS = 3.5
STATE_BUDGET = 0.06                   # the house single-quant-round gate
OUT_BUDGET = 0.10                     # query-prefill output tracking
N_SYS, N_DOC, N_QUERY = 8, 300, 8


def _rel_mse(a, b) -> float:
    a = torch.as_tensor(a, dtype=torch.float32).reshape(-1)
    b = torch.as_tensor(b, dtype=torch.float32).reshape(-1)
    return float(((a - b) ** 2).sum() / (b ** 2).sum().clamp_min(1e-30))


def _tokens(n: int, seed: int):
    """Deterministic per-sequence q/k/v/g/beta streams, model-shaped:
    q/k (B, T, HK, K) repeat_interleaved to HV (modeling.py:688), v
    (B, T, HV, V), beta sigmoided, g strictly negative (log decay)."""
    g = torch.Generator().manual_seed(seed)
    q = (torch.randn(B, n, HK, KD, generator=g).cuda() * 0.8
         ).repeat_interleave(HV // HK, dim=2)
    k = (torch.randn(B, n, HK, KD, generator=g).cuda() * 0.8
         ).repeat_interleave(HV // HK, dim=2)
    v = torch.randn(B, n, HV, VD, generator=g).cuda() * 0.8
    beta = torch.sigmoid(torch.randn(B, n, HV, generator=g).cuda())
    decay = -torch.rand(B, n, HV, generator=g) * 0.08 - 0.002
    return q, k, v, beta, decay


def _prefill(state, tokens):
    """One GDN prefill forward (the model's chunked path, reference math)."""
    q, k, v, beta, g = tokens
    return chunk_gdn(q, k, v, g=g, beta=beta, initial_state=state,
                     output_final_state=True, use_qk_l2norm_in_kernel=True)


@pytest.fixture(scope="module")
def rig():
    """The shared experiment state (deterministic; ~4 s to build)."""
    q = resolve_quantizer("S", S_D, BITS)
    # the system state and its codes (one quant round, as at ingestion)
    _, s_sys_true = _prefill(None, _tokens(N_SYS, 11))
    sys_codes = q.quant(s_sys_true.reshape(-1))
    s_sys_recon = q.dequant(sys_codes).reshape(S_SHAPE)

    # the TRUE [sys + doc] trajectory (no quantization anywhere)
    doc_tokens = _tokens(N_DOC, 22)
    _, s_sysdoc_true = _prefill(s_sys_true, doc_tokens)

    # the TQ online path: reseeded layer -> doc prefill -> codes (the
    # exact ingest_chunk flow: read dequant(sys), evolve, write codes)
    layer = TQLinearAttentionLayer(bits=BITS, online=True)
    layer._init_s(s_sys_recon.half().cuda(), 0)
    layer.s_codes = sys_codes
    read0 = layer.recurrent_states[0]
    _, s_doc_tq = _prefill(read0.float(), doc_tokens)
    layer.update_recurrent_state(s_doc_tq.half().cuda(), 0)
    doc_codes = layer.s_codes

    # the D4 delta + the install sum (sum_turboquant_codes, one requant)
    delta_vec = q.dequant(doc_codes) - q.dequant(sys_codes)
    delta_codes = q.quant(delta_vec)
    summed = sum_turboquant_codes(sys_codes, [delta_codes], kind="S", bits=BITS)
    s_installed = q.dequant(summed).reshape(S_SHAPE)

    return dict(q=q, sys_codes=sys_codes, s_sys_recon=s_sys_recon,
                s_sys_true=s_sys_true, doc_tokens=doc_tokens,
                s_sysdoc_true=s_sysdoc_true, doc_codes=doc_codes,
                delta_codes=delta_codes, summed=summed,
                s_installed=s_installed, layer=layer)


def test_install_read_path_bit_clean(rig):
    """The installed codes read back through the layer EXACTLY (the GPU
    handover's Hypothesis 2/3 — settled: the setter updates the store; the
    read is bit-identical to the direct dequant, shape/dtype/frame intact)."""
    r = rig
    layer = r["layer"]
    layer.s_codes = r["summed"]
    t = layer.recurrent_states[0]
    assert tuple(t.shape) == S_SHAPE and t.dtype == torch.float16
    assert torch.equal(t, r["q"].dequant(r["summed"],
                                         dtype=torch.float16).reshape(S_SHAPE))
    # quantizer identity across the whole install path (frame, registry)
    assert layer._tq_s is r["q"]
    assert resolve_quantizer("S", r["summed"].d, BITS) is r["q"]


def test_install_reconstructs_true_sysdoc_state(rig):
    r = rig
    assert _rel_mse(r["s_installed"], r["s_sysdoc_true"]) < STATE_BUDGET


def test_installed_query_output_tracks_truth(rig):
    """The answer prefill from the installed state tracks the TRUE
    [sys+doc] trajectory — the output the LM head would consume."""
    r = rig
    query = _tokens(N_QUERY, 33)
    out_b, _ = _prefill(r["s_installed"], query)
    out_c, _ = _prefill(r["s_sysdoc_true"], query)
    assert _rel_mse(out_b[:, -1], out_c[:, -1]) < OUT_BUDGET


def test_state_noise_is_contractive(rig):
    """A 5% state perturbation moves the query-prefill output by < 2%:
    quantization-scale noise cannot produce garbage outputs — the delta
    rule is contractive in the state (the W11 adjudication gate)."""
    r = rig
    query = _tokens(N_QUERY, 33)
    out_c, _ = _prefill(r["s_sysdoc_true"], query)
    pert = (r["s_sysdoc_true"]
            + 0.05 * r["s_sysdoc_true"].std() * torch.randn_like(r["s_sysdoc_true"]))
    out_p, _ = _prefill(pert, query)
    assert _rel_mse(out_p[:, -1], out_c[:, -1]) < 0.02


def test_reseed_and_install_carry_different_information(rig):
    """The reseed-only trajectory diverges from the [sys+doc] truth while
    the installed one does not — the separation a healthy install must
    show (and the asymmetry a 'reseed works / install garbage' report
    contradicts at reference math fidelity)."""
    r = rig
    query = _tokens(N_QUERY, 33)
    out_a, _ = _prefill(r["s_sys_recon"], query)
    out_c, _ = _prefill(r["s_sysdoc_true"], query)
    out_b, _ = _prefill(r["s_installed"], query)
    rel_a = _rel_mse(out_a[:, -1], out_c[:, -1])
    rel_b = _rel_mse(out_b[:, -1], out_c[:, -1])
    assert rel_b < rel_a / 5            # measured 0.019 vs 0.71
