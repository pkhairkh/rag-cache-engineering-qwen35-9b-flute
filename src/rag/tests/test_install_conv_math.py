"""test_install_conv_math.py — W12: the S+conv install COMBINATION at real
GDN-layer fidelity.

The W11 adjudication (test_install_real_math.py) pinned the S path in
isolation — its own header says "No conv (S-path isolation)". The W12 GPU
bisection then reported the failure requiring BOTH the summed S codes AND
the chunk conv codes installed together (s-only/conv-only generate, full
does not), a combination no CPU gate had ever exercised through the model
math. This file closes that gap: the REAL vendored
`Qwen3_5GatedDeltaNet.forward` (conv windowing + the chunked gated delta
rule, the torch reference route) driven through the REAL TQCache, over the
full protocol — system prefill, chunk ingestion (reseed + prefill + delta),
the four bisect variants, query prefill, and a teacher-forced decode loop.

THE GATES (rig: 4 GDN layers, residual stream h <- h + GDN(h), d_S = 2,048,
conv window 768 = 192 x 4 — multiple of 32, NON-power-of-two, FHT segments
512 + 256, structurally the box's 16,384 + 8,192; all codebooks committed):

  1. INSTALL-VS-TRUTH: the full install reconstructs the raw
     [system + doc] end state per layer — S rel-MSE <= 0.15 (the ONLINE
     ingestion drift over a 300-token chunk compounds the roundtrip; the
     install math itself is ~2%, see the W11 file), conv <= 0.06 (the
     window is rewritten every step — no accumulation).
  2. QUERY-PREFILL OUTPUT TRACKING: the installed state's query-prefill
     hidden tracks the raw true-state continuation <= 0.10.
  3. NO VARIANT ASYMMETRY: full's hidden deviation <= 2x the WORST of the
     half-installed variants (s-only / conv-only). The W12 GPU matrix's
     cliff (s-only OK / full GARBAGE) has NO counterpart on the reference
     path — combined with the artifact proof (the bisect's S-read rows),
     the install path is exonerated on CPU; whatever remains on the box
     lives in a GPU-only route or the generation semantics, which
     scripts/gpu/bisect_install.py's frame check + true-doc control
     isolate.
  4. DECODE STABILITY: 12 teacher-forced decode steps keep the per-layer
     S drift <= 0.8 (the online dequant->update->quant roundtrip's
     compounded drift bound at this rig; teacher forcing isolates the
     cache roundtrip from greedy argmax chaos).
"""
from __future__ import annotations

import pytest
import torch

import _paths  # noqa: F401
import codebooks
import modeling as modeling_mod
from transformers import DynamicCache
from transformers.models.qwen3_5 import Qwen3_5Config, Qwen3_5TextConfig

from ingest import SystemState, reseed_cache, ingest_chunk
from install import install_snapshot
from tq_cache import TQCache

# every (b, d) here resolves against committed codebooks (no npz writes):
# cb_b{3,4}_d2048.npz (S) and cb_b{3,4}_d768.npz (conv).
assert codebooks.CACHE_DIR  # the committed dir; nothing to redirect

B, VOCAB, HIDDEN, N_LAYERS = 1, 512, 256, 4
LAYER_TYPES = ["linear_attention"] * N_LAYERS
BITS = 3.5
N_SYS, N_DOC, N_QUERY, N_DECODE = 12, 300, 8, 12
SEED = 20261010

# the gates (see the module docstring; measured margins noted per gate)
S_TRUTH_BUDGET = 0.15      # measured 0.08 - 0.11 across seeds
CONV_TRUTH_BUDGET = 0.06   # measured 0.018 - 0.024
HIDDEN_BUDGET = 0.10       # measured 0.024 - 0.048
ASYM_RATIO_MAX = 2.0       # measured ~0.94 (full is NOT worse than s-only)
DECODE_DRIFT_BUDGET = 0.8  # measured 0.15 - 0.63 across seeds


def _rel_mse(a, b) -> float:
    a = torch.as_tensor(a, dtype=torch.float32).reshape(-1)
    b = torch.as_tensor(b, dtype=torch.float32).reshape(-1)
    return float(((a - b) ** 2).sum() / (b ** 2).sum().clamp_min(1e-30))


def _tokens(n: int, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, VOCAB, (B, n), generator=g).cuda()


def _config() -> Qwen3_5Config:
    tc = Qwen3_5TextConfig(
        hidden_size=HIDDEN, num_hidden_layers=N_LAYERS,
        layer_types=LAYER_TYPES,
        linear_num_key_heads=2, linear_num_value_heads=8,
        linear_key_head_dim=16, linear_value_head_dim=16,
        linear_conv_kernel_dim=4)
    return Qwen3_5Config(text_config=tc)


class _ResidualGDNStack(torch.nn.Module):
    """Embedding + 4 real GDN layers under a residual stream
    (h <- h + GDN(h), the decoder-layer form without the inter-layer
    norm — a slightly harsher, residual-faithful harness)."""

    def __init__(self, cfg: Qwen3_5Config, seed: int):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.embed = torch.nn.Embedding(VOCAB, HIDDEN)
        self.gdn = torch.nn.ModuleList(
            [modeling_mod.Qwen3_5GatedDeltaNet(cfg.text_config, i)
             for i in range(N_LAYERS)])
        for m in self.modules():
            for p in m.parameters():
                if p.dim() > 1:
                    p.data = torch.randn(p.shape, generator=g).cuda() * 0.1
                else:
                    p.data.fill_(1.0)
        for lyr in self.gdn:
            torch.nn.init.uniform_(lyr.A_log, -2.0, 0.0)

    def forward(self, input_ids, past_key_values=None, use_cache=True):
        h = self.embed(input_ids)
        for lyr in self.gdn:
            h = h + lyr(h, past_key_values)
        return h


def _snap_raw(cache: DynamicCache):
    s = {i: cache.layers[i].recurrent_states[0].clone()
         for i in range(N_LAYERS)}
    c = {i: cache.layers[i].conv_states[0].clone()
         for i in range(N_LAYERS)}
    return s, c


def _raw_state_cache(model, cfg, s, c) -> DynamicCache:
    """A fresh raw cache carrying exactly (s, c): one 1-token forward
    lazily initializes the layers (full-attn stays empty — the G5
    geometry), then the states are loaded in place."""
    cache = DynamicCache(config=cfg)
    with torch.no_grad():
        model(input_ids=torch.zeros(B, 1, dtype=torch.long),
              past_key_values=cache)
    for i in range(N_LAYERS):
        cache.layers[i].recurrent_states[0].copy_(s[i])
        cache.layers[i].conv_states[0].copy_(c[i])
        cache.layers[i].has_previous_state[0] = True
    return cache


@pytest.fixture(scope="module")
def rig():
    """The shared experiment state (deterministic)."""
    torch.manual_seed(SEED)
    cfg = _config()
    model = _ResidualGDNStack(cfg, SEED).eval()
    sys_ids, doc_ids = _tokens(N_SYS, 1), _tokens(N_DOC, 2)
    qry_ids, dec_ids = _tokens(N_QUERY, 3), _tokens(N_DECODE, 4)

    with torch.no_grad():
        # raw ground truth: system -> doc (one raw cache, no quantization)
        rc = DynamicCache(config=cfg)
        model(input_ids=sys_ids, past_key_values=rc)
        s_sys, c_sys = _snap_raw(rc)
        model(input_ids=doc_ids, past_key_values=rc)
        s_chunk, c_chunk = _snap_raw(rc)

        # the raw GT query-prefill hiddens per variant (own-GT comparison)
        def gt_hidden(s, c):
            cc = _raw_state_cache(model, cfg, s, c)
            return model(input_ids=qry_ids, past_key_values=cc)

        gt = {
            "reseed": gt_hidden(s_sys, c_sys),
            "s-only": gt_hidden(s_chunk, c_sys),
            "conv-only": gt_hidden(s_sys, c_chunk),
            "full": gt_hidden(s_chunk, c_chunk),
        }

        # the TQ protocol: system prefill -> SystemState -> chunk ingest
        tq = TQCache(layer_types=LAYER_TYPES, bits=BITS)
        model(input_ids=sys_ids, past_key_values=tq)
        system = SystemState.from_cache(tq, system_ref="w12rig", bits=BITS)
        tq2 = TQCache(layer_types=LAYER_TYPES, bits=BITS)
        record = ingest_chunk(model, doc_ids, tq2, system)
        record.chunk_idx = 0
        snap = record.to_snapshot(system)

        def tq_hidden(variant: str):
            with torch.no_grad():
                cache = TQCache(layer_types=LAYER_TYPES, bits=BITS)
                reseed_cache(cache, system)
                if variant == "s-only":
                    install_snapshot(cache, system, [snap])
                    for L, c in system.conv_codes.items():
                        cache.set_conv_codes(L, c)
                elif variant == "conv-only":
                    for L, c in snap.conv_codes.items():
                        if c is not None:
                            cache.set_conv_codes(L, c)
                elif variant == "full":
                    install_snapshot(cache, system, [snap])
                return model(input_ids=qry_ids, past_key_values=cache)

        # the full install's standing state (gate 1)
        full_cache = TQCache(layer_types=LAYER_TYPES, bits=BITS)
        reseed_cache(full_cache, system)
        install_snapshot(full_cache, system, [snap])
        installed = full_cache

        # the teacher-forced decode drift (gate 4): query prefill then
        # N_DECODE single-token steps on both flows, same token inputs
        raw_dec = _raw_state_cache(model, cfg, s_chunk, c_chunk)
        model(input_ids=qry_ids, past_key_values=raw_dec)
        for t in range(N_DECODE):
            model(input_ids=dec_ids[:, t:t + 1], past_key_values=raw_dec)
        tq_dec = TQCache(layer_types=LAYER_TYPES, bits=BITS)
        reseed_cache(tq_dec, system)
        install_snapshot(tq_dec, system, [snap])
        model(input_ids=qry_ids, past_key_values=tq_dec)
        for t in range(N_DECODE):
            model(input_ids=dec_ids[:, t:t + 1], past_key_values=tq_dec)

    return {
        "cfg": cfg, "model": model, "system": system, "snap": snap,
        "s_sys": s_sys, "c_sys": c_sys,
        "s_chunk": s_chunk, "c_chunk": c_chunk,
        "gt": gt, "tq_hidden": tq_hidden, "installed": installed,
        "raw_dec": raw_dec, "tq_dec": tq_dec,
    }


# ============================================ 1. install-vs-truth ===========
def test_full_install_reconstructs_true_state(rig):
    """Gate 1: the full install (S sums + the chunk conv codes) reads back
    as the raw [system + doc] end state per layer — S within the online
    ingestion drift budget, conv within the single-round budget."""
    r = rig
    for i in range(N_LAYERS):
        s_rel = _rel_mse(r["installed"].layers[i].recurrent_states[0],
                         r["s_chunk"][i])
        c_rel = _rel_mse(r["installed"].layers[i].conv_states[0],
                         r["c_chunk"][i])
        assert s_rel <= S_TRUTH_BUDGET, (
            f"layer {i}: installed S rel-MSE {s_rel:.3e} > "
            f"{S_TRUTH_BUDGET} — the S install drifted beyond the online "
            f"ingestion bound")
        assert c_rel <= CONV_TRUTH_BUDGET, (
            f"layer {i}: installed conv rel-MSE {c_rel:.3e} > "
            f"{CONV_TRUTH_BUDGET} — the verbatim conv install drifted")


# ======================================= 2. query-prefill tracking ==========
def test_installed_query_prefill_tracks_truth(rig):
    """Gate 2: the full install's query-prefill hidden tracks the raw
    true-state continuation."""
    dev = _rel_mse(rig["tq_hidden"]("full"), rig["gt"]["full"])
    assert dev <= HIDDEN_BUDGET, (
        f"full install query-prefill hidden rel-MSE {dev:.3e} > "
        f"{HIDDEN_BUDGET} — the installed state no longer tracks the raw "
        f"continuation")


# ======================================= 3. no variant asymmetry ============
def test_no_variant_asymmetry_w12_signature(rig):
    """Gate 3: the W12 GPU matrix's cliff (s-only/conv-only generate,
    full does not) has no numerical counterpart on the reference path —
    full's hidden deviation stays within 2x the WORST half-install's."""
    devs = {v: _rel_mse(rig["tq_hidden"](v), rig["gt"][v])
            for v in ("reseed", "s-only", "conv-only", "full")}
    worst_half = max(devs["s-only"], devs["conv-only"])
    ratio = devs["full"] / max(worst_half, 1e-12)
    assert ratio <= ASYM_RATIO_MAX, (
        f"full's deviation {devs['full']:.3e} is {ratio:.1f}x the worst "
        f"half-install {worst_half:.3e} — the W12 asymmetry signature "
        f"REPRODUCED on the reference path (it must not: the install "
        f"content is the same sum, only the conv source differs)")


# ============================================ 4. decode stability ===========
def test_teacher_forced_decode_drift_bounded(rig):
    """Gate 4: 12 teacher-forced decode steps keep the online cache's S
    drift bounded — the dequant->update->quant roundtrip does not
    diverge from the raw continuation."""
    for i in range(N_LAYERS):
        drift = _rel_mse(rig["tq_dec"].layers[i].recurrent_states[0],
                         rig["raw_dec"].layers[i].recurrent_states[0])
        assert drift <= DECODE_DRIFT_BUDGET, (
            f"layer {i}: decode S drift {drift:.3e} > "
            f"{DECODE_DRIFT_BUDGET} after {N_DECODE} teacher-forced steps "
            f"— the online roundtrip diverged")
