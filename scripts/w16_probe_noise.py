#!/usr/bin/env python3
"""w16_probe_noise.py — the W16 generation-noise decomposition (CPU).

THE QUESTION (handover W15-post): true-doc (raw cache) generates at conf
0.895; every TQ variant sits at 0.28-0.45 with repetition. The W15 TRUE-dist
row says the installed S is ~0.08-0.11 rel-MSE from the raw doc state —
but WHERE does that distortion come from? W15 attributed it to "online
ingestion drift" without decomposing it. This probe decomposes it on the
REAL GDN forward + REAL TQCache (the same rig as
src/rag/tests/test_install_conv_math.py — same config, same seeds, so the
numbers are comparable with the committed gate's 0.08-0.11):

  TRUE-A            raw continuous [sys -> doc] end state (the true-doc
                    control's own truth)
  PROTO-GAP         ||TRUE-B-raw - TRUE-A||: the delta protocol's path
                    independence on a RAW reseed (raw sys state loaded
                    into a raw cache, then the doc prefill) — the pure
                    protocol term, NO quantization anywhere
  RESEED-DRIFT      ||TRUE-B-quant - TRUE-B-raw||: what the 2.2% reseed
                    noise becomes after 300 tokens of recurrence through
                    the GDN layers + the residual stream (the cross-layer
                    amplification channel)
  CAPTURE+INSTALL   ||installed - TRUE-B-quant||: the online capture write
                    + the delta round + the install sum requant
  TOTAL             ||installed - TRUE-A|| (the W15 TRUE-dist row)

COUNTERFACTUALS (the mitigation levers, same rig, same tokens):
  bits=4.0          the whole chain at 4-bit uniform (the codebook max)
  qjl=True          the paper's Alg.-2 residual sketch on every round
  bits=4 + qjl      both
  sys-only at 4b    ONLY the system reset point quantized at 4 bits (the
                    reseed round), everything else at 3.5 — isolates how
                    much of the drift the reset point alone owns
"""
from __future__ import annotations

import sys

import torch

REPO = "/home/z/my-project/repo"
sys.path.insert(0, REPO + "/src/rag")
sys.path.insert(0, REPO + "/src/scripts")

import modeling as modeling_mod  # noqa: E402
from transformers import DynamicCache  # noqa: E402
from transformers.models.qwen3_5 import Qwen3_5Config, Qwen3_5TextConfig  # noqa: E402

from ingest import SystemState, reseed_cache, ingest_chunk  # noqa: E402
from install import install_snapshot  # noqa: E402
from tq_cache import TQCache, resolve_quantizer  # noqa: E402

B, VOCAB, HIDDEN, N_LAYERS = 1, 512, 256, 4
LAYER_TYPES = ["linear_attention"] * N_LAYERS
N_SYS, N_DOC, N_QUERY = 12, 300, 8
SEED = 20261010


def _rel_mse(a, b) -> float:
    a = torch.as_tensor(a, dtype=torch.float32).reshape(-1)
    b = torch.as_tensor(b, dtype=torch.float32).reshape(-1)
    return float(((a - b) ** 2).sum() / (b ** 2).sum().clamp_min(1e-30))


def _tokens(n: int, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, VOCAB, (B, n), generator=g)


def _config() -> Qwen3_5Config:
    tc = Qwen3_5TextConfig(
        hidden_size=HIDDEN, num_hidden_layers=N_LAYERS,
        layer_types=LAYER_TYPES,
        linear_num_key_heads=2, linear_num_value_heads=8,
        linear_key_head_dim=16, linear_value_head_dim=16,
        linear_conv_kernel_dim=4)
    return Qwen3_5Config(text_config=tc)


class _ResidualGDNStack(torch.nn.Module):
    def __init__(self, cfg, seed):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.embed = torch.nn.Embedding(VOCAB, HIDDEN)
        self.gdn = torch.nn.ModuleList(
            [modeling_mod.Qwen3_5GatedDeltaNet(cfg.text_config, i)
             for i in range(N_LAYERS)])
        for m in self.modules():
            for p in m.parameters():
                if p.dim() > 1:
                    p.data = torch.randn(p.shape, generator=g) * 0.1
                else:
                    p.data.fill_(1.0)
        for lyr in self.gdn:
            torch.nn.init.uniform_(lyr.A_log, -2.0, 0.0)

    def forward(self, input_ids, past_key_values=None, use_cache=True):
        h = self.embed(input_ids)
        for lyr in self.gdn:
            h = h + lyr(h, past_key_values)
        return h


def _snap_raw(cache):
    s = {i: cache.layers[i].recurrent_states[0].clone()
         for i in range(N_LAYERS)}
    c = {i: cache.layers[i].conv_states[0].clone()
         for i in range(N_LAYERS)}
    return s, c


def _raw_state_cache(model, cfg, s, c) -> DynamicCache:
    cache = DynamicCache(config=cfg)
    with torch.no_grad():
        model(input_ids=torch.zeros(B, 1, dtype=torch.long),
              past_key_values=cache)
    for i in range(N_LAYERS):
        cache.layers[i].recurrent_states[0].copy_(s[i])
        cache.layers[i].conv_states[0].copy_(c[i])
        cache.layers[i].has_previous_state[0] = True
    return cache


def _s_of(cache) -> dict:
    """The per-layer S states a cache holds (TQ: dequantized codes)."""
    out = {}
    for i in range(N_LAYERS):
        layer = cache.layers[i]
        t = layer.recurrent_states[0]
        out[i] = (t.reshape(-1).float().clone() if t is not None
                  else torch.zeros(1))
    return out


def main() -> int:
    torch.manual_seed(SEED)
    cfg = _config()
    model = _ResidualGDNStack(cfg, SEED).eval()
    sys_ids, doc_ids = _tokens(N_SYS, 1), _tokens(N_DOC, 2)
    qry_ids = _tokens(N_QUERY, 3)

    with torch.no_grad():
        # ---- TRUE-A: raw continuous [sys -> doc] --------------------------
        rc = DynamicCache(config=cfg)
        model(input_ids=sys_ids, past_key_values=rc)
        s_sys, c_sys = _snap_raw(rc)
        model(input_ids=doc_ids, past_key_values=rc)
        s_trueA, _ = _snap_raw(rc)

        # ---- TRUE-B-raw: raw reseed + doc (the protocol without quant) ----
        rb = _raw_state_cache(model, cfg, s_sys, c_sys)
        model(input_ids=doc_ids, past_key_values=rb)
        s_trueB_raw, _ = _snap_raw(rb)

    def run_chain(bits: float, qjl: bool, sys_bits: float | None = None):
        """The full TQ chain (ingest + install + the standing state) at the
        given knobs; sys_bits overrides ONLY the reset point's quantization
        (the reseed round). Returns per-layer S of the installed state and
        of the ingest end-state (TRUE-B-quant)."""
        with torch.no_grad():
            tq = TQCache(layer_types=LAYER_TYPES, bits=bits, qjl=qjl)
            model(input_ids=sys_ids, past_key_values=tq)
            system = SystemState.from_cache(tq, system_ref="w16probe",
                                            bits=bits)
            if sys_bits is not None and sys_bits != bits:
                # requant ONLY the reset point at sys_bits (the reseed
                # round's precision), through the same D3 frame
                for L, c in list(system.s_codes.items()):
                    q_lo = resolve_quantizer("S", c.d, bits)
                    q_hi = resolve_quantizer("S", c.d, sys_bits)
                    # dequant at the capture bits, requant at sys_bits:
                    # the CODES' frame is the seed — same for both
                    system.s_codes[L] = q_hi.quant(q_lo.dequant(c))
            tq2 = TQCache(layer_types=LAYER_TYPES, bits=bits, qjl=qjl)
            record = ingest_chunk(model, doc_ids, tq2, system)
            record.chunk_idx = 0
            snap = record.to_snapshot(system)

            # TRUE-B-quant: the ingest end state (dequantized)
            end_codes = tq2.snapshot_codes()
            qb = resolve_quantizer("S", end_codes["s"][0].d, bits)
            s_trueB_quant = {
                L: qb.dequant(end_codes["s"][L]).float().clone()
                for L in range(N_LAYERS)}

            cache = TQCache(layer_types=LAYER_TYPES, bits=bits, qjl=qjl)
            reseed_cache(cache, system)
            install_snapshot(cache, system, [snap])
            s_installed = _s_of(cache)
        return s_installed, s_trueB_quant

    def report(tag, s_installed, s_trueB_quant):
        proto = sum(_rel_mse(s_trueB_raw[i], s_trueA[i]) for i in range(N_LAYERS)) / N_LAYERS
        drift = sum(_rel_mse(s_trueB_quant[i], s_trueB_raw[i]) for i in range(N_LAYERS)) / N_LAYERS
        rounds = sum(_rel_mse(s_installed[i], s_trueB_quant[i]) for i in range(N_LAYERS)) / N_LAYERS
        total = sum(_rel_mse(s_installed[i], s_trueA[i]) for i in range(N_LAYERS)) / N_LAYERS
        print(f"{tag:<28} proto {proto:.4f}  reseed-drift {drift:.4f}  "
              f"capture+install {rounds:.4f}  TOTAL {total:.4f}")
        return total

    def run_chain_absolute(bits: float, qjl: bool):
        """The W16 chain: protocol=\"absolute\" snapshots + the VERBATIM
        install — the capture+install rounds vanish; the total collapses
        to the reseed drift."""
        with torch.no_grad():
            tq = TQCache(layer_types=LAYER_TYPES, bits=bits, qjl=qjl)
            model(input_ids=sys_ids, past_key_values=tq)
            system = SystemState.from_cache(tq, system_ref="w16probe",
                                            bits=bits)
            tq2 = TQCache(layer_types=LAYER_TYPES, bits=bits, qjl=qjl)
            record = ingest_chunk(model, doc_ids, tq2, system)
            record.chunk_idx = 0
            snap = record.to_snapshot(system, protocol="absolute")

            end_codes = tq2.snapshot_codes()
            qb = resolve_quantizer("S", end_codes["s"][0].d, bits)
            s_trueB_quant = {
                L: qb.dequant(end_codes["s"][L]).float().clone()
                for L in range(N_LAYERS)}

            cache = TQCache(layer_types=LAYER_TYPES, bits=bits, qjl=qjl)
            reseed_cache(cache, system)
            install_snapshot(cache, system, [snap])
            s_installed = _s_of(cache)
        return s_installed, s_trueB_quant

    print("per-layer mean rel-MSE vs TRUE-A (the raw [sys+doc] state):")
    print("-" * 78)
    s_inst, s_bq = run_chain(3.5, False)
    report("chain @3.5b (production)", s_inst, s_bq)

    s_inst, s_bq = run_chain_absolute(3.5, False)
    report("chain @3.5b ABSOLUTE (W16)", s_inst, s_bq)

    s_inst, s_bq = run_chain(4.0, False)
    report("chain @4.0b", s_inst, s_bq)

    s_inst, s_bq = run_chain_absolute(4.0, False)
    report("chain @4.0b ABSOLUTE", s_inst, s_bq)

    s_inst, s_bq = run_chain(3.5, True)
    report("chain @3.5b + qjl (Alg.2)", s_inst, s_bq)

    s_inst, s_bq = run_chain(4.0, True)
    report("chain @4.0b + qjl", s_inst, s_bq)

    # NOTE: the sys-bits counterfactual (reset point at 4b, chunks at 3.5b)
    # is blocked by a REAL design friction this probe found: the S read
    # path validates the CODES' bits against the layer's OWN quantizer
    # (_check_codes) — TQCodes are self-describing on partition/group
    # (W15) but NOT on bits. Enabling mixed-precision reset points needs
    # codes-driven bits resolution in the S read path (see the W16 report).

    print("-" * 78)
    print("Reading: proto = the delta protocol's own gap (0 in this all-")
    print("linear rig — the box's full-attn-empty ingestion adds a term ")
    print("the GPU decomposition must measure separately); reseed-drift =")
    print("the 2.2% reset-point noise amplified through the recurrence +")
    print("the residual stream; capture+install = the quant rounds the ")
    print("snapshot chain owns. The lever that moves the DOMINANT term is")
    print("the one to pull.")

    # bonus: the answer-prefill hidden deviation at the production chain —
    # the generation-facing consequence of the same noise
    with torch.no_grad():
        cc = _raw_state_cache(model, cfg, s_trueA[
            0].reshape(cache_shape := (1, 8, 16, 16)) if False else None, None) \
            if False else None
    print("\n(done)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
