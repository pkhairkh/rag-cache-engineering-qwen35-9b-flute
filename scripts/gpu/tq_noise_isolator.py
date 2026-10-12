#!/usr/bin/env python3
"""tq_noise_isolator.py — W30: THE E2E NOISE ISOLATOR (no M1/M2 anywhere).

The W29 box matrix verdict (pasted verbatim, 2026-10-12): the per-head
repair reached the FLOOR at the probe — [W29 per-head units] worst-block
rel-MSE 0.0234, spread-independent, kernel parity exact — yet G2/G4
(TQ + manual/generate) are STILL garbage (conf 0.527, rep 0.87, the
text degenerates at token 3). The script's own decision tree said it:
"the wiring differs from the probe". The dev box then re-read EVERY hop
of the real path (modeling.py's decode loop -> the cache read -> the
GDN recurrence -> the write) and found NO wiring defect: the read
dequantizes through the SAME quantizer instance (the frame cannot
mismatch), the write reshapes (B,H,K,V) -> (B*H, K*V) exactly as
quant_units expects, the dequant computes fp32 and casts once.

THE REMAINING CANDIDATE (F2c, the compounding hazard): the ONLINE
regime quantizes-on-write at EVERY decode step and the dequantized
state FEEDS BACK into the recurrence (modeling.py reads
recurrent_states[0] as the fused kernel's initial_state every step).
One round of the 3.5-bit floor is rel-MSE ~0.022 = a 15% NORM
perturbation (sqrt) of the state PER GDN LAYER PER STEP. The paper's
3.5-bit parity claim (LongBench 50.06 vs 50.06) is a KV-cache claim:
each token's d=128 vector is quantized ONCE at write and the softmax
averages the read noise over the whole context — no recurrence, no
feedback, no requant. The GDN state has NONE of that averaging: the
error compounds as E_t = lambda*E_{t-1} + eps_t across steps AND is
injected at every GDN layer simultaneously. Also: the repo's bit-width
ceiling is 4 (the Lloyd-Max codebook range, turboquant's constructor
refuses eff_bits > 4), so "raise the bits" cannot reach fp16-level
noise (~0.05% norm) — the structural controls below are the decision.

[1] FIVE CELLS, one model load, the manual greedy loop (the failure is
    loop-level — G2 == G4 in the W29 matrix):
      C0  raw DynamicCache                     the reference
      C1  TQCache online 3.5                   the failure, TELEMETRY ON
      C2  TQCache online 4.0                   the bits ceiling (the
          constructor's 1..4 cap — the ladder's top rung)
      C3  TQCache online 3.5 + THE S-EXACT INSTRUMENT (K=16): the
          layer stores the EXACT state at every write and the READ
          returns it verbatim; the codes requant every K-th write
          (first write always — the init); the CONV store stays fully
          online (its lazy re-capture keeps quantizing every step).
          This is a MEASUREMENT instrument patched onto the cache
          INSTANCE (the tq_gen_matrix --trace idiom) — the library is
          untouched. It answers: with S exact in the loop, is the
          remaining (conv + plumbing) noise enough to garbage?
      C4  TQCache online=False                 the D4 regime: raw
          tensors during forward, codes only at snapshot — the
          plumbing control (C4 == C0 expected).
    Each cell prints the [gen gate] line + text + the FIRST-DIVERGENCE
    token index vs C0 (greedy is deterministic — the index is exact).

[2] THE C1 TELEMETRY (the real path, instrumented): per S write, per
    layer: the round-trip rel-MSE ||dequant(quant(s)) - s||^2/||s||^2
    (the write's own return — the direct measurement of the floor IN
    THE REAL PATH; if this is NOT ~0.022 the wiring DOES differ from
    the probe and the table names the layer), the state's energy
    trajectory (first/last — the amplification check), the per-head
    norm spread (W28's 38x, re-confirmed on live states), and the conv
    window's round-trip rel-MSE at every lazy re-capture.

[3] THE VERDICT TREE (printed with the cells' verdicts filled in):
      C4 garbage                    -> PLUMBING (beyond the stubs):
                                      paste everything; re-run
                                      tq_gen_matrix.py --trace.
      C4 clean, C3 clean, C1 garbage-> F2c CONFIRMED: the S per-step
                                      requant noise is the killer; the
                                      W31 mitigation decision (the
                                      requant cadence / D4-for-decode /
                                      bits) rides these numbers.
      C4 clean, C3 garbage          -> the CONV store contributes (or
                                      the S+conv interplay): W31 takes
                                      the conv into the cadence too.
      C1 telemetry off the floor    -> a REAL wiring divergence —
                                      paste the telemetry table whole.
      C2 vs C1                      -> the bits direction at the
                                      ceiling (both expected weak;
                                      the delta is the datum).

GPU-ONLY: every number comes from this box (the GPU-only law). No
corpus, no M1/M2, no retrieval — the model, the caches, five runs.

Examples:
    python3 scripts/gpu/tq_noise_isolator.py
    python3 scripts/gpu/tq_noise_isolator.py --max-new-tokens 64
    python3 scripts/gpu/tq_noise_isolator.py --s-exact-every 8
"""
from __future__ import annotations

import argparse
import sys
import time
from collections import defaultdict

import torch

from _bootstrap import DEFAULTS, boot, load_model

boot()


# ------------------------------------------------------------- gate utils --
def gen_gate(tokenizer, new_token_ids, token_confs, ref_ids=None):
    """The [gen gate] line (the matrix's metrics) + the FIRST-DIVERGENCE
    token index vs the C0 reference (greedy decode is deterministic —
    the index is exact, 1-based, "identical" = never diverged)."""
    answer = tokenizer.decode(new_token_ids, skip_special_tokens=True)
    toks = list(new_token_ids)
    conf = (sum(token_confs) / len(token_confs)) if token_confs else 0.0
    rep = (sum(1 for a, b in zip(toks, toks[1:]) if a == b)
           / max(1, len(toks) - 1))
    non_ascii = (sum(1 for ch in answer if not ch.isascii())
                 / max(1, len(answer)))
    verdict = ("CLEAN-RANGE" if conf >= 0.60 else
               ("WEAK — judge the TEXT" if conf >= 0.20 else
                "GARBAGE-RANGE"))
    div = None
    if ref_ids is not None:
        for i, (a, b) in enumerate(zip(ref_ids, toks)):
            if a != b:
                div = i + 1
                break
        if div is None and len(ref_ids) != len(toks):
            div = min(len(ref_ids), len(toks)) + 1
    div_s = "identical" if div is None else f"token {div}"
    print(f"    [gen gate] conf {conf:.3f} | rep {rep:.2f} | "
          f"non-ascii {non_ascii:.2f} | {verdict} | vs C0: {div_s}")
    print(f"    text: {answer[:160]!r}")
    return verdict, conf


# ---------------------------------------------------- the driving style ----
def manual_greedy(model, ids, cache, max_new_tokens):
    """The W14 control's exact style (the matrix's manual_greedy,
    verbatim): model() calls + argmax feedback through `cache`."""
    out = model(input_ids=ids, past_key_values=cache, use_cache=True)
    logits = out.logits if hasattr(out, "logits") else out[0]
    new_ids, confs = [], []
    cur = logits[:, -1, :]
    for _ in range(max_new_tokens):
        probs = torch.softmax(cur.float(), dim=-1)
        conf, nxt = probs.max(dim=-1)
        nxt = nxt.view(1, 1)
        new_ids.append(int(nxt))
        confs.append(float(conf))
        with torch.no_grad():
            out = model(input_ids=nxt, past_key_values=cache,
                        use_cache=True)
        logits = out.logits if hasattr(out, "logits") else out[0]
        cur = logits[:, -1, :]
    return new_ids, confs


# ---------------------------------------------------- [2] the telemetry ----
def attach_telemetry(cache, stats):
    """Instrument every TQ layer's S write + conv re-capture on the
    INSTANCE (the --trace idiom — the library is untouched). The S
    write's return IS the round-trip: rel = ||out - s||^2 / ||s||^2."""
    from tq_cache import TQLinearAttentionLayer
    n_layers = 0
    for li, layer in enumerate(cache.layers):
        if not isinstance(layer, TQLinearAttentionLayer):
            continue
        n_layers += 1
        orig_update = layer.update_recurrent_state
        orig_sync = layer._sync_conv

        def make_update(orig, li, layer):
            def update(s, state_idx=0, **kw):
                out = orig(s, state_idx=state_idx, **kw)
                sf, of = s.detach().float(), out.detach().float()
                denom = sf.pow(2).sum().clamp_min(1e-30)
                rel = float(((of - sf) ** 2).sum() / denom)
                stats["s_rel"].append(rel)
                stats["s_rel_by_layer"][li].append(rel)
                if stats["s_energy_first"] is None:
                    stats["s_energy_first"] = float(sf.norm())
                stats["s_energy_last"] = float(sf.norm())
                units = getattr(layer, "_s_units", None)
                if units:
                    hn = sf.reshape(units, -1).norm(dim=1)
                    spread = float(hn.max() / hn.min().clamp_min(1e-30))
                    stats["s_spread_max"] = max(
                        stats["s_spread_max"], spread)
                stats["s_writes"] += 1
                return out
            return update

        def make_sync(orig, layer):
            def sync():
                t = layer._handed_conv
                codes_before = layer._conv_codes
                orig()
                if (t is not None and layer._conv_codes is not None
                        and layer._conv_codes is not codes_before):
                    rec = layer._dequant_conv(t.dtype)
                    tf = t.detach().float()
                    rf = rec.detach().float().reshape(tf.shape)
                    denom = tf.pow(2).sum().clamp_min(1e-30)
                    stats["conv_rel"].append(
                        float(((rf - tf) ** 2).sum() / denom))
            return sync

        layer.update_recurrent_state = make_update(orig_update, li, layer)
        layer._sync_conv = make_sync(orig_sync, layer)
    return n_layers


def report_telemetry(stats, n_layers):
    print("\n[2] C1 TELEMETRY (the real path, the instrumented failure)")
    if not stats["s_rel"]:
        print("    (no S writes captured — the layer list is empty?)")
        return
    mean = sum(stats["s_rel"]) / len(stats["s_rel"])
    print(f"    S round-trip rel-MSE : mean {mean:.4f} | "
          f"max {max(stats['s_rel']):.4f} | min {min(stats['s_rel']):.4f}")
    by_layer = [(li, sum(v) / len(v)) for li, v in
                sorted(stats["s_rel_by_layer"].items()) if v]
    if by_layer:
        worst = max(by_layer, key=lambda kv: kv[1])
        best = min(by_layer, key=lambda kv: kv[1])
        print(f"    per-layer mean       : worst L{worst[0]} {worst[1]:.4f} "
              f"| best L{best[0]} {best[1]:.4f} "
              f"({len(by_layer)} TQ layers)")
    floor_note = ("AT THE FLOOR — the real path == the probe; the wiring "
                  "is clean (F2c, the compounding hazard)"
                  if mean < 0.035 else
                  "OFF THE FLOOR — the wiring DOES differ from the probe; "
                  "PASTE THIS TABLE WHOLE (the dev box re-reads the named "
                  "layer)")
    print(f"    (verdict: {floor_note})")
    if stats["conv_rel"]:
        cm = sum(stats["conv_rel"]) / len(stats["conv_rel"])
        print(f"    conv round-trip      : mean {cm:.4f} | "
              f"max {max(stats['conv_rel']):.4f} "
              f"({len(stats['conv_rel'])} re-captures)")
    if stats["s_energy_first"] is not None:
        r = stats["s_energy_last"] / max(stats["s_energy_first"], 1e-30)
        print(f"    S energy             : first {stats['s_energy_first']:.2f}"
              f" -> last {stats['s_energy_last']:.2f} "
              f"(ratio {r:.2f} — the amplification check)")
    if stats.get("s_spread_max", 1.0) > 1.0:
        print(f"    S head-norm spread   : max {stats['s_spread_max']:.0f}x "
              f"(W28's 38x measurement, on live states)")
    if n_layers:
        per_layer = stats["s_writes"] // n_layers
        print(f"    writes               : S {per_layer}/layer "
              f"(1 prefill + {per_layer - 1} decode) x {n_layers} layers "
              f"= {stats['s_writes']} total")


# -------------------------------------- [C3] the S-exact instrument --------
class _RawSView(dict):
    """The S-exact instrument's read side: [0] returns the layer's EXACT
    last-written state; pre-write reads return None (the model handles
    a None initial state — modeling.py's `else None` branch). A dict
    subclass so every isinstance/duck-typing contract survives."""

    def __init__(self, raw):
        super().__init__()
        self._raw = raw          # {"s": tensor | None} — this layer's

    def __getitem__(self, state_idx):
        t = self._raw.get("s")
        if t is not None:
            return t
        return dict.get(self, state_idx)


def attach_s_exact(cache, every_k):
    """THE S-EXACT INSTRUMENT (C3): every write stores the exact state
    and the read returns it verbatim; the codes requant every K-th
    write (the first write ALWAYS — the layer's init). The conv store
    stays fully online. Patched onto the cache INSTANCE only — the
    library is untouched (the --trace idiom; a measurement, not a
    regime change)."""
    from tq_cache import TQLinearAttentionLayer
    n = 0
    for layer in cache.layers:
        if not isinstance(layer, TQLinearAttentionLayer):
            continue
        n += 1
        orig_update = layer.update_recurrent_state
        raw = {"s": None}
        count = {"n": 0}

        def make_update(orig, raw, count):
            def update(s, state_idx=0, **kw):
                raw["s"] = s
                count["n"] += 1
                if count["n"] == 1 or count["n"] % every_k == 0:
                    orig(s, state_idx=state_idx, **kw)
                return s
            return update

        layer.update_recurrent_state = make_update(orig_update, raw, count)
        layer.recurrent_states = _RawSView(raw)
    return n


# ------------------------------------------------------------- the run ----
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--artifacts-dir", default=DEFAULTS["artifacts_dir"])
    ap.add_argument("--heads-dir", default=DEFAULTS["heads_dir"])
    ap.add_argument("--model-name", default=DEFAULTS["model_name"])
    ap.add_argument("--bits", type=float, default=3.5)
    ap.add_argument("--bits-ceiling", type=float, default=4.0,
                    help="the C2 cell's bit-width (the constructor's "
                         "1..4 Lloyd-Max range is the hard ceiling)")
    ap.add_argument("--max-new-tokens", type=int, default=48)
    ap.add_argument("--prompt", default="How do I deploy perf-canary?")
    ap.add_argument("--s-exact-every", type=int, default=16,
                    help="C3's requant cadence K (the codes refresh every "
                         "K-th write; the READ always rides the exact state)")
    args = ap.parse_args()

    L0 = "C0 raw DynamicCache (the reference)"
    L1 = f"C1 TQ online {args.bits} (the failure)"
    L2 = f"C2 TQ online {args.bits_ceiling} (the bits ceiling)"
    L3 = (f"C3 TQ online {args.bits} + S-EXACT (K={args.s_exact_every}),"
          f" conv online")
    L4 = "C4 TQ online=False (the D4 plumbing control)"

    print("=" * 60)
    print("TQ NOISE ISOLATOR (scripts/gpu/tq_noise_isolator.py) — W30")
    print("=" * 60)

    import turboquant
    C = turboquant._load_tq_kernel()
    print(f"[0] flute_extended._C: "
          f"{'present' if C is not None else 'ABSENT (torch twin only)'}"
          f" | torch {torch.__version__} | cuda "
          f"{torch.cuda.get_device_name(0)}")
    print("    (the W29 matrix already proved kernel parity EXACT; the "
          "per-head d=16,384 path rides the torch twin by dispatch)")
    print("    THE FLOOR MATH: 3.5-bit rel-MSE ~0.022 = 15% NORM "
          "perturbation per GDN layer per step (sqrt);")
    print("    the paper's D_mse(1..4) = 0.36/0.117/0.03/0.009 is a "
          "ONE-SHOT per-unit guarantee (Theorem 1) — no requant,")
    print("    no recurrence; the KV parity claim rides softmax "
          "averaging over the context. The GDN recurrence has none.")
    print("    THE BITS CEILING: eff bits > 4 are refused (the Lloyd-Max "
          "codebook range) — bits cannot reach fp16-level noise.")

    print("\n[1] Loading the model (no M1/M2)...")
    model, _ = load_model(args.artifacts_dir, args.heads_dir,
                          args.model_name, use_m1m2=False)
    model = model.eval()
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    dev = next(model.parameters()).device
    ids = tokenizer.encode(args.prompt, return_tensors="pt").to(dev)
    print(f"    prompt {args.prompt!r} -> {tuple(ids.shape)} tokens")

    from transformers import DynamicCache
    from tq_cache import TQCache, TQLinearAttentionLayer

    def fresh_raw():
        return DynamicCache(config=model.config)

    def fresh_tq(bits, online=True):
        c = TQCache(config=model.config, bits=bits, online=online)
        n = sum(1 for l in c.layers
                if isinstance(l, TQLinearAttentionLayer))
        print(f"    (built TQCache bits={bits} online={online}: "
              f"{n} TQ linear-attn layers of {len(c.layers)})")
        return c

    print(f"\n[1] FIVE CELLS ({args.max_new_tokens} new tokens, greedy, "
          f"manual loop, the same prompt)")
    results = {}
    ref_ids = None

    def run_cell(label, cache):
        t0 = time.time()
        with torch.no_grad():
            new_ids, confs = manual_greedy(model, ids, cache,
                                           args.max_new_tokens)
        dt = time.time() - t0
        print(f"\n    ---- {label} ----  ({dt:.1f}s)")
        verdict, conf = gen_gate(tokenizer, new_ids, confs,
                                 ref_ids=ref_ids)
        results[label] = (verdict, conf)
        return new_ids

    # C0 — the reference (sets the divergence baseline)
    ref_ids = run_cell(L0, fresh_raw())

    # C1 — the failure, instrumented
    stats = {"s_rel": [], "s_rel_by_layer": defaultdict(list),
             "conv_rel": [], "s_energy_first": None,
             "s_energy_last": None, "s_spread_max": 1.0, "s_writes": 0}
    c1 = fresh_tq(args.bits)
    n_layers = attach_telemetry(c1, stats)
    run_cell(L1, c1)
    report_telemetry(stats, n_layers)

    # C2 — the bits ceiling
    run_cell(L2, fresh_tq(args.bits_ceiling))

    # C3 — the S-exact instrument (S exact in the loop, conv online)
    c3 = fresh_tq(args.bits)
    attach_s_exact(c3, args.s_exact_every)
    run_cell(L3, c3)

    # C4 — the D4 offline control (everything raw during forward)
    run_cell(L4, fresh_tq(args.bits, online=False))

    # ---------------------------------------------------------- verdict --
    print("\n[3] VERDICT TABLE")
    for label, (verdict, conf) in results.items():
        print(f"    {label:52s}: {verdict:22s} conf {conf:.2f}")

    c0v, c1v, c3v, c4v = (results[L0][0], results[L1][0],
                          results[L3][0], results[L4][0])
    mean_s_rel = (sum(stats["s_rel"]) / len(stats["s_rel"])
                  if stats["s_rel"] else float("nan"))

    print("\n    THE DECISION TREE:")
    if c4v != c0v:
        print("    * C4 (offline) != C0 -> THE PLUMBING (beyond the "
              "stubs): paste EVERYTHING; re-run tq_gen_matrix.py "
              "--trace --max-new-tokens 8 (the protocol evidence).")
    if c4v == c0v and c3v == c0v and c1v != c0v:
        print("    * C4 == C3 == C0 clean, C1 garbage -> F2c CONFIRMED: "
              "the S per-step requant noise compounds through the "
              "recurrence; the conv store is exonerated. W31 = the "
              "mitigation decision (requant cadence with an exact "
              "working copy / D4-for-decode / bits) — THESE numbers "
              "decide it.")
    if c4v == c0v and c3v != c0v:
        print("    * C4 clean, C3 garbage -> THE CONV STORE contributes "
              "(or the S+conv interplay): W31 takes the conv into the "
              "cadence/offline set too — paste the conv round-trip line.")
    if mean_s_rel >= 0.035:
        print("    * C1 S round-trip rel-MSE is OFF THE FLOOR "
              f"(mean {mean_s_rel:.4f} >> 0.022) -> the real path DOES "
              "differ from the probe — PASTE THE TELEMETRY TABLE WHOLE.")
    else:
        print(f"    * C1 S round-trip rel-MSE mean {mean_s_rel:.4f} — "
              "at the floor: the wiring is clean, the quantizer is "
              "clean, the noise is the design's (F2c).")
    print("    * C2 vs C1 above -> the bits direction at the 4-bit "
          "ceiling (the delta is the datum; the ceiling itself cannot "
          "reach fp16-level noise).")
    print("\n    THE HONEST MEMO: the online regime's value is "
          "multi-stream serving memory; at B=1 decode the exact S "
          "working copy costs ~1 MiB/layer (~30 MiB total) — the "
          "cadence/D4 designs trade that against per-step noise.")
    print("\nPASTE THIS WHOLE OUTPUT BACK (the GPU-only law).")
    return 0


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("tq_noise_isolator.py: GPU-ONLY "
              "(torch.cuda.is_available() is False) — run on the A10G box",
              file=sys.stderr)
        raise SystemExit(2)
    raise SystemExit(main())
