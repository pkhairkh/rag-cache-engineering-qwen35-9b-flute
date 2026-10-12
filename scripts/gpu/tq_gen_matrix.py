#!/usr/bin/env python3
"""tq_gen_matrix.py — W28: THE TQ MALFUNCTION LOCALIZER (no M1/M2 anywhere).

The user's evidence (2026-10-12, the box, verbatim): the palettized model
WITHOUT TurboQuant generates very well; WITH TurboQuant
(TQCache(config, bits=3.5, online=True) + model.generate) it produces
multilingual soup from the first tokens. The repo's own history says the
OPPOSITE for the manual driving loop (the box's 2026-10-10 session:
TQCache + a hand-rolled greedy loop generated "Paris<|im_end|>" cleanly
— but that predates W15/W20 and was never re-gated). This script closes
exactly that gap and LOCALIZES the malfunction in one run:

[0] THE BUILD + PARITY PROBE (the kernel question, "tq on kernel works
    fine?"): reports whether flute_extended._C carries the TQ kernel
    pair, then measures kernel-path vs torch-twin round trips at the
    PRODUCTION geometry (S: d=524,288, the 3/4 split) on THREE input
    distributions — the suite's Gaussian, a heavy-tailed draw, and a
    per-head-spread GDN-like state (32 blocks with a realistic norm
    spread). The last one ALSO reports PER-BLOCK rel-MSE: the W28
    finding is that ONE norm + ONE codebook over the concatenated
    32-head unit is a dynamic-range hazard (dev-box numpy sim:
    worst-head rel-MSE 10.7x at a 55x head-norm spread, 297x at 525x —
    the TOTAL rel-MSE stays ~2%, which is exactly why every green gate
    missed it). A kernel/torch divergence here = a STALE EXTENSION
    (the repo is a copy; rebuild: see the handover) — the probe names
    it with numbers.
[1] THE 2x2 GENERATION MATRIX — the same prompt, the same model, the
    same tokenizer, four cells:
      G1  raw DynamicCache + MANUAL greedy loop   (the W14 control's
          exact driving style)
      G2  TQCache(online=True) + MANUAL greedy loop (the Oct-10
          session's style — quantizer distortion only, no generate
          machinery)
      G3  raw DynamicCache + model.generate()     (the generate
          machinery alone)
      G4  TQCache(online=True) + model.generate() (the user's failing
          test, verbatim)
    Each cell prints the [gen gate] line (mean conf, repetition,
    non-ascii, verdict band) and the answer text.
[2] THE VERDICT TABLE — the decision tree, printed:
      G1 clean, G2 garbage  -> THE QUANTIZER/WIRING on real states:
                               read [0]'s per-block report; the
                               per-head unit split is the repair (a
                               re-ingest decision — tell the dev box).
      G2 clean, G4 garbage  -> THE GENERATE() INTERPLAY: run again with
                               --trace (every Cache/layer method call
                               printed verbatim); the protocol guards
                               in tq_cache.py already turn the known
                               silent mutations (crop /
                               batch_repeat_interleave) into loud
                               failures that name themselves.
      G3 garbage too        -> NOT TQ: the model/generate interplay —
                               paste all four cells; the dev box
                               re-plans.
      all four clean        -> the malfunction was the stale extension
                               or the environment: [0] failed or the
                               copy drifted — rebuild flute_extended
                               and re-run.
[3] --trace: wraps the TQCache and every TQLinearAttentionLayer method
    in a printing shim — the exact call sequence generate() makes,
    verbatim (the protocol evidence the generate-interplay leg needs).

GPU-ONLY: every number comes from this box (the GPU-only law). No
corpus, no M1/M2, no retrieval — the model, the cache, and four runs.

Examples:
    python3 scripts/gpu/tq_gen_matrix.py
    python3 scripts/gpu/tq_gen_matrix.py --max-new-tokens 64
    python3 scripts/gpu/tq_gen_matrix.py --trace --max-new-tokens 8
"""
from __future__ import annotations

import argparse
import contextlib
import math
import sys

import torch

from _bootstrap import DEFAULTS, boot, load_model

boot()


# ------------------------------------------------------------- gate utils --
def gen_gate(tokenizer, new_token_ids, token_confs):
    """The [gen gate] line (W27's honest metrics — conf alone passed the
    box's soup at 0.382 once already)."""
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
    print(f"    [gen gate] conf {conf:.3f} | rep {rep:.2f} | "
          f"non-ascii {non_ascii:.2f} | {verdict}")
    print(f"    text: {answer[:160]!r}")
    return conf, rep, non_ascii


@contextlib.contextmanager
def _torch_twin():
    """Force the torch reference path ON CUDA (the dispatcher's ONE probe
    patched off — the kernel tests' idiom, restored on exit)."""
    import turboquant
    original = turboquant._tq_kernel_available
    turboquant._tq_kernel_available = lambda x=None: False
    try:
        yield
    finally:
        turboquant._tq_kernel_available = original


# ------------------------------------------------------- [0] the probe ----
def build_and_parity_report():
    import turboquant
    from turboquant import TurboQuant

    C = turboquant._load_tq_kernel()
    have_kernels = C is not None
    print("[0] BUILD + PARITY PROBE")
    print(f"    flute_extended._C: "
          f"{'present' if C is not None else 'ABSENT (torch twin only)'}")
    if C is not None:
        pair = [n for n in ("tq_quant_batched", "tq_dequant_batched")
                if hasattr(C, n)]
        print(f"    TQ kernel pair: {len(pair)}/2 present {pair}")
    print(f"    torch {torch.__version__} | cuda "
          f"{torch.cuda.get_device_name(0)}")

    if not torch.cuda.is_available():
        print("    (no CUDA — the probe and the matrix below refuse; "
              "this script is GPU-only)")
        return have_kernels

    dev = torch.device("cuda")
    d = 1 << 19
    H, DK, DV = 32, 128, 128
    q = TurboQuant(kind="S", bits=3.5, d=d, seed=101)

    def roundtrip(x, kernel):
        if kernel:
            return q.dequant(q.quant(x))
        with _torch_twin():
            return q.dequant(q.quant(x))

    g = torch.Generator(device="cpu").manual_seed(20281012)
    dists = {}
    dists["gaussian (the suite's state)"] = \
        torch.randn(d, generator=g).to(dev)
    # heavy tails: t-distributed df=2 (normalized) — outlier coordinates
    t = torch.randn(d, generator=g) / torch.sqrt(
        torch.rand(d, generator=g) / 2.0).clamp_min(1e-6)
    dists["heavy-tail (t, df=2)"] = (t / t.norm()).to(dev)
    # the GDN-like state: 32 head blocks with a realistic norm spread
    xg = torch.randn(H, DK * DV, generator=g)
    scales = torch.pow(10.0, torch.rand(H, generator=g) * 1.75 - 1.75)
    dists["per-head-spread (GDN-like)"] = \
        (xg * scales.view(H, 1)).reshape(-1).to(dev)

    print("    distribution                    | kernel vs torch "
          "max|diff| | round-trip rel-MSE (kernel) | worst-block rel-MSE")
    for name, x in dists.items():
        xk = roundtrip(x, kernel=True)
        xt = roundtrip(x, kernel=False)
        diff = float((xk - xt).abs().max())
        rel = float(((xk - x) ** 2).sum() / max((x ** 2).sum(), 1e-30))
        # per-block fidelity (the dynamic-range hazard)
        xb = x.reshape(H, -1)
        xkb = xk.reshape(H, -1)
        block_rel = (((xkb - xb) ** 2).sum(-1)
                     / ((xb ** 2).sum(-1).clamp_min(1e-30)))
        worst = float(block_rel.max())
        print(f"    {name:32s} | {diff:.3e} | {rel:.4f} | {worst:.2f}")
        if name.startswith("per-head"):
            print(f"      (the block spread was "
                  f"{float(scales.max()/scales.min()):.0f}x; the TOTAL "
                  f"rel-MSE hides exactly this — see FINDINGS W28)")
    if have_kernels:
        print("    PARITY: kernel-vs-torch divergences above ~1e-4 on "
              "ANY row = a STALE/MISMATCHED EXTENSION BUILD — rebuild "
              "flute_extended (HANDOVER W28) and re-run")
    return have_kernels


# ---------------------------------------------------- the driving styles --
def manual_greedy(model, ids, cache, max_new_tokens):
    """The W14 control's exact style: model() calls + argmax feedback."""
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


def generate_greedy(model, ids, cache, max_new_tokens):
    """The user's test, verbatim: model.generate with the cache."""
    out = model.generate(ids, max_new_tokens=max_new_tokens,
                         do_sample=False, past_key_values=cache)
    # input echo excluded: the NEW tokens only
    return [int(t) for t in out[0][ids.shape[1]:]], []


# ------------------------------------------------------- [3] the trace ----
@contextlib.contextmanager
def trace_cache_calls(cache):
    """Wrap every public method of the cache + its TQ layers in a
    printing shim — the exact protocol call sequence, verbatim (the
    generate-interplay evidence)."""
    from tq_cache import TQLinearAttentionLayer

    def wrap(obj, name, depth=0):
        prefix = "  " * (depth + 1)
        for attr in list(vars(obj).keys()):
            pass  # instance attrs are state, not methods
        for meth in _public_methods(obj.__class__):
            if meth.startswith("__"):
                continue
            try:
                fn = getattr(obj, meth)
            except Exception:
                continue
            if not callable(fn) or getattr(fn, "_tq_traced", False):
                continue

            def make(m, f):
                def shim(*a, **kw):
                    print(f"{prefix}{type(obj).__name__}.{m}("
                          f"{', '.join([type(a).__name__ for a in a])}"
                          f"{', '.join(f'{k}=' for k in kw)})")
                    return f(*a, **kw)
                shim._tq_traced = True
                return shim
            setattr(obj, meth, make(meth, fn))

    def _public_methods(cls):
        seen, out = set(), []
        for klass in cls.__mro__:
            for k, v in vars(klass).items():
                if callable(v) and k not in seen:
                    seen.add(k)
                    out.append(k)
        return out

    from transformers.cache_utils import DynamicCache
    print("[trace] wrapping TQCache + TQLinearAttentionLayer methods")
    wrap(cache, "cache", 0)
    for layer in cache.layers:
        if isinstance(layer, TQLinearAttentionLayer):
            wrap(layer, f"layer", 1)
    # keep isinstance(x, DynamicCache) honest: methods are shadowed on
    # the INSTANCE, the class is untouched
    assert isinstance(cache, DynamicCache)
    try:
        yield
    finally:
        pass  # instance-level shadows die with the cache object


# ------------------------------------------------------------- the run ----
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--artifacts-dir", default=DEFAULTS["artifacts_dir"])
    ap.add_argument("--heads-dir", default=DEFAULTS["heads_dir"])
    ap.add_argument("--model-name", default=DEFAULTS["model_name"])
    ap.add_argument("--bits", type=float, default=3.5)
    ap.add_argument("--max-new-tokens", type=int, default=48)
    ap.add_argument("--prompt", default="How do I deploy perf-canary?")
    ap.add_argument("--trace", action="store_true",
                    help="print every Cache/layer method call the "
                         "generation makes (the protocol evidence)")
    args = ap.parse_args()

    print("=" * 60)
    print("TQ GEN MATRIX (scripts/gpu/tq_gen_matrix.py) — W28")
    print("=" * 60)

    build_and_parity_report()

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
    from tq_cache import TQCache

    def fresh_raw():
        return DynamicCache(config=model.config)

    def fresh_tq():
        return TQCache(config=model.config, bits=args.bits, online=True)

    cells = [
        ("G1 raw   + manual loop ", fresh_raw, manual_greedy),
        ("G2 TQ    + manual loop ", fresh_tq, manual_greedy),
        ("G3 raw   + generate()   ", fresh_raw, generate_greedy),
        ("G4 TQ    + generate()   ", fresh_tq, generate_greedy),
    ]

    print(f"\n[2] THE 2x2 MATRIX ({args.max_new_tokens} new tokens, "
          f"greedy, the same prompt)")
    results = {}
    for label, cache_fn, drive in cells:
        print(f"\n    ---- {label} ----")
        cache = cache_fn()
        cm = trace_cache_calls(cache) if args.trace else contextlib.nullcontext()
        try:
            with torch.no_grad(), cm:
                new_ids, confs = drive(model, ids, cache,
                                       args.max_new_tokens)
        except Exception as exc:  # a loud guard firing IS the evidence
            print(f"    CRASH: {type(exc).__name__}: {exc}")
            results[label] = ("CRASH", 0.0, 0.0, 1.0)
            continue
        results[label] = gen_gate(tokenizer, new_ids, confs)

    print("\n[3] VERDICT TABLE")
    g1, g2, g3, g4 = (results[c[0]] for c in cells)
    print(f"    G1(raw+manual): {g1[0]:7s} conf {g1[1]:.2f}")
    print(f"    G2(TQ+manual):  {g2[0]:7s} conf {g2[1]:.2f}")
    print(f"    G3(raw+gen):    {g3[0]:7s} conf {g3[1]:.2f}")
    print(f"    G4(TQ+gen):     {g4[0]:7s} conf {g4[1]:.2f}")
    print("\n    THE DECISION TREE (read G2 and G4 first):")
    if g2[0] != "CLEAN-RANGE":
        print("    * G2 garbage -> THE QUANTIZER/WIRING on real states: "
              "read [0]'s per-block row; the per-head unit split is the "
              "repair (a re-ingest decision — report to the dev box).")
    if g2[0] == "CLEAN-RANGE" and g4[0] != "CLEAN-RANGE":
        print("    * G2 clean, G4 garbage -> THE GENERATE() INTERPLAY: "
              "re-run with --trace (the protocol call sequence, "
              "verbatim); the tq_cache guards name the culprit if it is "
              "crop/batch_repeat_interleave.")
    if g3[0] != "CLEAN-RANGE":
        print("    * G3 garbage too -> NOT TQ: the model/generate "
              "interplay — paste all four cells.")
    if all(r[0] == "CLEAN-RANGE" for r in (g1, g2, g3, g4)):
        print("    * all clean -> the malfunction was the ENVIRONMENT "
              "(stale extension / drifted copy): check [0]'s parity "
              "numbers, rebuild flute_extended, re-run.")
    print("\nPASTE THIS WHOLE OUTPUT BACK (the GPU-only law).")
    return 0


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("tq_gen_matrix.py: GPU-ONLY (torch.cuda.is_available() is "
              "False) — run on the A10G box", file=sys.stderr)
        raise SystemExit(2)
    raise SystemExit(main())
