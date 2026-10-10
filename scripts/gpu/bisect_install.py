#!/usr/bin/env python3
"""bisect_install.py — pinpoint any residual install-time corruption.

The W11 CPU adjudication proved the S-install math, the read path and the
install semantics CLEAN at reference-kernel fidelity
(src/rag/tests/test_install_real_math.py), and fixed the conv geometry
(the 16,384 truncation that zeroed 8,192 of every 24,576-coordinate conv
window — the only proven corruption in the pushed pipeline). This harness
bisects anything that REMAINS broken on the GPU box, one axis at a time:

  axis A — install content:  reseed-only / S-only / conv-only / full
  axis B — kernel route:     FLA Triton kernels vs FLUTE_NO_FLA=1
                             (pure-torch decode, bit-identical contract)
  checks —
    S-read   the layer-0 S state the model actually reads, vs the exact
             expected dequant(sys)+dequant(delta) (catches frame/device/
             quantizer drift — proven clean on CPU, so a GPU failure here
             is a device/kernel bug);
    conv-tail the read-back conv window's coordinate energy split
             (coords 0:16384 vs 16384:24576 — the truncation's signature
             was a dead tail; equal energy = healthy full window);
    gen      greedy decode conf/repetition + text (the human-readable
             bottom line).

Run AFTER re-ingesting with the W11 code (the pre-W11 snapshots carry
16,384-truncated conv codes — the frame guard will say so loudly).

Examples:
    python3 scripts/gpu/bisect_install.py
    python3 scripts/gpu/bisect_install.py --fla-off
    python3 scripts/gpu/bisect_install.py --chunk 0 --question "What is 2+2?"
"""
from __future__ import annotations

import argparse
import os

import torch

from _bootstrap import DEFAULTS, boot, load_model

boot()


def _greedy(model, cache, first_ids, n, tokenizer):
    ids = first_ids
    out = model(input_ids=ids, past_key_values=cache, use_cache=True)
    logits = out.logits if hasattr(out, "logits") else (out[0] if isinstance(out, tuple) else out)
    toks, confs = [], []
    for _ in range(n):
        last = logits[:, -1, :].float()
        probs = torch.softmax(last, dim=-1)
        conf, next_id = probs.max(dim=-1)
        toks.append(int(next_id))
        confs.append(float(conf))
        if int(next_id) == tokenizer.eos_token_id:
            break
        ids = torch.cat([ids, next_id.unsqueeze(0)], dim=1)
        out = model(input_ids=next_id.unsqueeze(0), past_key_values=cache,
                    use_cache=True)
        logits = out.logits if hasattr(out, "logits") else (out[0] if isinstance(out, tuple) else out)
    return toks, confs


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--artifacts-dir", default=DEFAULTS["artifacts_dir"])
    ap.add_argument("--heads-dir", default=DEFAULTS["heads_dir"])
    ap.add_argument("--model-name", default=DEFAULTS["model_name"])
    ap.add_argument("--disk-dir", default=DEFAULTS["disk_dir"])
    ap.add_argument("--chunk", type=int, default=0)
    ap.add_argument("--question", default="What is 2+2?")
    ap.add_argument("--max-new-tokens", type=int, default=12)
    ap.add_argument("--bits", type=float, default=3.5)
    ap.add_argument("--fla-off", action="store_true",
                    help="FLUTE_NO_FLA=1 — pure-torch decode kernels")
    args = ap.parse_args()
    if args.fla_off:
        os.environ["FLUTE_NO_FLA"] = "1"

    from transformers import AutoTokenizer
    from ingest import load_system_state, reseed_cache
    from install import install_snapshot
    from snapshot import load_chunk
    from tq_cache import TQCache, resolve_quantizer

    print("=" * 60)
    print("RAGGA INSTALL BISECTION (scripts/gpu/bisect_install.py)")
    print(f"{'FLUTE_NO_FLA=1 (torch decode) ' if args.fla_off else ''}")
    print("=" * 60)

    model, _ = load_model(args.artifacts_dir, args.heads_dir, args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    system = load_system_state(args.disk_dir)
    snap = load_chunk(os.path.join(args.disk_dir, "snapshots",
                                   f"chunk_{args.chunk:05d}.npz"))
    q = resolve_quantizer("S", system.s_codes[0].d, args.bits)

    def fresh_cache():
        return TQCache(config=model.config, bits=args.bits, online=True)

    def s_read_check(cache):
        """Layer-0 S state the model reads vs the exact expected sum."""
        delta0 = snap.s_codes.get(0)
        expected = q.dequant(system.s_codes[0])
        if delta0 is not None:
            expected = expected + q.dequant(delta0)
        got = cache.layers[0].recurrent_states[0].reshape(-1).float().cpu()
        rel = float(((got - expected) ** 2).sum()
                    / (expected ** 2).sum().clamp_min(1e-30))
        print(f"    S-read: rel-MSE {rel:.2e} "
              f"({'OK' if rel < 1e-4 else 'DRIFT — frame/device bug'})")
        return rel < 1e-4

    def conv_tail_check(cache):
        """The conv window's coordinate energy split (truncation detector)."""
        w = cache.layers[0].conv_states[0].reshape(-1).float().cpu()
        head, tail = w[:16384].abs().mean().item(), w[16384:24576].abs().mean().item()
        healthy = tail > 0.1 * head
        print(f"    conv-tail energy: head {head:.4f} tail {tail:.4f} "
              f"({'OK' if healthy else 'DEAD TAIL — truncated conv codes (re-ingest)'})")
        return healthy

    def gen_check(cache, name):
        with torch.no_grad():
            ids = tokenizer.encode(args.question, return_tensors="pt").cuda()
            toks, confs = _greedy(model, cache, ids, args.max_new_tokens, tokenizer)
        text = tokenizer.decode(toks, skip_special_tokens=True)
        conf = sum(confs) / len(confs) if confs else 0.0
        rep = (sum(1 for a, b in zip(toks, toks[1:]) if a == b)
               / max(1, len(toks) - 1))
        print(f"    gen: conf {conf:.3f} rep {rep:.2f} text {text!r}")
        return conf >= 0.20 and rep <= 0.6

    variants = ["reseed", "s-only", "conv-only", "full"]
    results = {}
    for variant in variants:
        print(f"\n[{variant}] " + "-" * 50)
        with torch.no_grad():
            cache = fresh_cache()
            reseed_cache(cache, system)
            if variant == "s-only":
                install_snapshot(cache, system, [snap])
                # keep conv at the system codes: reinstall them over the
                # chunk conv the full install just wrote
                for L, c in system.conv_codes.items():
                    cache.set_conv_codes(L, c)
            elif variant == "conv-only":
                for L, c in snap.conv_codes.items():
                    if c is not None:
                        cache.set_conv_codes(L, c)
            elif variant == "full":
                install_snapshot(cache, system, [snap])
            ok_read = s_read_check(cache)
            ok_tail = conv_tail_check(cache)
            ok_gen = gen_check(cache, variant)
            results[variant] = (ok_read, ok_tail, ok_gen)

    print("\n" + "=" * 60)
    print(f"BISECTION MATRIX{' (FLA OFF)' if args.fla_off else ''}")
    print(f"{'variant':<12} {'S-read':<8} {'conv-tail':<10} {'gen':<6}")
    for v, (a, b, c) in results.items():
        print(f"{v:<12} {'OK' if a else 'DRIFT':<8} "
              f"{'OK' if b else 'DEAD':<10} {'OK' if c else 'GARBAGE':<6}")
    print("-" * 60)
    print("Reading the matrix: S-read DRIFT => device/frame bug in the quant")
    print("path; conv-tail DEAD => stale pre-W11 snapshots (re-ingest); gen")
    print("GARBAGE with both checks OK => the FLA Triton decode route — rerun")
    print("with --fla-off and compare.")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
