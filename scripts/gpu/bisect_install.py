#!/usr/bin/env python3
"""bisect_install.py — pinpoint any residual install-time corruption.

W12 REVISION — THE S-READ CHECK WAS WRONG, and its W12 output sent the
session chasing a ghost:

  * the OLD s_read_check compared the layer-0 S read against
    dequant(sys) + dequant(delta) UNCONDITIONALLY. For the reseed and
    conv-only variants the delta was NEVER installed, so the check
    printed the delta's energy fraction ||delta||^2/||sys+delta||^2
    (~0.85 uncorrelated; the box measured 7.66e-01) and called it DRIFT.
  * for the s-only/full variants the printed number was the EXPECTED
    single-requant error of install(sys, delta) (~1.5e-02, inside the
    0.06 house budget) judged against an impossible 1e-4 threshold
    that was written for the bit-clean reseed case.

The corrected check below uses the per-variant reference with both
thresholds. The W12 "S-read DRIFT in every variant" finding is hereby
RETRACTED: the S path was and is clean — pinned end-to-end by
src/rag/tests/test_install_real_math.py (the S install math) and
test_install_conv_math.py (the S+conv COMBINATION through the real GDN
forward — the gap the W12 matrix pointed at, now closed).

Axes and checks:

  axis A — install content:  reseed / s-only / conv-only / full +
                             true-doc (the raw semantic control, new)
  axis B — kernel route:     FLA Triton kernels vs FLUTE_NO_FLA=1
                             (pure-torch decode, bit-identical contract)
  checks —
    S-read   the layer-0 S state the model reads vs the CORRECT
             per-variant reference: dequant(sys) for reseed/conv-only
             (bit-clean, 1e-4), dequant(sys)+dequant(delta) for
             s-only/full (the single-requant budget, 0.06);
    frame    the one GPU-only frame risk: quant() on a CUDA input takes
             the FHT CUDA kernel (the conv window's 24,576 = 16,384 +
             8,192 is kernel-eligible) while dequant() ALWAYS runs the
             torch reference (numpy codebook lookup -> from_numpy ->
             CPU butterfly, then .to(device)). A kernel/reference
             transform mismatch would put the conv codes in a rotation
             no read path can undo. The check quantizes a random tensor
             ON DEVICE and dequantizes through the normal path: rel-MSE
             at the Lloyd-Max level (~0.02) = frames consistent; ~1.0 =
             frame split (the smoking gun);
    conv     the cache-read conv window vs the direct reference dequant
             of the installed codes (read-path validation) + the dead
             tail detector (a zeroed 16,384:24,576 slice = stale
             pre-W11 snapshots — re-ingest);
    gen      greedy decode conf/rep/text — now printed WITH the numbers
             in the matrix (the W12 matrix's OK/GARBAGE hid them; a
             0.21-vs-0.19 conf cliff and a 0.9-vs-0.01 cliff read very
             differently);
    true-doc the decisive SEMANTIC control for the G5 failure: prefill
             [system + chunk tokens] on a RAW cache (no TQ anywhere),
             drop the full-attn KV (DynamicLayer.reset — the G5
             geometry: the linear state carries the document, full-attn
             starts empty), then the same query + greedy decode. If
             true-doc ALSO fails the gen gate, the G5 "garbage" is the
             model faithfully CONTINUING FROM THE DOCUMENT STATE (the
             installed state equals exactly this raw state within
             quantization) — a design-level behavior, not install
             corruption. If true-doc generates confidently, the gap
             between true-doc and full isolates the TQ path (then read
             the frame/S-read/conv rows).

Run AFTER re-ingesting with the W11+ code. Examples:
    python3 scripts/gpu/bisect_install.py
    python3 scripts/gpu/bisect_install.py --fla-off
    python3 scripts/gpu/bisect_install.py --chunk 0 --question "What is 2+2?"
"""
from __future__ import annotations

import argparse
import json
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
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--artifacts-dir", default=DEFAULTS["artifacts_dir"])
    ap.add_argument("--heads-dir", default=DEFAULTS["heads_dir"])
    ap.add_argument("--model-name", default=DEFAULTS["model_name"])
    ap.add_argument("--disk-dir", default=DEFAULTS["disk_dir"])
    ap.add_argument("--corpus", default=DEFAULTS["corpus"],
                    help="documents.jsonl (the true-doc control rebuilds "
                         "the chunk's tokens from it; MUST be the corpus "
                         "run_ingestion.py used)")
    ap.add_argument("--system-prompt",
                    default="You are a helpful AI assistant.",
                    help="MUST match run_ingestion.py's --system-prompt "
                         "(the true-doc control re-prefills it)")
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
    from transformers.cache_utils import DynamicLayer, LinearAttentionCacheLayerMixin
    from transformers import DynamicCache
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
    device = next(model.parameters()).device
    q_s = resolve_quantizer("S", system.s_codes[0].d, args.bits)
    delta0 = snap.s_codes.get(0)

    def fresh_cache():
        return TQCache(config=model.config, bits=args.bits, online=True)

    # ------------------------------------------------ the frame check ----
    # quant() on-device (CUDA kernel route for kernel-eligible d — the
    # conv window's 24,576) vs dequant() (always the torch reference):
    # the ONE spot where a GPU-only transform mismatch could split the
    # D3 frame. ~0.02 = consistent; ~1.0 = frame split.
    print("\n[frame] on-device quant -> reference dequant roundtrip")
    frame_ok = True
    for kind, codes in (("S", system.s_codes[0]),
                        ("conv", next(iter(system.conv_codes.values())))):
        q = resolve_quantizer(kind, codes.d, args.bits)
        g = torch.Generator(device="cpu").manual_seed(777 + codes.d)
        x = torch.randn(codes.d, generator=g).to(device)
        codes_rt = q.quant(x)                       # the device route
        xr = q.dequant(codes_rt)                    # the reference route
        rel = float(((xr - x.float().cpu()) ** 2).sum()
                    / (x.float().cpu() ** 2).sum())
        ok = rel < 0.06                             # the Lloyd-Max level
        frame_ok = frame_ok and ok
        verdict = "OK" if ok else (
            "FRAME SPLIT — the CUDA FHT kernel and the reference "
            "disagree; every conv quant on this box writes codes no "
            "read can undo")
        print(f"    {kind:<5} d={codes.d:>7}: rel-MSE {rel:.2e} "
              f"({verdict})")
    q = q_s  # the S quantizer, reused by the checks below

    # --------------------------------------------------- the checks ----
    def s_read_check(cache, variant):
        """Layer-0 S state the model reads vs the CORRECT reference:
        the system codes for reseed/conv-only (bit-clean), the exact sum
        for s-only/full (the single-requant budget)."""
        expected = q.dequant(system.s_codes[0])
        threshold = 1e-4
        if variant in ("s-only", "full") and delta0 is not None:
            expected = expected + q.dequant(delta0)
            threshold = 0.06
        got = cache.layers[0].recurrent_states[0].reshape(-1).float().cpu()
        rel = float(((got - expected) ** 2).sum()
                    / (expected ** 2).sum().clamp_min(1e-30))
        print(f"    S-read: rel-MSE {rel:.2e} (<= {threshold:g} "
              f"{'OK' if rel < threshold else 'DRIFT — frame/device bug'})")
        return rel < threshold

    def conv_check(cache):
        """The conv window: read-path fidelity (the cache read vs the
        direct reference dequant of the installed codes — catches device/
        frame drift in the read) + the truncation detector (dead tail)."""
        layer = cache.layers[0]
        codes = layer.conv_codes
        if codes is None:
            print("    conv: no codes installed")
            return False
        w = layer.conv_states[0].reshape(-1).float().cpu()
        w_ref = resolve_quantizer("conv", codes.d, args.bits) \
            .dequant(codes).float().cpu()
        rel = float(((w - w_ref) ** 2).sum() / (w_ref ** 2).sum())
        head = w[:16384].abs().mean().item()
        tail = w[16384:24576].abs().mean().item()
        ok_read = rel < 5e-3
        ok_tail = tail > 0.1 * head
        read_verdict = "OK" if ok_read else (
            "READ DRIFT — device/frame bug in the conv read path")
        tail_verdict = "OK" if ok_tail else (
            "DEAD TAIL — stale pre-W11 snapshots (re-ingest)")
        print(f"    conv: read-vs-reference {rel:.2e} ({read_verdict}); "
              f"tail energy head {head:.4f} tail {tail:.4f} "
              f"({tail_verdict})")
        return ok_read and ok_tail

    def state_scale(cache):
        s = cache.layers[0].recurrent_states[0].reshape(-1).float()
        print(f"    S scale: std {s.std().item():.4f} "
              f"(system ~0.11, doc-loaded ~0.30 on the W12 box)")

    def gen_check(cache):
        with torch.no_grad():
            ids = tokenizer.encode(args.question, return_tensors="pt").to(device)
            toks, confs = _greedy(model, cache, ids, args.max_new_tokens, tokenizer)
        text = tokenizer.decode(toks, skip_special_tokens=True)
        conf = sum(confs) / len(confs) if confs else 0.0
        rep = (sum(1 for a, b in zip(toks, toks[1:]) if a == b)
               / max(1, len(toks) - 1))
        ok = conf >= 0.20 and rep <= 0.6
        print(f"    gen: conf {conf:.3f} rep {rep:.2f} text {text!r}")
        return ok, conf, rep

    # --------------------------------------------- the true-doc flow ----
    def true_doc_flow():
        """The raw semantic control: [system + chunk] prefilled on a RAW
        cache, full-attn KV dropped (the G5 geometry), query + decode."""
        path = args.corpus
        if not os.path.exists(path):
            print(f"    [true-doc SKIPPED — corpus not found: {path}]")
            return None, None, None
        try:
            with open(path) as f:
                for i, line in enumerate(f):
                    if i == args.chunk:
                        doc = json.loads(line)
                        doc = doc.get("text", doc.get("content", ""))
                        break
                else:
                    raise IndexError(args.chunk)
        except (IndexError, json.JSONDecodeError) as exc:
            print(f"    [true-doc SKIPPED — chunk read failed: {exc}]")
            return None, None, None
        sys_ids = tokenizer.encode(args.system_prompt,
                                   add_special_tokens=False)
        doc_ids = tokenizer.encode(doc, add_special_tokens=False)
        with torch.no_grad():
            cache = DynamicCache(config=model.config)
            model(input_ids=torch.tensor([sys_ids + doc_ids],
                                         dtype=torch.long,
                                         device=device),
                  past_key_values=cache)
            # drop the full-attn KV: G5's geometry (fresh full-attn; the
            # linear state carries the doc). DynamicLayer.reset() drops
            # keys/values; the linear layers are untouched.
            for layer in cache.layers:
                if isinstance(layer, DynamicLayer) \
                        and not isinstance(layer, LinearAttentionCacheLayerMixin):
                    layer.reset()
            ids = tokenizer.encode(args.question, return_tensors="pt").to(device)
            toks, confs = _greedy(model, cache, ids, args.max_new_tokens,
                                  tokenizer)
        text = tokenizer.decode(toks, skip_special_tokens=True)
        conf = sum(confs) / len(confs) if confs else 0.0
        rep = (sum(1 for a, b in zip(toks, toks[1:]) if a == b)
               / max(1, len(toks) - 1))
        ok = conf >= 0.20 and rep <= 0.6
        print(f"    true-doc ({len(doc_ids)} doc tokens): conf {conf:.3f} "
              f"rep {rep:.2f} text {text!r}")
        return ok, conf, rep

    # ------------------------------------------------------- the run ----
    variants = ["reseed", "s-only", "conv-only", "full", "true-doc"]
    results = {}
    for variant in variants:
        print(f"\n[{variant}] " + "-" * 50)
        with torch.no_grad():
            if variant == "true-doc":
                ok_gen, conf, rep = true_doc_flow()
                results[variant] = (None, None, ok_gen, conf, rep)
                continue
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
            ok_read = s_read_check(cache, variant)
            ok_conv = conv_check(cache)
            state_scale(cache)
            ok_gen, conf, rep = gen_check(cache)
            results[variant] = (ok_read, ok_conv, ok_gen, conf, rep)

    print("\n" + "=" * 60)
    print(f"BISECTION MATRIX{' (FLA OFF)' if args.fla_off else ''}")
    print(f"{'variant':<12} {'S-read':<8} {'conv':<8} {'gen':<8} "
          f"{'conf':>6} {'rep':>5}")
    for v, (a, b, c, conf, rep) in results.items():
        s = "-" if a is None else ("OK" if a else "DRIFT")
        cv = "-" if b is None else ("OK" if b else "DRIFT")
        gen = "??" if c is None else ("OK" if c else "GARBAGE")
        conf_s = "nan" if conf is None else f"{conf:.3f}"
        rep_s = "nan" if rep is None else f"{rep:.2f}"
        print(f"{v:<12} {s:<8} {cv:<8} {gen:<8} {conf_s:>6} {rep_s:>5}")
    print("-" * 60)
    print(f"frame check: {'OK' if frame_ok else 'FRAME SPLIT'}")
    print("Reading the matrix:")
    print("  * S-read/conv DRIFT => a REAL read-path bug (device/frame);")
    print("  * frame SPLIT => the CUDA FHT kernel disagrees with the")
    print("    reference at a kernel-eligible d (conv) — every conv code")
    print("    on the box is then written in an unreadable rotation;")
    print("  * true-doc GARBAGE (with reads + frame OK) => the G5 failure")
    print("    is SEMANTIC: the model faithfully continues from the")
    print("    document state — a prompt/flow design matter, not")
    print("    corruption. Compare true-doc's text with full's text;")
    print("  * reads/frame/true-doc all OK but full GARBAGE => the TQ")
    print("    quantization layer itself — rerun with --fla-off and")
    print("    compare (the Triton decode route is then the suspect).")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
