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

W15 REVISION — the checks the matrix still lacked:

  * frame (PURE): the old frame check's floor WAS the codebook noise
    (~0.02 rel-MSE = the Lloyd-Max distortion itself) — a subtle
    kernel/reference transform mismatch HIDES under it. The new pure
    check compares the TRANSFORMS with no quantization in between:
    (a) adjoint∘apply == identity (T is orthogonal — exact to fp32
    rounding), (b) the CUDA kernel vs the torch reference output at the
    production d's. Pass/fail at 1e-4, no noise floor.
  * TRUE-dist (NEW): the write-path distortion the read checks CANNOT
    see — the installed codes' dequant vs the RAW doc state the true-doc
    control materializes (the W14 matrix compared reads against the
    codes' own reference dequant, never against the truth). Printed
    per-variant for S and conv with the W15 budgets.
  * --qjl (NEW): the paper Alg.-2 A/B — TQCache(qjl=True) plumbs the
    residual-sketch compensation through the whole flow (ingest-free:
    the bisect quantizes live).

Axes and checks:

  axis A — install content:  reseed / s-only / conv-only / full +
                             true-doc (the raw semantic control)
  axis B — kernel route:     FLA Triton kernels vs FLUTE_NO_FLA=1
                             (pure-torch decode, bit-identical contract)
  axis C — W15 recipe:       --split-half (legacy fixed coordinate split)
                             vs default (the paper's outlier split) —
                             the conv write-path A/B
  checks —
    S-read   the layer-0 S state the model reads vs the CORRECT
             per-variant reference: dequant(sys) for reseed/conv-only
             (bit-clean, 1e-4), dequant(sys)+dequant(delta) for
             s-only/full (the single-requant budget, 0.06);
    frame    the PURE transform checks above (no codebook noise floor);
    conv     the cache-read conv window vs the direct reference dequant
             of the installed codes (read-path validation) + the dead
             tail detector (a zeroed 16,384:24,576 slice = stale
             pre-W11 snapshots — re-ingest);
    TRUE-dist the installed dequant vs the RAW [system+doc] state
             (write-path validation — the NEW W15 row);
    gen      greedy decode conf/rep/text — printed WITH the numbers;
    true-doc the decisive SEMANTIC control for the G5 failure.

Run AFTER re-ingesting with the W11+ code. Examples:
    python3 scripts/gpu/bisect_install.py
    python3 scripts/gpu/bisect_install.py --fla-off
    python3 scripts/gpu/bisect_install.py --chunk 0 --question "What is 2+2?"
    python3 scripts/gpu/bisect_install.py --qjl
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
    ap.add_argument("--qjl", action="store_true",
                    help="W15: the paper Alg.-2 A/B — TQCache(qjl=True), "
                         "the residual-sketch compensation end-to-end")
    ap.add_argument("--split-half", action="store_true",
                    help="W15 A/B: force the legacy fixed coordinate "
                         "half-split for conv (the pre-W15 recipe) — "
                         "isolates the paper outlier-split's effect")
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
    q_s = resolve_quantizer("S", system.s_codes[0].d, args.bits,
                            qjl=args.qjl)
    delta0 = snap.s_codes.get(0)

    def fresh_cache():
        return TQCache(config=model.config, bits=args.bits, online=True,
                       qjl=args.qjl)

    # -------------------------------------------- the PURE frame check ----
    # W15: the old check's floor WAS the codebook noise (~0.02) — a subtle
    # kernel/reference mismatch hides under it. The pure check compares
    # the TRANSFORMS with no quantization in between:
    #   (a) adjoint(apply(x)) == x  (T orthogonal — exact to fp32 rounding)
    #   (b) the CUDA kernel output vs the torch reference output
    # Both at 1e-4 rel — no noise floor. d's: the production S unit, the
    # conv window's split sub-units (16,384) and the legacy 24,576.
    import fht
    print("\n[frame] PURE transform checks (no codebook noise floor)")
    frame_ok = True
    for dd in (16384, 24576, system.s_codes[0].d):
        signs = fht.rotation_signs(dd, seed=dd)
        g = torch.Generator(device="cpu").manual_seed(777 + dd)
        x = torch.randn(1, dd, generator=g).to(device)
        y = fht.fht_apply(x, signs)                    # auto: kernel if CUDA
        xr = fht.fht_adjoint(y, signs)                 # the exact inverse
        rel_rt = float(((xr - x) ** 2).sum() / (x ** 2).sum())
        y_ref = fht.fht_reference(x.cpu(), signs).to(device)
        rel_kr = float(((y - y_ref) ** 2).sum() / (y_ref ** 2).sum())
        ok = rel_rt < 1e-4 and rel_kr < 1e-4
        frame_ok = frame_ok and ok
        verdict = "OK" if ok else (
            "FRAME SPLIT — the CUDA FHT kernel and the reference "
            "disagree; every quant on this box writes codes no read "
            "can undo")
        print(f"    d={dd:>7}: adjoint-roundtrip {rel_rt:.2e} "
              f"kernel-vs-reference {rel_kr:.2e} ({verdict})")
    # the W15 write-path A/B: --split-half forces the legacy fixed split
    if args.split_half:
        from tq_cache import resolve_quantizer as _rq
        import turboquant as _tq
        for key in [k for k in list(_tq._REGISTRY) if k.startswith("conv")]:
            del _tq._REGISTRY[key]
        _orig_init = _tq.TurboQuant.__init__

        def _no_split_init(self, *a, **kw):
            kw["group"] = 1          # force the legacy flat split
            _orig_init(self, *a, **kw)
        _tq.TurboQuant.__init__ = _no_split_init
        print("    [--split-half] conv quantizers forced to the legacy "
              "fixed coordinate half-split (the pre-W15 recipe)")
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
        frame drift in the read) + the truncation detector (dead tail).
        W15: the reference resolves at the CODES' group (outlier-partition
        units carry it; legacy units are flat)."""
        layer = cache.layers[0]
        codes = layer.conv_codes
        if codes is None:
            print("    conv: no codes installed")
            return False
        w = layer.conv_states[0].reshape(-1).float().cpu()
        grp = int(codes.group) if codes.group is not None else 1
        w_ref = resolve_quantizer("conv", codes.d, args.bits, group=grp) \
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
    raw_truth = {}   # layer -> {"s": tensor, "conv": tensor} (the W15
                     # TRUE-dist reference: the raw [system+doc] end state)

    def true_doc_flow():
        """The raw semantic control: [system + chunk] prefilled on a RAW
        cache, full-attn KV dropped (the G5 geometry), query + decode.
        W15: also snapshots the raw linear-layer end states — the WRITE-
        PATH truth the new TRUE-dist row compares the installed codes
        against (the read checks only ever compare against the codes'
        OWN reference dequant)."""
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
            # W15: capture the raw doc-loaded states (layer 0 + a middle
            # layer) BEFORE the full-attn reset — the write-path truth
            for L in (0, 12):
                lin = cache.layers[L]
                if isinstance(lin, LinearAttentionCacheLayerMixin):
                    raw_truth[L] = {
                        "s": lin.recurrent_states[0].reshape(-1)
                                .float().cpu().clone(),
                        "conv": lin.conv_states[0].reshape(-1)
                                  .float().cpu().clone()}
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

    def true_dist_check(cache, variant):
        """W15 — the write-path row the W14 matrix lacked: the installed
        codes' dequant vs the RAW doc state (layer 0). The read checks
        cannot see this — they compare the cache read against the SAME
        codes' reference dequant. Budgets: S <= 0.15 (the online
        ingestion drift compounds through the recurrence; the install
        math itself is ~1.3x single-shot — test_install_real_math),
        conv <= 0.03 (the split's measured write-path level; the legacy
        fixed split sat at 0.02-0.025)."""
        if not raw_truth or variant not in ("s-only", "full",
                                            "conv-only", "reseed"):
            return None
        L = 0
        truth = raw_truth[L]
        # the expected TRUE state per variant: reseed/conv-only -> the
        # SYSTEM state (the doc never installed); s-only/full -> the doc
        # state. The system truth is dequant(sys); compare only where it
        # is meaningful (reseed/conv-only vs the system reference).
        got_s = cache.layers[L].recurrent_states[0].reshape(-1).float().cpu()
        got_c = cache.layers[L].conv_states[0].reshape(-1).float().cpu()
        if variant in ("reseed", "conv-only"):
            exp_s = q.dequant(system.s_codes[L])
            rel_s = float(((got_s - exp_s) ** 2).sum()
                          / (exp_s ** 2).sum().clamp_min(1e-30))
            rel_c = None
            rel_s_budget = 1e-4
        else:
            exp_s = truth["s"]
            rel_s = float(((got_s - exp_s) ** 2).sum()
                          / (exp_s ** 2).sum().clamp_min(1e-30))
            rel_c = float(((got_c - truth["conv"]) ** 2).sum()
                          / (truth["conv"] ** 2).sum().clamp_min(1e-30))
            rel_s_budget = 0.15
        line = (f"    TRUE-dist: S {rel_s:.3f} (<= {rel_s_budget:g})")
        if rel_c is not None:
            line += f" conv {rel_c:.3f} (<= 0.03)"
        print(line)
        if rel_c is not None:
            return rel_s <= rel_s_budget and rel_c <= 0.03
        return rel_s <= rel_s_budget

    # ------------------------------------------------------- the run ----
    # true-doc runs FIRST so the raw_truth capture feeds TRUE-dist
    variants = ["true-doc", "reseed", "s-only", "conv-only", "full"]
    results = {}
    for variant in variants:
        print(f"\n[{variant}] " + "-" * 50)
        with torch.no_grad():
            if variant == "true-doc":
                ok_gen, conf, rep = true_doc_flow()
                results[variant] = (None, None, None, ok_gen, conf, rep)
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
            ok_true = true_dist_check(cache, variant)
            state_scale(cache)
            ok_gen, conf, rep = gen_check(cache)
            results[variant] = (ok_read, ok_conv, ok_true, ok_gen, conf, rep)

    print("\n" + "=" * 60)
    print(f"BISECTION MATRIX{' (FLA OFF)' if args.fla_off else ''}"
          f"{' (QJL ON)' if args.qjl else ''}"
          f"{' (LEGACY HALF SPLIT)' if args.split_half else ''}")
    print(f"{'variant':<12} {'S-read':<8} {'conv':<8} {'TRUE':<6} "
          f"{'gen':<8} {'conf':>6} {'rep':>5}")
    for v, (a, b, t, c, conf, rep) in results.items():
        s = "-" if a is None else ("OK" if a else "DRIFT")
        cv = "-" if b is None else ("OK" if b else "DRIFT")
        tv = "-" if t is None else ("OK" if t else "HIGH")
        gen = "??" if c is None else ("OK" if c else "GARBAGE")
        conf_s = "nan" if conf is None else f"{conf:.3f}"
        rep_s = "nan" if rep is None else f"{rep:.2f}"
        print(f"{v:<12} {s:<8} {cv:<8} {tv:<6} {gen:<8} {conf_s:>6} {rep_s:>5}")
    print("-" * 60)
    print(f"frame check: {'OK' if frame_ok else 'FRAME SPLIT'}")
    print("Reading the matrix:")
    print("  * S-read/conv DRIFT => a REAL read-path bug (device/frame);")
    print("  * frame SPLIT => the CUDA FHT kernel disagrees with the")
    print("    reference at a kernel-eligible d (conv) — every conv code")
    print("    on the box is then written in an unreadable rotation;")
    print("  * TRUE-dist HIGH with reads OK => the WRITE path is the")
    print("    distortion owner (the codes decode fine, they ENCODE")
    print("    badly) — the W15 recipe (outlier split / --qjl) is the")
    print("    lever; re-ingest with the W15 code to activate it;")
    print("  * true-doc GARBAGE (with reads + frame OK) => the G5 failure")
    print("    is SEMANTIC: the model faithfully continues from the")
    print("    document state — a prompt/flow design matter, not")
    print("    corruption. Compare true-doc's text with full's text;")
    print("  * reads/frame/true-doc all OK but full GARBAGE => the TQ")
    print("    quantization layer itself — rerun with --fla-off and")
    print("    compare (the Triton decode route is then the suspect),")
    print("    then --qjl and --split-half for the W15 write-path A/Bs.")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
