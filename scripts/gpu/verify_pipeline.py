#!/usr/bin/env python3
"""verify_pipeline.py — staged generation gates for the GPU box.

Consolidates the W10-GPU session's root-level test_generation /
test_tq_generation / test_system_only / test_install_layer0 /
test_install_gen / test_answer_query (the handover's ✅/❌ isolation table,
now as one runnable ladder). Every stage prints the generated text plus
coherence metrics; the exit code is the number of failed stages.

Stages:
  G1  pure generation, no cache                (model sanity)
  G2  TQCache generation, fresh cache          (cache protocol sanity)
  G3  reseed(system) + query                   (the reset point)
  G4  G3 + layer-0 S-only install              (the handover's isolation)
  G5  G3 + full install_snapshot               (the §6 install)
  G6  answer_query end-to-end (retrieval included)

Metrics per stage: the generated text, the mean top-softmax confidence of
the greedy steps, and the immediate-repetition rate (fraction of steps
whose token equals the previous one — high repetition with low confidence
is the classic corrupted-state signature). A stage PASSES when it emits
>= 1 token with confidence >= 0.20 and repetition <= 0.6; the text itself
is printed for the human call.

FLA A/B: --fla-off sets FLUTE_NO_FLA=1 (the pure-torch decode fallbacks —
bit-identical numerics per the modeling.py contract) to rule the Triton
kernels in/out of any corruption.

Examples:
    python3 scripts/gpu/verify_pipeline.py
    python3 scripts/gpu/verify_pipeline.py --fla-off
    python3 scripts/gpu/verify_pipeline.py --chunk 7 --max-new-tokens 12
"""
from __future__ import annotations

import argparse
import os

import torch

from _bootstrap import DEFAULTS, boot, load_model

boot()


def _greedy(model, cache, first_ids, n, tokenizer):
    """Greedy decode from `first_ids` (the prefill ids); returns
    (token_ids, confidences)."""
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


def _report(name, toks, confs, tokenizer):
    text = tokenizer.decode(toks, skip_special_tokens=True)
    conf = sum(confs) / len(confs) if confs else 0.0
    rep = (sum(1 for a, b in zip(toks, toks[1:]) if a == b)
           / max(1, len(toks) - 1))
    ok = len(toks) >= 1 and conf >= 0.20 and rep <= 0.6
    flag = "PASS" if ok else "FAIL"
    print(f"  [{flag}] {name}: conf {conf:.3f} rep {rep:.2f} "
          f"n {len(toks)}\n         text: {text!r}")
    return ok


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--artifacts-dir", default=DEFAULTS["artifacts_dir"])
    ap.add_argument("--heads-dir", default=DEFAULTS["heads_dir"])
    ap.add_argument("--model-name", default=DEFAULTS["model_name"])
    ap.add_argument("--disk-dir", default=DEFAULTS["disk_dir"])
    ap.add_argument("--chunk", type=int, default=0,
                    help="the snapshot id to install (G4/G5/G6)")
    ap.add_argument("--question", default="What is 2+2?")
    ap.add_argument("--prompt", default="The capital of France is")
    ap.add_argument("--max-new-tokens", type=int, default=12)
    ap.add_argument("--bits", type=float, default=3.5)
    ap.add_argument("--fla-off", action="store_true",
                    help="FLUTE_NO_FLA=1 — pure-torch decode kernels")
    ap.add_argument("--qjl", action="store_true",
                    help="W15: the paper Alg.-2 A/B — TQCache(qjl=True) "
                         "(the residual-sketch compensation, end-to-end)")
    args = ap.parse_args()
    if args.fla_off:
        os.environ["FLUTE_NO_FLA"] = "1"

    from transformers import AutoTokenizer
    from ingest import (check_m1m2_geometry, load_system_state,  # noqa: E402
                        m1m2_mem_size_from_system, reseed_cache)
    from install import install_snapshot, sum_turboquant_codes
    from snapshot import load_chunk
    from tq_cache import TQCache

    failures = 0

    print("=" * 60)
    print("RAGGA PIPELINE VERIFICATION (scripts/gpu/verify_pipeline.py)")
    print(f"{'FLUTE_NO_FLA=1 (torch decode) ' if args.fla_off else ''}")
    print("=" * 60)

    print("\n[load] model + tokenizer + system state...")
    system = load_system_state(args.disk_dir)
    mem_size = m1m2_mem_size_from_system(system)
    model, _ = load_model(args.artifacts_dir, args.heads_dir,
                          args.model_name,
                          m1m2_mem_size=mem_size or 128)
    check_m1m2_geometry(model, system, mem_size)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    snap = load_chunk(os.path.join(args.disk_dir, "snapshots",
                                   f"chunk_{args.chunk:05d}.npz"))

    def fresh_cache():
        return TQCache(config=model.config, bits=args.bits, online=True,
                       qjl=args.qjl)

    # G1 — pure generation (no cache)
    print(f"\nG1  pure generation: {args.prompt!r}")
    with torch.no_grad():
        ids = tokenizer.encode(args.prompt, return_tensors="pt").cuda()
        toks, confs = _greedy(model, None, ids, args.max_new_tokens, tokenizer)
    failures += 0 if _report("G1 pure model", toks, confs, tokenizer) else 1

    # G2 — TQCache, fresh (no reseed, no install)
    print(f"\nG2  fresh TQCache: {args.prompt!r}")
    with torch.no_grad():
        cache = fresh_cache()
        ids = tokenizer.encode(args.prompt, return_tensors="pt").cuda()
        toks, confs = _greedy(model, cache, ids, args.max_new_tokens, tokenizer)
    failures += 0 if _report("G2 TQCache", toks, confs, tokenizer) else 1

    # G3 — reseed only
    print(f"\nG3  reseed(system) only: {args.question!r}")
    with torch.no_grad():
        cache = fresh_cache()
        reseed_cache(cache, system)
        ids = tokenizer.encode(args.question, return_tensors="pt").cuda()
        toks, confs = _greedy(model, cache, ids, args.max_new_tokens, tokenizer)
    failures += 0 if _report("G3 reseed", toks, confs, tokenizer) else 1

    # G4 — reseed + layer-0 S-only install (the handover's isolation case)
    print(f"\nG4  reseed + layer-0 S-only install (chunk {args.chunk})")
    with torch.no_grad():
        cache = fresh_cache()
        reseed_cache(cache, system)
        delta0 = snap.s_codes.get(0)
        if delta0 is not None:
            summed = sum_turboquant_codes(system.s_codes[0], [delta0],
                                          kind="S", bits=args.bits)
            cache.set_s_codes(0, summed)
        ids = tokenizer.encode(args.question, return_tensors="pt").cuda()
        toks, confs = _greedy(model, cache, ids, args.max_new_tokens, tokenizer)
    failures += 0 if _report("G4 S-only install", toks, confs, tokenizer) else 1

    # G5 — reseed + full install_snapshot
    print(f"\nG5  reseed + full install_snapshot (chunk {args.chunk})")
    with torch.no_grad():
        cache = fresh_cache()
        reseed_cache(cache, system)
        report = install_snapshot(cache, system, [snap])
        ids = tokenizer.encode(args.question, return_tensors="pt").cuda()
        toks, confs = _greedy(model, cache, ids, args.max_new_tokens, tokenizer)
    print(f"         install report L0: {report.get(0)}")
    failures += 0 if _report("G5 full install", toks, confs, tokenizer) else 1

    # G6 — answer_query end-to-end
    print(f"\nG6  answer_query end-to-end (chunk {args.chunk})")
    from index import ChunkVectorLoader
    from query import answer_query
    loader = ChunkVectorLoader(args.disk_dir, bits=args.bits)
    with torch.no_grad():
        ids = tokenizer.encode(args.question, return_tensors="pt").cuda()
        result = answer_query(
            model=model, query_token_ids=ids, system=system, loader=loader,
            cache_factory=fresh_cache, retrieved_ids=[args.chunk],
            max_new_tokens=args.max_new_tokens)
    toks = result.new_token_ids
    # W12: the conf gate now applies to G6 too (answer_query tracks the
    # per-step confidences in QueryResult.token_confs). The old criteria
    # (repetition only) FALSE-PASSED low-confidence garbage — the W12
    # matrix's "G6 PASS vs G5 FAIL" contradiction was partly this.
    conf = (sum(result.token_confs) / len(result.token_confs)
            if result.token_confs else 0.0)
    rep = (sum(1 for a, b in zip(toks, toks[1:]) if a == b)
           / max(1, len(toks) - 1))
    text = tokenizer.decode(toks, skip_special_tokens=True)
    ok = len(toks) >= 1 and conf >= 0.20 and rep <= 0.6
    print(f"  [{'PASS' if ok else 'FAIL'}] G6 e2e: conf {conf:.3f} "
          f"rep {rep:.2f}\n         text: {text!r}")
    failures += 0 if ok else 1

    print("\n" + "=" * 60)
    print(f"VERIFICATION: {6 - failures}/6 stages passed"
          f"{'' if failures == 0 else f' ({failures} FAILED — see bisect_install.py)'}")
    print("=" * 60)
    return failures


if __name__ == "__main__":
    raise SystemExit(main())
