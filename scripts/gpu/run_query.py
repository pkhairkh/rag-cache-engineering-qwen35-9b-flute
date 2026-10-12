#!/usr/bin/env python3
"""run_query.py — the §6 query flow on the GPU box.

Consolidates the W10-GPU session's root-level run_query.py: query prefill
through a reseeded TQCache -> the query vector (§4) -> retrieval (direct
rerank at small corpus scale, or the built index) -> install -> decode.

Examples:
    python3 scripts/gpu/run_query.py --question "What is 2+2?"
    python3 scripts/gpu/run_query.py --questions-file ... --n-questions 5 --max-new-tokens 128
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch

from _bootstrap import DEFAULTS, boot, load_model

boot()

from index import (ChunkVectorLoader, corpus_memory_state,  # noqa: E402
                    load_retrieval_frame, rerank)
from ingest import (check_m1m2_geometry, load_system_state,  # noqa: E402
                    m1m2_mem_size_from_system, reseed_cache)
from query import answer_query, query_cache_vector  # noqa: E402
from tq_cache import TQCache  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--artifacts-dir", default=DEFAULTS["artifacts_dir"])
    ap.add_argument("--heads-dir", default=DEFAULTS["heads_dir"])
    ap.add_argument("--model-name", default=DEFAULTS["model_name"])
    ap.add_argument("--disk-dir", default=DEFAULTS["disk_dir"])
    ap.add_argument("--questions-file", default=DEFAULTS["questions"])
    ap.add_argument("--n-questions", type=int, default=5)
    ap.add_argument("--question", default=None,
                    help="a single ad-hoc question (overrides --questions-file)")
    ap.add_argument("--top-k", type=int, default=3)
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--bits", type=float, default=3.5)
    ap.add_argument("--qjl", action="store_true",
                    help="W15: the paper Alg.-2 A/B — TQCache(qjl=True)")
    ap.add_argument("--retrieval-frame", choices=("auto", "centered",
                                                    "absolute"),
                    default="auto",
                    help="W16: 'auto' uses the centered frame when "
                         "retrieval_frame.npz exists (run_index.py writes "
                         "it); 'absolute' forces the legacy metric")
    ap.add_argument("--m1m2-mem-size", type=int, default=None,
                    help="W17: override the M1/M2 slot count (default: "
                         "resolved from the corpus's system_state.npz — "
                         "ingest and query MUST agree; a mismatch fails "
                         "loudly)")
    ap.add_argument("--m1m2-gates", default=None,
                    help="W17: trained-gates .npz — the SAME artifact the "
                         "corpus was ingested with (geometry-validated)")
    ap.add_argument("--m1m2-read-topk", type=int, default=0,
                    help="W22: sparse memory retrieval — softmax over the "
                         "TOP-k M1 slots only (the PKM/Memorizing-"
                         "Transformers read; 0 = the spec §2.2 dense "
                         "softmax). Retrieval-quality knob, A/B it with "
                         "eval_retrieval.py; pass the SAME k to ingest "
                         "and query for read consistency")
    ap.add_argument("--no-answer-m1m2", action="store_true",
                    help="W23: skip the M1/M2 read/write block during the "
                         "ANSWER phase only (the true-doc control's "
                         "regime; the retrieval-side query prefill keeps "
                         "it ON, so the query vector is unaffected) — "
                         "the memory's generation effect becomes a "
                         "measured A/B, not an assumption")
    ap.add_argument("--install-gdn", choices=("sum", "best"),
                    default="sum",
                    help="W26 (FINDINGS.md F1): the n>=2 GDN S-state "
                         "install. 'sum' = the §6 contract "
                         "(Σ abs_i − (n−1)·sys — the measured-broken "
                         "superposition; the E1-E3 ladder runs on it as "
                         "the A/B baseline); 'best' = S+conv VERBATIM "
                         "from the highest-scoring chunk (algebra-exact: "
                         "the delta-rule recurrence is non-additive), "
                         "M1/M2 still summed (their write is additive). "
                         "n=1 is always verbatim either way")
    ap.add_argument("--no-m1m2", action="store_true",
                    help="W17: load without the wiring (only valid for a "
                         "corpus ingested --no-m1m2")
    args = ap.parse_args()

    print("=" * 60)
    print("RAGGA QUERY (scripts/gpu/run_query.py)")
    print("=" * 60)

    print("\n[1] Loading ingested data from " + args.disk_dir + "...")
    loader = ChunkVectorLoader(args.disk_dir, bits=args.bits)
    system = load_system_state(args.disk_dir)
    # W17: resolve the corpus's M1/M2 geometry from the reset point —
    # the model-side module must match it (loud drift guard)
    mem_size = args.m1m2_mem_size
    if mem_size is None:
        mem_size = m1m2_mem_size_from_system(system)
        if mem_size is None and not args.no_m1m2:
            mem_size = 128  # corpus without M1/M2 + wired model: P3 no-op
    chunks = list(loader.chunk_ids())
    print(f"    system {system.reference()}; {len(chunks)} chunks; layout "
          f"{loader.manifest.get('chunk_protocol', 'delta-v1')}; "
          f"m1m2 mem_size {mem_size if mem_size else 'OFF'}")

    # W27: the corpus's M1/M2 ingest regime — the retrieval-side query
    # prefill must run in the regime the chunk vectors were CAPTURED in
    # (a loaded gates artifact on a zero-write corpus produces a query
    # vector whose memory pieces and S capture have NO counterpart in
    # any chunk vector — regime mismatch, measured as score deflation
    # + rank distortion; the box's old snapshots are exactly this case)
    mem_state = None
    if not args.no_m1m2:
        mem_state = corpus_memory_state(loader)
        s_dims = sum(int(c.d) for c in system.s_codes.values())
        all_dims = s_dims + sum(
            int(c.d) for c in (system.m1_codes, system.m2_codes)
            if c is not None)
        if mem_state is None:
            print("    [W27] corpus M1/M2 memories: absent (no memory "
                  "codes) — no memory regime to match")
        elif mem_state:
            print(f"    [W27] corpus M1/M2 memories EMPTY (zero-write "
                  f"ingest: every sampled chunk's memory codes "
                  f"dequantize to the reset point's) — the memory "
                  f"subspace is inert; the §4 metric rides the S "
                  f"subspace alone ({s_dims}/{all_dims} dims)")
            if args.m1m2_gates:
                print("    [W27] --m1m2-gates on a zero-write corpus: the "
                      "chunks carry no memory content, so the gates' "
                      "query-time writes cannot match anything — the "
                      "retrieval prefill runs M1/M2-OFF (regime match); "
                      "the ANSWER phase keeps the gates A/B; re-ingest "
                      "with the gates to activate the memory subspace")
        elif not args.m1m2_gates:
            print("    [W27] corpus memories carry content but no "
                  "--m1m2-gates loaded — the query's memory writes stay "
                  "empty; pass the ingest-time artifact for a matched "
                  "retrieval regime")

    print("\n[2] Loading model + tokenizer "
          f"(m1m2_mem_size={mem_size or 128})...")
    model, _ = load_model(args.artifacts_dir, args.heads_dir,
                          args.model_name,
                          use_m1m2=not args.no_m1m2,
                          m1m2_mem_size=mem_size or 128,
                          m1m2_gates_path=args.m1m2_gates)
    check_m1m2_geometry(model, system, mem_size)
    # W22 — the sparse read knob (validated loudly inside M1M2.read;
    # 0 = the dense spec §2.2 contract, the default)
    inner = getattr(model, "model", model)
    if not args.no_m1m2 and args.m1m2_read_topk:
        inner.m1m2.read_topk = args.m1m2_read_topk
        print(f"    [W22] read_topk {args.m1m2_read_topk} — sparse memory "
              f"retrieval (the dense softmax is the default A/B)")
    dev = next(model.parameters()).device  # W18: ids follow the model
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    # W16: the centered retrieval frame (auto-detected from the side
    # file run_index.py writes; --retrieval-frame absolute = the A/B)
    import os
    frame = None
    if args.retrieval_frame in ("auto", "centered") and os.path.exists(
            f"{args.disk_dir.rstrip('/')}/retrieval_frame.npz"):
        frame = load_retrieval_frame(args.disk_dir)
        frame.check_loader(loader)
        print(f"    [W16] centered retrieval frame ON "
              f"(mean over {frame.n_mean_chunks} deltas)")
    elif args.retrieval_frame == "centered":
        raise SystemExit(
            "--retrieval-frame centered but no retrieval_frame.npz — "
            "run scripts/gpu/run_index.py first")
    else:
        print("    [W16] centered retrieval frame OFF (absolute metric) — "
              "run scripts/gpu/run_index.py for the centered content "
              "metric (the absolute frame's shared system component "
              "flattens the score spread; W16 measured the floor on "
              "this box)")

    if args.question:
        questions = [args.question]
    else:
        print(f"\n[3] Loading questions from {args.questions_file}...")
        questions = []
        with open(args.questions_file) as f:
            for i, line in enumerate(f):
                if i >= args.n_questions:
                    break
                q = json.loads(line)
                questions.append(q.get("text", q.get("question", "")))

    def cache_factory():
        return TQCache(config=model.config, bits=args.bits, online=True,
                       qjl=args.qjl)

    print(f"\n[4] Running {len(questions)} queries "
          f"(direct rerank over {len(chunks)} chunks)...")
    # W26: the regime line — the box log must always record which GDN
    # install the answers ran on (the E-ladder's A/B reads this)
    if args.install_gdn == "best":
        print("    [W26] GDN S install (n>=2): best — S+conv VERBATIM "
              "from the top-scoring chunk, M1/M2 summed (the FINDINGS "
              "F1 remedy)")
    else:
        print("    [W26] GDN S install (n>=2): sum — the §6 contract "
              "(SEE FINDINGS.md F1: non-additive at n>=2; the E1-E3 "
              "ladder baseline)")
    # W27: the shared module handle (the retrieval-prefill regime match
    # below and the W23 answer-phase A/B both flip `enabled` on it)
    m1m2_mod = getattr(inner, "m1m2", None)
    # the retrieval prefill runs M1/M2-OFF when the corpus's memories
    # are EMPTY and gates are loaded (the regime match — see the [W27]
    # diagnosis at [1]); gateless runs are already regime-consistent
    # (zero-init write gates leave the codes at the reset point's)
    retrieval_match_off = bool(mem_state) and args.m1m2_gates is not None
    for i, question in enumerate(questions):
        print(f"\n    Query {i + 1}: {question[:80]}...")
        # add_special_tokens=False: matches the ingestion's chunk and
        # system encoding (the query must live in the same token regime)
        # W23: 2-D (1, T) + dtype long + device — the model's forward
        # expects batched ids (a 1-D tensor crashes the reshape; the
        # box-side sed hack around this is now the repo's proper fix)
        query_ids = torch.tensor(
            [tokenizer.encode(question, add_special_tokens=False)],
            dtype=torch.long, device=dev)

        cache = cache_factory()
        reseed_cache(cache, system)
        if retrieval_match_off and m1m2_mod is not None:
            m1m2_mod.enabled = False
            print("    [W27] retrieval prefill: M1/M2 OFF — regime match "
                  "to the corpus's zero-write ingest")
        try:
            with torch.no_grad():
                model(input_ids=query_ids, past_key_values=cache,
                      use_cache=True)
            qvec = query_cache_vector(cache, system)
        finally:
            if retrieval_match_off and m1m2_mod is not None:
                m1m2_mod.enabled = True
        ids, scores = rerank(loader, qvec, np.array(chunks),
                             k=args.top_k, frame=frame)
        print(f"    Top {args.top_k}: {[int(x) for x in ids]} "
              f"scores {[f'{s:.4f}' for s in scores]}")

        # W24: the answer context is built by the §6 INSTALL ONLY (the
        # retrieved chunks' codes, summed on the system reset point).
        # The W23 text-reprefill branch is REMOVED — it bypassed the
        # installed codes (the venture's core mechanism).

        # W23: the answer-phase M1/M2 A/B — disabled AFTER the retrieval
        # prefill above (the query vector keeps the M1/M2 codes),
        # re-enabled right after the answer
        if args.no_answer_m1m2 and m1m2_mod is not None:
            m1m2_mod.enabled = False
            print("    [W23] M1/M2 read/write OFF for the answer "
                  "(the true-doc control's regime)")
        try:
            result = answer_query(
                model=model,
                query_token_ids=query_ids,
                system=system,
                loader=loader,
                cache_factory=cache_factory,
                retrieved_ids=[int(x) for x in ids],
                max_new_tokens=args.max_new_tokens,
                gdn_install=args.install_gdn)
        finally:
            if m1m2_mod is not None:
                m1m2_mod.enabled = True
        answer = tokenizer.decode(result.new_token_ids,
                                  skip_special_tokens=True)
        toks = result.new_token_ids
        conf = (sum(result.token_confs) / len(result.token_confs)
                if result.token_confs else 0.0)
        rep = (sum(1 for a, b in zip(toks, toks[1:]) if a == b)
               / max(1, len(toks) - 1))
        # W27: the script-mix ratio — the box's E1 'passed' the conf bar
        # at 0.382 while the answer was multilingual soup; a conf number
        # alone cannot see that. English answers about a retrieved doc
        # measure ~0.00-0.05 here; the soup measured ~0.2-0.4.
        non_ascii = (sum(1 for ch in answer if not ch.isascii())
                     / max(1, len(answer)))
        verdict = ("CLEAN-RANGE" if conf >= 0.60 else
                   ("WEAK — soup can pass the 0.20 bar; judge the TEXT "
                    "(the control measured 0.895)" if conf >= 0.20 else
                    "GARBAGE-RANGE"))
        print(f"    Answer: {answer[:300]}")
        print(f"    [gen gate] mean conf {conf:.3f} | repetition {rep:.2f} "
              f"| non-ascii {non_ascii:.2f} | {verdict} "
              f"(bisect_install's bar: conf >= 0.20, rep <= 0.6; the "
              f"true-doc control measured 0.895)")

    print("\n" + "=" * 60)
    print("QUERY COMPLETE")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
