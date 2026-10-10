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

import numpy as np
import torch

from _bootstrap import DEFAULTS, boot, load_model

boot()

from index import ChunkVectorLoader, load_retrieval_frame, rerank  # noqa: E402
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
    ap.add_argument("--no-m1m2", action="store_true",
                    help="W17: load without the wiring (only valid for a "
                         "corpus ingested --no-m1m2)")
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

    print("\n[2] Loading model + tokenizer "
          f"(m1m2_mem_size={mem_size or 128})...")
    model, _ = load_model(args.artifacts_dir, args.heads_dir,
                          args.model_name,
                          use_m1m2=not args.no_m1m2,
                          m1m2_mem_size=mem_size or 128,
                          m1m2_gates_path=args.m1m2_gates)
    check_m1m2_geometry(model, system, mem_size)
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
        print("    [W16] centered retrieval frame OFF (absolute metric)")

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
    for i, question in enumerate(questions):
        print(f"\n    Query {i + 1}: {question[:80]}...")
        # add_special_tokens=False: matches the ingestion's chunk and
        # system encoding (the query must live in the same token regime)
        query_ids = torch.tensor(
            tokenizer.encode(question, add_special_tokens=False)).cuda()

        cache = cache_factory()
        reseed_cache(cache, system)
        with torch.no_grad():
            model(input_ids=query_ids, past_key_values=cache, use_cache=True)
        qvec = query_cache_vector(cache, system)

        ids, scores = rerank(loader, qvec, np.array(chunks),
                             k=args.top_k, frame=frame)
        print(f"    Top {args.top_k}: {[int(x) for x in ids]} "
              f"scores {[f'{s:.4f}' for s in scores]}")

        result = answer_query(
            model=model,
            query_token_ids=query_ids,
            system=system,
            loader=loader,
            cache_factory=cache_factory,
            retrieved_ids=[int(x) for x in ids],
            max_new_tokens=args.max_new_tokens)
        answer = tokenizer.decode(result.new_token_ids, skip_special_tokens=True)
        print(f"    Answer: {answer[:300]}")

    print("\n" + "=" * 60)
    print("QUERY COMPLETE")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
