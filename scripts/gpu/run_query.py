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

from index import ChunkVectorLoader, rerank  # noqa: E402
from ingest import load_system_state, reseed_cache  # noqa: E402
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
    args = ap.parse_args()

    print("=" * 60)
    print("RAGGA QUERY (scripts/gpu/run_query.py)")
    print("=" * 60)

    print("\n[1] Loading model + tokenizer...")
    model, _ = load_model(args.artifacts_dir, args.heads_dir, args.model_name)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    print(f"\n[2] Loading ingested data from {args.disk_dir}...")
    loader = ChunkVectorLoader(args.disk_dir, bits=args.bits)
    system = load_system_state(args.disk_dir)
    chunks = list(loader.chunk_ids())
    print(f"    system {system.reference()}; {len(chunks)} chunks")

    if args.question:
        questions = [args.question]
    else:
        print(f"[3] Loading questions from {args.questions_file}...")
        questions = []
        with open(args.questions_file) as f:
            for i, line in enumerate(f):
                if i >= args.n_questions:
                    break
                q = json.loads(line)
                questions.append(q.get("text", q.get("question", "")))

    def cache_factory():
        return TQCache(config=model.config, bits=args.bits, online=True)

    print(f"\n[4] Running {len(questions)} queries "
          f"(direct rerank over {len(chunks)} chunks)...")
    for i, question in enumerate(questions):
        print(f"\n    Query {i + 1}: {question[:80]}...")
        query_ids = tokenizer.encode(question, return_tensors="pt").cuda()

        cache = cache_factory()
        reseed_cache(cache, system)
        with torch.no_grad():
            model(input_ids=query_ids, past_key_values=cache, use_cache=True)
        qvec = query_cache_vector(cache, system)

        ids, scores = rerank(loader, qvec, np.array(chunks), k=args.top_k)
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
