#!/usr/bin/env python3
"""eval_retrieval.py — W16: the retrieval evaluation ladder (GPU box).

THE QUESTION (handover W15-post): "retrieval relevance POOR — wrong
documents retrieved"; run_query.py only ever printed the top-3. This
script measures retrieval QUALITY against the benchmark's gold documents
in THREE metric frames on the SAME query vectors:

  absolute   the W14/W15 metric (cos of the raw §4 vectors)
  sys        cos of (vector - system_vector) on both sides
  sys+mean   the W16 centered frame (sys + corpus-mean subtraction)

Per question: the top-3 ids/scores per frame and the gold hit; per
frame: hit@1 / hit@3 / hit@5, the mean top-3 spread, and the evidence
row cos(q, sys) (the common component — the box's 0.70+ floor means the
query vector is mostly system direction).

THE READING:
  * absolute hit-rates at/below chance with a ~1.0-floor on the scores
    and a tiny spread = the W16 diagnosis confirmed on the real corpus
    (the common-component collapse);
  * the centered frame's hit-rates = the architecture's real query->doc
    signal. If it retrieves: DONE (run_index.py --frame centered).
    If it does not: the signal itself is too weak for the current
    weights — the spec §7 fine-tune (N28: "so S is discriminative") is
    the designed next lever, and the numbers here are its baseline.

Examples:
    python3 scripts/gpu/eval_retrieval.py
    python3 scripts/gpu/eval_retrieval.py --n-questions 50 --top-k 5
    python3 scripts/gpu/eval_retrieval.py --disk-dir /home/ubuntu/RAGGA/disk/ingested_1k
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch

from _bootstrap import DEFAULTS, boot, load_model

boot()

from index import (ChunkVectorLoader, build_retrieval_frame,  # noqa: E402
                   rerank)
from ingest import (check_m1m2_geometry, load_system_state,  # noqa: E402
                    m1m2_mem_size_from_system, reseed_cache)
from query import query_cache_vector  # noqa: E402
from tq_cache import TQCache  # noqa: E402


def _doc_id_to_chunk(corpus_path: str, n_docs: int) -> dict:
    """documents.jsonl line order == the ingestion order == chunk id."""
    mapping = {}
    with open(corpus_path) as f:
        for i, line in enumerate(f):
            if i >= n_docs:
                break
            try:
                doc = json.loads(line)
            except json.JSONDecodeError:
                continue
            did = doc.get("id") or doc.get("doc_id") or doc.get("dsid")
            if did is not None:
                mapping[str(did)] = i
    return mapping


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--artifacts-dir", default=DEFAULTS["artifacts_dir"])
    ap.add_argument("--heads-dir", default=DEFAULTS["heads_dir"])
    ap.add_argument("--model-name", default=DEFAULTS["model_name"])
    ap.add_argument("--disk-dir", default=DEFAULTS["disk_dir"])
    ap.add_argument("--corpus", default=DEFAULTS["corpus"])
    ap.add_argument("--questions-file", default=DEFAULTS["questions"])
    ap.add_argument("--n-questions", type=int, default=20)
    ap.add_argument("--top-k", type=int, default=3)
    ap.add_argument("--bits", type=float, default=3.5)
    ap.add_argument("--m1m2-mem-size", type=int, default=None,
                    help="W17: override the M1/M2 slot count (default: "
                         "resolved from the corpus's system_state.npz)")
    ap.add_argument("--m1m2-gates", default=None,
                    help="W17: trained-gates .npz — the SAME artifact the "
                         "corpus was ingested with")
    ap.add_argument("--no-m1m2", action="store_true",
                    help="W17: load without the wiring")
    args = ap.parse_args()

    print("=" * 60)
    print("RAGGA RETRIEVAL EVAL (scripts/gpu/eval_retrieval.py)")
    print("=" * 60)

    print(f"\n[2] Loading corpus vectors from {args.disk_dir}...")
    loader = ChunkVectorLoader(args.disk_dir, bits=args.bits)
    system = load_system_state(args.disk_dir)
    mem_size = args.m1m2_mem_size
    if mem_size is None:
        mem_size = m1m2_mem_size_from_system(system)
        if mem_size is None and not args.no_m1m2:
            mem_size = 128
    chunks = np.array(loader.chunk_ids())
    print(f"    {len(chunks)} chunks; layout "
          f"{loader.manifest.get('chunk_protocol', 'delta-v1')}; "
          f"m1m2 mem_size {mem_size if mem_size else 'OFF'}")

    print("\n[1] Loading model + tokenizer "
          f"(m1m2_mem_size={mem_size or 128})...")
    model, _ = load_model(args.artifacts_dir, args.heads_dir,
                          args.model_name,
                          use_m1m2=not args.no_m1m2,
                          m1m2_mem_size=mem_size or 128,
                          m1m2_gates_path=args.m1m2_gates)
    check_m1m2_geometry(model, system, mem_size)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    print("\n[3] Building the centered frame (sys + corpus mean)...")
    frame = build_retrieval_frame(loader)
    sys_vec = loader.system_vector()

    n_docs = len(chunks)
    doc_map = _doc_id_to_chunk(args.corpus, n_docs)
    print(f"    gold map: {len(doc_map)} of {n_docs} docs keyed by id")

    print(f"\n[4] Loading {args.n_questions} questions...")
    questions = []
    with open(args.questions_file) as f:
        for i, line in enumerate(f):
            if i >= args.n_questions:
                break
            q = json.loads(line)
            text = q.get("text") or q.get("question") or ""
            gold = [doc_map[str(d)] for d in (q.get("expected_doc_ids") or [])
                    if str(d) in doc_map]
            if text and gold:
                questions.append((text, gold))
    print(f"    {len(questions)} usable (text + in-corpus gold doc)")

    frames = {"absolute": None, "sys": _SysFrame(sys_vec),
              "sys+mean": frame}
    hits = {name: {1: 0, 3: 0, 5: 0} for name in frames}
    spreads = {name: [] for name in frames}
    cos_sys_rows = []

    for qi, (text, gold) in enumerate(questions):
        query_ids = torch.tensor(
            tokenizer.encode(text, add_special_tokens=False)).cuda()
        cache = TQCache(config=model.config, bits=args.bits, online=True)
        reseed_cache(cache, system)
        with torch.no_grad():
            model(input_ids=query_ids, past_key_values=cache, use_cache=True)
        qvec = query_cache_vector(cache, system)

        # the common-component evidence: cos(q, sys)
        cos_sys = float(np.dot(qvec, sys_vec)
                        / (np.linalg.norm(qvec) * np.linalg.norm(sys_vec)))
        cos_sys_rows.append(cos_sys)

        row = f"    q{qi:03d} cos(q,sys) {cos_sys:.3f} |"
        for name, fr in frames.items():
            ids, scores = rerank(loader, qvec, chunks, k=args.top_k,
                                frame=fr)
            top = [int(x) for x in ids]
            hit = any(g in top[:1] for g in gold)
            hit3 = any(g in top[:3] for g in gold)
            hit5 = any(g in top[:5] for g in gold)
            hits[name][1] += hit
            hits[name][3] += hit3
            hits[name][5] += hit5
            spreads[name].append(float(scores.max() - scores.min()))
            row += (f" {name}: [{' '.join(map(str, top))}]"
                    f" {float(scores[0]):.3f}"
                    f"{'*' if hit else ''}")
        print(row)

    n = max(1, len(questions))
    print("\n" + "=" * 60)
    print(f"RETRIEVAL MATRIX (n={len(questions)} questions, "
          f"top-{args.top_k})")
    print(f"{'frame':<10} {'hit@1':>7} {'hit@3':>7} {'hit@5':>7} "
          f"{'top-k spread':>13} {'top score':>10}")
    for name in frames:
        h = hits[name]
        sp = float(np.mean(spreads[name])) if spreads[name] else 0.0
        print(f"{name:<10} {h[1] / n:>7.2f} {h[3] / n:>7.2f} "
              f"{h[5] / n:>7.2f} {sp:>13.4f}")
    print(f"mean cos(q, sys) = {np.mean(cos_sys_rows):.4f} "
          f"(the common component; high = the absolute metric's floor)")
    print("-" * 60)
    print("Reading: absolute at chance + a ~1.0 floor + tiny spread = the")
    print("W16 common-component collapse confirmed; the centered frames'")
    print("hit rates ARE the architecture's real query->doc signal. If the")
    print("centered frame does not retrieve either, the spec §7 fine-tune")
    print("(N28: train the linear-attention params so S is discriminative)")
    print("is the designed next lever — these numbers are its baseline.")
    print("=" * 60)
    return 0


class _SysFrame:
    """A minimal sys-only centering (the second A/B frame): center_query
    = q - sys, center_delta = delta (no corpus mean)."""

    def __init__(self, sys_vector):
        self.sys_vector = np.asarray(sys_vector, dtype=np.float32)
        self.dims = int(self.sys_vector.shape[0])
        self.mean_vector = np.zeros_like(self.sys_vector)
        self.system_ref = ""
        self.n_mean_chunks = 0
        self.chunk_protocol = "sys"

    def center_query(self, qvec):
        q = np.asarray(qvec, dtype=np.float32)
        return q - self.sys_vector

    def center_delta(self, dvec):
        return np.asarray(dvec, dtype=np.float32)

    def check_loader(self, loader):
        if int(self.dims) != int(loader.dims):
            raise ValueError("sys-frame dims drift")


if __name__ == "__main__":
    raise SystemExit(main())
