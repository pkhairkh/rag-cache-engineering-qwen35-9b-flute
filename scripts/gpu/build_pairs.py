#!/usr/bin/env python3
"""build_pairs.py — the general-similarity pairs builder (W18, outstanding
issue #3 of the handover: "/home/ubuntu/RAGGA/data/pairs.jsonl MISSING").

Converts the GENERAL similarity datasets to the §7 trainer's pairs JSONL:

    {"text1": "...", "text2": "...", "label": 1}

(label 1 = similar — the rows the trainer's loader consumes; label-0 rows
ride along for future hard-negative work and are skipped by
finetune_m1m2.py's load_pairs). Sources (ALL general, NONE the served
corpus — corpus independence is the design):

  * MRPC   (sentence1, sentence2, label)          — paraphrase
  * QQP    (question1, question2, is_duplicate)   — paraphrase
  * PAWS   (sentence1, sentence2, label)          — hard paraphrase
  * SNLI   (premise, hypothesis, entailment)      — 1 iff entail
  * MNLI   (premise, hypothesis, entailment)      — 1 iff entail
  * STS-B  (sentence1, sentence2, score >= 4.0)   — graded similarity
  * SQuAD  (question, the answer-bearing sentence of its context) —
           the retrieval-shaped pair (question ~ passage)

Datasets are read through the HF `datasets` library (download or local
cache); the script NEVER touches the served corpus. Deterministic:
fixed seed, per-source caps, exact dedup, a manifest line per source.

Examples:
    # the handover's target (all sources, ~5k rows each)
    python3 scripts/gpu/build_pairs.py \
        --out /home/ubuntu/RAGGA/data/pairs.jsonl

    # one source, offline from a prepared HF cache
    python3 scripts/gpu/build_pairs.py --sources mrpc \
        --out pairs.jsonl --offline

    # format self-check without any dataset (writes 3 synthetic rows)
    python3 scripts/gpu/build_pairs.py --self-check --out /tmp/p.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys


# ------------------------------------------------------------- sources ----
def _clean(text: str) -> str:
    return " ".join(str(text or "").split())


def _ok(t: str, min_chars: int, max_chars: int) -> bool:
    return min_chars <= len(t) <= max_chars


def rows_mrpc(ds, min_chars, max_chars):
    for r in ds:
        a, b = _clean(r["sentence1"]), _clean(r["sentence2"])
        if _ok(a, min_chars, max_chars) and _ok(b, min_chars, max_chars):
            yield a, b, int(r["label"])


def rows_qqp(ds, min_chars, max_chars):
    for r in ds:
        a, b = _clean(r["question1"]), _clean(r["question2"])
        if _ok(a, min_chars, max_chars) and _ok(b, min_chars, max_chars):
            yield a, b, int(r["is_duplicate"])


def rows_paws(ds, min_chars, max_chars):
    for r in ds:
        a, b = _clean(r["sentence1"]), _clean(r["sentence2"])
        if _ok(a, min_chars, max_chars) and _ok(b, min_chars, max_chars):
            yield a, b, int(r["label"])


def _rows_nli(ds, min_chars, max_chars):
    # HF label convention for snli/mnli: 0 = entail, 1 = neutral,
    # 2 = contradiction, -1 = no gold (skip)
    for r in ds:
        lab = r["label"]
        if lab is None or int(lab) < 0:
            continue
        a, b = _clean(r["premise"]), _clean(r["hypothesis"])
        if _ok(a, min_chars, max_chars) and _ok(b, min_chars, max_chars):
            yield a, b, 1 if int(lab) == 0 else 0


def rows_snli(ds, min_chars, max_chars):
    yield from _rows_nli(ds, min_chars, max_chars)


def rows_mnli(ds, min_chars, max_chars):
    yield from _rows_nli(ds, min_chars, max_chars)


def rows_stsb(ds, min_chars, max_chars):
    for r in ds:
        a, b = _clean(r["sentence1"]), _clean(r["sentence2"])
        if _ok(a, min_chars, max_chars) and _ok(b, min_chars, max_chars):
            yield a, b, 1 if float(r["label"]) >= 4.0 else 0


def _answer_sentence(context: str, answer: str) -> str:
    """The first sentence of the context containing the answer span
    (the retrieval-shaped passage member); the leading context if no
    sentence matches (answers paraphrased across splits)."""
    import re
    sentences = [s for s in re.split(r"(?<=[.!?])\s+", context) if s.strip()]
    for s in sentences:
        if answer and answer.strip() in s:
            return s
    return sentences[0] if sentences else context


def rows_squad(ds, min_chars, max_chars):
    for r in ds:
        q = _clean(r["question"])
        ans = ""
        answers = r.get("answers") or {}
        if isinstance(answers, dict) and answers.get("text"):
            ans = str(answers["text"][0])
        p = _clean(_answer_sentence(str(r["context"]), ans))
        if _ok(q, min_chars, max_chars) and _ok(p, min_chars, max_chars):
            yield q, p, 1


# source registry: (HF repo id, config, split, row generator)
SOURCES = {
    "mrpc": ("glue", "mrpc", "train", rows_mrpc),
    "qqp": ("glue", "qqp", "train", rows_qqp),
    "paws": ("paws", "labeled_final", "train", rows_paws),
    "snli": ("snli", None, "train", rows_snli),
    "mnli": ("glue", "mnli", "train", rows_mnli),
    "stsb": ("glue", "stsb", "train", rows_stsb),
    "squad": ("squad", None, "train", rows_squad),
}


# ------------------------------------------------------------ the build ---
def build(sources, out_path, per_source, min_chars, max_chars,
          max_pairs, seed, offline, from_cache=None):
    try:
        from datasets import load_dataset
    except ImportError:
        raise SystemExit(
            "the `datasets` library is not installed — pip install "
            "datasets (the builder reads the GENERAL similarity "
            "datasets through it; nothing here touches the served "
            "corpus)")

    rng = random.Random(seed)
    rows = []
    manifest = {}
    for name in sources:
        repo, config, split, rowgen = SOURCES[name]
        print(f"  [{name}] loading {repo}:{config or ''} ({split})...")
        try:
            ds = load_dataset(
                repo, config, split=split,
                cache_dir=from_cache,
                download_mode="reuse_dataset_if_exists")
        except Exception as e:  # noqa: BLE001 — loud, per-source, continue
            if offline:
                print(f"  [{name}] OFFLINE miss ({type(e).__name__}: "
                      f"{e}) — skipped; prepare the HF cache first")
                manifest[name] = 0
                continue
            raise
        n = 0
        for a, b, lab in rowgen(ds, min_chars, max_chars):
            if n >= per_source:
                break
            rows.append({"text1": a, "text2": b, "label": lab})
            n += 1
        manifest[name] = n
        print(f"  [{name}] {n} pairs (cap {per_source})")

    # exact dedup on (text1, text2) either order; deterministic shuffle
    seen = set()
    uniq = []
    for r in rows:
        key = (r["text1"], r["text2"])
        rkey = (r["text2"], r["text1"])
        if key in seen or rkey in seen:
            continue
        seen.add(key)
        uniq.append(r)
    rng.shuffle(uniq)
    if max_pairs:
        uniq = uniq[:max_pairs]

    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".",
                exist_ok=True)
    n_pos = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for r in uniq:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            n_pos += int(r["label"] == 1)
    meta = {
        "total": len(uniq), "positives": n_pos,
        "per_source": manifest, "seed": seed,
        "per_source_cap": per_source,
        "min_chars": min_chars, "max_chars": max_chars,
    }
    with open(out_path + ".manifest.json", "w") as f:
        json.dump(meta, f, indent=1)
    print(f"\nwrote {len(uniq)} pairs ({n_pos} positive) -> {out_path}")
    print(f"manifest: {out_path}.manifest.json")
    if n_pos < 2:
        raise SystemExit(
            "fewer than 2 positive pairs — the trainer needs >= 2 "
            "(in-batch negatives); raise --per-source or fix the "
            "filters")
    return meta


def self_check(out_path: str) -> int:
    """Write 3 synthetic rows and validate the EXACT contract the
    trainer's loader expects (keys, label type, jsonl framing)."""
    rows = [
        {"text1": "The cat sat on the mat.", "text2": "A cat rested on a mat.", "label": 1},
        {"text1": "Prices rose sharply.", "text2": "Costs fell slightly.", "label": 0},
        {"text1": "A man is playing guitar.", "text2": "Someone plays a guitar.", "label": 1},
    ]
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".",
                exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    # validate: every line parses, has the 3 keys, label is 0/1
    n = 0
    with open(out_path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            assert set(r) == {"text1", "text2", "label"}, r
            assert isinstance(r["text1"], str) and r["text1"]
            assert isinstance(r["text2"], str) and r["text2"]
            assert r["label"] in (0, 1)
            n += 1
    print(f"self-check OK: {n} rows at {out_path} "
          f"(format {{text1, text2, label}})")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="/home/ubuntu/RAGGA/data/pairs.jsonl",
                    help="output JSONL path (the trainer's --pairs-file)")
    ap.add_argument("--sources", default="mrpc,qqp,paws,snli,mnli,stsb,squad",
                    help="comma list from: " + ", ".join(sorted(SOURCES)))
    ap.add_argument("--per-source", type=int, default=5000)
    ap.add_argument("--max-pairs", type=int, default=0,
                    help="0 = keep everything after the shuffle")
    ap.add_argument("--min-chars", type=int, default=12)
    ap.add_argument("--max-chars", type=int, default=1800)
    ap.add_argument("--seed", type=int, default=20261018)
    ap.add_argument("--offline", action="store_true",
                    help="skip (loudly, per source) datasets missing from "
                         "the local HF cache instead of downloading")
    ap.add_argument("--from-cache", default=None,
                    help="an explicit HF datasets cache dir")
    ap.add_argument("--self-check", action="store_true",
                    help="format self-check only (no datasets needed)")
    args = ap.parse_args()

    if args.self_check:
        return self_check(args.out)

    sources = [s.strip().lower() for s in args.sources.split(",") if s.strip()]
    unknown = [s for s in sources if s not in SOURCES]
    if unknown:
        raise SystemExit(
            f"unknown source(s) {unknown}; known: {sorted(SOURCES)}")

    print("=" * 60)
    print("RAGGA GENERAL-SIMILARITY PAIRS BUILDER (build_pairs.py)")
    print("=" * 60)
    print(f"sources: {sources}")
    build(sources, args.out, args.per_source, args.min_chars,
          args.max_chars, args.max_pairs, args.seed, args.offline,
          args.from_cache)
    print("\nnext: python3 scripts/gpu/finetune_m1m2.py "
          f"--pairs-file {args.out} --gates-out <path>.npz "
          "--m1m2-mem-size 1024")
    return 0


if __name__ == "__main__":
    sys.exit(main())
