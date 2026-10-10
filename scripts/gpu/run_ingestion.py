#!/usr/bin/env python3
"""run_ingestion.py — the §5 ingestion driver on the GPU box.

Consolidates the W10-GPU session's root-level run_ingestion.py (one arg per
knob, no hardcoded-only paths). The model runs as the INNER TextModel
(model.model — the CausalLM wrapper adds only the LM head; cache states are
identical), the system prompt is the delta protocol's zero point, and each
document is one chunk.

Examples:
    python3 scripts/gpu/run_ingestion.py
    python3 scripts/gpu/run_ingestion.py --n-docs 1000 --out-dir /home/ubuntu/RAGGA/disk/ingested_1k
    python3 scripts/gpu/run_ingestion.py --system-prompt "You are a RAG assistant." --bits 3.5
"""
from __future__ import annotations

import argparse
import json

import torch

from _bootstrap import DEFAULTS, boot, load_model

boot()

from ingest import IngestDriver  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--artifacts-dir", default=DEFAULTS["artifacts_dir"])
    ap.add_argument("--heads-dir", default=DEFAULTS["heads_dir"])
    ap.add_argument("--model-name", default=DEFAULTS["model_name"])
    ap.add_argument("--corpus", default=DEFAULTS["corpus"],
                    help="documents.jsonl (field 'text' or 'content')")
    ap.add_argument("--out-dir", default=DEFAULTS["disk_dir"])
    ap.add_argument("--n-docs", type=int, default=100,
                    help="how many documents to ingest (from the top)")
    ap.add_argument("--max-tokens", type=int, default=0,
                    help="truncate each document to N tokens (0 = full text)")
    ap.add_argument("--system-prompt", default="You are a helpful AI assistant.")
    ap.add_argument("--bits", type=float, default=3.5)
    ap.add_argument("--m1m2-mem-size", type=int, default=1024,
                    help="W17: the M1/M2 memories' slot count (spec §2.2 "
                         "default 128; the handover's retrieval "
                         "experiments: 1024/4096/8192). The snapshot "
                         "gains 2 x (32*mem*128)-dim m1/m2 code units; "
                         "query/index must use the SAME value (the "
                         "loaders fail loudly on drift)")
    ap.add_argument("--m1m2-gates", default=None,
                    help="W17: trained-gates .npz (finetune_m1m2.py's "
                         "artifact) — without it the gates stay at the "
                         "P3 zero init and the m1/m2 units are zero (the "
                         "retrieval signal appears only after training)")
    ap.add_argument("--no-m1m2", action="store_true",
                    help="W17: load without the M1/M2 wiring (the "
                         "W16-exact A/B; snapshots carry no m1/m2 units)")
    ap.add_argument("--chunk-protocol", choices=("absolute", "delta-v1"),
                    default="absolute",
                    help="W16: 'absolute' (default) stores the cache's own "
                         "end-of-chunk codes — the single-chunk install is "
                         "VERBATIM (the W16 noise decomposition measured the "
                         "delta path's 2 extra rounds at ~70 percent of the "
                         "write-path distortion); 'delta-v1' keeps the D4 "
                         "legacy layout (the A/B)")
    args = ap.parse_args()

    print("=" * 60)
    print("RAGGA INGESTION (scripts/gpu/run_ingestion.py)")
    print("=" * 60)

    print("\n[1] Loading model...")
    model, _ = load_model(args.artifacts_dir, args.heads_dir, args.model_name,
                          use_m1m2=not args.no_m1m2,
                          m1m2_mem_size=args.m1m2_mem_size,
                          m1m2_gates_path=args.m1m2_gates)

    from transformers import AutoTokenizer
    print("[2] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    print(f"\n[3] Loading documents from {args.corpus} (n={args.n_docs})...")
    docs = []
    with open(args.corpus) as f:
        for i, line in enumerate(f):
            if i >= args.n_docs:
                break
            doc = json.loads(line)
            docs.append(doc.get("text", doc.get("content", "")))
    if not docs:
        raise SystemExit(f"no documents loaded from {args.corpus}")
    print(f"    {len(docs)} documents")

    print("\n[4] Preparing the system prompt (the delta-protocol zero point)...")
    system_ids = tokenizer.encode(args.system_prompt, add_special_tokens=False)
    system_ids = torch.tensor([system_ids], dtype=torch.long)
    print(f"    {system_ids.shape[-1]} tokens")

    print("\n[5] Tokenizing chunks (one document = one chunk)...")
    chunks = []
    for i, doc in enumerate(docs):
        if i and i % 10000 == 0:
            print(f"    tokenizing doc {i}/{len(docs)}...")
        ids = tokenizer.encode(doc, add_special_tokens=False)
        if args.max_tokens:
            ids = ids[: args.max_tokens]
        chunks.append(torch.tensor([ids], dtype=torch.long))

    print(f"\n[6] Ingesting to {args.out_dir} (restartable via the manifest)...")
    drv = IngestDriver(
        model=model.model,  # inner TextModel: states, not logits
        system_token_ids=system_ids,
        chunks=chunks,
        out_dir=args.out_dir,
        bits=args.bits,
        chunk_protocol=args.chunk_protocol)
    stats = drv.run()
    print("\n" + "=" * 60)
    print(f"INGESTION COMPLETE: {stats}")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
