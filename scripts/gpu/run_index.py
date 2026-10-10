#!/usr/bin/env python3
"""run_index.py — build the retrieval index over an ingested corpus.

Consolidates the W10-GPU session's root-level run_index.py. Small corpora
(< 256 chunks, where FAISS IVFPQ's training floor sits) build a flat
IndexFlatIP; larger corpora use the repo's IVFADC builder
(src/rag/index.py::build_index — the production path, spec §5).

Examples:
    python3 scripts/gpu/run_index.py
    python3 scripts/gpu/run_index.py --disk-dir /home/ubuntu/RAGGA/disk/ingested_1k
"""
from __future__ import annotations

import argparse

import numpy as np

from _bootstrap import DEFAULTS, boot

boot()

import faiss  # noqa: E402

from index import ChunkVectorLoader, build_index  # noqa: E402

FLAT_FLOOR = 256  # faiss IVFPQ's train-vector floor at nbits=8


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--disk-dir", default=DEFAULTS["disk_dir"])
    ap.add_argument("--bits", type=float, default=3.5)
    ap.add_argument("--index-path", default=None,
                    help="output path (default: <disk-dir>/ivfadc_cache.index)")
    args = ap.parse_args()
    index_path = args.index_path or f"{args.disk_dir.rstrip('/')}/ivfadc_cache.index"

    print("=" * 60)
    print("RAGGA INDEX BUILD (scripts/gpu/run_index.py)")
    print("=" * 60)

    print(f"\n[1] Loading chunk vectors from {args.disk_dir}...")
    loader = ChunkVectorLoader(args.disk_dir, bits=args.bits)
    chunks = list(loader.chunk_ids())
    print(f"    {len(chunks)} chunks; manifest protocol "
          f"{loader.manifest.get('protocol', 'unknown')}")

    if len(chunks) >= FLAT_FLOOR:
        print("\n[2] Building the IVFADC index (build_index, the §5 path)...")
        build_index(loader.iter_vectors(), path=index_path)
        print(f"    written to {index_path}")
    else:
        print(f"\n[2] {len(chunks)} < {FLAT_FLOOR} chunks — flat IndexFlatIP "
              f"(exact IP; the IVFPQ training floor would under-train)")
        vecs = np.vstack([
            loader.vector(c) for c in chunks]).astype(np.float32)
        vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
        index = faiss.IndexFlatIP(vecs.shape[1])
        index.add(vecs)
        faiss.write_index(index, index_path)
        print(f"    written to {index_path}")

    print("\n" + "=" * 60)
    print("INDEX BUILD COMPLETE")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
