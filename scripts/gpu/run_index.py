#!/usr/bin/env python3
"""run_index.py — build the retrieval index over an ingested corpus.

Consolidates the W10-GPU session's root-level run_index.py. Small corpora
(< 256 chunks, where FAISS IVFPQ's training floor sits) build a flat
IndexFlatIP; larger corpora use the repo's IVFADC builder
(src/rag/index.py::build_index — the production path, spec §5).

W16 — THE CENTERED RETRIEVAL FRAME (default): before the faiss build, the
script builds the RetrievalFrame (the reset point's own vector + the
corpus-delta mean) and persists it to <disk-dir>/retrieval_frame.npz; the
index is then built over the CENTERED content vectors
(iter_centered_vectors). run_query.py / answer_query pick the frame up
automatically. The W15 behavior (absolute vectors, no frame file) is the
--frame absolute A/B.

The flat path at small corpora centers too (same frame, exact IP over the
centered+normalized vectors).

Examples:
    python3 scripts/gpu/run_index.py
    python3 scripts/gpu/run_index.py --frame absolute     # the W15 A/B
    python3 scripts/gpu/run_index.py --disk-dir /home/ubuntu/RAGGA/disk/ingested_1k
"""
from __future__ import annotations

import argparse

import numpy as np

from _bootstrap import DEFAULTS, boot

boot()

import faiss  # noqa: E402

from index import (ChunkVectorLoader, build_index,  # noqa: E402
                   build_retrieval_frame, iter_centered_vectors,
                   save_retrieval_frame)

FLAT_FLOOR = 256  # faiss IVFPQ's train-vector floor at nbits=8


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--disk-dir", default=DEFAULTS["disk_dir"])
    ap.add_argument("--bits", type=float, default=3.5)
    ap.add_argument("--index-path", default=None,
                    help="output path (default: <disk-dir>/ivfadc_cache.index)")
    ap.add_argument("--frame", choices=("centered", "absolute"),
                    default="centered",
                    help="W16: 'centered' (default) builds the retrieval "
                         "frame + the index over content vectors; "
                         "'absolute' keeps the legacy W15 behavior")
    ap.add_argument("--mean-chunks", type=int, default=0,
                    help="cap the corpus-mean sample (0 = all chunks)")
    args = ap.parse_args()
    index_path = args.index_path or f"{args.disk_dir.rstrip('/')}/ivfadc_cache.index"

    print("=" * 60)
    print("RAGGA INDEX BUILD (scripts/gpu/run_index.py)")
    print("=" * 60)

    print(f"\n[1] Loading chunk vectors from {args.disk_dir}...")
    loader = ChunkVectorLoader(args.disk_dir, bits=args.bits)
    chunks = list(loader.chunk_ids())
    print(f"    {len(chunks)} chunks; manifest protocol "
          f"{loader.manifest.get('protocol', 'unknown')}, chunk layout "
          f"{loader.manifest.get('chunk_protocol', 'delta-v1')}")

    frame = None
    if args.frame == "centered":
        print("\n[2] Building the centered retrieval frame (W16)...")
        frame = build_retrieval_frame(
            loader, max_chunks=args.mean_chunks or None)
        save_retrieval_frame(loader, frame)
        print(f"    frame written to {args.disk_dir.rstrip('/')}/retrieval_frame.npz")
        print(f"    mean over {frame.n_mean_chunks} chunk deltas "
              f"(dims {frame.dims}); chunk layout {frame.chunk_protocol}")

    # W17: the corpus's actual M1/M2 unit dims (mem_size experiments) —
    # pinned into the side metadata's codebook digests
    unit_dims = [int(c.d) for c in (loader.system.m1_codes,
                                    loader.system.m2_codes)
                 if c is not None]
    unit_dims = sorted(set(unit_dims)) or None

    if len(chunks) >= FLAT_FLOOR:
        print("\n[3] Building the IVFADC index (build_index, the §5 path)...")
        stream = iter_centered_vectors(loader, frame) if frame is not None \
            else loader.iter_vectors()
        build_index(stream, path=index_path,
                    extra_metadata={"vector_frame": args.frame},
                    unit_dims=unit_dims)
        print(f"    written to {index_path}")
    else:
        print(f"\n[3] {len(chunks)} < {FLAT_FLOOR} chunks — flat IndexFlatIP "
              f"(exact IP; the IVFPQ training floor would under-train)")
        if frame is not None:
            vecs = np.vstack([
                frame.center_delta(loader.delta_vector(c))
                for c in chunks]).astype(np.float32)
        else:
            vecs = np.vstack([
                loader.vector(c) for c in chunks]).astype(np.float32)
        vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
        index = faiss.IndexFlatIP(vecs.shape[1])
        index.add(vecs)
        faiss.write_index(index, index_path)
        print(f"    written to {index_path}")

    print("\n" + "=" * 60)
    print(f"INDEX BUILD COMPLETE (frame: {args.frame})")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
