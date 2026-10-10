#!/usr/bin/env python3
"""w16_calibrate_stub.py — calibrate the common-component stub for the
retrieval test (finds constants where the ABSOLUTE frame fails and the
CENTERED frame retrieves, through the real TQ machinery at d=128)."""
from __future__ import annotations

import sys
import zlib

import numpy as np
import torch

REPO = "/home/z/my-project/repo"
sys.path.insert(0, REPO + "/src/rag")

from ingest import IngestDriver, load_system_state, reseed_cache  # noqa: E402
from index import (ChunkVectorLoader, build_retrieval_frame,  # noqa: E402
                   rerank)
from query import query_cache_vector  # noqa: E402
from tq_cache import TQCache  # noqa: E402

LIN, FULL = "linear_attention", "full_attention"
LAYER_TYPES = [LIN, FULL, LIN, FULL]
LINEARS = [0, 2]
S_SHAPE, S_D = (1, 8, 16), 128
CONV_D, KERNEL = 32, 4
M_SHAPE, M_D = (2, 4, 16), 128
BITS = 3.5
N_TOPICS, N_CHUNKS = 5, 260
MARKER, NOISE, M_NOISE = 6.0, 0.05, 0.1
JITTER = 0.25
QUERY_TOPIC = 2
QUERY_IDS = (200, 201)
SYSTEM_IDS = torch.tensor([[5, 6, 7, 8]])


def _make_cache():
    return TQCache(layer_types=LAYER_TYPES, bits=BITS)


class Stub:
    def __init__(self, common, q_marker):
        g = torch.Generator().manual_seed(99)
        self.topic_dirs = [torch.randn(S_SHAPE, generator=g)
                           for _ in range(N_TOPICS)]
        self.common_dir = torch.randn(S_SHAPE, generator=g)
        self.chunk_topics = {}
        self.chunk_jitter = {}
        self.q_marker = q_marker
        self.common = common

    def __call__(self, input_ids, past_key_values, use_cache=True):
        ids = tuple(int(t) for t in input_ids.flatten().tolist())
        kh = zlib.crc32(np.asarray(ids, dtype=np.int64).tobytes())
        scale = self.chunk_jitter.get(ids, 1.0)
        topic = self.chunk_topics.get(ids)
        for L in LINEARS:
            cur = past_key_values.layers[L].recurrent_states[0]
            if cur is None:
                cur = torch.zeros(S_SHAPE, dtype=torch.float16)
            g = torch.Generator().manual_seed(31 * (L + 1) + kh)
            add = NOISE * torch.randn(S_SHAPE, generator=g) \
                + scale * self.common * self.common_dir
            if ids == QUERY_IDS:
                add = add + self.q_marker * self.topic_dirs[QUERY_TOPIC]
            elif topic is not None:
                add = add + scale * MARKER * self.topic_dirs[topic]
            new = (cur.float() + add.float()).half()
            past_key_values.update_recurrent_state(new, L)
            conv_in = torch.randn(1, CONV_D, max(1, len(ids)), generator=g)
            past_key_values.update_conv_state(
                conv_in.half(), L, conv_kernel_size=KERNEL)
        for which, seed in (("m1", 7001), ("m2", 7002)):
            m = getattr(past_key_values, f"read_{which}")()
            if m is None:
                m = torch.zeros(*M_SHAPE, dtype=torch.float16)
            g = torch.Generator().manual_seed(seed + kh)
            getattr(past_key_values, f"update_{which}")(
                (m.float() + M_NOISE * torch.randn(*M_SHAPE,
                                                   generator=g)).half())
        return torch.zeros(1, max(1, len(ids)), N_TOPICS)


def run(common, q_marker, disk):
    model = Stub(common, q_marker)
    chunks = []
    for i in range(N_CHUNKS):
        tok = (100 + 7 * i, 101 + 7 * i, 102 + 7 * i)
        model.chunk_topics[tok] = i % N_TOPICS
        model.chunk_jitter[tok] = 1.0 + JITTER * (
            ((i * 2654435761) % 1000) / 1000.0 - 0.5)
        chunks.append(torch.tensor([list(tok)]))
    IngestDriver(model, SYSTEM_IDS, chunks, disk,
                 cache_factory=_make_cache).run()
    loader = ChunkVectorLoader(disk)
    system = load_system_state(disk)
    frame = build_retrieval_frame(loader)

    cache = _make_cache()
    reseed_cache(cache, system)
    with torch.no_grad():
        model(input_ids=torch.tensor([list(QUERY_IDS)]),
              past_key_values=cache, use_cache=True)
    qvec = query_cache_vector(cache, system)

    out = {}
    for name, fr in (("absolute", None), ("centered", frame)):
        ids, scores = rerank(loader, qvec,
                             np.array(loader.chunk_ids()), k=3, frame=fr)
        purity = sum(1 for i in ids if int(i) % N_TOPICS == QUERY_TOPIC)
        out[name] = (list(map(int, ids)), purity,
                     [f"{s:.4f}" for s in scores])
    return out


if __name__ == "__main__":
    import tempfile
    for common, qm in ((40.0, 6.0), (40.0, 2.0), (40.0, 1.0),
                       (20.0, 1.0), (60.0, 3.0), (60.0, 2.0)):
        with tempfile.TemporaryDirectory() as disk:
            res = run(common, qm, disk)
        print(f"COMMON={common} QMARKER={qm}:")
        for name, (ids, purity, sc) in res.items():
            print(f"  {name:<9} purity {purity}/3 top {ids} scores {sc}")
