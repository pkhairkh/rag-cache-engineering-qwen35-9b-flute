"""src/rag — the cache-engineered RAG build (SPECIFICATION.md §2–§9).

Package layout (PROPOSAL.md §3), one module per spec section:

  codebooks.py    §3.1   Lloyd-Max codebooks on the Beta density (per bit-width)
  turboquant.py   §3     the quantizer: units, FHT rotation, 3.5-bit split,
                         norm handling, code pack/unpack + serialization
  tq_cache.py     §3.2   the online cache wrapper: quantize-on-write,
                         dequantize-on-read (the cache never holds fp16)
  m1m2.py         §2.2   the two global memories (gated additive writes,
                         softmax(q @ M1^T) @ M2 reads)
  hooks.py        §1     the 9-hook capture harness (24 S tensors)
  snapshot.py     §5/§11 the per-chunk TurboQuant-code disk format
  ingest.py       §5     the delta-protocol ingestion + batch driver
  index.py        §8     IVFADC build + preselect + cos-sim rerank
  install.py      §6     dequant-sum-requant install math
  query.py        §6     the eight-step query flow
  finetune.py     §7     the lean straight-through LUT fine-tune loop
  lut_export.py   §7/§11 fine-tuned LUT artifact export/import
  evals.py        gates  the phase-gate measurement harness (GPU-box entry)

House contract: this package's modules import each other top-level after
`import _paths` (see _paths.py). Tests live in tests/ and run on the CPU
box via `python3 -m pytest src/rag/tests -q` (TASKS.md §2).
"""
from __future__ import annotations

__all__: list[str] = []
