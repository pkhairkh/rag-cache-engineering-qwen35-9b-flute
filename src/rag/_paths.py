"""_paths.py — the sys.path anchor for the RAG package (house convention).

The repo's import convention is same-dir top-level imports with sys.path
insertion (see `src/scripts/modeling.py`'s lazy `attn_sm86` import and
`src/scripts/palettized_modules.py`'s flute path insert). This module
centralizes it for the RAG build:

  - this dir  (src/rag)          -> `import turboquant`, `import codebooks`, ...
  - ../scripts (src/scripts)     -> `import modeling`, `import palettized_modules`
  - ../flute_extended            -> `import fht` (the standalone CPU-legal FHT)

Import this once (conftest does it for the tests; entry modules do it in
main()) BEFORE importing any sibling. Stdlib-only at import, per the house
module contract.
"""
from __future__ import annotations

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_SCRIPTS = os.path.normpath(os.path.join(_HERE, os.pardir, "scripts"))
_FLUTE = os.path.normpath(os.path.join(_HERE, os.pardir, "flute_extended"))


def anchor() -> None:
    """Idempotently put src/rag, src/scripts, src/flute_extended on sys.path."""
    for p in (_HERE, _SCRIPTS, _FLUTE):
        if p not in sys.path:
            sys.path.insert(0, p)


anchor()
