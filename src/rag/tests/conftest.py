"""conftest.py — pytest anchors + shared fixtures for the RAG test suite.

Puts the repo's three code roots on sys.path (house convention, see
src/rag/_paths.py) so tests can `import turboquant`, `import fht`,
`import modeling` directly.
"""
from __future__ import annotations

import os
import sys

_RAG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _RAG not in sys.path:
    sys.path.insert(0, _RAG)

import _paths  # noqa: E402,F401  (anchors src/rag, src/scripts, src/flute_extended)

import pytest  # noqa: E402


@pytest.fixture(scope="session")
def rng():
    """A deterministic torch RNG shared across the session (seed pinned)."""
    import torch

    return torch.Generator().manual_seed(20261009)


@pytest.fixture()
def unit_vector_factory():
    """Factory: deterministic random unit vectors of a requested dim."""

    def make(d: int, count: int = 1, gen=None):
        import torch

        g = gen
        if g is None:
            g = torch.Generator().manual_seed(1234 + d)
        x = torch.randn(count, d, generator=g, dtype=torch.float32)
        x = x / x.norm(dim=-1, keepdim=True)
        return x if count > 1 else x[0]

    return make
