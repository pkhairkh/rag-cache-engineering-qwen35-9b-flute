#!/usr/bin/env python3
"""w16_probe_retrieval.py — the W16 retrieval-failure probe (CPU).

THE QUESTION (handover W15-post): "does not find the right document / document
selection" — top-3 cosines [0.7232, 0.7072, 0.7071], spread 0.016, gold doc
absent from the top-3. Why does the §4 cosine degenerate, and what fixes it
WITHOUT leaving the architecture (N17: cache-state vectors, cos-sim =
content overlap)?

THE MODEL OF THE FAILURE (this script quantifies it):
Both the query vector and every chunk vector are ABSOLUTE cache states:
      q_abs  = sys + q_delta          (query prefill from the reseed point)
      c_abs  = sys + c_delta          (the loader's reconstruction)
They share (a) the system-prompt state and (b) the GENERIC-TEXT response —
the model's mean state response to English prose (every doc and every query
produces it). The shared part does not discriminate; cos(q_abs, c_abs) is
dominated by it:
      cos_abs ~= (|shared|^2 + <q_sig, c_sig>) / (|q_abs| |c_abs|)
so every chunk scores ~ the same 0.70 and the RANKING is driven by chunk
norm differences (length effects), not content. The observed signature
(spread 0.016 across the top-3) is exactly this regime.

THE FIX CANDIDATES (all in-frame, no embedder, no chunk text):
  frame "absolute"  : cos(q_abs, c_abs)                       — the W14/W15 path
  frame "sys"       : cos(q_delta, c_delta)                   — center on the
                      delta protocol's zero (the system reset point)
  frame "sys+mean"  : cos(q_delta - mu, c_delta - mu), mu = mean of chunk
                      deltas (removes the residual generic-text direction)

This probe runs the REAL repo machinery: resolve_quantizer("S", d, 3.5) with
the committed d=2048 codebooks (the D3 frame, the FHT rotation, the Lloyd-Max
codebooks, the packed codes) — the same quantization noise the box sees —
the REAL delta protocol (sys codes, delta codes, the loader's
double-round absolute reconstruction), and scores every frame on the same
vectors.

GEOMETRY (calibrated to the box's measured scales):
  - d = 2048, 100 chunks, 5 topics
  - |sys| : |delta| = 0.11 : 0.29 in std terms (the W12 box's state_scale row)
  - chunk deltas: generic-common g + topic marker u_t + idiosyncratic noise,
    with per-chunk norm modulation (chunk lengths vary) and the generic
    fraction f swept — f = the share of the delta's energy in the common
    direction (the failure driver)
  - query delta: the same common g (a question is English text too) + a WEAK
    topic marker (a question names the topic less directly than the doc
    states it) + noise

OUTPUT: per frame — hit@1 / hit@3, the top-3 scores, the spread, and the
gold doc's rank. The box's failure signature (all ~0.70, spread ~0.016,
gold outside) should appear in the absolute frame at high f and disappear
in the centered frames.
"""
from __future__ import annotations

import sys

import numpy as np
import torch

REPO = "/home/z/my-project/repo"
sys.path.insert(0, REPO + "/src/rag")

from tq_cache import resolve_quantizer  # noqa: E402

D = 2048
N_CHUNKS = 100
N_TOPICS = 5
BITS = 3.5
GOLD = 37                      # the gold chunk (topic GOLD % N_TOPICS)
SEED = 20261011

# the state scales (std of the components, box-calibrated)
SYS_STD = 0.11
DELTA_STD = 0.29               # the doc-loaded state std minus the sys part
QUERY_DELTA_STD = 0.18         # a ~20-token question builds less state
TOPIC_SHARE_DOC = 0.30         # the doc's topic-marker energy share
NOISE_SHARE = 0.25             # idiosyncratic noise share
NORM_JITTER = 0.25             # per-chunk norm modulation (length effects)
QRY_TOPIC_SWEEP = (0.15, 0.08, 0.05, 0.03, 0.02)   # the query's topic share


def _unit(rng: np.random.Generator, d: int) -> np.ndarray:
    v = rng.standard_normal(d)
    return v / np.linalg.norm(v)


def main() -> int:
    rng = np.random.default_rng(SEED)
    q = resolve_quantizer("S", D, BITS)   # the real frame: seed 101 + FHT

    def quant_dequant(x: np.ndarray) -> np.ndarray:
        codes = q.quant(torch.from_numpy(x.astype(np.float32)))
        return q.dequant(codes).numpy()

    # the shared directions
    g = _unit(rng, D)                                   # generic-text common
    topics = [_unit(rng, D) for _ in range(N_TOPICS)]   # topic markers

    # the system state (quantized once — the reset point)
    sys_raw = SYS_STD * np.sqrt(D) * _unit(rng, D)
    sys_hat = quant_dequant(sys_raw)

    # the generic fraction sweep: the failure driver
    for f in (0.10, 0.30, 0.50, 0.70, 0.80, 0.90):
        sig_share = 1.0 - f - NOISE_SHARE
        if sig_share <= 0.05:
            continue
        # per-chunk deltas (the doc states, on the quantization lattice)
        deltas = []
        for i in range(N_CHUNKS):
            jitter = 1.0 + NORM_JITTER * (rng.random() - 0.5)
            comp = (np.sqrt(f) * g * DELTA_STD
                    + np.sqrt(sig_share) * topics[i % N_TOPICS] * DELTA_STD
                    + np.sqrt(NOISE_SHARE)
                    * _unit(rng, D) * DELTA_STD)
            deltas.append(quant_dequant(jitter * np.sqrt(D) * comp))

        # the query-SNR sweep: the question's topic-signal strength
        for qts in QRY_TOPIC_SWEEP:
            # the query delta: common g + gold-topic marker + noise
            qsig = (np.sqrt(f) * g * QUERY_DELTA_STD
                    + np.sqrt(qts) * topics[GOLD % N_TOPICS]
                    * QUERY_DELTA_STD
                    + np.sqrt(max(0.0, 1.0 - f - qts))
                    * _unit(rng, D) * QUERY_DELTA_STD)
            q_delta = quant_dequant(np.sqrt(D) * qsig)

            # the frames
            q_abs = sys_hat + q_delta
            c_abs = [sys_hat + d for d in deltas]
            mu = np.mean(deltas, axis=0)

            frames = {
                "absolute": (q_abs, c_abs),
                "sys": (q_delta, deltas),
                "sys+mean": (q_delta - mu, [d - mu for d in deltas]),
            }

            print(f"\n=== f = {f:.2f} (doc topic share {sig_share:.2f}), "
                  f"query topic share {qts:.2f} ===")
            gold_topic = GOLD % N_TOPICS
            for name, (qv, cvs) in frames.items():
                qn = np.linalg.norm(qv)
                scores = np.array([np.dot(qv, c) / (qn * np.linalg.norm(c))
                                   for c in cvs])
                order = np.argsort(-scores)
                top3 = order[:3]
                purity = sum(1 for i in top3
                             if int(i) % N_TOPICS == gold_topic)
                best_mate = next(pos for pos, i in enumerate(order)
                                 if int(i) % N_TOPICS == gold_topic)
                mate_scores = scores[[i for i in range(N_CHUNKS)
                                      if i % N_TOPICS == gold_topic]]
                other_scores = scores[[i for i in range(N_CHUNKS)
                                       if i % N_TOPICS != gold_topic]]
                gap = (float(mate_scores.max())
                       - float(other_scores.max()))
                print(f"  {name:<9} top3-purity {purity}/3 "
                      f"best-mate-rank {best_mate:>3} "
                      f"disc-gap {gap:+.4f} "
                      f"top3 [{scores[top3[0]]:.3f} "
                      f"{scores[top3[1]]:.3f} {scores[top3[2]]:.3f}]")
    print("\nReading the table: the ABSOLUTE frame is the W14/W15 path; "
          "its top-3 collapse to a common score with the gold doc outside "
          "as f grows (the box: 0.72/0.71/0.71, spread 0.016). The centered "
          "frames keep ranking by content.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
