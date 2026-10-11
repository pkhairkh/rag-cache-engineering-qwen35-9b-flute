#!/usr/bin/env python3
"""finetune_m1m2.py — the §7 M1/M2 gate fine-tune on the GPU box.

Trains ONLY the M1/M2 gate vectors (write_gate_k/write_gate_v —
per-(layer, head) since W21; read_gate optionally) of the W17-wired
model, on GENERAL semantic-similarity pairs (paraphrase / entailment /
QA pairs — MRPC, QQP, PAWS, SNLI, MNLI, STS-B, SQuAD ...), NEVER on
the served corpus. The goal (the handover's design): similar texts ->
similar M1/M2 states -> the §4 vector's M1/M2 units carry the
query-document alignment the S-only vectors lack (measured cos(q, doc)
~0.50-0.58, 50% hit rate).

W21 — THE ADDRESSING A/B (the position-lock, answered): the v1 write
addressed memory slots by POSITION (slot = p mod mem_size), so the
pair-similarity the InfoNCE trains is a position-aligned key
correlation — surface overlap, not content; the box's runs plateaued
at spread ~0.13 with cos_neg parked at ~+0.36 (the anisotropy
pedestal). This driver now defaults to the W21 module behavior —
CONTENT-addressed slots (simhash of the per-(b,h) mean-centered key;
shared-content tokens match anywhere in the two texts) + per-head
write gates — with the v1 behavior one flag away:

    --write-addr positional      # the v1 slot rule (bit-exact A/B)
    --no-center-write            # raw keys/values (the pedestal stays)

THE DATA CONTRACT (corpus-independent): a JSONL of
    {"text1": "...", "text2": "...", "label": 1}
(label 1 = similar; label 0 rows are skipped by the trainer — the
InfoNCE takes its negatives IN-BATCH from the other pairs; they ride
along as the held-out CONTROL group). Convert any general dataset
to this format offline (the box does not need internet at train time);
EXAMPLES of supported sources: MRPC/QQP/PAWS (paraphrase), SNLI/MNLI
(entailment=1, contradiction=0), STS-B (score>4 -> 1), SQuAD
(question, answer sentence).

The deployment protocol (ORDER MATTERS): train -> save gates ->
RE-INGEST with the trained gates (run_ingestion.py --m1m2-gates ...
--m1m2-mem-size SAME) -> run_index.py -> query (run_query.py
--m1m2-gates ...). The snapshot geometry (m1_shape in system_state.npz)
pins the mem_size; the query-side loader fails loudly on drift. W21:
a gates .npz trained before W21 ((L,) gate vectors) is REFUSED loudly
by load_gates — the addressing changed the semantics, retrain.

Examples:
    # the W21 default run (content-addressed, centered, per-head gates)
    python3 scripts/gpu/finetune_m1m2.py --pairs-file data/pairs.jsonl \
        --gates-out m1m2_gates_trained.npz --m1m2-mem-size 1024 \
        --max-steps 300 --refresh-every 4 --gate-init 0.1

    # the v1 A/B (position-addressed, uncentered — the pre-W21 regime)
    python3 scripts/gpu/finetune_m1m2.py --pairs-file data/pairs.jsonl \
        --gates-out m1m2_gates_v1.npz --m1m2-mem-size 1024 \
        --write-addr positional --no-center-write

    # mechanics smoke test on synthetic pairs (NOT training data)
    python3 scripts/gpu/finetune_m1m2.py --self-test --max-steps 5

THE HELD-OUT LADDER ([6], the W21 instrument): the captured k/v of the
held-out pairs are replayed under EVERY addressing config — the run's
own, the kappa ladder (1/2/8 blocks), uncentered content, centered
positional, and the v1 corner — and each row reports
cos(pos)/cos(neg)/spread vs the in-batch AND the label-0 curated
negatives, plus the lexical-overlap buckets. One capture, N configs
(the replay is ~seconds; the prefill is not repeated). The row
differences ARE the architectural measurement: the v1 row vs the
content row on the LOW-overlap bucket is the position-lock ceiling,
made visible; the cos_neg column is the pedestal (centered rows
should sit near 0, the uncentered rows near +0.36).

OOM HARDENING (the box's W20.15 run crashed INSIDE [5] and lost the
trained artifact): the gates artifact + history now save BEFORE any
held-out report ([5] before [6]); the ladder streams per-pair-side
vectors (no stacked (B, mem, H, D) states — the 4 x 1.07 GiB stacks
that OOM'd), frees the replay graphs per side, and every report
section is a try/except that WARNS and continues (a report failure
never loses the artifact again).

W22 — THE LADDER OOM, ROOT-CAUSED (the box's W21.6 run): the W21
streaming kept the units ON THE GPU — 128 captured pair-sides (~2.6
GiB k/v) + four (64, dim) unit stacks + their python lists (~6 GiB
transient) on top of the model and the allocator's un-returnable
residue: the first ladder row hit the 22 GiB ceiling at a 16 MiB
allocation. The ladder now runs HOST-STAGED: the capture moves every
recorded k/v + m_init to (pinned) host memory the moment a side's
prefill ends, the replay streams one side's calls to the device and
frees them, the units stack on the HOST, and _sim_from_units stages
only (chunk, dim) row/col blocks onto the device for the matmul —
the ladder's GPU footprint beyond the model is bounded at < 1 GiB
no matter how many pairs or rows. The [4] residue (the epoch
iterator, the trailing prefetch, the system cache) is released
before [6] (gc + empty_cache). The UnboundLocalError crash after the
ladder (the box's traceback: `del val_group` then a later
`if val_group is not None` — `del` UNBINDS the name) is fixed: the
groups rebind to None, never del.

W22 — THE CURATED HARD NEGATIVES (--neg-pairs N, default 8): the
label-0 rows the loader already carries join the TRAINING loss as
shared negative columns (see m1m2_finetune's W22 note — DPR/MoCo's
proven small-batch fix; the box's B=8 loss swung 0.28..1.64 on seven
random in-batch draws). The train negatives draw from the label-0
pool EXCLUDING the held-out control slice (the first n_eval rows
after the seeded shuffle — the control line stays untouched by
training), re-batched per epoch under the same seed. --neg-pairs 0
restores the exact W21 loss (the A/B).
"""
from __future__ import annotations

import argparse
import json
import os
import random
import time

import torch

from _bootstrap import DEFAULTS, boot, load_model

boot()

from ingest import prefill_system  # noqa: E402
from m1m2_finetune import (GatesTrainConfig, bucketed_spread_from_sim,  # noqa: E402
                            epoch_batches, flat_unit,
                            lexical_overlap,
                            prefill_capture_calls, replay_states,
                            sim_matrix_stats, save_gates, train_gates)
from tq_cache import TQCache  # noqa: E402


def load_pairs(path: str, limit: int = 0):
    """The positive pairs AND the label-0 curated non-pairs from a
    general-similarity JSONL.

    W20.15: the label-0 rows ride along as the held-out CONTROL GROUP
    (their own pair cosines — the datasets' own NON-similar pairs, a
    cleaner discrimination floor than in-batch negatives, which can be
    accidentally-similar cross pairs). They stay OUT of training — the
    v1 in-batch-negative design is unchanged (a future wave may mine
    them as explicit hard negatives)."""
    pairs, negatives = [], []
    with open(path) as f:
        for i, line in enumerate(f):
            if limit and len(pairs) >= limit:
                break
            row = json.loads(line)
            label = int(row.get("label", 1))
            t1, t2 = row.get("text1"), row.get("text2")
            if not (t1 and t2):
                continue
            if label == 1:
                pairs.append((t1, t2))
            else:
                negatives.append((t1, t2))
    if len(pairs) < 2:
        raise SystemExit(
            f"{path}: need >= 2 positive pairs (in-batch negatives); "
            f"got {len(pairs)}")
    return pairs, negatives


def synthetic_pairs(n: int, vocab: int, length: int = 96, seed: int = 5):
    """Mechanics smoke pairs: a random base and a light perturbation of
    it as the 'positive' (the in-batch negatives are the other bases).
    NOT semantic data — only verifies the loop moves the gates and the
    artifact saves; retrieval quality needs the general datasets."""
    g = torch.Generator().manual_seed(seed)
    out = []
    for _ in range(n):
        base = torch.randint(0, vocab, (length,), generator=g).tolist()
        pert = list(base)
        for _ in range(max(1, length // 8)):
            j = int(torch.randint(0, length, (1,), generator=g))
            pert[j] = int(torch.randint(0, vocab, (1,), generator=g))
        out.append((base, pert))
    return out


def _parse_layer_spec(spec: str):
    """'all' -> None (every layer's write gates train — the default);
    otherwise a sorted unique index list from comma ints / 'lo-hi'
    ranges (the driver side of GatesTrainConfig.train_layers, the W20.14
    H6 knob; train_gates validates the range against the module)."""
    if spec is None or spec.strip().lower() in ("", "all"):
        return None
    out = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-", 1)
            out.update(range(int(lo), int(hi) + 1))
        else:
            out.add(int(part))
    if not out:
        raise SystemExit(f"--train-layers {spec!r} parses to no layers — "
                          f"use 'all' or e.g. '5' / '0-11,16'")
    return sorted(out)


def _host_pinned(ids) -> torch.Tensor:
    """W20.11.2 — the loader pinning: token-id tensors are built on the
    HOST as PINNED (page-locked) tensors, so the capture loop's
    per-refresh H2D staging rides async DMA (non_blocking). Transport
    only — the values are bit-identical to the plain host tensors;
    without CUDA the plain tensor is returned (same values, no pin)."""
    t = torch.tensor([ids])
    return t.pin_memory() if torch.cuda.is_available() else t


# ------------------------------------------------------ the data report ---
def _overlap_profile(text_pairs, cap: int = 4000):
    """[3.5] — the data-quality profile of a pair list: the lexical
    overlap distribution (lexical_overlap — the W20.15 instrument,
    word-multiset containment) plus length stats, text-only, no model.

    The question it answers with numbers ("is the data good?"): a
    positive-pair profile concentrated at low overlap with a label-0
    profile that MATCHES it means the pairs carry no surface signal —
    fine for content-based training (W21) but fatal for the v1
    position-locked write (the W20.15 ceiling); positives clearly above
    the label-0 profile mean the signal exists and the addressing is
    the only question. Capped at `cap` pairs (the mean is stable long
    before 4k; the report is a probe, not a census)."""
    if not text_pairs:
        return None
    sample = text_pairs if len(text_pairs) <= cap else \
        random.Random(7).sample(text_pairs, cap)
    ovs, lens = [], []
    for t1, t2 in sample:
        ovs.append(lexical_overlap(t1, t2))
        lens.append((len(t1.split()) + len(t2.split())) / 2.0)
    ovs.sort()
    n = len(ovs)

    def q(f):
        return ovs[min(n - 1, int(f * n))]
    return {"n": n, "mean": sum(ovs) / n, "median": q(0.5),
            "p25": q(0.25), "p75": q(0.75),
            "frac_hi": sum(1 for o in ovs if o >= 0.5) / n,
            "frac_mid": sum(1 for o in ovs if 0.2 <= o < 0.5) / n,
            "frac_lo": sum(1 for o in ovs if o < 0.2) / n,
            "mean_len": sum(lens) / len(lens)}


# ----------------------------------------------- the [6] streaming ladder --
def _to_host(t, pin: bool):
    """W22 — one tensor to (optionally pinned) HOST memory with its own
    storage (the ladder's host staging; a no-op returning the same
    object when already there — .to() semantics)."""
    if t is None:
        return None
    h = t.detach().to("cpu")
    return h.pin_memory() if pin and h.device.type == "cpu" else h


def _capture_group(model, system, cache_factory, id_pairs,
                   host_stage: bool = False):
    """Streaming capture of a group of pairs: per pair, per side —
    prefill through the production path and KEEP ONLY the calls +
    m_init (the ladder replays them under every addressing config; the
    cache and the graphs are dropped per side). Returns the list of
    [(calls_a, m1i_a, m2i_a), (calls_b, m1i_b, m2i_b)] per pair.

    W22 — host_stage=True (the ladder's mode): the recorded k/v and
    the m_init land on (pinned) HOST memory the moment the side's
    prefill ends — the GPU keeps nothing per side beyond the prefill
    transients (the W21.6 box run OOM'd at the first ladder row with
    ~2.6 GiB of retained device-side k/v on top of the unit stacks).
    The ladder streams each side back to the device at replay time
    (_side_units) — the values are bit-identical, only the residence
    moved. host_stage=False keeps the W21 device-resident behavior
    for any other caller."""
    pin = host_stage and torch.cuda.is_available()
    sides = []
    for ids_a, ids_b in id_pairs:
        ps = []
        for ids in (ids_a, ids_b):
            c = cache_factory()
            calls, m1i, m2i = prefill_capture_calls(model, ids, c, system)
            del c
            if host_stage:
                calls = [(_to_host(k, pin), _to_host(v, pin), li,
                          _to_host(pos, pin))
                         for (k, v, li, pos) in calls]
                m1i, m2i = _to_host(m1i, pin), _to_host(m2i, pin)
            ps.append((calls, m1i, m2i))
        sides.append(ps)
    return sides


def _side_units(module, calls, m1_i, m2_i, dev=None):
    """One pair side -> (u1, u2): the L2-normalized flat DELTA vectors
    (fp32, (dim,), HOST) — the pair_sim_matrix operands. Runs the
    replay under no_grad and frees the graph before returning (the
    [6] OOM fix: no stacked states, no retained autograd; the gates'
    requires_grad is irrelevant under no_grad — this is the metric
    path, not the training path).

    W22 — device streaming: the calls/m_init may live on HOST (the
    host-staged capture); each call's k/v (+ positions) stages to
    `dev` (the module's device) for the replay and frees after, so
    the replay's device footprint is one call's k/v + the states
    (~O(50 MiB)) regardless of the group size. Values bit-identical
    (transport only). The returned units land on HOST (the stacks
    and the sim-matmul chunks stage back per block — bounded)."""
    if dev is None:
        dev = module.write_gate_k.device
    fast = dev.type == "cuda"
    staged = [(k.to(dev, non_blocking=fast),
               v.to(dev, non_blocking=fast), li,
               pos.to(dev, non_blocking=fast) if pos is not None else None)
              for (k, v, li, pos) in calls]
    m1d, m2d = m1_i.to(dev), m2_i.to(dev)
    with torch.no_grad():
        m1, m2 = replay_states(module, staged, m1d, m2d)
        u1 = flat_unit(m1 - m1d)[0]
        u2 = flat_unit(m2 - m2d)[0]
    u1, u2 = u1.detach().to("cpu"), u2.detach().to("cpu")
    del m1, m2, m1d, m2d, staged
    return u1, u2


def _group_units(module, group, dev=None):
    """Replay every side of a captured group under the module's CURRENT
    addressing knobs -> (A1, A2, B1, B2): the four (B, dim) fp32 unit
    matrices (a-side m1/m2, b-side m1/m2).

    W22 — the stacks build on the HOST (each side's unit comes back
    from _side_units resident there): 4 x 1.07 GiB of HOST fp32 at 64
    pairs, zero device pressure (the W21 device-side stacks + their
    lists were the OOM's second term)."""
    u1a, u2a, u1b, u2b = [], [], [], []
    for (ca, i1a, i2a), (cb, i1b, i2b) in group:
        a1, a2 = _side_units(module, ca, i1a, i2a, dev)
        b1, b2 = _side_units(module, cb, i1b, i2b, dev)
        u1a.append(a1)
        u2a.append(a2)
        u1b.append(b1)
        u2b.append(b2)
    return (torch.stack(u1a), torch.stack(u2a),
            torch.stack(u1b), torch.stack(u2b))


def _sim_from_units(A1, A2, B1, B2, chunk: int = 16, dev=None):
    """0.5*(cos_m1 + cos_m2) from the unit matrices — the pair_sim_matrix
    arithmetic, CHUNKED so no full (B, dim) @ (dim, B) fp32 operand pair
    is ever live (the [6] OOM fix: the transients are (chunk, dim) x
    (dim, chunk)). The values are identical to the unchunked matmul up
    to fp32 accumulation order (a report, not a gradient).

    W22 — the units are HOST-stacked; when `dev` is a CUDA device each
    (chunk, dim) row block and (dim, chunk) col block stages to the
    device for the matmul and the (chunk, chunk) result lands back on
    host — the matmul's device footprint is ~2 x chunk x dim fp32
    (~0.5 GiB at chunk 16 / dim 2^22) regardless of B. dev=None or a
    cpu device computes the blocks in place (the no-CUDA self-test
    path)."""
    B = A1.shape[0]
    fast = (dev if dev is not None else A1.device)
    if fast.type != "cuda" or A1.device.type != "cpu":
        fast = A1.device       # compute where the units live
    sim = torch.zeros(B, B, device=A1.device, dtype=torch.float32)
    for r in range(0, B, chunk):
        ra1 = A1[r:r + chunk].to(fast).float()
        ra2 = A2[r:r + chunk].to(fast).float()
        for c in range(0, B, chunk):
            cb1 = B1[c:c + chunk].to(fast).float().T
            cb2 = B2[c:c + chunk].to(fast).float().T
            block = 0.5 * (ra1 @ cb1 + ra2 @ cb2)
            sim[r:r + chunk, c:c + chunk] = block.to(sim.device)
            del cb1, cb2, block
        del ra1, ra2
    return sim


_ADDR_LADDER = [
    # (label, write_addr, center_write, addr_blocks) — the run's config
    # rides FIRST (dict-built at call time), then the kappa ladder and
    # the corners. Every row replays the SAME captured k/v (offline).
    ("kappa=1", "content", True, 1),
    ("kappa=2", "content", True, 2),
    ("kappa=8", "content", True, 8),
    ("content uncentered", "content", False, 4),
    ("positional centered", "positional", True, 4),
    ("positional v1 (A/B)", "positional", False, 4),
]


def _fmt_stats(st):
    return (f"cos(pos) {st['cos_pos']:+.4f} cos(neg) {st['cos_neg']:+.4f} "
            f"spread {st['spread']:+.4f} [worst neg {st['off_max']:+.4f} "
            f"weakest pos {st['diag_min']:+.4f}]")


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--artifacts-dir", default=DEFAULTS["artifacts_dir"])
    ap.add_argument("--heads-dir", default=DEFAULTS["heads_dir"])
    ap.add_argument("--model-name", default=DEFAULTS["model_name"])
    ap.add_argument("--device", default=DEFAULTS["device"])
    ap.add_argument("--pairs-file", default=None,
                    help="JSONL {text1, text2, label} — general similarity "
                         "pairs (NOT the served corpus)")
    ap.add_argument("--self-test", action="store_true",
                    help="synthetic mechanics pairs (smoke only)")
    ap.add_argument("--gates-out", default="m1m2_gates.npz",
                    help="output .npz (save_gates artifact)")
    ap.add_argument("--m1m2-mem-size", type=int, default=1024,
                    help="the memories' slot count — the architecture's "
                         "band is [1024, 4096] (two GLOBAL states, "
                         "shared across all 24 linear layers; the gates "
                         "are per-layer per-head); ingest+query must use "
                         "the SAME value")
    ap.add_argument("--write-addr", default="content",
                    choices=("content", "positional"),
                    help="W21: the slot ADDRESS — 'content' (default: "
                         "simhash of the centered key; shared-content "
                         "tokens match anywhere in the two texts) or "
                         "'positional' (the v1 slot = p mod mem_size, "
                         "the position-locked A/B)")
    ap.add_argument("--no-center-write", action="store_true",
                    help="W21 A/B: write RAW keys/values (skip the "
                         "per-(b,h) mean-centering that removes the "
                         "anisotropy pedestal — the +0.36 random-pair "
                         "cosine floor)")
    ap.add_argument("--addr-blocks", type=int, default=4,
                    help="W21: kappa — the number of independent sign "
                         "blocks (each token writes kappa slots; the "
                         "collision probability of similar keys rises "
                         "with kappa, the per-slot resolution falls). "
                         "The [6] ladder sweeps 1/2/4/8 offline")
    ap.add_argument("--no-memopt", action="store_true",
                    help="A/B: the W17 plain autograd replay chain (retains "
                         "the 24 boundary states per side — the A10G OOM "
                         "path; the default is the W18 exact-gradient "
                         "memory-safe replay)")
    ap.add_argument("--gate-init", type=float, default=0.1,
                    help="W19: fill both write gates with this before step 0 "
                         "— the P3 zero-init is an EXACT saddle of the "
                         "InfoNCE (loss ln(B), gradient identically zero; "
                         "the flat 2.0794 run). 0 = train from the loaded "
                         "gates (provably dead at P3)")
    ap.add_argument("--refresh-every", type=int, default=4,
                    help="W19: the capture pool — ONE batch prefilled, "
                         "trained on for N GRADIENT STEPS (the 16 "
                         "prefills/step are ~95%% of the wall clock; 1 = "
                         "fresh capture every step, the W17 regime). The "
                         "replay re-runs per step so gradients sit at the "
                         "current gates; only the k/v freeze (stale by "
                         "<= lr x N steps). W20: 8 = the THROUGHPUT "
                         "regime — a first-class A/B row; at refresh 8 "
                         "the staleness bound is lr x (N - 1) = lr x 7 "
                         "(the tight form of the N bound above: the "
                         "frozen k/v trail the gates by at most 7 "
                         "optimizer steps, and the prefill count halves "
                         "again). The default 4 stays quality-first — "
                         "the operator's call on the box")
    ap.add_argument("--absolute-loss", action="store_true",
                    help="A/B: score the ABSOLUTE states (m_final) instead "
                         "of the W19 default deltas (m_final - m_init; the "
                         "W16 centered retrieval frame the serving side "
                         "scores)")
    ap.add_argument("--no-m1m2", action="store_true",
                    help="load without the wiring (the W16-exact A/B)")
    ap.add_argument("--bits", type=float, default=3.5)
    ap.add_argument("--system-prompt", default="You are a helpful AI assistant.")
    ap.add_argument("--max-tokens", type=int, default=256,
                    help="truncate each pair member to N tokens (capture "
                         "memory scales with T)")
    ap.add_argument("--max-steps", type=int, default=300)
    ap.add_argument("--batch-pairs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--tau", type=float, default=0.07)
    ap.add_argument("--neg-pairs", type=int, default=8,
                    help="W22: curated label-0 SIDES attached to every "
                         "capture pool as shared InfoNCE hard-negative "
                         "columns (DPR/MoCo's proven small-batch fix; the "
                         "B=8 in-batch pool is 7 random draws). 0 = the "
                         "exact W21 loss (the A/B). Drawn from the label-0 "
                         "rows EXCLUDING the held-out control slice")
    ap.add_argument("--train-read-gate", action="store_true",
                    help="unfreeze read_gate too (the contrastive loss "
                         "itself never sees it — the read affects "
                         "generation, not the state; pair this with a "
                         "next-token objective in a later wave)")
    ap.add_argument("--train-layers", default="all",
                    help="W20.14 (the H6 investigation): write-gate layer "
                         "subset to train — 'all' (default), a comma list "
                         "('5,17'), or ranges ('0-11,16'); every other "
                         "layer's gates freeze at their gate-init value "
                         "(zero grads + weight_decay 0) — the "
                         "isolate-single-layer experiment")
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--val-pairs", type=int, default=64,
                    help="W20.15: held-out pairs scored in [6] (the v1 "
                         "hard-coded 32 starved the bucketed report; "
                         "capped at the val split)")
    ap.add_argument("--shuffle-seed", type=int, default=1234,
                    help="W20.15: the regime seed — the val slice AND the "
                         "per-epoch re-batching order derive from it "
                         "(deterministic run to run)")
    ap.add_argument("--fixed-batches", action="store_true",
                    help="W20.15 A/B: the v1 regime — the fixed "
                         "file-order batch cycle (a pair's in-batch "
                         "negatives stay its file-layout neighbors "
                         "forever; the val split is still the seeded "
                         "shuffle)")
    ap.add_argument("--limit-pairs", type=int, default=0)
    args = ap.parse_args()

    print("=" * 60)
    print("RAGGA M1/M2 GATE FINE-TUNE (scripts/gpu/finetune_m1m2.py)")
    print("=" * 60)

    # the architecture's band: two GLOBAL memories shared across all 24
    # linear layers (the handover's contract) — outside it is a loud
    # refusal, never a silent geometry change
    if not 1024 <= args.m1m2_mem_size <= 4096:
        raise SystemExit(
            f"--m1m2-mem-size {args.m1m2_mem_size} is outside the "
            f"architecture's band [1024, 4096] (two global memories, "
            f"shared across the 24 linear layers) — refuse")

    print(f"\n[1] Loading model (m1m2_mem_size={args.m1m2_mem_size})...")
    model, _ = load_model(args.artifacts_dir, args.heads_dir,
                          args.model_name, args.device,
                          use_m1m2=not args.no_m1m2,
                          m1m2_mem_size=args.m1m2_mem_size)

    # W21 — apply the addressing config to the shared module BEFORE the
    # system prefill (the reset point, the captures, the ladder, and
    # the artifact's provenance all live under this config)
    inner = getattr(model, "model", model)
    if not args.no_m1m2:
        inner.m1m2.write_addr = args.write_addr
        inner.m1m2.center_write = not args.no_center_write
        inner.m1m2.addr_blocks = args.addr_blocks
        print(f"    [W21] write_addr {args.write_addr} / "
              f"center_write {not args.no_center_write} / "
              f"addr_blocks (kappa) {args.addr_blocks}")

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    print("\n[2] Building the system reset point (the delta zero)...")
    # W20.11.2 — the loader pinning: the system prefill's inputs stage
    # through the same pinned transport (host-built -> pinned ->
    # non_blocking H2D; values identical)
    system_ids = _host_pinned(
        tokenizer.encode(args.system_prompt, add_special_tokens=False)
    ).to(args.device, non_blocking=True)
    cache = TQCache(config=model.config, bits=args.bits, online=True)
    system = prefill_system(model.model, system_ids, cache,
                            system_ref="m1m2-finetune", bits=args.bits)
    print(f"    system {system.reference()}; "
          f"m1 {'present' if system.m1_codes is not None else 'zero-init'}")

    print("\n[3] Loading pairs...")
    # W20.11.2 — the loader pinning: pairs are built on HOST as PINNED
    # tensors and the H2D moves INTO the capture loop — each pool
    # refresh stages its prefill inputs there, the staging window
    # being the refresh cycle (transport only, values bit-identical)
    pair_texts: list = []   # W20.15: the (t1, t2) texts, aligned with
    neg_texts: list = []    # `pairs` / the label-0 control texts
    if args.self_test:
        raw = synthetic_pairs(64, vocab=tokenizer.vocab_size)
        pairs = [(_host_pinned(p[0]), _host_pinned(p[1])) for p in raw]
        print("    [SELF-TEST] 64 synthetic pairs (mechanics only)")
    else:
        if not args.pairs_file:
            raise SystemExit("--pairs-file is required (or --self-test)")
        texts, neg_texts = load_pairs(args.pairs_file,
                                      limit=args.limit_pairs)
        print(f"    {len(texts)} positive pairs "
              f"({len(neg_texts)} label-0 control rows) from "
              f"{args.pairs_file}")
        # W20.15 — the seeded regime: the val slice is a RANDOM sample
        # (v1 took the file's head block) and the epoch re-batching is
        # reproducible; --fixed-batches restores the v1 batch cycle.
        rng = random.Random(args.shuffle_seed)
        rng.shuffle(texts)
        rng.shuffle(neg_texts)
        pairs = []
        for t1, t2 in texts:
            ids1 = tokenizer.encode(t1, add_special_tokens=False)
            ids2 = tokenizer.encode(t2, add_special_tokens=False)
            if args.max_tokens:
                ids1, ids2 = ids1[:args.max_tokens], ids2[:args.max_tokens]
            if not ids1 or not ids2:
                continue
            pairs.append((_host_pinned(ids1), _host_pinned(ids2)))
            pair_texts.append((t1, t2))
    n_val = max(2, int(len(pairs) * args.val_frac)) if len(pairs) > 4 else 0
    # W20.11.2: the held-out prefills' ids stage ONCE here (the same
    # pinned non_blocking transport — [6] runs the unchanged path)
    if n_val:
        val_pairs = [(a.to(args.device, non_blocking=True),
                      b.to(args.device, non_blocking=True))
                     for a, b in pairs[:n_val]]
        val_texts = pair_texts[:n_val]      # W20.15 — the bucket labels
    else:
        val_pairs = []
        val_texts = []
    train_pairs = pairs[n_val:] if n_val else pairs
    # W20.15 — the regime: per-epoch re-batching by default (fresh
    # in-batch negative sets every epoch, deterministic per seed); the
    # v1 fixed file-order cycle stays one flag away (the A/B)
    if args.fixed_batches:
        print(f"    [W20.15 A/B] FIXED file-order batches (the v1 "
              f"regime; seed only fixes the val slice)")
        train_iter = _cycle_batches(train_pairs, args.batch_pairs)
    else:
        train_iter = epoch_batches(train_pairs, args.batch_pairs,
                                   seed=args.shuffle_seed)
    consumed = {"batches": 0}   # W20.15 — the coverage tally

    def _tallied(source):
        while True:
            batch = next(source)
            consumed["batches"] += 1
            yield batch

    train_iter = _tallied(train_iter)

    # W22 — the curated hard-negative feed: the label-0 rows EXCLUDING
    # the held-out control slice (the first n_ctrl rows after the seeded
    # shuffle — [6]'s control line stays untouched by training), encoded
    # as single b-SIDES (the loss's candidate columns: an anchor must
    # rank its OWN text2 above a curated non-pair's text2). Capped at
    # 2048 sides (75 pools x 8 = 600 uses — one epoch, no reuse; the
    # cap is an encode-time guard, not a semantic one). The feed rides
    # the same epoch-rebatch regime (a different seed so the negative
    # order never correlates with the pair order).
    n_ctrl = min(args.val_pairs, len(val_pairs)) if val_pairs else 0
    neg_iter = None
    if args.neg_pairs and not args.self_test and len(neg_texts) > n_ctrl + 1:
        if args.neg_pairs < 2:
            print("    [W22] --neg-pairs < 2 — the shared-negative design "
                  "needs a batch (training on in-batch negatives only)")
        else:
            cap = 2048
            neg_train = neg_texts[n_ctrl:n_ctrl + cap]
            neg_sides = []
            for _t1, t2 in neg_train:
                ids = tokenizer.encode(t2, add_special_tokens=False)
                if args.max_tokens:
                    ids = ids[:args.max_tokens]
                if ids:
                    neg_sides.append(_host_pinned(ids))
            if len(neg_sides) >= args.neg_pairs:
                neg_iter = epoch_batches(neg_sides, args.neg_pairs,
                                         seed=args.shuffle_seed + 1)
                print(f"    [W22] hard negatives: {len(neg_sides)} curated "
                      f"label-0 sides (of {len(neg_texts) - n_ctrl} "
                      f"non-control rows), {args.neg_pairs} per pool")
            else:
                print("    [W22] too few encodable label-0 sides — "
                      "training on in-batch negatives only")
    elif args.neg_pairs and args.self_test:
        print("    [W22] the self-test carries no label-0 rows — no "
              "curated negatives (mechanics path)")

    # [3.5] — the data-quality profile (text-only, no model): the
    # lexical-overlap distribution of the positives vs the label-0
    # controls — "is the data good" answered with numbers
    if pair_texts:
        pos_prof = _overlap_profile(pair_texts)
        print("    [W21 data] positives: overlap mean "
              f"{pos_prof['mean']:.3f} median {pos_prof['median']:.3f} "
              f"(hi {pos_prof['frac_hi']:.0%} / mid {pos_prof['frac_mid']:.0%}"
              f" / lo {pos_prof['frac_lo']:.0%}), mean pair len "
              f"{pos_prof['mean_len']:.0f} words")
        neg_prof = _overlap_profile(neg_texts)
        if neg_prof:
            print("    [W21 data] label-0:    overlap mean "
                  f"{neg_prof['mean']:.3f} median {neg_prof['median']:.3f} "
                  f"(hi {neg_prof['frac_hi']:.0%} / mid "
                  f"{neg_prof['frac_mid']:.0%} / lo {neg_prof['frac_lo']:.0%})"
                  f", mean pair len {neg_prof['mean_len']:.0f} words")
            print("    [W21 data] the pos-vs-label0 overlap separation "
                  f"{pos_prof['mean'] - neg_prof['mean']:+.3f} "
                  f"(0 = the pairs carry no surface signal — only a "
                  f"content-addressed write can separate them)")

    print(f"\n[4] Training the gates ({args.max_steps} steps, "
          f"batch {args.batch_pairs}, tau {args.tau}, lr {args.lr})...")
    cfg = GatesTrainConfig(
        lr=args.lr, tau=args.tau, max_steps=args.max_steps,
        batch_pairs=args.batch_pairs,
        freeze_read_gate=not args.train_read_gate,
        gate_init=args.gate_init if args.gate_init != 0 else None,
        refresh_every=args.refresh_every,
        delta_loss=not args.absolute_loss,
        train_layers=_parse_layer_spec(args.train_layers),
        neg_pairs=(args.neg_pairs if neg_iter is not None else 0),
        memopt=False if args.no_memopt else None)
    if args.no_memopt:
        print("    [A/B] plain autograd replay (the W17 OOM path) — "
              "boundary states retained")
    else:
        print("    [W18] exact-gradient memory-safe replay "
              "(O(state + k/v), no boundary-state chain)")
    print(f"    [W19] gate_init {args.gate_init:+.2f} (saddle escape), "
          f"refresh_every {args.refresh_every} (capture pool), "
          f"{'delta' if not args.absolute_loss else 'absolute'} loss "
          f"(retrieval-frame matched)")
    print(f"    [W20.15] {'FIXED file-order' if args.fixed_batches else 'epoch-rebatched'} "
          f"batches, seed {args.shuffle_seed}; train {len(train_pairs)} / "
          f"val {len(val_pairs)} (seeded random split); held-out eval "
          f"{min(args.val_pairs, len(val_pairs)) if val_pairs else 0} "
          f"pairs + the label-0 curated control")
    if neg_iter is not None:
        print(f"    [W22] loss: in-batch negatives + {args.neg_pairs} "
              f"curated hard-negative columns per pool (the control "
              f"slice stays held out)")
    else:
        print("    [W22] loss: in-batch negatives only (the W21 "
              "regime; pass --neg-pairs 8 for the curated columns)")

    def cache_factory():
        return TQCache(config=model.config, bits=args.bits, online=True)

    t0 = time.perf_counter()
    cfg = train_gates(model, train_iter, system, cache_factory, cfg,
                      negatives=neg_iter)
    print(f"    done in {time.perf_counter() - t0:.0f}s")
    # W20.15 — the coverage tally: how much of the train split this run
    # actually consumed (the box's v1 run: 600 of 14,469 = 4.1% — the
    # single most under-reported number of the whole regime)
    used = consumed["batches"] * args.batch_pairs
    n_train = len(train_pairs)
    print(f"    [W20.15] coverage: {consumed['batches']} capture batches "
          f"~ {used} pair-uses of {n_train} train pairs "
          f"({100.0 * min(used, n_train) / max(1, n_train):.1f}% unique "
          f"this run; the tally includes the trailing prefetch)")

    # W21 — THE ARTIFACT FIRST (the box's v1 run crashed inside the
    # held-out report and lost 575 s of training): save_gates + the
    # history BEFORE any report; every [6] section is a warn-and-
    # continue try/except. The artifact records the addressing config
    # (provenance: a W21 file names its regime; load_gates refuses the
    # wrong-shape v1 files loudly).
    out = save_gates(inner.m1m2, args.gates_out, extra_meta={
        "pairs_file": args.pairs_file or "self-test",
        "max_steps": args.max_steps, "tau": args.tau,
        "final_loss": cfg.history[-1]["loss"] if cfg.history else -1.0,
        "write_addr": args.write_addr,
        "center_write": not args.no_center_write,
        "addr_blocks": args.addr_blocks,
        "neg_pairs": (args.neg_pairs if neg_iter is not None else 0)})
    print(f"\n[5] Gates artifact: {out}")
    hist_path = os.path.splitext(out)[0] + "_history.json"
    with open(hist_path, "w") as f:
        json.dump(cfg.history, f, indent=1)
    print(f"    history: {hist_path}")

    # W22 — release the [4] residue before the reports: the epoch
    # iterators (their cached batches + the trailing prefetch), and the
    # [2] system cache hand their device blocks back; empty_cache alone
    # only frees UNALLOCATED cached segments (the W21.6 run's 21.6 GiB
    # "allocated" at the first ladder row — live references, not
    # fragmentation). gc.collect() first: the generators' frames hold
    # the staged tensors. (val_pairs STAYS — [6] captures from it; the
    # ids are ~1 MiB.)
    run_ladder = bool(val_pairs) and not args.no_m1m2
    train_iter = None
    neg_iter = None
    cache = None
    system_ids = None
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if run_ladder:
        print("\n[6] Held-out discrimination ladder (W21: one capture, "
              "every addressing config replayed offline)...")
        module = inner.m1m2
        n_eval = min(args.val_pairs, len(val_pairs))
        # the ladder: the run's config first, then the kappa ladder and
        # the corners (all replayed from the SAME captured k/v)
        ladder = [(f"run config ({args.write_addr}, "
                   f"{'centered' if not args.no_center_write else 'raw'}, "
                   f"kappa={args.addr_blocks})",
                   args.write_addr, not args.no_center_write,
                   args.addr_blocks)] + _ADDR_LADDER

        # ---- capture the held-out group ONCE (streaming; the labels
        # and the ctrl texts ride along). W22: HOST-STAGED — the k/v +
        # m_init land on (pinned) host per side; the ladder replays each
        # side streamed to the module's device ----
        ladder_dev = module.write_gate_k.device
        try:
            val_group = _capture_group(model, system, cache_factory,
                                       val_pairs[:n_eval], host_stage=True)
        except RuntimeError as exc:
            print(f"    [WARN] held-out capture failed ({exc}) — "
                  f"skipping the ladder (artifact already saved)")
            val_group = None
        labels = []
        if val_group is not None and val_texts:
            for t1, t2 in val_texts[:n_eval]:
                ov = lexical_overlap(t1, t2)
                labels.append("high" if ov >= 0.5 else
                              ("med" if ov >= 0.2 else "low"))

        ctrl_ids = []
        ctrl_group = None
        if val_group is not None and neg_texts:
            try:
                for t1, t2 in neg_texts[:n_eval]:
                    c1 = tokenizer.encode(t1, add_special_tokens=False)
                    c2 = tokenizer.encode(t2, add_special_tokens=False)
                    if args.max_tokens:
                        c1, c2 = c1[:args.max_tokens], c2[:args.max_tokens]
                    if not c1 or not c2:
                        continue
                    ctrl_ids.append((c1, c2))
                ctrl_group = _capture_group(
                    model, system, cache_factory,
                    [(_host_pinned(a).to(args.device, non_blocking=True),
                      _host_pinned(b).to(args.device, non_blocking=True))
                     for a, b in ctrl_ids[:n_eval]], host_stage=True)
            except RuntimeError as exc:
                print(f"    [WARN] control capture failed ({exc}) — "
                      f"the ladder runs without the label-0 line")
                ctrl_group = None
                ctrl_ids = []
        else:
            ctrl_group = None

        # ---- the ladder sweep: per config, replay BOTH groups (val
        # first, ctrl after — the units freed between) and report ----
        for (label, write_addr, center_write, addr_blocks) in ladder:
            if val_group is None:
                print("    (ladder skipped — the held-out capture "
                      "failed; the artifact is saved)")
                break
            try:
                module.write_addr = write_addr
                module.center_write = center_write
                module.addr_blocks = addr_blocks
                A1, A2, B1, B2 = _group_units(module, val_group,
                                              dev=ladder_dev)
                sim = _sim_from_units(A1, A2, B1, B2, dev=ladder_dev)
                del A1, A2, B1, B2
                st = sim_matrix_stats(sim)
                print(f"    {label:34s} {_fmt_stats(st)}")
                if labels:
                    rep = bucketed_spread_from_sim(sim, labels)
                    for lbl in ("high", "med", "low"):
                        if lbl in rep["buckets"]:
                            bb = rep["buckets"][lbl]
                            print(f"      overlap {lbl:4s} (n={bb['n']:3d}): "
                                  f"cos_pos {bb['cos_pos']:+.4f} "
                                  f"gap {bb['gap']:+.4f}")
                del sim
                if ctrl_group is not None:
                    C1, C2, D1, D2 = _group_units(module, ctrl_group,
                                                  dev=ladder_dev)
                    csim = _sim_from_units(C1, C2, D1, D2, dev=ladder_dev)
                    del C1, C2, D1, D2
                    cst = sim_matrix_stats(csim)
                    print(f"      label-0 control (the curated negatives, "
                          f"{csim.shape[0]} pairs): cos(pos) "
                          f"{cst['cos_pos']:+.4f} — true-vs-curated gap "
                          f"{st['cos_pos'] - cst['cos_pos']:+.4f}")
                    del csim
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except RuntimeError as exc:
                print(f"    [WARN] ladder row {label!r} failed ({exc}) — "
                      f"continuing (artifact already saved)")
        # restore the run's config on the module (the artifact's regime)
        module.write_addr = args.write_addr
        module.center_write = not args.no_center_write
        module.addr_blocks = args.addr_blocks
        # W22 — REBIND, never `del`: the trailing report below reads the
        # binding, and `del val_group` UNBINDS the name (the box's W21.6
        # traceback — UnboundLocalError at the exact `if val_group is
        # not None` that follows a `del` four lines up). None releases
        # the group's host tensors just as completely, without the
        # unbinding.
        ladder_ran = val_group is not None
        val_group = None
        ctrl_group = None
        ctrl_ids = None
        import gc as _gc
        _gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if ladder_ran:
            print("    (cos(neg) ~0.36 on uncentered rows = the anisotropy "
                  "pedestal; the low-overlap bucket separating on content "
                  "rows but not on positional rows = the position lock, "
                  "measured)")
    elif not val_pairs:
        print("\n[6] (self-test: no held-out pairs — the ladder is a "
              "data-path report)")

    val_pairs = None

    print("\n" + "=" * 60)
    print("GATE FINE-TUNE COMPLETE — re-ingest with --m1m2-gates "
          "--m1m2-mem-size " + str(args.m1m2_mem_size))
    print("=" * 60)
    return 0


def _cycle_batches(pairs, batch):
    """Endless batches (the loop stops at max_steps)."""
    i = 0
    while True:
        batch_pairs = [pairs[(i + j) % len(pairs)] for j in range(batch)]
        i = (i + batch) % len(pairs)
        yield batch_pairs


if __name__ == "__main__":
    raise SystemExit(main())
