#!/usr/bin/env python3
"""finetune_m1m2.py — the §7 M1/M2 gate fine-tune on the GPU box.

Trains ONLY the M1/M2 gate vectors (write_gate_k/write_gate_v, optionally
read_gate) of the W17-wired model, on GENERAL semantic-similarity pairs
(paraphrase / entailment / QA pairs — MRPC, QQP, PAWS, SNLI, MNLI, STS-B,
SQuAD ...), NEVER on the served corpus. The goal (the handover's design):
similar texts -> similar M1/M2 states -> the §4 vector's M1/M2 units carry
the query-document alignment the S-only vectors lack (measured cos(q, doc)
~0.50-0.58, 50% hit rate).

THE DATA CONTRACT (corpus-independent): a JSONL of
    {"text1": "...", "text2": "...", "label": 1}
(label 1 = similar; label 0 rows are skipped in v1 — the InfoNCE takes
its negatives IN-BATCH from the other pairs). Convert any general dataset
to this format offline (the box does not need internet at train time);
EXAMPLES of supported sources: MRPC/QQP/PAWS (paraphrase), SNLI/MNLI
(entailment=1, contradiction=0), STS-B (score>4 -> 1), SQuAD
(question, answer sentence).

The deployment protocol (ORDER MATTERS): train -> save gates ->
RE-INGEST with the trained gates (run_ingestion.py --m1m2-gates ...
--m1m2-mem-size SAME) -> run_index.py -> query (run_query.py
--m1m2-gates ...). The snapshot geometry (m1_shape in system_state.npz)
pins the mem_size; the query-side loader fails loudly on drift.

Examples:
    # general pairs from a converted dataset
    python3 scripts/gpu/finetune_m1m2.py --pairs-file mrpc_pairs.jsonl \
        --gates-out disk/m1m2_gates.npz --m1m2-mem-size 1024

    # mechanics smoke test on synthetic pairs (NOT training data)
    python3 scripts/gpu/finetune_m1m2.py --self-test --max-steps 5
"""
from __future__ import annotations

import argparse
import json
import os
import time

import torch

from _bootstrap import DEFAULTS, boot, load_model

boot()

from ingest import prefill_system  # noqa: E402
from m1m2_finetune import (GatesTrainConfig, cosine_spread,  # noqa: E402
                            prefill_capture_calls, replay_states,
                            save_gates, train_gates)
from tq_cache import TQCache  # noqa: E402


def load_pairs(path: str, limit: int = 0):
    """The positive pairs from a general-similarity JSONL."""
    pairs = []
    with open(path) as f:
        for i, line in enumerate(f):
            if limit and len(pairs) >= limit:
                break
            row = json.loads(line)
            if int(row.get("label", 1)) != 1:
                continue
            t1, t2 = row.get("text1"), row.get("text2")
            if t1 and t2:
                pairs.append((t1, t2))
    if len(pairs) < 2:
        raise SystemExit(
            f"{path}: need >= 2 positive pairs (in-batch negatives); "
            f"got {len(pairs)}")
    return pairs


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


def _host_pinned(ids) -> torch.Tensor:
    """W20.11.2 — the loader pinning: token-id tensors are built on the
    HOST as PINNED (page-locked) tensors, so the capture loop's
    per-refresh H2D staging rides async DMA (non_blocking). Transport
    only — the values are bit-identical to the plain host tensors;
    without CUDA the plain tensor is returned (same values, no pin)."""
    t = torch.tensor([ids])
    return t.pin_memory() if torch.cuda.is_available() else t


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
                         "are per-layer); ingest+query must use the SAME "
                         "value")
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
    ap.add_argument("--train-read-gate", action="store_true",
                    help="unfreeze read_gate too (the contrastive loss "
                         "itself never sees it — the read affects "
                         "generation, not the state; pair this with a "
                         "next-token objective in a later wave)")
    ap.add_argument("--val-frac", type=float, default=0.1)
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
    if args.self_test:
        raw = synthetic_pairs(64, vocab=tokenizer.vocab_size)
        pairs = [(_host_pinned(p[0]), _host_pinned(p[1])) for p in raw]
        print("    [SELF-TEST] 64 synthetic pairs (mechanics only)")
    else:
        if not args.pairs_file:
            raise SystemExit("--pairs-file is required (or --self-test)")
        texts = load_pairs(args.pairs_file, limit=args.limit_pairs)
        print(f"    {len(texts)} positive pairs from {args.pairs_file}")
        pairs = []
        for t1, t2 in texts:
            ids1 = tokenizer.encode(t1, add_special_tokens=False)
            ids2 = tokenizer.encode(t2, add_special_tokens=False)
            if args.max_tokens:
                ids1, ids2 = ids1[:args.max_tokens], ids2[:args.max_tokens]
            if not ids1 or not ids2:
                continue
            pairs.append((_host_pinned(ids1), _host_pinned(ids2)))
    n_val = max(2, int(len(pairs) * args.val_frac)) if len(pairs) > 4 else 0
    # W20.11.2: the held-out prefills' ids stage ONCE here (the same
    # pinned non_blocking transport — [5] runs the unchanged path)
    if n_val:
        val_pairs = [(a.to(args.device, non_blocking=True),
                      b.to(args.device, non_blocking=True))
                     for a, b in pairs[:n_val]]
    else:
        val_pairs = []
    train_iter = _cycle_batches(pairs[n_val:] if n_val else pairs,
                                args.batch_pairs)

    print(f"\n[4] Training the gates ({args.max_steps} steps, "
          f"batch {args.batch_pairs}, tau {args.tau}, lr {args.lr})...")
    inner = getattr(model, "model", model)
    cfg = GatesTrainConfig(
        lr=args.lr, tau=args.tau, max_steps=args.max_steps,
        batch_pairs=args.batch_pairs,
        freeze_read_gate=not args.train_read_gate,
        gate_init=args.gate_init if args.gate_init != 0 else None,
        refresh_every=args.refresh_every,
        delta_loss=not args.absolute_loss,
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

    def cache_factory():
        return TQCache(config=model.config, bits=args.bits, online=True)

    t0 = time.perf_counter()
    cfg = train_gates(model, train_iter, system, cache_factory, cfg)
    print(f"    done in {time.perf_counter() - t0:.0f}s")

    if val_pairs:
        print("\n[5] Held-out discrimination spread...")
        module = inner.m1m2
        states = []
        with torch.no_grad():
            for ids_a, ids_b in val_pairs[:32]:
                ps = []
                for ids in (ids_a, ids_b):
                    c = cache_factory()
                    calls, m1i, m2i = prefill_capture_calls(
                        model, ids, c, system)
                    m1, m2 = replay_states(module, calls, m1i, m2i)
                    # W19: the spread over the DELTAS — the centered
                    # retrieval frame's geometry (cos of the full states
                    # is ~+1 trivially off the shared m_init)
                    ps.append((m1 - m1i, m2 - m2i))
                states.append(ps)
        m1_a = torch.stack([s[0][0] for s in states])
        m2_a = torch.stack([s[0][1] for s in states])
        m1_b = torch.stack([s[1][0] for s in states])
        m2_b = torch.stack([s[1][1] for s in states])
        pos, neg = cosine_spread(m1_a, m2_a, m1_b, m2_b)
        print(f"    cos(true pairs) {pos:+.4f} vs cos(shifted) {neg:+.4f} "
              f"— gap {pos - neg:+.4f} (the discrimination signal; "
              f"the §4 vector's M1/M2 units carry this)")

    out = save_gates(inner.m1m2, args.gates_out, extra_meta={
        "pairs_file": args.pairs_file or "self-test",
        "max_steps": args.max_steps, "tau": args.tau,
        "final_loss": cfg.history[-1]["loss"] if cfg.history else -1.0})
    print(f"\n[6] Gates artifact: {out}")
    hist_path = os.path.splitext(out)[0] + "_history.json"
    with open(hist_path, "w") as f:
        json.dump(cfg.history, f, indent=1)
    print(f"    history: {hist_path}")

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
