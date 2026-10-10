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
                    help="the memories' slot count (spec default 128; the "
                         "handover's retrieval experiments: 1024/4096/8192; "
                         "ingest+query must use the SAME value)")
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

    print(f"\n[1] Loading model (m1m2_mem_size={args.m1m2_mem_size})...")
    model, _ = load_model(args.artifacts_dir, args.heads_dir,
                          args.model_name, args.device,
                          use_m1m2=not args.no_m1m2,
                          m1m2_mem_size=args.m1m2_mem_size)

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    print("\n[2] Building the system reset point (the delta zero)...")
    system_ids = torch.tensor(
        [tokenizer.encode(args.system_prompt, add_special_tokens=False)],
        dtype=torch.long).to(args.device)
    cache = TQCache(config=model.config, bits=args.bits, online=True)
    system = prefill_system(model.model, system_ids, cache,
                            system_ref="m1m2-finetune", bits=args.bits)
    print(f"    system {system.reference()}; "
          f"m1 {'present' if system.m1_codes is not None else 'zero-init'}")

    print("\n[3] Loading pairs...")
    if args.self_test:
        raw = synthetic_pairs(64, vocab=tokenizer.vocab_size)
        pairs = [(torch.tensor([p[0]]).to(args.device),
                  torch.tensor([p[1]]).to(args.device)) for p in raw]
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
            pairs.append((torch.tensor([ids1]).to(args.device),
                          torch.tensor([ids2]).to(args.device)))
    n_val = max(2, int(len(pairs) * args.val_frac)) if len(pairs) > 4 else 0
    val_pairs = pairs[:n_val] if n_val else []
    train_iter = _cycle_batches(pairs[n_val:] if n_val else pairs,
                                args.batch_pairs)

    print(f"\n[4] Training the gates ({args.max_steps} steps, "
          f"batch {args.batch_pairs}, tau {args.tau}, lr {args.lr})...")
    inner = getattr(model, "model", model)
    cfg = GatesTrainConfig(
        lr=args.lr, tau=args.tau, max_steps=args.max_steps,
        batch_pairs=args.batch_pairs,
        freeze_read_gate=not args.train_read_gate)

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
                    ps.append((m1, m2))
                states.append(ps)
        m1_a = torch.stack([s[0][0] for s in states])
        m2_a = torch.stack([s[0][1] for s in states])
        m1_b = torch.stack([s[1][0] for s in states])
        m2_b = torch.stack([s[1][1] for s in states])
        pos, neg = cosine_spread(m1_a, m2_a, m1_b, m2_b)
        print(f"    cos(true pairs) {pos:+.4f} vs cos(shifted) {neg:+.4f} "
              f"— gap {pos - neg:+.4f} (the discrimination signal)")

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
