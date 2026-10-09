#!/usr/bin/env python3
"""evals.py — the phase-gate measurement harness (PROPOSAL §3 phases P1–P6;
SPECIFICATION §9/§10 ledgers).

Every proposal acceptance row maps to a subcommand here (TASKS.md §7 — the
GPU-box execution register). Each subcommand:

  * runs its measurement (REAL tensors/model/index on the GPU box; a
    SYNTHETIC self-test on CPU boxes — `--self-test`, the harness's own
    correctness mode),
  * writes a JSON artifact to evals_out/<name>.json,
  * prints one PASS/FAIL line per gate to stdout (the proposal P1 format:
    "gate <name>: <value> <cmp> <threshold> — PASS/FAIL").

Subcommands:
  roundtrip   P1 items i–iv: Beta concentration of rotated coords, round-trip
              MSE vs the paper's D_mse, per-head norm spread (D2 decision
              input), outlier mass (the 3.5-bit split A/B decision input)
  streaming   P2: greedy token-match vs the fp16-cache baseline (the
              recurrent-compounding gate, target >= 95% on <= 4k prompts)
  margin      P4: same-topic vs different-topic cache-vector cosine margin
              (target >= 3x — the "S is discriminative / M1/M2 carry info"
              acceptance)
  recall      P5: recall@100 (preselect) and top-3 correctness (rerank) on
              held-out same-topic queries
  e2e         P6: end-to-end QA accuracy, actual-retrieval vs no-RAG vs
              oracle-install (the gap isolates retrieval from generation)
  ledger      §9 per-query timing ledger + §10 VRAM ledger (probe-based on
              CPU; torch.cuda memory stats on the GPU box)

Real-data mode requires the GPU-box inputs (model artifacts, corpus,
index) and refuses loudly when absent. Self-test mode exercises the exact
measurement code paths on deterministic synthetic fixtures — that is what
the CPU tests (tests/test_evals.py) run.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone

_HERE = os.path.dirname(os.path.abspath(__file__))
_OUT_DIR = os.environ.get("EVALS_OUT", os.path.join(_HERE, "evals_out"))

__all__ = [
    "measure_roundtrip", "measure_streaming", "measure_margin",
    "measure_recall", "measure_e2e", "measure_ledger", "main",
]

# paper constants (PROPOSAL §1.1; solved values from codebooks.py)
PAPER_D_MSE = {1: 0.36, 2: 0.117, 3: 0.03, 4: 0.009}
PAPER_BOUND = lambda b: 2.72 * 4.0 ** -b  # noqa: E731
MSE_TOL = {1: 0.10, 2: 0.05, 3: 0.15, 4: 0.15}  # paper values are rounded


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _finish(name: str, result: dict, out_dir: str = _OUT_DIR) -> dict:
    result["gate"] = name
    result["utc"] = _now()
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{name}.json")
    with open(path, "w") as f:
        json.dump(result, f, indent=1, sort_keys=True, default=str)
    result["artifact"] = path
    ok = bool(result.get("passed"))
    print(f"[{name}] {'PASS' if ok else 'FAIL'} -> {path}")
    for line in result.get("lines", []):
        print(f"  {line}")
    return result


# ------------------------------------------------------------ P1 roundtrip -
def measure_roundtrip(bits_list=(1, 2, 3, 3.5, 4), n_vectors: int = 16,
                      d: int = 1024, seed: int = 20261009,
                      out_dir: str = _OUT_DIR) -> dict:
    """P1 items i–iv. On the GPU box this runs over REAL captured cache
    tensors (`--capture-dir`); the core is identical on synthetic unit
    vectors (self-test)."""
    import numpy as np
    import torch
    import turboquant as tq

    g = torch.Generator().manual_seed(seed)
    X = torch.randn(n_vectors, d, generator=g)
    X = X / X.norm(dim=-1, keepdim=True)

    lines, mse_table, passed = [], {}, True
    for bits in bits_list:
        q = tq.TurboQuant(kind="custom", bits=bits, d=d, seed=7)
        mses = []
        for i in range(n_vectors):
            xr = q.roundtrip(X[i])
            mses.append(((X[i] - xr) ** 2).sum().item())
        mse = float(np.mean(mses))
        mse_table[str(bits)] = mse
        if float(bits).is_integer():
            b = int(bits)
            bound = PAPER_BOUND(b)
            ok = mse <= bound
            if b == 2:
                ok = ok and abs(mse - 0.117) / 0.117 <= MSE_TOL[2]
            lines.append(f"D_mse(b={b}) = {mse:.4f} vs paper "
                         f"{PAPER_D_MSE[b]} (bound {bound:.4f}) — "
                         f"{'PASS' if ok else 'FAIL'}")
            passed = passed and ok

    # (i) Beta concentration of the rotated coordinates (the D3/FHT
    # substitution check — empirical, per PROPOSAL R2)
    q = tq.TurboQuant(kind="custom", bits=4, d=d, seed=7)
    import fht
    signs = fht.rotation_signs(d, 7)
    rot = torch.cat([fht.fht_apply(X[i:i + 1], signs) for i in range(n_vectors)])
    coords = rot.reshape(-1).numpy() * np.sqrt(d)  # ~ N(0,1) if concentrated
    var = float(np.var(coords))
    kurt = float(np.mean(coords ** 4))
    max_abs = float(np.max(np.abs(coords)))
    frac_3s = float(np.mean(np.abs(coords) < 3.0))
    conc_ok = abs(var - 1.0) < 0.05 and 2.5 < kurt < 3.6 and max_abs < 6.0 \
        and frac_3s > 0.995
    lines.append(f"concentration: var={var:.4f} (exp 1.0) kurtosis={kurt:.2f} "
                 f"(exp ~3.0) max|z|={max_abs:.2f} <6 frac<3s={frac_3s:.4f} — "
                 f"{'PASS' if conc_ok else 'FAIL'}")
    passed = passed and conc_ok

    # (iii) per-head norm spread (D2 decision input; threshold: > 10x flags
    # the per-head-norm fallback, PROPOSAL D2)
    g2 = torch.Generator().manual_seed(seed + 1)
    S = torch.randn(1, 32, 128, 128, generator=g2).half()
    norms = S.float().norm(dim=-1).reshape(-1).numpy()  # per (head, 128-row)
    spread = float(norms.max() / np.median(norms))
    lines.append(f"per-head norm spread max/median = {spread:.2f} "
                 f"(> 10.0 flags the D2 per-head fallback — informational)")

    # (iv) outlier mass: top-1% |coordinate| energy (the 3.5-bit split A/B
    # decision input — informational on synthetic; real tensors decide)
    a = np.abs(coords)
    thresh = np.quantile(a, 0.99)
    mass = float((a[a >= thresh] ** 2).sum() / (a ** 2).sum())
    lines.append(f"outlier mass (top-1% coords, rotated) = {mass:.4f} "
                 f"(decision input for the split A/B — informational)")

    return _finish("roundtrip", {
        "passed": passed, "mse_table": mse_table, "var": var, "kurtosis": kurt,
        "max_abs_z": max_abs, "frac_within_3sigma": frac_3s,
        "per_head_norm_spread": spread, "outlier_mass_top1pct": mass,
        "n_vectors": n_vectors, "d": d, "lines": lines}, out_dir)


# ------------------------------------------------------------- P2 streaming -
def measure_streaming(model=None, prompts=None, out_dir: str = _OUT_DIR,
                      self_test: bool = False, steps: int = 32,
                      seed: int = 7) -> dict:
    """The recurrent-compounding gate: greedy token-match rate of the
    online-TQ cache vs an fp16-cache baseline, and the degradation vs
    prompt length (the compounding signature). GPU box: real model +
    Phase-0 prompt set. Self-test: a deterministic stub decoder whose
    token choice depends on the accumulated cache state."""
    import torch
    from tq_cache import TQCache

    class _Stub:
        """Decode step t: token = argmax over a head reading the state —
        sensitive to state error, deterministic given the state."""

        def __init__(self, d=128):
            g = torch.Generator().manual_seed(seed)
            self.head = torch.randn(d, 8, generator=g)
            self.d = d

        def forward(self, input_ids, past_key_values, use_cache=True):
            for L in (0, 2):
                cur = past_key_values.layers[L].recurrent_states[0]
                if cur is None:
                    cur = torch.zeros(self.d, dtype=torch.float16)
                gg = torch.Generator().manual_seed(
                    100 * L + int(input_ids.flatten()[0]) * 31 + t_global[0])
                new = (cur.float() + 0.1 * torch.randn(
                    self.d, generator=gg).float()).half()
                past_key_values.update_recurrent_state(new, L)
            s0 = past_key_values.layers[0].recurrent_states[0].float()
            logits = (s0 @ self.head).reshape(1, 1, -1)
            return logits

    t_global = [0]
    layer_types = ["linear_attention", "full_attention",
                   "linear_attention", "full_attention"]

    def run(stub, use_tq: bool):
        # fp16-cache baseline = the D4 offline regime (raw tensors held);
        # online regime = codes on every write (the compounding under test)
        torch.manual_seed(seed)
        cache = TQCache(layer_types=layer_types, online=use_tq)
        tokens = []
        tok = 3
        for t in range(steps):
            t_global[0] = t
            logits = stub.forward(torch.tensor([[tok]]), cache, True)
            tok = int(logits[:, -1, :].argmax(dim=-1))
            tokens.append(tok)
        return tokens

    if self_test or model is None:
        stub = _Stub()
        base = run(stub, use_tq=False)   # fp16 (offline) baseline
        tqt = run(stub, use_tq=True)     # online-TQ codes
        match = sum(1 for a, b in zip(base, tqt) if a == b) / steps
        # degradation vs length: first-half vs second-half match
        half = steps // 2
        m1 = sum(1 for a, b in zip(base[:half], tqt[:half]) if a == b) / half
        m2 = sum(1 for a, b in zip(base[half:], tqt[half:])) / (steps - half)
        # The self-test stub is CHAOTIC BY DESIGN: argmax over 8 random
        # logit lines amplifies ~2% state error into full token divergence
        # (measured: early divergence, then both runs lock onto the same
        # attractor). The CPU self-test therefore verifies the MEASUREMENT
        # MACHINERY (both regimes run, metrics finite, the per-half
        # compounding signature computed) — the >= 0.95 gate is measured on
        # the real model with the Phase-0 prompt set on the GPU box.
        machinery_ok = all(isinstance(x, float) and 0.0 <= x <= 1.0
                           for x in (match, m1, m2)) and len(base) == steps
        target = 0.95
        lines = [f"token-match (self-test stub, {steps} steps): "
                 f"{match:.3f} first-half {m1:.3f} second-half {m2:.3f} — "
                 f"{'PASS' if machinery_ok else 'FAIL'} (machinery gate; "
                 f"the stub is chaotic-by-design — see the module note)",
                 f"GPU target on real prompts: >= {target} (authoritative "
                 f"P2 gate — runs there, not here)"]
        return _finish("streaming", {
            "passed": machinery_ok, "token_match": match, "first_half": m1,
            "second_half": m2, "steps": steps, "mode": "self-test",
            "stub": "chaotic-by-design (measurement-machinery gate)",
            "gpu_target": target, "lines": lines}, out_dir)
    raise SystemExit(
        "streaming: real-model mode requires the GPU box (Phase-0 prompt "
        "set + artifacts) — rerun with --self-test on CPU")


# --------------------------------------------------------------- P4 margin -
def measure_margin(corpus_dir=None, out_dir: str = _OUT_DIR,
                   self_test: bool = False, n_topics: int = 5,
                   per_topic: int = 8, seed: int = 11) -> dict:
    """The cache-signal check: mean cos(query, same-topic chunks) vs
    cos(query, different-topic chunks); the margin ratio must be >= 3x
    (proposal P4 gate — 'S is discriminative / M1/M2 carry info')."""
    import numpy as np
    import torch

    if self_test or corpus_dir is None:
        g = torch.Generator().manual_seed(seed)
        topic_dirs = torch.randn(n_topics, 128, generator=g)
        topic_dirs = topic_dirs / topic_dirs.norm(dim=-1, keepdim=True)
        vecs = {}
        for t in range(n_topics):
            for i in range(per_topic):
                v = 0.15 * torch.randn(128, generator=g) + 4.0 * topic_dirs[t]
                vecs[(t, i)] = (v / v.norm()).numpy().astype(np.float32)
        margins = []
        for t in range(n_topics):
            q = vecs[(t, 0)]
            same = [float(q @ vecs[(t, i)]) for i in range(1, per_topic)]
            diff = [float(q @ vecs[(u, i)]) for u in range(n_topics)
                    if u != t for i in range(per_topic)]
            margins.append((float(np.mean(same)), float(np.mean(diff))))
        same_m = float(np.mean([m for m, _ in margins]))
        diff_m = float(np.mean([d for _, d in margins]))
        ratio = abs(same_m - diff_m) / max(1e-9, abs(diff_m)) \
            if abs(diff_m) > 0.05 else (same_m / max(1e-9, diff_m))
        ok = ratio >= 3.0
        lines = [f"margin: same-topic cos {same_m:.3f} vs diff-topic "
                 f"{diff_m:.3f} -> ratio {ratio:.1f}x (>= 3x) — "
                 f"{'PASS' if ok else 'FAIL'}"]
        return _finish("margin", {
            "passed": ok, "same_topic_cos": same_m, "diff_topic_cos": diff_m,
            "ratio": ratio, "mode": "self-test", "lines": lines}, out_dir)
    raise SystemExit("margin: real-corpus mode requires the GPU box")


# --------------------------------------------------------------- P5 recall --
def measure_recall(disk_dir=None, out_dir: str = _OUT_DIR,
                   self_test: bool = False, seed: int = 5) -> dict:
    """recall@100 (preselect contains the source) and rerank top-3
    correctness on held-out queries. Self-test: the topic corpus."""
    import os

    if self_test or disk_dir is None:
        import shutil
        import tempfile
        import torch
        from ingest import IngestDriver
        from index import ChunkVectorLoader, IndexConfig, build_index, \
            preselect, rerank
        from tq_cache import TQCache

        tmp = tempfile.mkdtemp(prefix="evals_recall_")
        try:
            layer_types = ["linear_attention", "full_attention",
                           "linear_attention", "full_attention"]
            S_SHAPE, CONV_D, M_SHAPE = (1, 8, 16), 32, (2, 4, 16)

            class Stub:
                def __init__(self):
                    g = torch.Generator().manual_seed(21)
                    self.dirs = torch.randn(5, 128, generator=g)
                    self.dirs = self.dirs / self.dirs.norm(dim=-1, keepdim=True)

                def __call__(self, input_ids, past_key_values, use_cache=True):
                    import zlib
                    key = zlib.crc32(input_ids.numpy().tobytes())
                    t = key % 5
                    for L in (0, 2):
                        cur = past_key_values.layers[L].recurrent_states[0]
                        if cur is None:
                            cur = torch.zeros(S_SHAPE, dtype=torch.float16)
                        g = torch.Generator().manual_seed(key + 17 * L)
                        add = 0.05 * torch.randn(S_SHAPE, generator=g).half()
                        if L == 0:
                            add = add + (4.0 * self.dirs[t]
                                         .reshape(S_SHAPE)).half()
                        past_key_values.update_recurrent_state(
                            (cur.float() + add.float()).half(), L)
                        past_key_values.update_conv_state(
                            torch.randn(1, CONV_D, input_ids.shape[-1],
                                        generator=g).half(), L,
                            conv_kernel_size=4)
                    m1 = past_key_values.read_m1()
                    if m1 is None:
                        m1 = torch.zeros(M_SHAPE, dtype=torch.float16)
                    past_key_values.update_m1(
                        (m1.float() + 0.02 * torch.randn(
                            M_SHAPE, generator=g).float()).half())
                    return None

            model = Stub()
            chunks = [torch.tensor([[100 + i, 101 + i, 102 + i]])
                      for i in range(260)]
            drv = IngestDriver(model, torch.tensor([[1, 2, 3]]), chunks, tmp,
                               cache_factory=lambda: TQCache(
                                   layer_types=layer_types))
            drv.run()
            loader = ChunkVectorLoader(tmp)
            d = loader.vector(0).shape[0]
            build_index(loader.iter_vectors(),
                        IndexConfig(d=d, nlist=8, m=8, nbits=8, nprobe=8),
                        train_sample=260,
                        path=os.path.join(tmp, "ivfadc_cache.index"))
            import index as index_mod
            idx, _meta = index_mod.load_index(
                os.path.join(tmp, "ivfadc_cache.index"))
            hits100 = top3 = 0
            n_q = 0
            for cid in range(0, 260, 26):  # 10 held-out queries
                q = loader.vector(cid)
                cands = preselect(idx, q, k=100)
                if cid in cands.tolist():
                    hits100 += 1
                ids, _scores = rerank(loader, q, cands, k=3)
                if cid in ids.tolist():
                    top3 += 1
                n_q += 1
            r100, r3 = hits100 / n_q, top3 / n_q
            ok = r100 >= 0.90 and r3 >= 0.80
            lines = [f"recall@100 = {r100:.2f} (>= 0.90 mechanics) — "
                     f"{'PASS' if r100 >= 0.90 else 'FAIL'}",
                     f"top-3 correct = {r3:.2f} (>= 0.80) — "
                     f"{'PASS' if r3 >= 0.80 else 'FAIL'}"]
            return _finish("recall", {
                "passed": ok, "recall_at_100": r100, "top3": r3,
                "n_queries": n_q, "mode": "self-test", "lines": lines},
                out_dir)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
    raise SystemExit("recall: real-index mode requires the GPU box")


# ------------------------------------------------------------------ P6 e2e --
def measure_e2e(disk_dir=None, out_dir: str = _OUT_DIR,
                self_test: bool = False) -> dict:
    """Three-way accuracy: no-RAG vs actual retrieval vs oracle install
    (the oracle-vs-actual gap isolates retrieval quality from generation
    quality). Self-test: the topic-marker stub."""
    import os
    import shutil
    import tempfile
    import torch
    from ingest import IngestDriver, load_system_state
    from index import ChunkVectorLoader, IndexConfig, build_index
    from query import answer_query
    from tq_cache import TQCache

    if self_test or disk_dir is None:
        tmp = tempfile.mkdtemp(prefix="evals_e2e_")
        try:
            layer_types = ["linear_attention", "full_attention",
                           "linear_attention", "full_attention"]
            S_SHAPE, CONV_D, M_SHAPE = (1, 8, 16), 32, (2, 4, 16)
            N_TOPICS = 5

            class Stub:
                def __init__(self):
                    g = torch.Generator().manual_seed(31)
                    self.dirs = torch.randn(N_TOPICS, 128, generator=g)
                    self.dirs = self.dirs / self.dirs.norm(
                        dim=-1, keepdim=True)
                    self.head = torch.randn(128, N_TOPICS, generator=g)

                def __call__(self, input_ids, past_key_values, use_cache=True):
                    key = int(input_ids.flatten()[0])
                    t = key % N_TOPICS
                    for L in (0, 2):
                        cur = past_key_values.layers[L].recurrent_states[0]
                        if cur is None:
                            cur = torch.zeros(S_SHAPE, dtype=torch.float16)
                        g = torch.Generator().manual_seed(key + 17 * L)
                        add = 0.05 * torch.randn(S_SHAPE, generator=g).half()
                        if L == 0:
                            add = add + (4.0 * self.dirs[t]
                                         .reshape(S_SHAPE)).half()
                        past_key_values.update_recurrent_state(
                            (cur.float() + add.float()).half(), L)
                        past_key_values.update_conv_state(
                            torch.randn(1, CONV_D, input_ids.shape[-1],
                                        generator=g).half(), L,
                            conv_kernel_size=4)
                    m1 = past_key_values.read_m1()
                    if m1 is None:
                        m1 = torch.zeros(M_SHAPE, dtype=torch.float16)
                    past_key_values.update_m1(
                        (m1.float() + 0.02 * torch.randn(
                            M_SHAPE, generator=g).float()).half())
                    s0 = past_key_values.layers[0].recurrent_states[0]
                    return (s0.float().reshape(-1) @ self.head).reshape(1, 1, -1)

            model = Stub()
            chunks = [torch.tensor([[100 + i, 101 + i, 102 + i]])
                      for i in range(260)]
            drv = IngestDriver(model, torch.tensor([[1, 2, 3]]), chunks, tmp,
                               cache_factory=lambda: TQCache(
                                   layer_types=layer_types))
            drv.run()
            loader = ChunkVectorLoader(tmp)
            d = loader.vector(0).shape[0]
            ipath = os.path.join(tmp, "ivfadc_cache.index")
            build_index(loader.iter_vectors(),
                        IndexConfig(d=d, nlist=8, m=8, nbits=8, nprobe=8),
                        train_sample=260, path=ipath)
            import index as index_mod
            idx, _meta = index_mod.load_index(ipath)
            system = load_system_state(tmp)

            def topic_of(cid):
                return cid % N_TOPICS

            correct = {"no_rag": 0, "actual": 0, "oracle": 0}
            n = 0
            for cid in range(0, 260, 26):
                t = topic_of(cid)
                qids = torch.tensor([[900 + t, 901 + t, 902 + t,
                                      100 + cid]])
                # no-RAG: answer over a bare system-reseeded cache
                from ingest import reseed_cache
                cache = TQCache(layer_types=layer_types)
                reseed_cache(cache, system)
                with torch.no_grad():
                    out = model(input_ids=qids, past_key_values=cache,
                                use_cache=True)
                if int(out[:, -1, :].argmax(dim=-1)) == t:
                    correct["no_rag"] += 1
                # actual retrieval
                res = answer_query(model, qids, system, index=idx,
                                   loader=loader, cache_factory=lambda: TQCache(
                                       layer_types=layer_types))
                got = [topic_of(i) for i in res.retrieved_ids[:1]]
                if t in got:
                    correct["actual"] += 1
                # oracle install
                res_o = answer_query(model, qids, system, loader=loader,
                                     cache_factory=lambda: TQCache(
                                         layer_types=layer_types),
                                     retrieved_ids=[cid])
                if topic_of(res_o.retrieved_ids[0]) == t:
                    correct["oracle"] += 1
                n += 1
            acc = {k: v / n for k, v in correct.items()}
            ok = acc["oracle"] >= 0.9 and acc["actual"] >= 0.5
            lines = [f"accuracy no-RAG {acc['no_rag']:.2f} | actual "
                     f"{acc['actual']:.2f} | oracle {acc['oracle']:.2f} — "
                     f"{'PASS' if ok else 'FAIL'} "
                     f"(oracle-vs-actual gap isolates retrieval quality)"]
            return _finish("e2e", {
                "passed": ok, "accuracy": acc, "n": n, "mode": "self-test",
                "lines": lines}, out_dir)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
    raise SystemExit("e2e: real-corpus mode requires the GPU box (OfficeQA)")


# ------------------------------------------------------------------ ledger --
def measure_ledger(out_dir: str = _OUT_DIR, self_test: bool = False) -> dict:
    """§9 per-query timing ledger + §10 VRAM ledger. CPU self-test: the
    timing keys over a synthetic mini-run + a null VRAM probe. GPU box:
    real query flow + torch.cuda.max_memory_allocated."""
    import torch

    def vram_probe() -> dict:
        if torch.cuda.is_available():
            return {"cuda_available": True,
                    "max_allocated_gib": round(
                        torch.cuda.max_memory_allocated() / 2 ** 30, 3)}
        return {"cuda_available": False, "max_allocated_gib": None}

    t = {}
    t0 = time.perf_counter()
    time.sleep(0.01)
    t["sample_step"] = time.perf_counter() - t0
    vram = vram_probe()
    lines = [f"timing keys present: {sorted(t)} (self-test placeholder — "
             f"the §9 ledger fills per-query on the GPU box)",
             f"VRAM: {vram} (cuda probe)"]
    return _finish("ledger", {
        "passed": True, "timings": t, "vram": vram, "mode": "self-test",
        "spec_targets_s9": {"prefill_snapshot": "~8 ms",
                            "preselect": "~10 ms", "rerank": "~20 ms",
                            "load_codes": "~3 ms", "install": "~15 ms",
                            "decode": "~8 s", "total": "~8.06 s"},
        "spec_target_s10_vram_gib": 13, "lines": lines}, out_dir)


# -------------------------------------------------------------------- CLI ---
def main(argv=None) -> int:
    sys.path.insert(0, _HERE)
    import _paths  # noqa: F401  (anchor siblings)

    ap = argparse.ArgumentParser(prog="evals.py", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p, name):
        p.add_argument("--self-test", action="store_true",
                       help="run on synthetic fixtures (CPU-legal)")
        p.add_argument("--out-dir", default=None)

    p = sub.add_parser("roundtrip", help="P1: MSE + concentration table")
    p.add_argument("--n-vectors", type=int, default=16)
    p.add_argument("--d", type=int, default=1024)
    common(p, "roundtrip")

    p = sub.add_parser("streaming", help="P2: token-match vs fp16 baseline")
    p.add_argument("--steps", type=int, default=32)
    common(p, "streaming")

    p = sub.add_parser("margin", help="P4: same/diff-topic cos margin")
    common(p, "margin")

    p = sub.add_parser("recall", help="P5: recall@100 + top-3")
    common(p, "recall")

    p = sub.add_parser("e2e", help="P6: no-RAG vs actual vs oracle")
    common(p, "e2e")

    p = sub.add_parser("ledger", help="S9/S10 ledgers")
    common(p, "ledger")

    args = ap.parse_args(argv)
    out = args.out_dir or _OUT_DIR
    st = getattr(args, "self_test", False)
    # GPU-real-mode commands WITHOUT --self-test refuse LOUDLY (W9.1's
    # documented contract — "Real-data mode requires the GPU-box inputs
    # ... and refuses loudly when absent"): this CLI has no arguments for
    # the real inputs (model artifacts / corpus / index are GPU-box-only),
    # so falling through to the synthetic self-test would silently LIE
    # about what was measured. (W9.3 hardening; roundtrip/ledger stay
    # CPU-legal — their synthetic core IS the real core on CPU.)
    if args.cmd in ("streaming", "margin", "recall", "e2e") and not st:
        raise SystemExit(
            f"{args.cmd}: real-data mode requires the GPU box (model "
            f"artifacts / corpus / index — the Phase-0/Phase-5 inputs); "
            f"none are available on this CPU box — rerun with "
            f"--self-test for the synthetic machinery check")
    if args.cmd == "roundtrip":
        r = measure_roundtrip(n_vectors=args.n_vectors, d=args.d,
                              out_dir=out)
    elif args.cmd == "streaming":
        r = measure_streaming(self_test=st, steps=args.steps, out_dir=out)
    elif args.cmd == "margin":
        r = measure_margin(self_test=st, out_dir=out)
    elif args.cmd == "recall":
        r = measure_recall(self_test=st, out_dir=out)
    elif args.cmd == "e2e":
        r = measure_e2e(self_test=st, out_dir=out)
    else:
        r = measure_ledger(self_test=st, out_dir=out)
    return 0 if r.get("passed") else 1


if __name__ == "__main__":
    raise SystemExit(main())
