"""test_evals.py — W9.3: the evals harness self-tests (src/rag/evals.py,
W9.1) + the W9.2 flag-parity gates (the P7 hardening DoD).

GATES (all deterministic; plain asserts; tmp out dirs):

  1. CLI SELF-TESTS: each of the six subcommands runs end-to-end IN-PROCESS
     via evals.main([cmd, "--self-test", "--out-dir", tmp]) -> exit code 0;
     the JSON artifact exists and parses; "passed" is true; "lines" is
     non-empty; mode == "self-test" where the measure reports it
     (roundtrip reports NO mode key — its synthetic core is the real core;
     assert what each command actually reports, per-command contract
     fields).
  2. ROUNDTRIP VALUES from the artifact (the JSON is the contract):
     D_mse(b=2) within 5% of the paper's 0.117; every integer b within the
     2.72*4^-b bound; the 3.5-bit split beats 3-bit.
  3. GPU-MODE REFUSALS: streaming/margin/recall/e2e WITHOUT --self-test
     exit non-zero via a loud SystemExit naming the GPU box (pytest.raises
     SystemExit) — the W9.1 contract "refuses loudly when absent", hardened
     in W9.3 (the CLI previously fell through to the synthetic self-test,
     silently lying about what was measured).
  4. FLAG PARITY — the W9.2 DoD (all flags default OFF; flag-off output
     bit-identical to the unflagged path; where the flag is a no-op on CPU,
     flag-on == flag-off bit-identically):
     (a) turboquant qjl: qjl=False == the pre-existing behavior (identical
         idx streams + fp32 norm bits, no qjl fields, identical dequant);
         qjl=True codes carry the residual fields, the MSE layer is
         UNCHANGED (the A/B is purely additive), dequant adds the Alg-2
         compensation (exact geometry: ||comp|| = sqrt(pi/2)*gamma, comp
         points along the residual), round-trip rel-MSE(qjl) <= rel-MSE
         (base) + 5% (measured: it IMPROVES — mean ratio ~0.64);
         serialization round-trips qjl codes; OLD-format arrays (no qjl
         keys) load with the fields None (backward compat);
     (b) snapshot use_mmap: save+load one small chunk (mixed qjl / qjl-free
         units) with use_mmap False vs True — loaded fields bit-identical;
         WHAT MMAP ACTUALLY DOES is asserted, not just documented (array
         members come back as zero-copy views onto np.memmap roots; the
         sha256 digest still gates the load; verify_chunk agrees);
     (c) tq_cache graph_safe: TQCache(layer_types=..., graph_safe=True) vs
         False — the same update/read sequence produces bit-identical
         (torch.equal) dequantized reads AND bit-identical code streams;
         before_graph_capture()/after_graph_capture() exist and are
         callable no-ops (state + counters untouched by calling them).
  5. EVALS DETERMINISM: the roundtrip self-test run twice into different
     out dirs -> identical mse_table (and concentration stats).

The streaming stub is chaotic-by-design (W9.1: the machinery gate) — this
suite asserts exactly what that gate reports (finite match rates in [0,1],
artifact + lines + the authoritative GPU target line), never the 0.95
token-match itself (that is the real-model P2 gate, GPU-box-only).
"""
from __future__ import annotations

import json
import math
import os

import numpy as np
import pytest
import torch

import evals
import fht
from snapshot import (ChunkSnapshot, load_chunk, save_chunk, verify_chunk)
from tq_cache import TQCache, TQLinearAttentionLayer
from turboquant import QJL_SEED_OFFSET, TQCodes, TurboQuant

COMMANDS = ("roundtrip", "streaming", "margin", "recall", "e2e", "ledger")
GPU_MODE_COMMANDS = ("streaming", "margin", "recall", "e2e")

# shared small dims (power-of-two, custom kinds keep the real D3 seeds)
S_D, CONV_D, M_D = 256, 128, 256
S_SEED, CONV_SEED, M1_SEED = 101, 202, 303


# ------------------------------------------------------------- gate 1/2 ---
_RUN_CACHE: dict = {}


@pytest.fixture(scope="module")
def cli_run(tmp_path_factory):
    """Run each subcommand ONCE, in-process, with --self-test into a tmp
    out dir; memoized per command for the whole module (recall/e2e are the
    heavy ones — one 260-chunk ingest + IVFADC build each)."""

    def get(cmd: str):
        if cmd not in _RUN_CACHE:
            out = tmp_path_factory.mktemp(f"evals_{cmd}_")
            code = evals.main([cmd, "--self-test", "--out-dir", str(out)])
            path = os.path.join(str(out), f"{cmd}.json")
            assert os.path.isfile(path), f"missing artifact {path}"
            with open(path) as fh:
                art = json.load(fh)
            _RUN_CACHE[cmd] = (code, art, str(out))
        return _RUN_CACHE[cmd]

    return get


@pytest.mark.parametrize("cmd", COMMANDS)
def test_cli_self_test(cmd, cli_run):
    """Gate 1: exit 0, artifact parses, passed true, lines non-empty, and
    the per-command contract fields each measure actually reports."""
    code, art, out = cli_run(cmd)
    assert code == 0
    assert art["gate"] == cmd
    assert art["passed"] is True
    assert isinstance(art["lines"], list) and len(art["lines"]) > 0
    # the five gate-emitting commands carry PASS verdict lines in the
    # artifact; ledger's lines are informational (its PASS verdict is the
    # stdout [ledger] summary line, not an artifact line)
    if cmd != "ledger":
        assert any("PASS" in line for line in art["lines"])
    # mode: what each measure ACTUALLY reports
    if cmd in GPU_MODE_COMMANDS or cmd == "ledger":
        assert art["mode"] == "self-test"
    else:  # roundtrip: no mode key — the synthetic core IS the real core
        assert "mode" not in art
    if cmd == "roundtrip":
        assert set(art["mse_table"]) == {"1", "2", "3", "3.5", "4"}
        assert art["n_vectors"] == 16 and art["d"] == 1024
    elif cmd == "streaming":
        for key in ("token_match", "first_half", "second_half"):
            v = art[key]
            assert isinstance(v, float) and 0.0 <= v <= 1.0
        assert art["steps"] == 32
        assert art["gpu_target"] == 0.95  # the authoritative P2 gate line
        assert "chaotic-by-design" in art["stub"]
    elif cmd == "margin":
        assert art["ratio"] >= 3.0
        assert art["same_topic_cos"] > art["diff_topic_cos"]
    elif cmd == "recall":
        assert art["recall_at_100"] >= 0.90
        assert art["top3"] >= 0.80
        assert art["n_queries"] == 10
    elif cmd == "e2e":
        acc = art["accuracy"]
        assert set(acc) == {"no_rag", "actual", "oracle"}
        assert acc["oracle"] >= 0.9 and acc["actual"] >= 0.5
    else:  # ledger
        assert art["timings"]["sample_step"] > 0.0
        assert art["vram"]["cuda_available"] is False  # honest CPU probe
        assert art["spec_target_s10_vram_gib"] == 13
        assert "spec_targets_s9" in art


def test_roundtrip_values_from_artifact(cli_run):
    """Gate 2: the JSON artifact is the contract — b=2 within 5% of the
    paper's 0.117, every integer b inside the 2.72*4^-b bound."""
    _, art, _ = cli_run("roundtrip")
    t = art["mse_table"]
    assert abs(t["2"] - 0.117) / 0.117 <= 0.05
    for b in (1, 2, 3, 4):
        assert t[str(b)] <= 2.72 * 4.0 ** -b, f"D_mse(b={b}) out of bound"
    assert t["3.5"] < t["3"]  # the 50/50 split beats uniform 3-bit
    # the concentration gate rode through the same artifact
    assert abs(art["var"] - 1.0) < 0.05 and 2.5 < art["kurtosis"] < 3.6


# --------------------------------------------------------------- gate 3 ---
@pytest.mark.parametrize("cmd", GPU_MODE_COMMANDS)
def test_gpu_mode_refusal(cmd):
    """Gate 3: without --self-test the GPU-real-mode commands exit non-zero
    with the loud SystemExit message (the W9.1 'refuses loudly' contract —
    hardened in W9.3: the CLI used to fall through to the synthetic
    self-test and exit 0, lying about what was measured)."""
    with pytest.raises(SystemExit) as excinfo:
        evals.main([cmd])
    msg = str(excinfo.value)
    assert "GPU" in msg
    assert "self-test" in msg
    assert cmd in msg


# --------------------------------------------------------------- gate 5 ---
def test_evals_determinism(tmp_path_factory):
    """Gate 5: the roundtrip self-test is bit-stable across runs — two
    runs into different out dirs give identical mse_table values (and the
    concentration / spread stats)."""
    outs = [tmp_path_factory.mktemp(f"evals_det{i}_") for i in (1, 2)]
    arts = []
    for out in outs:
        assert evals.main(
            ["roundtrip", "--self-test", "--out-dir", str(out)]) == 0
        with open(os.path.join(str(out), "roundtrip.json")) as fh:
            arts.append(json.load(fh))
    a, b = arts
    assert a["mse_table"] == b["mse_table"]
    for key in ("var", "kurtosis", "max_abs_z", "frac_within_3sigma",
                "per_head_norm_spread", "outlier_mass_top1pct"):
        assert a[key] == b[key]


# ------------------------------------------------- gate 4(a): turboquant ---
def test_qjl_flag_parity_and_backward_compat():
    """The W9.2 --qjl A/B: flag-off == pre-existing behavior bit-for-bit;
    flag-on carries the residual sketch, compensates dequant exactly per
    Alg. 2, never makes round-trip MSE meaningfully worse; serialization
    is backward compatible (old arrays -> fields None)."""
    g = torch.Generator().manual_seed(20261009)
    x = torch.randn(256, generator=g)
    x = (x / x.norm()).float()

    # (a) flag OFF: identical idx streams + fp32 norm bits + dequant,
    # whether the kwarg is absent (pre-existing constructor) or False
    q_plain = TurboQuant(kind="custom", bits=3.5, d=256, seed=1)
    q_off = TurboQuant(kind="custom", bits=3.5, d=256, seed=1, qjl=False)
    assert q_plain.qjl is False and q_off.qjl is False
    c_plain, c_off = q_plain.quant(x), q_off.quant(x)
    assert np.array_equal(c_plain.idx_lo, c_off.idx_lo)
    assert np.array_equal(c_plain.idx_hi, c_off.idx_hi)
    assert c_plain.norm.tobytes() == c_off.norm.tobytes()
    assert c_plain.qjl_signs is None and c_plain.gamma is None
    assert c_off.qjl_signs is None and c_off.gamma is None
    assert torch.equal(q_plain.dequant(c_plain), q_off.dequant(c_off))

    # flag ON: fields present; the MSE layer is UNTOUCHED (additive A/B)
    q_on = TurboQuant(kind="custom", bits=3.5, d=256, seed=1, qjl=True)
    assert q_on.qjl is True
    c_on = q_on.quant(x)
    assert c_on.qjl_signs is not None and c_on.gamma is not None
    assert c_on.qjl_signs.dtype == np.int8
    assert c_on.qjl_signs.shape == (256,)
    assert set(np.unique(c_on.qjl_signs).tolist()) <= {-1, 1}
    assert isinstance(c_on.gamma, np.float32) and float(c_on.gamma) > 0.0
    assert np.array_equal(c_on.idx_lo, c_plain.idx_lo)
    assert np.array_equal(c_on.idx_hi, c_plain.idx_hi)
    assert c_on.norm.tobytes() == c_plain.norm.tobytes()
    # the QJL sign vector is a SECOND draw: seed+7777 != the rotation seed
    assert not torch.equal(fht.rotation_signs(256, 1 + QJL_SEED_OFFSET),
                           fht.rotation_signs(256, 1))

    # dequant carries the compensation, with the exact Alg-2 geometry:
    # ||comp|| == sqrt(pi/2) * gamma (deterministic), and comp points
    # along the original-frame residual (positive inner product)
    d_off, d_on = q_off.dequant(c_off), q_on.dequant(c_on)
    comp = d_on - d_off
    assert float(comp.abs().max()) > 0.0  # the compensation is nonzero
    assert torch.allclose(
        comp.norm(),
        torch.tensor(math.sqrt(math.pi / 2.0) * float(c_on.gamma)),
        rtol=1e-4)
    resid = x - d_off
    assert float((resid * comp).sum()) > 0.0

    # round-trip rel-MSE gate: qjl never meaningfully worse (+5% bound;
    # measured here: 16/16 vectors IMPROVE, mean ratio ~0.64)
    gg = torch.Generator().manual_seed(5)
    worst, ratios = 0.0, []
    for _ in range(16):
        v = torch.randn(256, generator=gg)
        v = v / v.norm()
        m_off = float(((v - q_off.roundtrip(v)) ** 2).sum())
        m_on = float(((v - q_on.dequant(q_on.quant(v))) ** 2).sum())
        ratios.append(m_on / m_off)
        worst = max(worst, m_on / m_off)
    assert worst <= 1.05, f"qjl round-trip rel-MSE worst ratio {worst:.3f}"
    assert sum(ratios) / len(ratios) < 1.0  # improves on average

    # serialization: qjl fields ride; flag-off codes write NO new keys;
    # OLD-format arrays (no qjl keys) load with the fields None
    arr = c_on.to_arrays()
    assert "qjl_signs" in arr and "gamma" in arr
    c_rt = TQCodes.from_arrays(dict(arr))
    assert np.array_equal(c_rt.qjl_signs, c_on.qjl_signs)
    assert np.float32(c_rt.gamma).tobytes() == c_on.gamma.tobytes()
    assert np.array_equal(c_rt.idx_lo, c_on.idx_lo)
    assert torch.equal(q_on.dequant(c_rt), d_on)
    arr_off = c_off.to_arrays()
    assert "qjl_signs" not in arr_off and "gamma" not in arr_off
    c_old = TQCodes.from_arrays(dict(arr_off))  # the pre-W9.2 key set
    assert c_old.qjl_signs is None and c_old.gamma is None
    assert torch.equal(q_off.dequant(c_old), d_off)
    # codes are self-describing: a qjl=False instance dequantizes qjl codes
    assert torch.equal(q_off.dequant(c_on), d_on)

    # a PARTIAL sketch (exactly one of the pair) is refused loudly
    c_part = TQCodes.from_arrays(dict(arr))
    c_part.gamma = None
    with pytest.raises(ValueError, match="PARTIAL QJL"):
        q_on.dequant(c_part)


# ------------------------------------------------- gate 4(b): snapshot -----
def test_snapshot_use_mmap_parity_and_qjl_codes(tmp_path):
    """The W9.2 use_mmap flag: one small chunk (MIXED qjl / qjl-free units
    — the A/B may flip per unit) saved once, loaded with use_mmap False vs
    True: loaded fields bit-identical; what mmap actually does is ASSERTED
    (zero-copy views onto np.memmap roots — numpy's asarray strips the
    subclass inside from_arrays); the digest still gates the load;
    verify_chunk agrees both ways; pre-W9.2 (qjl-free) snapshots load with
    the fields None under both readers."""
    qs_on = TurboQuant(kind="custom", bits=3.5, d=S_D, seed=S_SEED,
                       qjl=True)
    qs_off = TurboQuant(kind="custom", bits=3.5, d=S_D, seed=S_SEED)
    qc = TurboQuant(kind="custom", bits=3.5, d=CONV_D, seed=CONV_SEED)
    qm = TurboQuant(kind="custom", bits=3.5, d=M_D, seed=M1_SEED, qjl=True)
    g = torch.Generator().manual_seed(9)
    vs, vs2 = torch.randn(S_D, generator=g), torch.randn(S_D, generator=g)
    vc, vm = torch.randn(CONV_D, generator=g), torch.randn(M_D, generator=g)

    snap = ChunkSnapshot(
        chunk_id=3, protocol="delta-v1",
        s_codes={0: qs_on.quant(vs), 1: qs_off.quant(vs2)},
        conv_codes={0: qc.quant(vc)},
        m1_codes=qm.quant(vm), m2_codes=None,
        system_ref="sys-selftest", extra={"n": 2})
    p = save_chunk(str(tmp_path), snap)

    eager = load_chunk(p)                 # flag OFF (default: pre-W9.2 path)
    mm = load_chunk(p, use_mmap=True)     # flag ON
    units = [(eager.s_codes[0], mm.s_codes[0], "s_0 (qjl)"),
             (eager.s_codes[1], mm.s_codes[1], "s_1 (plain)"),
             (eager.conv_codes[0], mm.conv_codes[0], "conv_0"),
             (eager.m1_codes, mm.m1_codes, "m1 (qjl)")]
    for a, b, name in units:
        assert np.array_equal(a.idx_lo, b.idx_lo), name
        assert np.array_equal(a.idx_hi, b.idx_hi), name
        assert b.idx_lo.dtype == np.uint8, name
        assert np.float32(a.norm).tobytes() == np.float32(b.norm).tobytes(), \
            name
        assert (a.qjl_signs is None) == (b.qjl_signs is None), name
        if a.qjl_signs is not None:
            assert np.array_equal(a.qjl_signs, b.qjl_signs), name
            assert np.float32(a.gamma).tobytes() == \
                np.float32(b.gamma).tobytes(), name
    assert eager.s_codes[0].qjl_signs is not None   # the qjl unit rides
    assert eager.s_codes[1].qjl_signs is None       # the plain unit rides
    for field in ("chunk_id", "protocol", "system_ref", "extra"):
        assert getattr(eager, field) == getattr(mm, field)
    assert eager.extra == {"n": 2}
    assert eager.m2_codes is None and mm.m2_codes is None

    # WHAT use_mmap ACTUALLY DOES (asserted so the flag cannot lie): the
    # array members are zero-copy views onto np.memmap roots into the
    # snapshot FILE; the eager reader holds plain heap ndarrays.
    def mm_root(arr):
        while arr.base is not None and not isinstance(arr.base, np.memmap):
            arr = arr.base
        return arr.base

    assert isinstance(mm_root(mm.s_codes[0].idx_lo), np.memmap)
    assert isinstance(mm_root(mm.m1_codes.qjl_signs), np.memmap)
    assert mm_root(eager.s_codes[0].idx_lo) is None
    assert not isinstance(eager.s_codes[0].idx_lo, np.memmap)

    # dequant through the mmap-loaded codes is bit-identical, and the
    # sha256 integrity digest still gated the load (load_chunk would have
    # raised on mismatch — the round-trip IS the assert)
    assert torch.equal(qs_on.dequant(mm.s_codes[0]),
                       qs_on.dequant(eager.s_codes[0]))
    assert torch.equal(qm.dequant(mm.m1_codes), qm.dequant(eager.m1_codes))

    ve, vm_v = verify_chunk(p), verify_chunk(p, use_mmap=True)
    for v in (ve, vm_v):
        assert v["ok"] and v["sha256_ok"]
        assert v["chunk_id"] == 3
        assert v["n_s"] == 2 and v["n_conv"] == 1
        assert v["has_m1"] and not v["has_m2"]

    # pre-W9.2 (qjl-free) snapshots: fields None under BOTH readers
    snap_old = ChunkSnapshot(chunk_id=4,
                             s_codes={0: qs_off.quant(vs)},
                             conv_codes={0: qc.quant(vc)},
                             m1_codes=None, m2_codes=None)
    p_old = save_chunk(str(tmp_path), snap_old)
    for loaded in (load_chunk(p_old), load_chunk(p_old, use_mmap=True)):
        assert loaded.s_codes[0].qjl_signs is None
        assert loaded.s_codes[0].gamma is None
        assert loaded.m1_codes is None


# ------------------------------------------------- gate 4(c): tq_cache ----
_LAYER_TYPES = ["linear_attention", "full_attention",
                "linear_attention", "full_attention"]
_S_SHAPE, _M_SHAPE = (1, 8, 16), (2, 4, 16)


def _drive_cache(cache: TQCache) -> list:
    """A fixed deterministic update/read sequence (S + conv writes on both
    linear layers, M1/M2 writes, then dequantized reads back)."""
    g = torch.Generator().manual_seed(777)
    outs = []
    for layer in (0, 2):
        s = torch.randn(_S_SHAPE, generator=g).half()
        outs.append(cache.update_recurrent_state(s, layer))
        c = torch.randn(1, 32, 3, generator=g).half()
        outs.append(cache.update_conv_state(c, layer, conv_kernel_size=4))
    outs.append(cache.update_m1(torch.randn(_M_SHAPE, generator=g).half()))
    outs.append(cache.update_m2(torch.randn(_M_SHAPE, generator=g).half()))
    outs.append(cache.layers[0].recurrent_states[0])   # dequantized read
    outs.append(cache.layers[2].conv_states[0])
    outs.append(cache.read_m1())
    outs.append(cache.read_m2())
    return outs


def test_tq_cache_graph_safe_parity():
    """The W9.2 graph_safe flag (P7): flag-on vs flag-off — the same
    update/read sequence gives bit-identical dequantized reads
    (torch.equal) AND bit-identical code streams; the capture hooks exist
    and are callable no-ops (calling them leaves codes + counters
    untouched — the provable CPU no-op)."""
    c_off = TQCache(layer_types=_LAYER_TYPES)                  # default OFF
    c_on = TQCache(layer_types=_LAYER_TYPES, graph_safe=True)  # flag ON
    assert c_off.layers[0].graph_safe is False
    assert c_on.layers[0].graph_safe is True

    o_off, o_on = _drive_cache(c_off), _drive_cache(c_on)
    assert len(o_off) == len(o_on) == 10  # 4 S/conv writes + 2 M + 4 reads
    for a, b in zip(o_off, o_on):
        assert torch.equal(a, b), "graph_safe changed a dequantized read"
    for layer in (0, 2):
        assert np.array_equal(c_off.layers[layer].s_codes.idx_lo,
                              c_on.layers[layer].s_codes.idx_lo)
        assert np.array_equal(c_off.layers[layer].s_codes.idx_hi,
                              c_on.layers[layer].s_codes.idx_hi)
        assert np.array_equal(c_off.layers[layer].conv_codes.idx_lo,
                              c_on.layers[layer].conv_codes.idx_lo)
    assert np.array_equal(c_off.m1_codes.idx_lo, c_on.m1_codes.idx_lo)
    assert np.array_equal(c_off.m2_codes.idx_hi, c_on.m2_codes.idx_hi)

    # the hooks: present, callable, and PROVABLE no-ops
    lay = c_on.layers[0]
    assert callable(lay.before_graph_capture)
    assert callable(lay.after_graph_capture)
    codes_before = lay.s_codes
    reads_before, writes_before = lay.reads.copy(), lay.writes.copy()
    assert lay.before_graph_capture() is None
    assert lay.after_graph_capture() is None
    assert lay.s_codes is codes_before
    assert lay.reads == reads_before and lay.writes == writes_before

    # default OFF on the bare layer class too
    assert TQLinearAttentionLayer().graph_safe is False
