"""test_finetune.py — the W8 CPU dry-run: the §7 fine-tune loop + the §11
LUT artifact codec (finetune.py W8.1, lut_export.py W8.2).

THE W8 DoD (TASKS.md): a synthetically constructed PalettizedLinear
inside a stub model that ALSO carries an M1M2 and performs a
GatedDeltaNet-like state write into a FRESH TQCache per forward;
`finetune.train` runs 3 steps and every §7 trainable group receives
gradients; `freeze_lut` snaps to the fp16 deployment grid; the
export/import round-trip is BIT-EQUAL (spec §7/§11: the fine-tuned LUTs
are served as FULL LUT artifacts under `pretrained_luts/` — not
adapters).

Derived PalettizedLinear synthetic layout (the gates that pin it, all
read off scripts/palettized_modules.py + flute_extended/idxN.py):
  * idxN eligibility — `make_trainable()` -> `_logical_indices_numpy()`
    -> `idxN.unpack_idxn(blob, N, K, bits)` -> `check_eligible` requires
    N % 128 == 0 and K % 64 == 0, so the MINIMUM honest module is
    N=128, K=64 (the constructor alone would accept N=8/K=16 on the
    reference path, but the straight-through cache would refuse);
  * indices: a uint8 torch blob of numel N*K*bitwidth/8, produced by
    idxN.pack_idxn((N, K) uint8 logical matrix, bitwidth) — the
    canonical producer; `_logical_indices_numpy()` unpacks it back to
    the logical matrix (asserted below: the layout derivation);
  * lut: fp16 (N // group_size, 2**bitwidth); stream 2 (the lm_head's
    mixed 4+2 module) shares the row-group grid with palette width
    2**bitwidth2, its own idx{bitwidth2} blob of numel N*K*bitwidth2/8;
  * reference=True (the only CPU-legal route), rotation_seed=None (the
    FHT fold is skipped for the synthetic), resA/resB fp16 residual
    factors on the stream-1 module (the LQER serving form).

Stub design notes (documented deviations, both deliberate):
  * the M1/M2 WRITE GATES are opened (0.5) at stub init: production
    zero-inits them (PROPOSAL P3 — the untrained model is bit-unchanged),
    but with zero gates AND zero fresh-cache memories every gate
    gradient is exactly zero, which would make the §7 gate-grad DoD
    vacuous. Opening them makes the dry-run assert real gradient flow.
  * the logits consume the POST-write M1/M2 read: `M1M2.forward` reads
    the PRE-write memories (causal), which on the loop's fresh
    per-batch cache are always zeros (a no-op for the loss). Reading
    the post-write states puts all three gate params in the loss graph.
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

import _paths  # noqa: F401  (house sys.path anchor — must precede sibling imports)

# (Historical note: the idxN loader in palettized_modules used to search
# only the upstream NESTED layout flute_extended/flute_extended/idxN.py;
# this repo ships the FLAT src/flute_extended/idxN.py. Fixed at source
# (the flat candidates now come first, matching the FHT loader) — this
# suite constructs PalettizedLinears with NO env override, exercising the
# fixed candidate list itself.)

import finetune  # noqa: E402
import lut_export  # noqa: E402
import palettized_modules as pm  # noqa: E402
from m1m2 import M1M2, m1m2_from_cache, push_to_cache  # noqa: E402
from tq_cache import TQCache  # noqa: E402

SRC_RAG = Path(__file__).resolve().parents[1]
IDXN = pm._get_idxn()  # the standalone canonical idxN producer/verifier

# ------------------------------------------------------------- geometry ---
# Stub sizes: the idxN minimum (N=128, K=64) for the stream-1 module; the
# lm_head module is N=VOCAB=128, K=128 (N%128==0, K%64==0 hold). The cache
# units are power-of-two FHT dims with COMMITTED codebooks (no solving at
# test time): S (1, 8, 16) = 128; conv window (1, 32, 4) = 128; M1/M2
# (2, 4, 64) = 512. H*D = 2*64 = 128 = N so the M1/M2 read reshapes onto
# the palettized output without a projection.
LIN = "linear_attention"
S_SHAPE = (1, 8, 16)
S_D = 128
CONV_D = 32
KERNEL = 4
M_H, M_MEM, M_D = 2, 4, 64
VOCAB = 128
D_EMB = 64
N_OUT = 128
BITS = 3.5
GROUP_SIZE = 32          # 128 / 32 = 4 LUT row groups
RES_RANK = 8             # the LQER residual rank


# ------------------------------------------------- the synthetic module ---- #

def _make_palettized(N: int, K: int, group_size: int, bits: int,
                     bits2: int = None, seed: int = 0,
                     n_groups: int = None, residual: bool = False):
    """A synthetically constructed PalettizedLinear (the derived layout,
    see the module docstring). `n_groups` overrides the LUT row-group
    count (N // group_size by default) — the constructor clamps row
    groups, so a SHORT grid is a legal module with different shapes
    (the import shape-mismatch gate needs one)."""
    rng = np.random.default_rng(seed)
    gen = torch.Generator().manual_seed(seed + 1)
    ng = N // group_size if n_groups is None else int(n_groups)

    logical = rng.integers(0, 1 << bits, size=(N, K), dtype=np.uint8)
    blob = IDXN.pack_idxn(logical, bits)          # canonical idxN producer
    lut = (0.1 * torch.randn(ng, 1 << bits, generator=gen)).half()

    indices2 = lut2 = None
    if bits2 is not None:                          # the W4 Route A stream 2
        logical2 = rng.integers(0, 1 << bits2, size=(N, K), dtype=np.uint8)
        blob2 = IDXN.pack_idxn(logical2, bits2)
        indices2 = torch.from_numpy(np.ascontiguousarray(blob2))
        lut2 = (0.1 * torch.randn(ng, 1 << bits2, generator=gen)).half()

    resA = resB = None
    if residual:                                   # LQER serving form
        resA = (0.05 * torch.randn(N, RES_RANK, generator=gen)).half()
        resB = (0.05 * torch.randn(RES_RANK, K, generator=gen)).half()

    mod = pm.PalettizedLinear(
        torch.from_numpy(np.ascontiguousarray(blob)), lut, bits, group_size,
        N, K, resA=resA, resB=resB, indices2=indices2, lut2=lut2,
        bitwidth2=bits2, reference=True)           # the CPU-legal route
    mod._synth_logical = logical                   # for the layout assert
    return mod


class Qwen3_5GatedDeltaNet(nn.Module):
    """The linear-attention stand-in carrying the PRODUCTION class name —
    `finetune.build_trainables` discovers group 1 by
    `type(mod).__name__ == "Qwen3_5GatedDeltaNet"`, so the stub
    masquerades under exactly that name (the real discovery route is
    what the dry-run exercises). Params: in_proj (the qkv/z/b/a
    projection stand-in), A_log + dt_bias (the GDN decay knobs)."""

    def __init__(self, d_model: int, seed: int = 0):
        super().__init__()
        gen = torch.Generator().manual_seed(seed + 2)
        self.in_proj = nn.Linear(d_model, d_model, bias=False)
        self.in_proj.weight.data.copy_(
            0.2 * torch.randn(d_model, d_model, generator=gen))
        self.A_log = nn.Parameter(torch.zeros(()))
        self.dt_bias = nn.Parameter(torch.full((d_model,), 0.1))

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        # every group-1 param sits in the graph (grads must reach all)
        return self.in_proj(h) * torch.exp(-torch.exp(self.A_log)) \
            + self.dt_bias


class FinetuneStubModel(nn.Module):
    """The §7 dry-run stub (the W5 stub-model pattern): a palettized
    in_proj-side GEMM + a palettized two-stream lm_head, an M1M2 whose
    read/write gates feed the logits, and one GatedDeltaNet-like state
    write into a TQCache per forward (S quantize-on-write, the conv
    windowing write, M1/M2 quantize-on-write)."""

    def __init__(self, seed: int = 0, plin_n_groups: int = None):
        super().__init__()
        gen = torch.Generator().manual_seed(seed + 100)
        self.embed = nn.Embedding(VOCAB, D_EMB)
        self.embed.weight.data.copy_(
            torch.randn(VOCAB, D_EMB, generator=gen))
        self.linear_attn = Qwen3_5GatedDeltaNet(D_EMB, seed=seed)
        # stream-1 module WITH the LQER residual; group-1 input path
        self.plin = _make_palettized(
            N_OUT, D_EMB, GROUP_SIZE, 4, seed=seed, residual=True,
            n_groups=plin_n_groups)
        # the vocab head: mixed 4+2 two-stream module (Route A)
        self.lm_head_plin = _make_palettized(
            VOCAB, N_OUT, GROUP_SIZE, 4, bits2=2, seed=seed + 50)
        self.m1m2 = M1M2(num_heads=M_H, head_dim=M_D, mem_size=M_MEM,
                         num_linear_layers=1)
        # opened write gates (see the module docstring: the P3 zero-init
        # is a production property; the dry-run needs nonzero gate grads)
        with torch.no_grad():
            self.m1m2.write_gate_k.fill_(0.5)
            self.m1m2.write_gate_v.fill_(0.5)
        self.last_cache = None

    def forward(self, input_ids, past_key_values=None, use_cache=True):
        cache = past_key_values
        B, T = input_ids.shape

        # (1) embed -> in_proj -> the palettized GEMM (reference path; the
        # fp16 input mirrors the kernel path's A-operand dtype contract)
        h = self.linear_attn(self.embed(input_ids))          # (B, T, 64)
        y = self.plin(h.half()).float()                      # (B, T, 128)

        # (2) the GatedDeltaNet-like state write into the TQCache: S
        # quantize-on-write (path h -> y -> S: the state follows the
        # palettized output), the conv windowing write
        cur = cache.layers[0].recurrent_states[0]
        cur = torch.zeros(S_D, dtype=torch.float16) if cur is None \
            else cur.reshape(-1)
        new_s = cur.float() + y.float().sum(dim=(0, 1))
        cache.update_recurrent_state(new_s.reshape(S_SHAPE).half(), 0)
        conv_in = h.detach()[..., :CONV_D].transpose(1, 2).contiguous()
        cache.update_conv_state(conv_in.half(), 0,
                                conv_kernel_size=KERNEL)

        # (3) M1/M2: read the current state (zeros on a fresh cache) ->
        # gated write (the §7 gate params) -> quantize-on-write push
        m1, m2 = m1m2_from_cache(cache, self.m1m2, dtype=torch.float16)
        qkv = y.reshape(B, T, M_H, M_D).permute(0, 2, 1, 3)  # (B, 2, T, 64)
        m1_new, m2_new = self.m1m2.write(qkv, qkv, m1, m2, 0)
        push_to_cache(cache, m1_new, m2_new)

        # (4) the POST-write read feeds the logits (all three gate params
        # land in the loss graph — the documented stub deviation)
        read_out = self.m1m2.read(qkv, m1_new, m2_new, 0)
        read_contrib = read_out.permute(0, 2, 1, 3).reshape(
            B, T, M_H * M_D)                                 # (B, T, 128)
        z = y + read_contrib

        logits = self.lm_head_plin(z.half()).float()         # (B, T, VOCAB)
        self.last_cache = cache
        return logits


def _make_cache() -> TQCache:
    return TQCache(layer_types=[LIN], bits=BITS)


def _batches(n: int = 3, seed: int = 7):
    gen = torch.Generator().manual_seed(seed)
    return [torch.randint(0, VOCAB, (1, 8), generator=gen)
            for _ in range(n)]


def _iter_plins(model):
    return dict(pm.iter_palettized_linears(model))


# ============================================ 1. THE W8 DoD: dry-run ========
def test_dry_run_straight_through_luts():
    """The §7 loop on the synthetic stub: 3 steps, loss finite at every
    step, LUT masters + linear-attn params + M1/M2 gate params all
    receive grads, and the cache traffic really went through the TQCache."""
    model = FinetuneStubModel(seed=0)
    init_lut = model.plin.lut.clone()

    # the derived-layout assert: the constructor's blob unpacks back to
    # the packed logical matrix (idxN round-trip through the module)
    assert np.array_equal(model.plin._logical_indices_numpy(),
                          model.plin._synth_logical)
    assert model.plin.has_residual()
    assert model.lm_head_plin.has_stream2
    assert model.lm_head_plin.bitwidth2 == 2

    res = finetune.train(model, _batches(3), _make_cache,
                         cfg=finetune.FinetuneConfig(max_steps=3,
                                                     log_every=1))
    assert res["steps"] == 3
    assert res["param_groups"] == ["linear_attn", "m1m2_gates", "luts"]
    assert len(res["history"]) == 3
    for step, loss in res["history"]:
        assert math.isfinite(loss) and not math.isnan(loss), res["history"]

    # LUT master grads: every "luts" group param (build_trainables put
    # both streams' masters there) — straight-through, nonzero
    lut_params = [model.plin.lut, model.lm_head_plin.lut,
                  model.lm_head_plin.lut2]
    for p in lut_params:
        assert isinstance(p, nn.Parameter) and p.dtype == torch.float32
        assert p.grad is not None, "LUT master received no grad"
        assert p.grad.abs().sum().item() > 0.0

    # group 1: the linear-attn-side params (in_proj / A_log / dt_bias)
    for p in model.linear_attn.parameters():
        assert p.grad is not None and p.grad.abs().sum().item() > 0.0

    # group 2: the M1/M2 read/write gates (the stub's loss depends on
    # them through the post-write read)
    for name in ("write_gate_k", "write_gate_v", "read_gate"):
        p = getattr(model.m1m2, name)
        assert p.grad is not None and p.grad.abs().sum().item() > 0.0, name

    # the optimizer really stepped the masters (fp32 diff > 0)
    moved = (model.plin.lut.detach() - init_lut.float()).abs().sum().item()
    assert moved > 0.0

    # the cache traffic: codes exist for S / conv / M1 / M2 after the
    # last forward (quantize-on-write through the TQCache)
    cache = model.last_cache
    assert cache.s_codes[0] is not None
    assert cache.conv_codes[0] is not None
    assert cache.m1_codes is not None
    assert cache.m2_codes is not None


# ================================================ 2. freeze snaps fp16 ======
def test_freeze_lut_snaps_fp16():
    """freeze_all_luts -> every lut buffer is fp16, snapped_lut() equals
    the buffer, and both freeze- and make_trainable->freeze are
    idempotent (bit-equal). The frozen module still forwards finite
    logits (the reference path on the fp16 deployment grid)."""
    model = FinetuneStubModel(seed=1)
    finetune.train(model, _batches(3), _make_cache,
                   cfg=finetune.FinetuneConfig(max_steps=3, log_every=1))

    n = finetune.freeze_all_luts(model)
    assert n == 2                       # plin + lm_head_plin
    for name, mod in _iter_plins(model).items():
        assert mod.lut.dtype == torch.float16, name
        assert torch.equal(mod.snapped_lut(), mod.lut), name
        assert not isinstance(mod.lut, nn.Parameter)
        if mod.has_stream2:
            assert mod.lut2.dtype == torch.float16, name
            assert torch.equal(mod.snapped_lut2(), mod.lut2), name

    head = model.lm_head_plin
    snap = head.lut.clone()
    snap2 = head.lut2.clone()

    # freeze is idempotent on an already-frozen module
    head.freeze_lut(snap_fp16=True)
    assert torch.equal(head.lut, snap) and torch.equal(head.lut2, snap2)

    # make_trainable -> freeze is bit-equal (fp16 -> fp32 -> fp16 snap)
    head.make_trainable()
    assert head.lut.dtype == torch.float32     # the fp32 master
    head.freeze_lut(snap_fp16=True)
    assert torch.equal(head.lut, snap) and torch.equal(head.lut2, snap2)

    # the frozen reference path forwards (deployment numerics)
    logits = model(input_ids=_batches(1, seed=11)[0],
                   past_key_values=_make_cache(), use_cache=True)
    assert torch.isfinite(logits).all().item()


# ==================================== 3. export/import round-trip ===========
def test_export_import_round_trip_bit_equal(tmp_path):
    """export_luts (after freeze) -> import_luts into a FRESH copy of the
    stub -> every lut buffer bit-equal (torch.equal); the manifest's
    sha256s verify; a tampered artifact is refused (the checksum gate)."""
    model = FinetuneStubModel(seed=0)
    finetune.train(model, _batches(3), _make_cache,
                   cfg=finetune.FinetuneConfig(max_steps=3, log_every=1))
    finetune.freeze_all_luts(model)
    exported = {name: mod.lut.clone()
                for name, mod in _iter_plins(model).items()}
    exported2 = {name: mod.lut2.clone() for name, mod in
                 _iter_plins(model).items() if mod.has_stream2}

    out_dir = str(tmp_path / "pretrained_luts")
    manifest = lut_export.export_luts(
        model, out_dir, extra_meta={"run": "w8-dry-run", "steps": 3})
    assert manifest["version"] == lut_export.LUT_EXPORT_VERSION == 1
    assert manifest["n_modules"] == 2
    assert set(manifest["files"]) == set(exported)
    assert manifest["extra_meta"] == {"run": "w8-dry-run", "steps": 3}
    for fname in manifest["files"].values():
        assert (tmp_path / "pretrained_luts" / fname).is_file()
        digest = lut_export._file_digest(os.path.join(out_dir, fname))
        assert digest == manifest["sha256"][fname]

    fresh = FinetuneStubModel(seed=999)     # different luts pre-import
    assert not torch.equal(fresh.plin.lut, exported["plin"])
    rep = lut_export.import_luts(fresh, out_dir)
    assert rep["checked"] == 2
    assert sorted(rep["loaded"]) == sorted(exported)
    assert rep["mismatches"] == []

    live = _iter_plins(fresh)
    for name, tensor in exported.items():
        assert torch.equal(live[name].lut, tensor), name
    for name, tensor in exported2.items():
        assert torch.equal(live[name].lut2, tensor), name
    # the fresh model still forwards with the imported grid
    logits = fresh(input_ids=_batches(1, seed=13)[0],
                   past_key_values=_make_cache(), use_cache=True)
    assert torch.isfinite(logits).all().item()

    # tamper gate: one flipped byte -> the whole-file sha256 refuses
    tam_dir = tmp_path / "tampered"
    tam_dir.mkdir()
    for f in (tmp_path / "pretrained_luts").iterdir():
        tam_dir.joinpath(f.name).write_bytes(f.read_bytes())
    victim = tam_dir / manifest["files"]["plin"]
    data = bytearray(victim.read_bytes())
    data[len(data) // 2] ^= 0xFF
    victim.write_bytes(bytes(data))
    with pytest.raises(ValueError, match="sha256 mismatch"):
        lut_export.import_luts(FinetuneStubModel(seed=2), str(tam_dir))


# ============================== 4. strict import refuses on drift ===========
def test_import_strict_drift_raises(tmp_path):
    """strict import: a MISMATCHED lut shape raises (the shape gate), and
    a live module not covered by the manifest raises (the coverage
    gate); strict=False reports the drift best-effort."""
    model = FinetuneStubModel(seed=0)
    finetune.freeze_all_luts(model)
    out_dir = str(tmp_path / "pretrained_luts")
    lut_export.export_luts(model, out_dir)

    # same geometry config, SHORT lut grid (2 row groups vs the exported
    # 4) — a pure shape mismatch
    short = FinetuneStubModel(seed=5, plin_n_groups=2)
    assert tuple(short.plin.lut.shape) == (2, 16)
    with pytest.raises(ValueError, match="shape/dtype"):
        lut_export.import_luts(short, out_dir, strict=True)

    # a live module the manifest does not cover (coverage drift)
    extra = FinetuneStubModel(seed=6)
    extra.extra_plin = _make_palettized(N_OUT, D_EMB, GROUP_SIZE, 4,
                                        seed=77)
    with pytest.raises(ValueError, match="not covered"):
        lut_export.import_luts(extra, out_dir, strict=True)

    # best-effort: the mismatch is reported, the rest still loads
    rep = lut_export.import_luts(extra, out_dir, strict=False)
    assert rep["checked"] == 2
    assert "plin" in rep["loaded"] and "lm_head_plin" in rep["loaded"]
    assert len(rep["mismatches"]) == 1 and "not covered" in rep["mismatches"][0]
    assert torch.equal(extra.plin.lut, model.plin.lut)


# ============================== 5. export refuses trainable masters =========
def test_export_refuses_trainable_master(tmp_path):
    """The deployment-grid contract: export_luts raises while a LUT is a
    trainable fp32 master (make_trainable state) and on a non-fp16
    buffer (freeze_lut(snap_fp16=False) residue); freeze_all_luts is the
    documented recovery."""
    model = FinetuneStubModel(seed=3)
    out_dir = str(tmp_path / "pretrained_luts")

    model.plin.make_trainable()
    with pytest.raises(RuntimeError, match="freeze_all_luts"):
        lut_export.export_luts(model, out_dir)

    # a master demoted WITHOUT the fp16 snap is also refused (the grid
    # is contractually fp16); NOTE freeze_lut only demotes Parameters,
    # so the snap_fp16=False residue must be re-promoted before the
    # documented recovery can snap it
    model.plin.freeze_lut(snap_fp16=False)
    assert model.plin.lut.dtype == torch.float32
    with pytest.raises(RuntimeError, match="fp16"):
        lut_export.export_luts(model, out_dir)

    # the documented recovery: re-promote, snap everything, export
    model.plin.make_trainable()
    assert finetune.freeze_all_luts(model) == 2
    manifest = lut_export.export_luts(model, out_dir)
    assert manifest["n_modules"] == 2


# ============================== 6. the W8 grep gate (TASKS.md DoD) ==========
def test_w8_dependency_grep_gate():
    """TASKS.md W8 DoD: the W0-deleted trainer plane must not creep back —
    ZERO occurrences of its name fragment anywhere under src/rag/. The
    needle is assembled at runtime so THIS file passes its own gate."""
    needle = "q" + "lora"
    py_files = sorted(SRC_RAG.rglob("*.py"))
    assert py_files, "expected python files under src/rag/"
    for path in py_files:
        text = path.read_text(encoding="utf-8").lower()
        assert needle not in text, \
            f"{path.name} mentions {needle!r} (the W8 grep gate)"


# ============================== 7. cosine + warmup sanity ===================
def test_cosine_warmup_schedule():
    """_cosine: small at step 0, rising through warmup to exactly 1.0,
    then monotone decay to exactly min_lr_frac at the end."""
    cfg = finetune.FinetuneConfig(max_steps=100, warmup_steps=20,
                                  min_lr_frac=0.05)

    def f(step: int) -> float:
        return finetune._cosine(step, 100, cfg)

    assert f(0) == pytest.approx(0.05)                 # small at the start
    warm = [f(s) for s in range(cfg.warmup_steps)]
    assert all(b > a for a, b in zip(warm, warm[1:]))  # rises through warmup
    assert f(cfg.warmup_steps - 1) == pytest.approx(1.0)
    assert f(100) == pytest.approx(cfg.min_lr_frac)    # exact landing
    mid = f(60)
    assert cfg.min_lr_frac < mid < 1.0
    decay = [f(s) for s in range(20, 101, 10)]
    assert all(b < a for a, b in zip(decay, decay[1:]))  # monotone decay


# ============================== 8. loud refusal: no trainables ==============
def test_train_refuses_without_trainable_groups():
    """train() with no trainable groups (an empty model, no param_groups
    override) is a loud ValueError, never a silent no-op."""
    with pytest.raises(ValueError, match="no trainable groups"):
        finetune.train(nn.Module(), _batches(1),
                       lambda: None,
                       cfg=finetune.FinetuneConfig(max_steps=3, log_every=1))
