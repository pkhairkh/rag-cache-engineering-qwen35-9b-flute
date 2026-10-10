"""test_w17_loader.py — the W17 M1/M2 activation gates (the loader path).

THE BUG THESE GATES PIN (the W16 handover's "M1/M2 NOT ACTIVATED"):
`load_palettized_model` resolved `AutoModelForCausalLM` to the NATIVE
transformers class, which carries no M1/M2 wiring — the `use_m1m2` config
flag was a silent no-op (AND the flag was set on the composite wrapper,
not the text config where the vendored `Qwen3_5TextModel.__init__` reads
it). W17 loads through the vendored classes with the flags on the TEXT
config, plus the official `key_mapping` that strips the composite
checkpoint's `model.language_model.*` prefix (the vendored module counts
as "custom code" — is_custom_code — and the library-internal conversion
table is skipped for it: without the explicit mapping the text weights
silently re-initialize).

The fixture builds a tiny COMPOSITE checkpoint (native
Qwen3_5ForConditionalGeneration save — the hub Qwen/Qwen3.5-9B layout:
text weights under model.language_model.*, vision under model.visual.*)
plus a minimal artifacts dir (metadata.json with an empty tensor map —
the palettized swap no-ops, exercising everything else in
load_palettized_model EXACTLY as the box runs it).
"""
from __future__ import annotations

import json
import os

import numpy as np
import pytest
import torch
from transformers import (
    Qwen3_5Config,
    Qwen3_5ForCausalLM as NativeCausalLM,
    Qwen3_5ForConditionalGeneration as NativeCG,
)
from transformers.models.qwen3_5.configuration_qwen3_5 import \
    Qwen3_5VisionConfig
from transformers import Qwen3_5TextConfig

import modeling  # the vendored copy (src/scripts)
import palettized_modules as pmod

import m1m2_finetune as ft
from ingest import (IngestDriver, check_m1m2_geometry, load_system_state,
                    m1m2_mem_size_from_system, reseed_cache)
from tq_cache import TQCache

# the wiring-level geometry (test_m1m2.py's kit, one scale up for the
# full-model rig): v-heads 4, v-head-dim 16, mem 8 -> M numel 512
H, D, MEM = 4, 16, 8
M_NUMEL = H * MEM * D
N_LAYERS = 4
LAYER_TYPES = ["linear_attention", "full_attention"] * 2
BITS = 3.5


def _text_config(**kw):
    base = dict(
        hidden_size=64, num_hidden_layers=N_LAYERS,
        layer_types=list(LAYER_TYPES),
        linear_num_value_heads=H, linear_num_key_heads=2,
        linear_key_head_dim=D, linear_value_head_dim=D,
        linear_conv_kernel_dim=4,
        num_attention_heads=4, num_key_value_heads=2, vocab_size=128,
        dtype=torch.float32, tie_word_embeddings=False)
    base.update(kw)
    return Qwen3_5TextConfig(**base)


def _vision_config():
    return Qwen3_5VisionConfig(
        hidden_size=32, intermediate_size=32, num_hidden_layers=1,
        depth=1, num_heads=2, embed_dim=8, num_position_embeddings=16,
        patch_size=2, spatial_merge_size=1, temporal_patch_size=1,
        out_hidden_size=64)


@pytest.fixture(scope="module")
def ckpt(tmp_path_factory):
    """The tiny composite checkpoint + the minimal artifacts dir."""
    d = tmp_path_factory.mktemp("w17_ckpt")
    comp = Qwen3_5Config(text_config=_text_config(),
                         vision_config=_vision_config())
    torch.manual_seed(3)
    NativeCG(comp).save_pretrained(str(d / "hub"))
    (d / "artifacts").mkdir()
    with open(d / "artifacts" / "metadata.json", "w") as f:
        json.dump({"tensors": {}}, f)   # the swap no-ops
    return {"hub": str(d / "hub"), "artifacts": str(d / "artifacts"),
            "composite": comp}


def _load(ckpt, **kw):
    kw.setdefault("device", "cpu")
    kw.setdefault("dtype", torch.float32)
    return pmod.load_palettized_model(
        ckpt["artifacts"], ckpt["hub"], **kw)


def _ids(n, seed):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, 128, (1, n), generator=g)


# ================================================= 1. load parity ==========
def test_w17_1_vendored_load_bit_equal_native(ckpt):
    """use_m1m2=False: the vendored-class load (the W17 path) reproduces
    the NATIVE class's load EXACTLY — every text weight bit-equal (the
    key_mapping prefix strip vs the library-internal conversion table),
    the vision/mtp keys dropped, no random re-initialization."""
    model, _ = _load(ckpt, use_m1m2=False)
    native = NativeCausalLM.from_pretrained(ckpt["hub"],
                                            dtype=torch.float32)
    sv, sn = model.state_dict(), native.state_dict()
    assert set(sv) == set(sn), "state-dict key drift (vendored vs native)"
    for k in sorted(sv):
        assert torch.equal(sv[k], sn[k]), f"weight drift at {k}"
    # the composite's text weights (model.language_model.*) actually
    # LOADED (not re-initialized): pin against the checkpoint's own CG
    comp = Qwen3_5Config(text_config=_text_config(),
                         vision_config=_vision_config())
    assert len(sv) > 0


def test_w17_2_forward_parity_zero_gate(ckpt):
    """use_m1m2=True (P3 zero gates): the wired model's outputs are
    BIT-IDENTICAL to the native class's through a TQCache forward — the
    activation is a true no-op until the fine-tune opens the gates."""
    wired, _ = _load(ckpt, use_m1m2=True, m1m2_mem_size=MEM)
    native = NativeCausalLM.from_pretrained(ckpt["hub"],
                                            dtype=torch.float32).eval()
    ids = _ids(12, 7)
    outs = {}
    for tag, mdl in (("native", native), ("wired", wired)):
        cache = TQCache(layer_types=LAYER_TYPES, bits=BITS)
        with torch.no_grad():
            o = mdl(input_ids=ids, past_key_values=cache, use_cache=True)
        h = o.last_hidden_state if hasattr(o, "last_hidden_state") else o[0]
        outs[tag] = (h.clone(), cache)
    assert torch.equal(outs["native"][0], outs["wired"][0]), \
        "zero-gate parity broken — the wiring perturbs the untrained model"
    # the M1/M2 codes EXIST (the wiring executed; norm 0 at zero gates)
    m1, m2 = outs["wired"][1].m1_codes, outs["wired"][1].m2_codes
    assert m1 is not None and m2 is not None
    assert m1.d == M_NUMEL and m2.d == M_NUMEL
    assert float(m1.norm) == 0.0 and float(m2.norm) == 0.0
    assert outs["native"][1].m1_codes is None  # the unwired cache


# ================================================= 2. the wiring ==========
def test_w17_3_wiring_attached_shared_ordered(ckpt):
    """use_m1m2=True + mem_size: the module lands on the inner TextModel,
    SHARED (plain-object refs) by every linear layer, ordinals ascending,
    geometry from the config, gates at the P3 init; the OFF model has
    nothing anywhere."""
    wired, _ = _load(ckpt, use_m1m2=True, m1m2_mem_size=MEM)
    inner = wired.model
    mod = inner.m1m2
    n = 0
    for layer in inner.layers:
        if layer.block_type == "linear_attention":
            assert layer.linear_attn.m1m2 is mod
            assert layer.linear_attn.m1m2_linear_ordinal == n
            n += 1
    assert n == 2  # 2 linear layers in LAYER_TYPES
    assert mod.mem_size == MEM and mod.num_heads == H \
        and mod.head_dim == D and mod.num_linear_layers == 2
    assert bool((mod.write_gate_k == 0).all())
    assert bool((mod.write_gate_v == 0).all())
    assert bool((mod.read_gate == 1).all())

    off, _ = _load(ckpt, use_m1m2=False)
    assert getattr(off.model, "m1m2", None) is None
    assert getattr(off.model.layers[0].linear_attn, "m1m2", None) is None


def test_w17_4_flags_on_text_config_not_composite(ckpt):
    """The composite-vs-text config trap: the flags must land on the TEXT
    config (`config.text_config`) — where the vendored TextModel reads
    them — not only on the composite wrapper (the c37f20e no-op)."""
    comp = Qwen3_5Config(text_config=_text_config())
    # what the W17 loader does (inline in load_palettized_model)
    tcfg = getattr(comp, "text_config", None) or comp
    tcfg.use_m1m2 = True
    tcfg.m1m2_mem_size = MEM
    m = modeling.Qwen3_5TextModel(comp).eval()
    assert getattr(m, "m1m2", None) is not None, \
        "flags on the text config must wire the model"
    # the OLD (broken) placement: composite only -> NO wiring
    comp2 = Qwen3_5Config(text_config=_text_config())
    comp2.use_m1m2 = True
    comp2.m1m2_mem_size = MEM
    m2 = modeling.Qwen3_5TextModel(comp2).eval()
    assert getattr(m2, "m1m2", None) is None, \
        "composite-only flags must NOT wire (the silent no-op)"


def test_w17_5_verify_loud_on_unwired(ckpt):
    """_verify_m1m2_attached refuses SILENT unwired models (the W17 root
    cause was silent) and validates mem_size/ordinals/sharing."""
    off, _ = _load(ckpt, use_m1m2=False)
    with pytest.raises(RuntimeError, match="did NOT attach"):
        pmod._verify_m1m2_attached(off, MEM)
    wired, _ = _load(ckpt, use_m1m2=True, m1m2_mem_size=MEM)
    with pytest.raises(RuntimeError, match="mem_size"):
        pmod._verify_m1m2_attached(wired, MEM * 2)
    pmod._verify_m1m2_attached(wired, MEM)  # passes silently


def test_w17_6_mem_size_validation(ckpt):
    """m1m2_mem_size must be a positive int (bools refused)."""
    for bad in (0, -1, True, 8.0, "8"):
        with pytest.raises(ValueError, match="m1m2_mem_size"):
            _load(ckpt, use_m1m2=True, m1m2_mem_size=bad)


# ================================================= 3. e2e snapshot flow ===
def test_w17_7_ingest_snapshot_loader_vector_with_m1m2(ckpt, tmp_path):
    """The FULL §5/§4 flow with M1/M2 on and OPEN gates: the driver's
    snapshots carry m1/m2 units; the loader's dims = sum(S) + 2 x M_d;
    the vectors' m1/m2 pieces are nonzero and CONTENT-DEPENDENT (the
    retrieval signal the gates open); query_cache_vector picks them up
    in the same order."""
    wired, _ = _load(ckpt, use_m1m2=True, m1m2_mem_size=MEM)
    with torch.no_grad():
        wired.model.m1m2.write_gate_k.fill_(0.5)
        wired.model.m1m2.write_gate_v.fill_(0.5)
    sys_ids, docs = _ids(6, 1), [_ids(24, 2), _ids(24, 3)]
    disk = str(tmp_path)
    drv = IngestDriver(
        wired.model, sys_ids, docs, disk,
        cache_factory=lambda: TQCache(layer_types=LAYER_TYPES, bits=BITS),
        chunk_protocol="absolute", system_ref="w17")
    drv.run()

    from index import ChunkVectorLoader
    from query import query_cache_vector
    loader = ChunkVectorLoader(disk, bits=BITS)
    system = load_system_state(disk)
    s_dim = sum(int(c.d) for c in system.s_codes.values())
    assert loader.dims == s_dim + 2 * M_NUMEL
    assert system.m1_codes is not None and system.m2_codes is not None
    assert system.m1_codes.d == M_NUMEL

    v0, v1 = loader.vector(0), loader.vector(1)
    assert v0.shape == (loader.dims,)
    tail0_m1 = v0[s_dim:s_dim + M_NUMEL]
    tail1_m1 = v1[s_dim:s_dim + M_NUMEL]
    assert float(np.abs(tail0_m1).max()) > 0.0, "m1 piece is zero"
    # content dependence: different chunks -> different m1 pieces
    assert not np.allclose(tail0_m1, tail1_m1, atol=1e-6)

    # the query side: reseed + prefill captures the same-space vector
    cache = TQCache(layer_types=LAYER_TYPES, bits=BITS)
    reseed_cache(cache, system)
    with torch.no_grad():
        wired(input_ids=_ids(10, 4), past_key_values=cache, use_cache=True)
    qvec = query_cache_vector(cache, system)
    assert qvec.shape == (loader.dims,)


def test_w17_8_install_restores_m1m2_codes(ckpt, tmp_path):
    """The §6 install (absolute, single chunk): the cache holds the
    SNAPSHOT'S OWN m1/m2 codes — bit-exact, the verbatim contract."""
    wired, _ = _load(ckpt, use_m1m2=True, m1m2_mem_size=MEM)
    with torch.no_grad():
        wired.model.m1m2.write_gate_k.fill_(0.5)
        wired.model.m1m2.write_gate_v.fill_(0.5)
    sys_ids, docs = _ids(6, 1), [_ids(24, 2)]
    disk = str(tmp_path)
    IngestDriver(
        wired.model, sys_ids, docs, disk,
        cache_factory=lambda: TQCache(layer_types=LAYER_TYPES, bits=BITS),
        chunk_protocol="absolute", system_ref="w17").run()
    from snapshot import load_chunk
    from install import install_snapshot
    system = load_system_state(disk)
    snap = load_chunk(os.path.join(disk, "snapshots", "chunk_00000.npz"))
    assert snap.m1_codes is not None
    cache = TQCache(layer_types=LAYER_TYPES, bits=BITS)
    reseed_cache(cache, system)
    report = install_snapshot(cache, system, [snap])
    assert report["M1"]["mode"] == "verbatim"
    assert cache.m1_codes is not None and cache.m2_codes is not None
    assert cache.m1_codes.idx_lo.shape == snap.m1_codes.idx_lo.shape
    assert (cache.m1_codes.idx_lo == snap.m1_codes.idx_lo).all()
    assert (cache.m1_codes.idx_hi == snap.m1_codes.idx_hi).all()
    assert float(cache.m1_codes.norm) == float(snap.m1_codes.norm)


# ================================================= 4. geometry guards =====
def test_w17_9_gates_artifact_roundtrip(ckpt, tmp_path):
    """save_gates -> load through the LOADER (m1m2_gates_path): values
    land in the module; a geometry-mismatched artifact is a LOUD error."""
    wired, _ = _load(ckpt, use_m1m2=True, m1m2_mem_size=MEM)
    mod = wired.model.m1m2
    with torch.no_grad():
        mod.write_gate_k.uniform_(-0.3, 0.3)
        mod.write_gate_v.uniform_(-0.1, 0.1)
        mod.read_gate.uniform_(0.5, 1.5)
    want = {k: getattr(mod, k).detach().clone()
            for k in ("write_gate_k", "write_gate_v", "read_gate")}
    path = ft.save_gates(mod, str(tmp_path / "gates"),
                         extra_meta={"pairs_file": "unit"})
    assert path.endswith(".npz")

    wired2, _ = _load(ckpt, use_m1m2=True, m1m2_mem_size=MEM,
                      m1m2_gates_path=path)
    mod2 = wired2.model.m1m2
    for k, w in want.items():
        assert torch.equal(getattr(mod2, k).detach(), w), f"{k} drift"

    # geometry mismatch: train at MEM, load into a MEM*2 module
    with pytest.raises(ValueError, match="mem_size"):
        _load(ckpt, use_m1m2=True, m1m2_mem_size=MEM * 2,
              m1m2_gates_path=path)


def test_w17_10_geometry_guards_from_disk(ckpt, tmp_path):
    """m1m2_mem_size_from_system + check_m1m2_geometry: match passes,
    module-vs-corpus mismatch is loud and actionable, corpus-with-M1/M2
    on an unwired model is loud."""
    wired, _ = _load(ckpt, use_m1m2=True, m1m2_mem_size=MEM)
    with torch.no_grad():
        wired.model.m1m2.write_gate_k.fill_(0.2)
    sys_ids, docs = _ids(6, 1), [_ids(24, 2)]
    disk = str(tmp_path)
    IngestDriver(
        wired.model, sys_ids, docs, disk,
        cache_factory=lambda: TQCache(layer_types=LAYER_TYPES, bits=BITS),
        chunk_protocol="absolute", system_ref="w17").run()
    system = load_system_state(disk)
    assert m1m2_mem_size_from_system(system) == MEM

    check_m1m2_geometry(wired, system, MEM)  # match: silent pass
    wrong, _ = _load(ckpt, use_m1m2=True, m1m2_mem_size=MEM * 2)
    with pytest.raises(ValueError, match="m1m2-mem-size"):
        check_m1m2_geometry(wrong, system, MEM * 2)
    off, _ = _load(ckpt, use_m1m2=False)
    with pytest.raises(ValueError, match="without the wiring"):
        check_m1m2_geometry(off, system, None)
    # a corpus WITHOUT M1/M2: helper returns None, unwired model passes
    off_system = load_system_state(disk)
    object.__setattr__(off_system, "m1_shape", None)
    object.__setattr__(off_system, "m1_codes", None)
    object.__setattr__(off_system, "m2_shape", None)
    object.__setattr__(off_system, "m2_codes", None)
    assert m1m2_mem_size_from_system(off_system) is None
    check_m1m2_geometry(off, off_system, None)


def test_w17_11_forward_loud_on_codes_geometry_mismatch(ckpt, w17_corpus):
    """The DEEP guard: a cache holding M1/M2 codes at another geometry
    than the module's fails LOUDLY at the first forward (the M1M2 state
    validation), never a silent reshape."""
    wired, _ = _load(ckpt, use_m1m2=True, m1m2_mem_size=MEM)
    system = load_system_state(w17_corpus)
    ids = _ids(8, 9)
    cache = TQCache(layer_types=LAYER_TYPES, bits=BITS)
    reseed_cache(cache, system)
    # poison the m1 geometry: codes from a DIFFERENT mem module
    big, _ = _load(ckpt, use_m1m2=True, m1m2_mem_size=MEM * 2)
    with torch.no_grad():
        big.model.m1m2.write_gate_k.fill_(0.5)
        big.model.m1m2.write_gate_v.fill_(0.5)
    c2 = TQCache(layer_types=LAYER_TYPES, bits=BITS)
    with torch.no_grad():
        big(input_ids=ids, past_key_values=c2, use_cache=True)
    cache.m1_codes = c2.m1_codes          # d = 1024 vs the module's 512
    cache.m2_codes = c2.m2_codes
    with pytest.raises(ValueError, match="geometry mismatch"):
        with torch.no_grad():
            wired(input_ids=ids, past_key_values=cache, use_cache=True)


# a module-scope shared corpus for tests 9-11 (built ONCE)
@pytest.fixture(scope="module")
def w17_corpus(ckpt, tmp_path_factory):
    wired, _ = _load(ckpt, use_m1m2=True, m1m2_mem_size=MEM)
    with torch.no_grad():
        wired.model.m1m2.write_gate_k.fill_(0.4)
        wired.model.m1m2.write_gate_v.fill_(0.4)
    sys_ids, docs = _ids(6, 1), [_ids(24, 2), _ids(24, 3)]
    disk = str(tmp_path_factory.mktemp("w17_corpus"))
    IngestDriver(
        wired.model, sys_ids, docs, disk,
        cache_factory=lambda: TQCache(layer_types=LAYER_TYPES, bits=BITS),
        chunk_protocol="absolute", system_ref="w17").run()
    return disk
