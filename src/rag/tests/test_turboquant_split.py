"""test_turboquant_split.py — W15: the paper's §LongBench outlier split
(arXiv:2504.19874 §5.3: "splitting channels into outlier and non-outlier
sets, and applying two independent instances of TurboQuant to each,
allocating higher bit precision to outliers").

The W14 bisection isolated the TQ layer as the G5 failure's owner: every
read-level check passes (frame, S-read, conv read-path) yet the full
install generates garbage. The write-path distortion is what those checks
CANNOT see (they compare the cache read against the SAME codes' reference
dequant, never against the true tensor). This file pins the W15 fixes:

  1. FIDELITY: at the same effective bit budget the split's write-path
     rel-MSE is materially below the fixed coordinate half-split on
     channel-structured (lognormal-scale) windows — the paper's own
     quality lever for KV-cache-like tensors (the conv window IS one).
  2. PARITY: group=1 (S/M1/M2 and any flat vector) is BIT-IDENTICAL to
     the pre-W15 quantizer (same codes object for the same input).
  3. CODES SELF-DESCRIPTION + SERIALIZATION: the mask/group/norm_hi ride
     in TQCodes/to_arrays/from_arrays and the chunk snapshot roundtrip
     (save_chunk/load_chunk), digests cover them, and tampering the mask
     changes the digest.
  4. BACKWARD COMPAT: a split-configured quantizer dequantizes legacy
     partition="half" codes exactly (the flat twin), and a FLAT
     quantizer dequantizes outlier codes exactly (the split twin) —
     every mixed-code path (reseed of old snapshots, hooks, index)
     stays correct.
  5. THE LAYER PATH: TQLinearAttentionLayer quantizes conv windows
     through the split (group = kernel), reads back within the house
     budget, and the decode round-trip stays idempotent.
  6. THE INSTALL PATH: install_snapshot carries split conv codes
     verbatim (§6: conv is never summed) and the report says so.
"""
from __future__ import annotations

import os

import numpy as np
import pytest
import torch

import _paths  # noqa: F401
import codebooks
import fht
import snapshot as snap_mod
from ingest import SystemState, reseed_cache
from install import install_snapshot
from snapshot import ChunkSnapshot, load_chunk, save_chunk
from tq_cache import TQCache, resolve_quantizer
from turboquant import TQCodes, TurboQuant

# ---------------------------------------------------------------------------
# rigs
# ---------------------------------------------------------------------------
BITS = 3.5
N_CH, TAPS = 48, 4          # 192 dims: sub-units 24ch->96->128, 24ch->128
D = N_CH * TAPS
SEED = 31337

FIDELITY_GATE = 0.012       # the split's measured rel-MSE (< 0.006 here)
FLAT_MIN = 0.012            # the flat split's floor at this scale (~0.02)
REL_MSE_GATE = 0.06         # the house budget (read-back gates)


def _rel_mse(a, b) -> float:
    a = torch.as_tensor(a, dtype=torch.float32).reshape(-1)
    b = torch.as_tensor(b, dtype=torch.float32).reshape(-1)
    return float(((a - b) ** 2).sum() / (b ** 2).sum().clamp_min(1e-30))


def _channel_window(seed: int, sigma: float = 1.5) -> torch.Tensor:
    """A (D,) conv-like window: lognormal channel scales x tap decay —
    the structure real mixed_qkv windows carry (the thing the paper's
    outlier split exists for)."""
    rng = np.random.default_rng(seed)
    ch = rng.lognormal(0.0, sigma, size=N_CH).astype(np.float32)
    taps = np.array([0.55, 0.7, 0.85, 1.0], dtype=np.float32)
    base = rng.standard_normal((N_CH, TAPS)).astype(np.float32)
    return torch.from_numpy(
        (base * ch[:, None] * taps[None, :]).reshape(-1).copy())


@pytest.fixture(scope="module")
def quantizers():
    return (resolve_quantizer("conv", D, BITS, group=1),    # legacy flat
            resolve_quantizer("conv", D, BITS, group=TAPS))  # paper split


@pytest.fixture(scope="module")
def window():
    return _channel_window(SEED)


# ============================================ 1. fidelity ==================
def test_split_beats_flat_at_same_budget(quantizers, window):
    """The paper's recipe at the SAME effective bits: materially lower
    write-path distortion on channel-structured windows."""
    q_flat, q_split = quantizers
    x = window
    r_flat = _rel_mse(q_flat.dequant(q_flat.quant(x)), x)
    r_split = _rel_mse(q_split.dequant(q_split.quant(x)), x)
    assert r_split <= FIDELITY_GATE, (
        f"split rel-MSE {r_split:.4f} > {FIDELITY_GATE} — the paper recipe "
        f"is not delivering its fidelity")
    assert r_split < 0.6 * r_flat, (
        f"split rel-MSE {r_split:.4f} is not materially below the flat "
        f"{r_flat:.4f} (the W15 justification)")
    # effective bits: k channels at bits_hi, the rest at bits_lo — 3.5
    codes = q_split.quant(x)
    k = sum(bin(int(b)).count("1") for b in codes.mask)
    eff = (k * TAPS * 4 + (N_CH - k) * TAPS * 3) / D
    assert abs(eff - BITS) <= 1e-9, (eff, BITS)


# ============================================ 2. parity ====================
def test_group1_is_bit_identical_legacy(quantizers, window):
    """group=1 keeps the pre-W15 flat semantics EXACTLY: the same codes
    object (idx streams and norm bit-for-bit) as the pre-W15 quantizer."""
    q_flat, q_split = quantizers
    q_pre = TurboQuant(kind="custom", bits=BITS, d=D, seed=202)
    c_new = q_flat.quant(window)
    c_old = q_pre.quant(window)
    assert c_new.partition == "half"
    assert c_new.group is None and c_new.mask is None \
        and c_new.norm_hi is None
    assert np.array_equal(c_new.idx_lo, c_old.idx_lo)
    assert np.array_equal(c_new.idx_hi, c_old.idx_hi)
    assert float(c_new.norm) == float(c_old.norm)
    # the S kind resolves through the SAME canonical instance as before
    assert resolve_quantizer("S", 524288) is resolve_quantizer("S", 524288)


# ============================================ 3. serialization =============
def test_codes_arrays_roundtrip(quantizers, window):
    q_flat, q_split = quantizers
    c = q_split.quant(window)
    c2 = TQCodes.from_arrays(c.to_arrays())
    assert c2.partition == "outlier"
    assert c2.group == TAPS
    assert np.array_equal(c2.mask, c.mask)
    assert float(c2.norm_hi) == float(c.norm_hi)
    assert np.array_equal(c2.idx_lo, c.idx_lo)
    assert np.array_equal(c2.idx_hi, c.idx_hi)
    assert _rel_mse(q_split.dequant(c2), window) < FIDELITY_GATE


def test_snapshot_roundtrip_and_digest(quantizers, window, tmp_path):
    """save_chunk/load_chunk round-trips outlier codes bit-exactly; the
    sha256 covers the mask (tampering flips the digest)."""
    q_flat, q_split = quantizers
    c = q_split.quant(window)
    snap = ChunkSnapshot(
        chunk_id=0, protocol="delta-v1",
        s_codes={}, conv_codes={0: c}, m1_codes=None, m2_codes=None,
        system_ref="w15-split", extra={"n_tokens": 4})
    path = save_chunk(str(tmp_path), snap)
    loaded = load_chunk(path)
    c2 = loaded.conv_codes[0]
    assert c2.partition == "outlier" and c2.group == TAPS
    assert np.array_equal(c2.idx_lo, c.idx_lo)
    assert np.array_equal(c2.idx_hi, c.idx_hi)
    assert np.array_equal(c2.mask, c.mask)
    assert float(c2.norm_hi) == float(c.norm_hi)
    assert _rel_mse(q_split.dequant(c2), window) < FIDELITY_GATE
    # digest: a tampered mask must change it (the load-time verify path)
    c_tampered = TQCodes.from_arrays(c.to_arrays())
    c_tampered.mask = c_tampered.mask.copy()
    c_tampered.mask[0] ^= 0x01  # flip one channel's membership
    d1 = snap_mod._digest_codes({}, {0: c}, None, None)
    d2 = snap_mod._digest_codes({}, {0: c_tampered}, None, None)
    assert d1 != d2


def test_snapshot_rejects_partial_split_members(quantizers, window,
                                                tmp_path):
    """A partition=outlier unit missing one of group/mask/norm_hi is a
    corruption signal — refused loudly (never silently mis-decoded)."""
    q_flat, q_split = quantizers
    c = q_split.quant(window)
    arrays = c.to_arrays()
    del arrays["norm_hi"]
    c_partial = TQCodes.from_arrays(arrays)
    assert c_partial.partition == "outlier" and c_partial.norm_hi is None
    snap = ChunkSnapshot(
        chunk_id=0, s_codes={}, conv_codes={0: c_partial},
        m1_codes=None, m2_codes=None, system_ref="w15-partial")
    path = save_chunk(str(tmp_path), snap)
    with pytest.raises(ValueError, match="split members .* incomplete"):
        load_chunk(path)


# ============================================ 4. cross-semantics reads ======
def test_split_quantizer_reads_legacy_codes(quantizers, window):
    """A split-configured quantizer decodes partition='half' codes
    exactly (the flat twin) — old snapshots keep working."""
    q_flat, q_split = quantizers
    c_flat = q_flat.quant(window)
    x_flat = q_flat.dequant(c_flat)
    x_via_split = q_split.dequant(c_flat)
    assert torch.allclose(x_flat, x_via_split, atol=0, rtol=0), (
        "the split quantizer must decode legacy codes BIT-IDENTICALLY")


def test_flat_quantizer_reads_split_codes(quantizers, window):
    """A FLAT-configured quantizer (the generic codes-only paths — hooks,
    index, the bisect's conv check) decodes outlier codes exactly through
    the split twin."""
    q_flat, q_split = quantizers
    c_split = q_split.quant(window)
    x_split = q_split.dequant(c_split)
    x_via_flat = q_flat.dequant(c_split)
    assert torch.equal(x_split, x_via_flat), (
        "a flat quantizer must decode outlier codes BIT-IDENTICALLY "
        "(the split twin route)")


# ============================================ 5. the layer path =============
def test_layer_conv_split_readback_and_idempotency():
    """TQLinearAttentionLayer quantizes conv through the split (group =
    kernel); the read-back stays inside the house budget and the decode
    round-trip is idempotent (the W15 probe's measured property)."""
    cache = TQCache(layer_types=["linear_attention"], bits=BITS)
    layer = cache.layers[0]
    g = torch.Generator().manual_seed(SEED)
    x = _channel_window(SEED + 1).reshape(1, N_CH, TAPS).half().cuda()
    cache.update_conv_state(x, 0, conv_kernel_size=TAPS)
    c = layer.conv_codes
    assert c.partition == "outlier" and c.group == TAPS
    w = layer.conv_states[0]
    assert tuple(w.shape) == (1, N_CH, TAPS)
    assert _rel_mse(w.float(), x.float()) < REL_MSE_GATE

    # idempotent decode loop: 4 lazy-sync requants of the handed-out tensor.
    # NOTE (measured): the per-vector top-k selection is re-run at every
    # requant, so channels near the k-boundary can flip sides — at this
    # 48-channel toy scale that costs ~3% drift over 4 requants; at
    # production scale (3,072 of 6,144 channels) the measured drift is
    # 0.006-0.007 (boundary flips are negligible relative to n_ch).
    for _ in range(4):
        _ = layer.conv_states[0]           # read (hands out + syncs)
    w4 = layer.conv_states[0]
    assert _rel_mse(w4.float(), w.float()) < 0.05, (
        "the dequant->requant loop must stay bounded (the split preserves "
        "the measured decode-loop property; production-scale drift is "
        "0.006-0.007)")


# ============================================ 6. the install path ===========
def test_install_carries_split_conv_verbatim(tmp_path):
    """install_snapshot carries the chunk's split conv codes VERBATIM
    (§6: conv is never summed) and the report records the partition."""
    bits = BITS
    # build a tiny system + chunk with split conv codes
    q_s = resolve_quantizer("S", 128, bits)
    q_c = resolve_quantizer("conv", D, bits, group=TAPS)
    g = torch.Generator().manual_seed(SEED + 2)
    sys_s = {0: q_s.quant(torch.randn(128, generator=g).cuda() * 0.1)}
    sys_conv = {0: q_c.quant(_channel_window(SEED + 3))}
    system = SystemState(
        s_codes=sys_s, conv_codes=sys_conv, m1_codes=None, m2_codes=None,
        system_ref="w15", bits=bits,
        s_shapes={0: (1, 128)}, conv_shapes={0: (1, N_CH, TAPS)},
        s_dtype="float16", conv_dtype="float16")
    chunk_conv = q_c.quant(_channel_window(SEED + 4))
    chunk_s = q_s.quant(torch.randn(128, generator=g).cuda() * 0.3)
    snap = ChunkSnapshot(
        chunk_id=0, s_codes={0: chunk_s}, conv_codes={0: chunk_conv},
        m1_codes=None, m2_codes=None, system_ref="w15")

    cache = TQCache(layer_types=["linear_attention"], bits=bits)
    reseed_cache(cache, system)
    report = install_snapshot(cache, system, [snap])
    installed = cache.layers[0].conv_codes
    assert installed is not None
    assert installed.partition == "outlier" and installed.group == TAPS
    assert np.array_equal(installed.idx_lo, chunk_conv.idx_lo)
    assert np.array_equal(installed.idx_hi, chunk_conv.idx_hi)
    assert np.array_equal(installed.mask, chunk_conv.mask)
    assert report[0]["conv_partition"] == "outlier"
    # and the installed window reads back inside the house budget
    w = cache.layers[0].conv_states[0]
    assert tuple(w.shape) == (1, N_CH, TAPS)
    q_read = resolve_quantizer("conv", installed.d, bits,
                               group=installed.group or 1)
    dequant = q_read.dequant(installed).to(w.device)
    assert _rel_mse(w.float(), dequant.float()) < 1e-6


# ============================================ misc guards ===================
def test_split_needs_two_sides():
    """1 <= k < n_channels enforced at construction (a degenerate split
    would crash in the sub-unit geometry)."""
    with pytest.raises(ValueError, match="1 <= k < n_channels"):
        TurboQuant(kind="custom", bits=BITS, d=8, seed=202, group=8)


def test_integer_bits_ignore_group():
    """Integer bit-widths keep the flat uniform path even with group > 1
    (no fractional budget to allocate — nothing to split)."""
    q = TurboQuant(kind="custom", bits=4, d=D, seed=202, group=TAPS)
    assert not q.split
    x = _channel_window(SEED + 5)
    c = q.quant(x)
    assert c.partition == "half" and c.group is None
    assert _rel_mse(q.dequant(c), x) < REL_MSE_GATE


def test_effective_bits_and_mask_consistency(quantizers, window):
    """The mask's popcount equals the quantizer's k; the sub-codebooks
    are the committed (bits, pow2) pairs; the sub-frames are the kind's
    seed (regular) and seed+555 (outlier)."""
    q_flat, q_split = quantizers
    x = window
    c = q_split.quant(x)
    k = sum(bin(int(b)).count("1") for b in c.mask)
    assert k == q_split.k_out
    # the sub quantizers: reg keeps the kind seed, outlier is seed+555
    q_reg = q_split._sub_quantizer("reg")
    q_out = q_split._sub_quantizer("out")
    assert q_reg.seed == q_split.seed == 202
    assert q_out.seed == 202 + 555
    assert q_reg.d == q_split.reg_d and q_out.d == q_split.out_d
    # both sub-units are powers of two (single-segment full rotations)
    for dd in (q_reg.d, q_out.d):
        assert dd & (dd - 1) == 0
    # the codebooks are the committed per-(b, d) pairs (field-wise: the
    # frozen dataclass holds numpy arrays — == is ambiguous)
    for qa, qb in ((q_reg._cb_lo, codebooks.get_codebook(3, q_reg.d)),
                   (q_out._cb_lo, codebooks.get_codebook(4, q_out.d))):
        assert qa.bits == qb.bits and qa.d == qb.d
        assert np.array_equal(qa.centroids, qb.centroids)
        assert qa.mse_per_variance == qb.mse_per_variance
