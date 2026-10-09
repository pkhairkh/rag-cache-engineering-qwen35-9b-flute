"""test_turboquant.py — W1.4 acceptance gates for the TurboQuant core.

Covers the W1.4 contract (SPECIFICATION.md §3.1/§3.3/§5; PROPOSAL.md §1.3,
D2, D3; TurboQuant, arXiv:2504.19874):

  1.  FHT binding: fht_apply(x, q._signs) == x @ build_rotation_matrix(d,
      seed) (the explicit ground-truth rotation) and fht_adjoint inverts
      fht_apply (T orthogonal).
  2.  Round-trip MSE at uniform b in {1,2,3,4}: E||x - x~||^2 <=
      2.72 * 4^-b (the paper's 2.7x-of-Shannon-LB guarantee) over random
      unit vectors; b=2 additionally within 5% of the paper constant 0.117.
  3.  3.5-bit split arithmetic (D2: the fractional part is the hi-set
      fraction -> 50/50 halves realize exactly 3.5 bits).
  4.  pack_bits/unpack_bits lossless bit packing (>= 5000 indices, both
      edge values, the 4-bit fast path, odd lengths) + loud rejection of
      out-of-range indices.
  5.  TQCodes serialization: quant -> to_arrays -> from_arrays (dict AND
      npz round-trip) -> bit-identical dequant (torch.equal).
  6.  Full-chunk size gate (spec §3.3/§5, THE full-scale test, @slow):
      24 S + 24 conv + 1 M1 + 1 M2 units at 3.5 bits -> ~6.0 MiB.
  7.  Zero-norm edge: norm 0.0, dequant returns exact zeros.
  8.  D3 frame consistency: same kind -> same seed/signs; get_quantizer
      registry identity; same vector -> byte-identical codes.
  9.  Mismatch rejection: codes from a different seed/kind/bits/d raise
      ValueError.
  10. fp16 round-trip: output dtype fp16, MSE within 1.15x of the fp32
      path for the same vector.

Determinism: every random draw goes through a torch.Generator pinned to a
fixed seed. No network, no GPU (the FHT runs its reference backend).
Small dims use kind="custom": d=1024 hits the committed cb_b*_d1024
caches (and d=256 solves fresh into a throwaway tmp dir — tests never
write into src/rag/codebooks/); full-scale d=2^19 is touched only by the
single @slow size-gate test.

NOTE on gate 3 (reported to the orchestrator, source untouched): the W1.4
brief expected bits=2.5 -> n_hi == 256. That traces to the paper's outlier
worked example — arXiv:2504.19874 main.tex: "32 outlier channels at 3
bits, 96 at 2 bits -> (32*3 + 96*2)/128 = 2.5" — whose arithmetic is
actually 2.25 (a typo in the paper; the repo's default split is the
data-oblivious 50/50 partition, the outlier split is the W9 flag). The
module's documented semantics (turboquant.py docstring "the fractional
part choosing the lo/hi coordinate ratio (0.5 -> half/half)"; PROPOSAL D2
"half the coordinates at 3 bits, half at 4 bits -> exactly 3.5") make the
fractional part the hi-set fraction, so 2.5 -> 512 at 3 bits + 512 at 2
bits = exactly 2.5 effective bits. This test pins THAT self-consistent
arithmetic (256 would realize only 2.25 bits).
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

import fht
import codebooks
from turboquant import (
    KINDS,
    SEEDS,
    TQCodes,
    TurboQuant,
    get_quantizer,
    pack_bits,
    unpack_bits,
)

D = 1024                          # small-dim workhorse (committed cb_b*_d1024)
N_VECS = 30                       # unit vectors per MSE measurement
MSE_B2_PAPER = 0.117              # the paper's rounded b=2 distortion constant
SHANNON_FACTOR = 2.72             # paper: D_mse within 2.7x of the Shannon LB
FHT_SEEDS = (7, 42)               # the brief's rotation-binding seeds


def _unit_vectors(d: int, n: int, seed: int) -> torch.Tensor:
    """n deterministic random unit vectors, shape (n, d), fp32."""
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, d, generator=g, dtype=torch.float32)
    return x / x.norm(dim=-1, keepdim=True)


def _quantizer(bits: float = 3.5, d: int = D, seed: int = 1) -> TurboQuant:
    """A small-dim custom-kind quantizer (tests never touch prod seeds)."""
    return TurboQuant(kind="custom", bits=bits, d=d, seed=seed)


# --- gate 1: FHT binding vs the explicit ground-truth rotation ------------ #
@pytest.mark.parametrize("seed", FHT_SEEDS)
def test_fht_binding_matches_rotation_matrix(seed):
    q = _quantizer(bits=3.5, d=D, seed=seed)
    # TurboQuant wires exactly fht.rotation_signs(d, seed) (the D3 frame)
    assert torch.equal(q._signs, fht.rotation_signs(D, seed))

    g = torch.Generator().manual_seed(seed)
    x = torch.randn(8, D, generator=g, dtype=torch.float32)
    T = fht.build_rotation_matrix(D, seed)          # the explicit (D, D) truth

    y = fht.fht_apply(x, q._signs)                  # y = x @ T
    assert (y - x @ T).abs().max().item() < 1e-4

    back = fht.fht_adjoint(y, q._signs)             # (x @ T) @ T^T = x
    assert (back - x).abs().max().item() < 1e-5


# --- gate 2: round-trip MSE at uniform bit widths -------------------------- #
@pytest.mark.parametrize("b", [1, 2, 3, 4])
def test_roundtrip_mse_uniform_bits(b):
    q = _quantizer(bits=b, d=D, seed=7)
    x = _unit_vectors(D, N_VECS, seed=1000 + b)

    sq_err = 0.0
    for i in range(N_VECS):
        x_rec = q.roundtrip(x[i])
        assert x_rec.shape == (D,) and x_rec.dtype == torch.float32
        sq_err += ((x[i] - x_rec) ** 2).sum().item()
    mse = sq_err / N_VECS

    gate = SHANNON_FACTOR * 4.0 ** (-b)
    assert mse <= gate, f"b={b}: mse={mse:.4f} > gate {gate:.5f}"
    if b == 2:  # paper-constant anchor (0.117 is the rounded 0.11748)
        assert abs(mse / MSE_B2_PAPER - 1.0) < 0.05, f"b=2: {mse:.4f} vs 0.117"


# --- gate 3: 3.5-bit split arithmetic (D2) ---------------------------------- #
def test_split_arithmetic(tmp_path, monkeypatch):
    q35 = _quantizer(bits=3.5, d=D, seed=1)
    assert (q35.bits_lo, q35.bits_hi) == (3, 4)
    assert q35.n_lo == 512 and q35.n_hi == 512

    # 2.5: frac 0.5 is the hi-set fraction (see the module docstring NOTE) —
    # 512 coords at 3 bits + 512 at 2 bits = exactly 2.5 effective bits.
    q25 = _quantizer(bits=2.5, d=D, seed=1)
    assert (q25.bits_lo, q25.bits_hi) == (2, 3)
    assert q25.n_hi == 512 and q25.n_lo == 512

    # uniform bit width: no hi set at all
    q4 = _quantizer(bits=4, d=D, seed=1)
    assert q4.bits_lo == q4.bits_hi == 4
    assert q4.n_lo == D and q4.n_hi == 0

    # a second, smaller power-of-two dim. No committed cache exists for
    # d=256 — solve into a throwaway dir so the repo tree stays pristine
    # (codebooks.get_codebook would otherwise save into src/rag/codebooks).
    monkeypatch.setattr(codebooks, "CACHE_DIR", str(tmp_path))
    q256 = _quantizer(bits=3.5, d=256, seed=1)
    assert q256.n_lo == 128 and q256.n_hi == 128

    # the invariant the split exists to enforce: realized average bits ==
    # requested bits (per coordinate; the fp32 norm is stored unquantized)
    for q_ in (q25, q35, q4, q256):
        assert q_.n_lo + q_.n_hi == q_.d
        assert q_.n_lo * q_.bits_lo + q_.n_hi * q_.bits_hi == \
            round(q_.bits_spec * q_.d)


# --- gate 4: lossless bit packing -------------------------------------------- #
@pytest.mark.parametrize("b", [1, 2, 3, 4])
def test_pack_bits_roundtrip_lossless(b):
    g = torch.Generator().manual_seed(2000 + b)
    idx = torch.randint(0, 2 ** b, (5000,), generator=g).numpy() \
        .astype(np.uint8)
    idx[0], idx[1] = 0, 2 ** b - 1                  # both edge values

    packed = pack_bits(idx, b)
    assert packed.dtype == np.uint8 and packed.ndim == 1
    assert packed.nbytes == (5000 * b + 7) // 8
    back = unpack_bits(packed, b, 5000)
    assert back.dtype == np.uint8
    assert np.array_equal(back, idx)

    # odd lengths: the 4-bit fast path pads to a whole nibble pair, the
    # generic path leaves a partial trailing byte — both must trim to n
    idx_odd = idx[:101].copy()
    p_odd = pack_bits(idx_odd, b)
    assert p_odd.nbytes == (101 * b + 7) // 8
    assert np.array_equal(unpack_bits(p_odd, b, 101), idx_odd)


def test_pack_bits_rejects_out_of_range():
    for b in (1, 2, 3, 4):
        with pytest.raises(ValueError):
            pack_bits(np.array([0, 2 ** b], dtype=np.uint8), b)
    # the bits domain itself is validated loudly too
    for bad in (0, 9):
        with pytest.raises(ValueError):
            pack_bits(np.zeros(4, np.uint8), bad)
        with pytest.raises(ValueError):
            unpack_bits(np.zeros(1, np.uint8), bad, 4)


# --- gate 5: code serialization round-trip ----------------------------------- #
def test_code_serialization_roundtrip(tmp_path):
    q = _quantizer(bits=3.5, d=D, seed=5)
    x = _unit_vectors(D, 1, seed=55)[0]
    codes = q.quant(x)

    # nbytes is exactly the two packed streams + the fp32 norm
    assert codes.nbytes() == codes.idx_lo.nbytes + codes.idx_hi.nbytes + 4
    assert codes.nbytes() == (512 * 3 + 7) // 8 + (512 * 4 + 7) // 8 + 4

    direct = q.dequant(codes)

    # dict round-trip (the snapshot.py embedding contract)
    via_dict = TQCodes.from_arrays(codes.to_arrays())
    assert via_dict.kind == codes.kind and via_dict.d == codes.d
    assert via_dict.seed == codes.seed
    assert via_dict.partition == codes.partition == "half"
    assert via_dict.bits_lo == codes.bits_lo and via_dict.bits_hi == codes.bits_hi
    assert via_dict.n_lo == codes.n_lo and via_dict.n_hi == codes.n_hi
    assert float(via_dict.norm) == float(codes.norm)
    assert np.array_equal(via_dict.idx_lo, codes.idx_lo)
    assert np.array_equal(via_dict.idx_hi, codes.idx_hi)
    assert torch.equal(q.dequant(via_dict), direct)

    # npz round-trip: exactly the path a chunk snapshot writes and reads
    path = str(tmp_path / "unit_codes.npz")
    np.savez(path, **codes.to_arrays())
    loaded = TQCodes.from_arrays(dict(np.load(path)))
    assert loaded.kind == codes.kind and loaded.d == codes.d
    assert loaded.seed == codes.seed and loaded.partition == codes.partition
    assert float(loaded.norm) == float(codes.norm)
    assert np.array_equal(loaded.idx_lo, codes.idx_lo)
    assert np.array_equal(loaded.idx_hi, codes.idx_hi)
    assert torch.equal(q.dequant(loaded), direct)


# --- gate 6: full-chunk size gate (spec §3.3/§5) — the full-scale test ------ #
@pytest.mark.slow
def test_full_chunk_size_gate():
    # ONE synthetic chunk = 24 S + 24 conv + 1 M1 + 1 M2 units at 3.5 bits.
    # Structural arithmetic (fp32 norm per unit, 4 B):
    #   S / M1 / M2 unit: ceil(262144*3/8) + 262144*4/8 + 4 = 229,380 B
    #   conv unit:        ceil(16384*3/8)  +  16384*4/8 + 4 =  14,340 B
    #   chunk total: 26*229,380 + 24*14,340 = 6,308,040 B ~= 6.016 MiB
    # (spec §3.3: ~6.0 MiB per chunk, 4.6x vs fp16.)
    units = (("S", 24), ("conv", 24), ("M1", 1), ("M2", 1))
    quants = {k: TurboQuant(kind=k, bits=3.5) for k, _ in units}
    g = torch.Generator().manual_seed(20260401)

    total = 0
    per_unit = {}
    for kind, count in units:
        q = quants[kind]
        kind_total = 0
        for _ in range(count):
            v = torch.randn(q.d, generator=g, dtype=torch.float32)
            v = v / v.norm()
            kind_total += q.quant(v).nbytes()       # quantize each unit once
        per_unit[kind] = kind_total // count
        total += kind_total

    assert per_unit["S"] == per_unit["M1"] == per_unit["M2"] == 229_380
    assert per_unit["conv"] == 14_340
    # the spec gate: 5.9 .. 6.1 MiB == [6_185_920, 6_396_288] bytes
    assert 6_185_920 <= total <= 6_396_288, f"chunk total = {total} bytes"


# --- gate 7: zero-norm edge --------------------------------------------------- #
def test_zero_norm_unit():
    q = _quantizer(bits=3.5, d=D, seed=1)
    codes = q.quant(torch.zeros(D))
    assert float(codes.norm) == 0.0
    # degenerate codes are well-formed all-zero packed streams
    assert codes.idx_lo.shape == ((512 * 3 + 7) // 8,)
    assert codes.idx_hi.shape == ((512 * 4 + 7) // 8,)
    assert not codes.idx_lo.any() and not codes.idx_hi.any()

    out = q.dequant(codes)
    assert out.dtype == torch.float32
    assert torch.equal(out, torch.zeros(D))
    out16 = q.dequant(codes, dtype=torch.float16)
    assert torch.equal(out16, torch.zeros(D, dtype=torch.float16))


# --- gate 8: D3 frame consistency ---------------------------------------------- #
def test_d3_frame_consistency():
    # two fresh instances of the same kind + bits share the seed and signs
    a = TurboQuant(kind="conv", bits=3.5)
    b = TurboQuant(kind="conv", bits=3.5)
    assert a.seed == b.seed == SEEDS["conv"] == KINDS["conv"][1]
    assert torch.equal(a._signs, b._signs)

    # the registry hands out the same object (one shared rotation per kind)
    assert get_quantizer("S") is get_quantizer("S")

    # the SAME vector under two independently constructed same-seed
    # instances -> byte-identical codes (deterministic, frame-stable)
    q1 = _quantizer(bits=3.5, d=D, seed=9)
    q2 = _quantizer(bits=3.5, d=D, seed=9)
    x = _unit_vectors(D, 1, seed=99)[0]
    c1, c2 = q1.quant(x), q2.quant(x)
    assert float(c1.norm) == float(c2.norm)
    assert np.array_equal(c1.idx_lo, c2.idx_lo)
    assert np.array_equal(c1.idx_hi, c2.idx_hi)
    assert torch.equal(q1.dequant(c1), q2.dequant(c2))


# --- gate 9: mismatch rejection ------------------------------------------------- #
def test_mismatch_rejected():
    q = _quantizer(bits=3.5, d=D, seed=1)
    codes = q.quant(_unit_vectors(D, 1, seed=123)[0])

    with pytest.raises(ValueError):    # different D3 seed (frame drift)
        _quantizer(bits=3.5, d=D, seed=2).dequant(codes)
    with pytest.raises(ValueError):    # different kind label
        TurboQuant(kind="custom-b", bits=3.5, d=D, seed=1).dequant(codes)
    with pytest.raises(ValueError):    # different bit-width split
        _quantizer(bits=3, d=D, seed=1).dequant(codes)
    with pytest.raises(ValueError):    # a real kind (d=32768, seed=202)
        TurboQuant(kind="conv", bits=3.5).dequant(codes)


# --- gate 10: fp16 dtype round-trip ---------------------------------------------- #
def test_fp16_dtype_roundtrip():
    q = _quantizer(bits=3.5, d=D, seed=3)
    x32 = _unit_vectors(D, 1, seed=77)[0]
    x16 = x32.to(torch.float16)

    out16 = q.dequant(q.quant(x16), dtype=torch.float16)
    assert out16.dtype == torch.float16
    assert out16.shape == (D,)

    mse16 = ((x16.float() - out16.float()) ** 2).sum().item()
    out32 = q.dequant(q.quant(x32))
    mse32 = ((x32 - out32) ** 2).sum().item()
    assert mse32 > 0.0
    # fp16 rounding on top of quantization noise: at most a 15% penalty
    assert mse16 <= 1.15 * mse32, f"mse16={mse16:.6f} vs mse32={mse32:.6f}"


# --- the kinds table (construction only; codebooks load from the cache) ---------- #
@pytest.mark.parametrize("kind,d,seed", [
    ("S", 524_288, 101),
    ("conv", 32_768, 202),
    ("M1", 524_288, 303),
    ("M2", 524_288, 404),
])
def test_kinds_d_and_seed(kind, d, seed):
    q = TurboQuant(kind=kind)
    assert q.d == d == KINDS[kind][0]
    assert q.seed == seed == SEEDS[kind]
