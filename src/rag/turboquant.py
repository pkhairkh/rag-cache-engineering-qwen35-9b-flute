"""turboquant.py — the TurboQuant quantizer core (SPECIFICATION.md §3).

Online, data-oblivious vector quantization (arXiv:2504.19874), mapped to
this repo's cache tensors per PROPOSAL.md §1–§2. Torch-native and
device-parametric end-to-end since the W20.3.1–W20.3.4 rewrites — the
REV-2 RUNTIME CONTRACT at the end of this docstring is the binding
statement of the runtime semantics:

  quant(x):    norm (fp, stored) → r = x/‖x‖ → FHT rotation (y = r @ T,
               the repo's `flute_extended/fht.py`, O(d log d)) → per-set
               scalar Lloyd-Max encode (`codebooks.py`) → packed uint8 codes
  dequant:     unpack → centroids → adjoint FHT (y @ Tᵀ) → × norm

(r itself is never materialized on the encode side: the
boundary-scaling identity in quant() folds the normalization into the
codebook boundaries.)

Quantization units (PROPOSAL D2 / SPECIFICATION §3.3–§5):

  kind    d (dims)   units per chunk   FHT stages
  S       524,288    24 (per layer)    19 (2^19, single segment)
  conv    32,768     24 (per layer)    15
  M1      524,288    1                 19
  M2      524,288    1                 19

The 3.5-bit recipe (§3.3, PROPOSAL §1.3/D2): a fixed 50/50 coordinate
split — the first half of coordinates at 3 bits, the second half at 4 —
(no calibration, data-oblivious; the paper's outlier-split variant is the
W9 flag, never the default). Per unit one fp32 norm — a 0-dim float32
TENSOR since the W20.2 codes contract (TQCodes rev-1: never a host
scalar) — is stored and never quantized (the paper's recipe).

Rotation sharing (PROPOSAL D3): ONE rotation instance per tensor kind —
every quantization of a given kind (system prompt, chunk deltas, query,
install sums) uses the same FHT sign vector, fixed seed per kind, so
dequantized deltas live in one shared rotated frame and the IVFADC index
sees a stable original-frame vector. Seeds persist in the index
side-metadata.

Entropy coding is deliberately skipped (paper: ~5% at b=4 — not worth it).
Codes are b-bit indices + one fp norm. Lloyd-Max is not additive:
Q(a+b) ≠ Q(a)+Q(b) — install math (SPECIFICATION §6) therefore requantizes
SUMS of dequantized vectors (see rag/install.py); this module intentionally
offers no code-plus-code operation.

THE W15 OUTLIER SPLIT (the paper's OWN non-integer-bit recipe — the
conv kind's default; see TurboQuant(group=...)): arXiv:2504.19874 §5.3
quantizes KV-cache-like tensors by "splitting channels into outlier and
non-outlier sets, and applying two independent instances of TurboQuant
to each, allocating higher bit precision to outliers". The conv window
(channel, tap) layout quantizes through TWO sub-instances: the top-k
energy channels (k = round(frac·n_ch)) at bits_hi with their OWN
power-of-two full rotation (seed+555) and own fp norm; the rest at
bits_lo with the kind's seed. The membership mask (packed channel
bits), the group and the second norm ride in TQCodes — the codes are
self-describing and dequant routes on codes.partition, so pre-W15
partition="half" snapshots decode UNCHANGED (through the flat twin) and
legacy-only paths (hooks/index) read split codes exactly (through the
split twin). Measured at production conv geometry (channel-structured
windows): ~3x lower write-path rel-MSE than the fixed coordinate
half-split at the SAME effective bits — the fixed split hands the 4-bit
half to a fixed coordinate range that ignores the channel-energy
structure, while the paper's split spends them where the energy is.
group=1 (S/M1/M2 — no channel structure; the full 2^19 rotation already
mixes everything) keeps the legacy flat split, BIT-IDENTICAL.

THE W9.2 `qjl` FLAG (P7 / D1's flagged prod-variant A/B — DEFAULT OFF):
`TurboQuant(..., qjl=True)` additionally computes the structured QJL
residual sketch (PROPOSAL §1.2, the paper's Alg. 2 "TurboQuant_prod"):

  r   = x − dequant_mse(codes)          (ORIGINAL-frame residual; the
                                         rotated-frame form fht(x/‖x‖) − y
                                         is identical by linearity — sign()
                                         is scale-invariant either way)
  γ   = ‖r‖₂                            (one fp32 scalar, never quantized)
  qjl = sign(FHT(r/γ, seed+7777))       (d int8 ±1 — the SECOND
                                         deterministic sign draw, own seed
                                         = the kind's seed + QJL_SEED_OFFSET,
                                         same d as the rotation)
  dequant: x̃ = x̃_mse + (√(π/2)/√d)·γ·FHT_adjoint(qjl)

EXACT FORMULATION (documented per the A/B contract): PROPOSAL §1.2 writes
the paper's estimator as x̃ = x̃_mse + (√(π/2)/d)·γ·Sᵀ·qjl with S iid
N(0,1) d×d — unit-variance entries, rows of norm ~√d. Our substitution
(PROPOSAL §1.5 item 4: "the QJL projection is likewise substituted with a
structured sketch") is S := √d·FHT(d, seed+7777): the sign-flipped
Hadamard scaled to the SAME per-entry variance, whose second moments
match the dense Gaussian exactly (E[S_ik·S_jl] = δ_ij·δ_kl over the ±1
draw). Since sign() is scale-invariant the SKETCH is computed on the
unscaled orthogonal FHT, and the adjoint carries the √d:

  (√(π/2)/d)·γ·Sᵀ·qjl  ==  (√(π/2)/√d)·γ·FHT_adjoint(qjl)

Unbiasedness of ⟨y, x̃⟩ (E = ⟨y, x⟩) survives any JL-valid projection with
the right moments to CLT accuracy — the FHT's projection coordinates are
equal-weight ±1 sums, Gaussian-marginal for generic x — and the VARIANCE
CONSTANT of the structured substitute is exactly what the GPU Phase-5 gate
re-measures (PROPOSAL §1.5(4), D1's A/B decision rule). Two deliberate A/B
simplifications vs. the paper's Alg. 2, both pinned by PROPOSAL D1: (i)
the MSE base layer keeps the SAME bit budget (the paper drops it to b-1;
here the A/B isolates the residual bit at the production budget — "one
extra bit per coordinate", so the effective rate is b+1); (ii) the
retrieval-plane consumer (IVFADC-side IP estimation) is the GPU box's
Phase-5 decision — index.py is deliberately untouched.

Round-trip note: with qjl=True the MSE typically IMPROVES (the residual
is compensated in expectation: E‖r−c‖² ≈ (π/2−1)·γ² vs. γ², with
‖c‖ = √(π/2)·γ exactly); the parity gate covers qjl=False only.
Serialization is BACKWARD-COMPATIBLE: codes quantized without the flag
carry qjl_signs/gamma == None and write NONE of the new keys; from_arrays
loads old npz/snapshots (no qjl keys) with the fields None.

REV-2 RUNTIME CONTRACT (W20.3.5 — the module contract after the
W20.3.1–W20.3.4 torch-native rewrites; this block reconciles the
docstring to the code as it stands):

(a) The torch path is THE device-parametric single path: the input's
    device decides the compute device — there is no CPU-reference
    variant and no device switch. CPU-resident tensors quantize and
    dequantize as CPU torch (the GPU-less dev box's path — this box
    runs static audits only), CUDA-resident tensors compute on CUDA
    with no host round trips. The documented host-visible exceptions:
    quant()'s zero-norm branch, its exact twin in _attach_qjl
    (γ == 0), and TQCodes' metadata-only type cops — every other
    value stays on-device (W20.7's kernel design removes the branch).

(b) The torch path is the differential twin for the kernel family —
    and, since W20.8.2, the CPU/fallback half of the kernel-first
    dispatcher (PROPOSAL §3.7; the REV-3 routing table below): on CUDA
    with the extension present and the geometry eligible, the
    kernel_tq.cu entries (tq_quant_batched / tq_dequant_batched, the
    flute_extended._C surface) take precedence ALWAYS; on CPU, with
    the extension absent, or on ineligible geometry, this torch path
    is the path (the fht.py auto-backend discipline — CUDA kernel
    when built + CUDA input, the torch butterfly otherwise — lifted
    from the rotation up to the TurboQuant level). The twin pins the
    kernels' semantics: a kernel/torch divergence is a bug, never
    drift.

(c) The legacy array era is confined to two OFFLINE edges — this
    module imports no array library. It survives ONLY at (i) the
    TQCodes.to_arrays/from_arrays npz disk edge (tq_codes.py) and
    (ii) the Lloyd-Max codebook solve, the documented
    OFFLINE-NUMPY-ISLAND in codebooks.py (authored offline, cached);
    the quant/dequant paths read its tables only through the 2.2
    DeviceCodebook mirrors — fp32 torch tensors on the compute
    device.

(d) The legacy semantics are pinned by the frozen fixtures (G20.P1):
    the byte layouts (pack_bits_t/unpack_bits_t, the MSB-first split
    mask), the searchsorted tie behavior (leftmost insertion), the
    stable-argsort determinism, and the qjl estimator round-trip.
    Behavioral verification is G20.P1 on the A10G box — this module
    is statically audited on the dev box: py_compile is the only
    box-side guarantee (no test runs, no repo-module imports on this
    box; the consumers are swept in W20.4).

REV-3 THE DISPATCH ROUTING TABLE (W20.8.2 — kernel-first, PROPOSAL
§3.7; device/geometry -> path, evaluated in this order):

  1. CPU input (quant: x's device; dequant: the CODES' device,
     idx_lo's) -> the TORCH reference path. Always — the GPU-less dev
     box's path, and every non-CUDA device's.
  2. Kernel pair absent (flute_extended._C not importable, or built
     without tq_quant_batched/tq_dequant_batched — the module-level
     lazy probe _tq_kernel_available, the fht.py kernel_available
     discipline at the TurboQuant level, probed once per process) ->
     the TORCH path. A performance fallback, never a correctness one.
  3. Ineligible geometry / shape -> the TORCH path:
       * the W15 outlier split (partition="outlier": the channel
         sort, the packed mask, norm_hi) is not fused (yet) — the
         split CONTAINER stays torch; its sub-quantizers are FLAT
         instances that dispatch on their own rows here (production
         conv: 16,384-d sub-units -> this row's d clause -> torch);
       * d <= 16,384 (one 64 KiB butterfly tile is the small-K
         paths' own — so the flat 24,576 production conv window and
         every 16,384-d sub-unit stay torch, their fht_apply still
         riding the small CUDA kernels), d not a power of two,
         d % 32 != 0 (the flat 32,768 canonical conv d and the
         524,288 S/M1/M2 d's ARE eligible);
       * n_lo % 16 != 0 (the 16-coordinate pack-slice ownership —
         misaligned fractional splits);
       * bits_lo / bits_hi outside {3, 4} (the b1/b2 recipes, bit
         specs past 4);
       * quant only: the qjl flag ON — the W9.2 A/B rides the torch
         twin this wave (the fused qjl kernel chain lands W20.9);
       * dequant only: codes carrying the QJL sketch (qjl_signs /
         gamma), or dtype outside {float32, float16} (the kernel
         decode's two output widths; the torch twin casts at the
         very end).
  4. CUDA input + kernel pair present + geometry eligible -> THE
     KERNEL PATH, ALWAYS (§3.7): quant = ONE tq_quant_batched call
     over the unit as a (1, d) row (norm -> batched large-K butterfly
     -> ladder/pack -> the zero-norm sweep — no host branch, no host
     round trip of any kind); dequant = ONE tq_dequant_batched call
     over the packed (1, L) code rows (bfe unpack + smem LUT gather +
     the large-K adjoint + the per-unit norm multiply + the single
     fp32/fp16 store). The decision is device-class + geometry —
     never codes provenance: either path's codes decode through
     either path's dequant. Both paths emit and consume the SAME
     TQCodes rev-1 contract — the kernel path packs the kernel's
     (codes_lo, codes_hi, norms) into exactly the fields the torch
     twin writes (codes as CUDA device tensors, the 0-dim fp32
     norm). The signs and the boundary/centroid tables come from the
     QUANTIZER'S OWN state: the D3 rotation draw (the same (d,) fp32
     vector the torch path feeds fht_apply) mirrored once per device
     (_device_signs), and the 2.2 DeviceCodebook mirrors
     (.boundaries / .centroids, UNSCALED — the kernel applies the
     boundary-scaling identity itself).
     Mixed-device kernel-path inputs (e.g. CUDA codes + a CPU norm,
     CUDA x + CPU-mirrored tables) raise an explicit RuntimeError
     naming the mismatch — never a silent fallback, never a crash
     deep in torch or a kernel-side generic refusal.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple
import math

import torch

import _paths  # noqa: F401  (anchors src/rag, src/scripts, src/flute_extended)
import codebooks
import fht
from tq_codes import TQCodes  # W20.2.1 — the torch-native codes container (rev-1 contract; re-export for callers)

__all__ = [
    # the consumer surface (swept in W20.4): tq_cache imports the
    # module handle (get_quantizer + the process registry) plus
    # KINDS/SEEDS/TQCodes/TurboQuant; install/hooks/snapshot import
    # TQCodes; index imports SEEDS; evals imports the module handle;
    # query/m1m2 consume through tq_cache (no direct import).
    "KINDS", "SEEDS", "TQCodes", "TurboQuant", "get_quantizer",
    "pack_bits_t", "unpack_bits_t", "UNITS_PERM_OFFSET",
]

# ---------------------------------------------------------------- kinds ----
# (kind -> (d, seed)); the seeds are the D3 rotation keys — persisted in the
# index side-metadata by rag/index.py; do not change them once an index
# exists (frame drift would silently corrupt retrieval).
# NOTE (W15): KINDS["conv"]'s canonical d (32,768) is the HISTORICAL
# power-of-two unit. The production conv window (Qwen3.5 in_proj Q+V:
# 1×6144×4 = 24,576) resolves through the CUSTOM-d path (kind="custom",
# seed 202 preserved — the D3 frame contract), and quantizes through the
# paper's outlier split (TurboQuant(group=kernel); see the module
# docstring). The canonical entry is kept UNCHANGED so pre-W15
# 24,576-dim "custom" codes still pass _check_codes against the custom
# path (flipping the canonical d would silently change the resolved
# kind string and strand every existing snapshot).
KINDS: Dict[str, Tuple[int, int]] = {
    "S":   (524_288, 101),   # per-layer recurrent state (spec §2.1)
    "conv": (32_768, 202),   # per-layer conv state (spec §2.3; see NOTE)
    "M1":  (524_288, 303),   # global key-memory (spec §2.2)
    "M2":  (524_288, 404),   # global value-memory (spec §2.2)
}
SEEDS: Dict[str, int] = {k: v[1] for k, v in KINDS.items()}

# W9.2 qjl flag: the QJL sketch's own sign-draw seed offset, added to the
# kind's D3 rotation seed (a SECOND deterministic draw — same d).
QJL_SEED_OFFSET = 7777
# The paper's §LongBench outlier-split recipe ("splitting channels into
# outlier and non-outlier sets, two independent instances of TurboQuant"):
# the OUTLIER sub-quantizer's own D3 frame seed offset, added to the kind's
# seed (the regular sub-quantizer keeps the kind's own seed).
SPLIT_SEED_OFFSET = 555
# W29 — the UNITS frame's coordinate-permutation seed offset (the
# per-head S repair): the units-mode rotation is y = FHT(P x) with P a
# FIXED random coordinate permutation drawn from the kind's seed + this
# offset (deterministic — the frame stays seed-addressed, so the codes
# need NO new serialization fields beyond `units`). WHY the permutation
# is REQUIRED (measured, scripts/w28_tq_sim.py + the Kronecker
# experiment): the d=2^14 Hadamard factors as H_(2^7) (x) H_(2^7), which
# EXACTLY aligns with the 128x128 per-head S matrix layout — a rank-1
# head k (x) v rotates to coordinatewise PRODUCTS of Gaussians
# (heavy-tailed), mismatching the Lloyd-Max codebook: per-head rel-MSE
# 0.087 (rank-1) / 0.056 (rank-8) vs the 0.022 floor. The fixed
# permutation scatters the (i, j) pairs before the butterfly, restoring
# the CLT: 0.0221 on rank-1, rank-8, full-rank and scaled states alike —
# structure-independent, the paper's per-unit floor.
UNITS_PERM_OFFSET = 4545
# The paper's Alg.-2 IP-estimator constant (PROPOSAL §1.2: (√(π/2)/d)·γ·Sᵀ·qjl
# with S unit-variance; folded with S = √d·FHT -> (√(π/2)/√d)·γ·FHT_adjoint).
QJL_C = math.sqrt(math.pi / 2.0)


# ------------------------------------------- kernel dispatch (W20.8.2) ------
# The kernel-first dispatcher's extension probe (PROPOSAL §3.7, the REV-3
# routing table in the module docstring): the fht.py kernel_available /
# _load_kernel lazy idiom, applied at the TurboQuant level. ONE extension
# module (flute_extended._C), ONE loader — fht._load_kernel owns the
# sys.path / stale-namespace-package retry dance; this probe adds the TQ
# surface check (an extension built before W20.7/W20.8.1 lacks the
# tq_quant_batched / tq_dequant_batched pair) and its own
# once-per-process cache. Soft by design: the torch twin is the fallback
# when the pair is absent (CPU boxes, pre-build), so the probe NEVER
# raises — presence is a routing input, not a precondition.
_tq_kernel_module = None
_tq_kernel_probed = False


def _load_tq_kernel():
    """flute_extended._C when it exposes the TQ batched pair, else None.

    Lazy and cached (probed once per process, the fht.py _load_kernel
    discipline): the first call resolves the extension and records the
    answer; every later call is a dict-free global read. Returns the
    module object (the caller binds tq_quant_batched /
    tq_dequant_batched off it) or None when absent."""
    global _tq_kernel_module, _tq_kernel_probed
    if _tq_kernel_probed:
        return _tq_kernel_module
    _tq_kernel_probed = True
    C = fht._load_kernel()
    if C is not None and hasattr(C, "tq_quant_batched") \
            and hasattr(C, "tq_dequant_batched"):
        _tq_kernel_module = C
    return _tq_kernel_module


def _tq_kernel_available(x: Optional[torch.Tensor] = None) -> bool:
    """True when the TQ kernel pair is importable (and x, if given, is a
    CUDA tensor) — the fht.py kernel_available signature discipline,
    applied to the TQ pair."""
    C = _load_tq_kernel()
    if C is None:
        return False
    if x is None:
        return True
    return bool(x.is_cuda)


# The TQ kernels' geometry envelope (kernel_tq.cu's check_args, mirrored —
# the dispatcher's copy; the kernels' own TORCH_CHECKs are the belt):
# d strictly above one 64 KiB butterfly tile, a power of two, 32-aligned
# (implied by pow2 past the tile — kept explicit, the kernel's clause
# order); n_lo a multiple of the 16-coordinate pack slice; bits in {3, 4}.
_TQ_KERNEL_TILE = 1 << 14     # 16,384 coords = one 64 KiB butterfly tile
_TQ_KERNEL_ALIGN = 32         # the kernel_fht.cu thread discipline
_TQ_PACK_SLICE = 16           # the 16-coordinate pack-slice ownership


def _check_kernel_devices(dev: torch.device, named: tuple) -> None:
    """The dispatcher's mixed-device belt: every (name, tensor) pair in
    `named` must live on `dev` — a mismatch raises an explicit
    RuntimeError naming BOTH devices (e.g. CUDA codes + a CPU norm),
    never a silent fallback and never a crash deep in torch or a
    kernel-side generic refusal. Metadata only: device identity, no
    values, no sync."""
    for name, t in named:
        if t.device != dev:
            raise RuntimeError(
                f"TurboQuant({dev.type}) kernel dispatch: the {name} "
                f"tensor lives on {t.device} while the compute device is "
                f"{dev} — a mixed-device input. Quantize and dequantize "
                f"on one device (TQCodes.to(device) moves every tensor "
                f"field), or keep the unit on the torch twin's device. "
                f"Never a silent cross-device fallback.")


# ------------------------------------------------------------- bit packs ---
def _check_pack_width(bits, ctx: str) -> int:
    """The shared pack/unpack width domain (W20.13.1): integer bits in
    1..8 — the LEGACY pack domain restored on the torch twin. The TQ
    index streams are {3, 4} (production); {1, 2} ride the evals
    bits_list sweep and the uniform-bits tests (the box's first full
    237-suite run: 15 failures at the authored {3, 4}-only guard —
    the W20.4.4 open item's resolution). The KERNEL pack/unpack stays
    {3, 4} by its own TORCH_CHECKs, and the dispatcher's
    _kernel_geometry_eligible gates the kernel path to production
    geometry, so no non-production width can ever reach a kernel.
    Accepts int-like values (built-in ints, host-integer types whose
    int() round-trips, x.0 floats — the old guard's float-equality
    semantics); rejects bools, non-integral floats, and out-of-domain
    widths loudly."""
    if isinstance(bits, bool):
        raise ValueError(
            f"{ctx}: bits must be an integer width in 1..8, got {bits!r}")
    try:
        b = int(bits)
        integral = (b == bits)
    except (TypeError, ValueError):
        integral = False
    if not integral or not 1 <= b <= 8:
        raise ValueError(
            f"{ctx}: bits must be an integer width in 1..8 — the TQ index "
            f"streams are {{3, 4}} (production; the kernel pack stays "
            f"there, dispatcher-gated), {{1, 2}} ride the evals sweep and "
            f"the uniform-bits tests (the legacy domain, W20.13.1) — got "
            f"{bits!r}")
    return b


def pack_bits_t(idx: torch.Tensor, bits: int) -> torch.Tensor:
    """Pack uint8 quantization indices into the packed uint8 bit-stream
    (the canonical W20.3.2 encode side — promoted from the W20.3.1
    stand-in it replaced). uint8 in -> uint8 out, same device, no host
    hop, no host arrays anywhere in the body.

    Byte layout — bit-precise (the EXACT pack_bits layout of the
    legacy array era):
    * bit order: LITTLE-ENDIAN — bit k of index i sits at flat stream
      position i*bits + k (each index's LSB first);
    * byte assembly: byte m holds flat bits 8m..8m+7 with flat bit 8m
      in the LSB (bytes LSB-first); length = ceil(n*bits/8) =
      _packed_len(n, bits) bytes;
    * tail padding: the final partial byte's unused high flat bits
      are zero;
    * bits == 4 fast path: two nibbles per byte — byte j = idx[2j] |
      (idx[2j+1] << 4), the EARLIER index in the LOW nibble; an odd
      tail pads with one zero nibble (the last byte's high nibble 0);
    * bits != 4: the general shift/mask path (1, 2, 3, 5..8; bits == 8
      degenerates to the identity byte stream).
    Supported widths: integer bits in 1..8 (W20.13.1 — the LEGACY
    domain restored). The TQ index streams are {3, 4} (production;
    the lo/hi split packs its halves separately, the callers already
    do), {1, 2} ride the evals bits_list sweep and the uniform-bits
    tests — and the KERNEL pack stays {3, 4} (the dispatcher's
    _kernel_geometry_eligible gates the kernel path to production
    geometry, so non-production widths can never reach it). Other
    widths raise a clear ValueError.
    Indices stay uint8 at the boundary; the general path widens to
    int64 internally for the shift/mask arithmetic (device-resident,
    documented widening — the output folds back to uint8).

    Checks are metadata-only (the host-side bits domain): NO device
    value probe — the array-era original's range check cost a host sync,
    the exact syndrome this rewrite removes; searchsorted bucketize
    indices are < 2**bits by construction, so the legacy range error
    is unreachable from the encode path.

    byte layout pinned by the frozen fixtures — G20.P1 is the pinning
    gate (pack_bits/unpack_bits parity against the legacy array era)."""
    bits = _check_pack_width(bits, "pack_bits_t")
    idx = idx.reshape(-1)
    n = idx.numel()
    if n == 0:
        return torch.zeros(0, dtype=torch.uint8, device=idx.device)
    if idx.dtype != torch.uint8:
        idx = idx.to(torch.uint8)
    if bits == 4:
        # fast path: two nibbles per byte, the EARLIER index in the
        # low nibble; an odd tail pads with one zero nibble
        if n % 2:
            idx = torch.cat(
                [idx, torch.zeros(1, dtype=torch.uint8, device=idx.device)])
        return (idx[0::2] | (idx[1::2] << 4)).to(torch.uint8)
    # general path: the LSB-first bit matrix (row i = index i's bits,
    # LSB first) flattens to the stream, zero-pads to whole bytes, then
    # folds each 8-tuple into one byte (stream bit 8m -> the LSB)
    shifts = torch.arange(bits, device=idx.device)
    flat = ((idx.to(torch.int64).reshape(-1, 1) >> shifts) & 1).reshape(-1)
    tail = (-flat.numel()) % 8
    if tail:
        flat = torch.cat(
            [flat, torch.zeros(tail, dtype=flat.dtype, device=idx.device)])
    weights = torch.tensor([1, 2, 4, 8, 16, 32, 64, 128], device=idx.device)
    return (flat.reshape(-1, 8) * weights).sum(dim=-1).to(torch.uint8)


def unpack_bits_t(packed: torch.Tensor, bits: int, n: int) -> torch.Tensor:
    """Unpack exactly `n` uint8 indices from the packed uint8 bit-stream
    (the canonical W20.3.2 decode side — the EXACT inverse of
    pack_bits_t). uint8 in -> uint8 out, same device, no host hop, no
    host arrays anywhere in the body.

    Byte layout — bit-precise (identical contract to pack_bits_t, the
    frozen-fixture legacy array-era layout):
    * bit order: LITTLE-ENDIAN — flat stream position i*bits + k holds
      bit k of output index i (each index's LSB first);
    * byte disassembly: byte m contributes flat bits 8m..8m+7, flat
      bit 8m read from the byte's LSB (bytes LSB-first);
    * tail padding: the zero pad bits beyond n*bits are DROPPED —
      exactly n indices come out;
    * bits == 4 fast path: two nibbles per byte — idx[2j] = byte & 0xF
      (the LOW nibble, the earlier index) and idx[2j+1] = byte >> 4
      (the HIGH nibble); an odd tail's last byte carries the zero pad
      in its high nibble (read, then trimmed by [:n]);
    * bits != 4: the general shift/mask regroup with weights 2**k
      (the exact inverse of pack's general path for every width).
    Supported widths: integer bits in 1..8 — matching pack_bits_t
    (W20.13.1, the legacy domain; the kernel unpack stays {3, 4}).
    Other widths raise a clear ValueError.
    Packed bytes in uint8, indices out uint8 (the general path widens
    to int64 internally for the shift/mask arithmetic — documented
    widening; the output folds back to uint8).

    byte layout pinned by the frozen fixtures — G20.P1 is the pinning
    gate (pack_bits/unpack_bits parity against the legacy array era)."""
    bits = _check_pack_width(bits, "unpack_bits_t")
    p = packed.reshape(-1)
    if p.dtype != torch.uint8:
        p = p.to(torch.uint8)
    if bits == 4:
        # fast path, pack's exact inverse: byte j -> idx[2j] (low
        # nibble) and idx[2j+1] (high nibble); an odd tail's zero pad
        # sits in the last byte's high nibble (trimmed by [:n])
        out = torch.empty(p.numel() * 2, dtype=torch.uint8,
                          device=p.device)
        out[0::2] = p & 0xF
        out[1::2] = p >> 4
        return out[:n]
    # general path (bits != 4): expand each byte to its 8 flat bits
    # LSB-first, keep the first n*bits, regroup into (n, bits) rows,
    # fold with the powers-of-two weights
    sh = torch.arange(8, device=p.device)
    flat = ((p.to(torch.int64).reshape(-1, 1) >> sh) & 1).reshape(-1)
    flat = flat[: n * bits]
    weights = torch.tensor([1 << k for k in range(bits)],
                           device=p.device)
    return (flat.reshape(n, bits) * weights).sum(dim=-1).to(torch.uint8)


# ------------------------------------------------------------ quantizer ----
class TurboQuant:
    """One quantizer instance per tensor KIND (PROPOSAL D3).

    Parameters
    ----------
    kind : "S" | "conv" | "M1" | "M2" — or "custom" with explicit d/seed.
    bits : float in (0, 8] or int. 3.5 (default) → 50/50 split at (3, 4)
        bits (spec §3.3). An int b → uniform b-bit. A non-integer x is
        realized as (floor(x), ceil(x)) with the fractional part choosing
        the lo/hi coordinate ratio (0.5 → half/half).
    seed : overrides the kind's D3 seed (tests only — never in production).
    qjl : bool, default False — the W9.2 P7/D1 prod-variant A/B flag. When
        True, quant() additionally computes the structured QJL residual
        sketch (see the module docstring for the exact formulation) and
        the returned TQCodes carry qjl_signs/gamma; dequant() then adds the
        compensation term (√(π/2)/√d)·γ·FHT_adjoint(qjl) whenever the CODES
        carry the fields (a qjl=False instance dequantizes qjl codes
        correctly — the codes are self-describing). Flag OFF is
        bit-identical to the pre-W9.2 behavior (parity gate).
    group : int, default 1 — W15, the paper's §LongBench outlier-split
        recipe ("splitting channels into outlier and non-outlier sets, and
        applying two independent instances of TurboQuant to each, allocating
        higher bit precision to outliers"). group > 1 selects it: the d
        coordinates are CHANNEL GROUPS of `group` coords (the conv window's
        (channel, tap) layout — group = the kernel width), k = round(frac·d/group)
        highest-energy channels quantize at bits_hi with their OWN
        power-of-two full rotation (frame seed = kind seed + 555) and their
        OWN fp norm; the remaining channels at bits_lo with the kind's own
        seed frame. The membership mask (packed, ceil(n_ch/8) bytes) and the
        second norm ride in the codes (TQCodes.mask / norm_hi) — measured
        at production conv geometry this is ~3x lower write-path rel-MSE
        than the fixed coordinate half-split at the SAME effective bits
        (the fixed split puts the 4 bits on a fixed coordinate half that
        ignores the channel-energy structure; the paper's split puts them
        on the channels that carry the energy). group == 1 (S/M1/M2 — and
        any flat vector) keeps the legacy coordinate half-split exactly
        (partition="half", bit-identical to pre-W15). dequant() routes on
        the CODES' partition field, so old "half" snapshots load and
        decode unchanged through a split-configured quantizer.

    Dispatch (W20.8.2, kernel-first — PROPOSAL §3.7): quant/dequant
        route CUDA inputs with the kernel pair present and the geometry
        eligible through ONE tq_quant_batched / tq_dequant_batched call
        each, and EVERYTHING else (CPU inputs, kernel absent, the
        outlier split, qjl, ineligible geometry) through the torch
        reference path — the module docstring's REV-3 routing table is
        the binding statement (device/geometry -> path). Either path
        emits and consumes the same TQCodes contract.
    """

    def __init__(self, kind: str = "S", bits: float = 3.5,
                 d: Optional[int] = None, seed: Optional[int] = None,
                 qjl: bool = False, group: int = 1):
        if kind not in KINDS:
            if d is None or seed is None:
                raise ValueError(
                    f"TurboQuant: unknown kind {kind!r}; pass d and seed "
                    f"explicitly for custom kinds (tests only)")
        self.kind = kind
        self.d = int(d if d is not None else KINDS[kind][0])
        self.seed = int(seed if seed is not None else KINDS[kind][1])
        # FHT kernel supports non-power-of-two via segmentation (as long as
        # K is a multiple of 32 and each segment fits in shared memory).
        # Only validate that d is a positive integer.
        if self.d < 1:
            raise ValueError(f"TurboQuant: d must be >= 1, got {self.d}")
        if not (0 < bits <= 8):
            raise ValueError(f"TurboQuant: bits must be in (0, 8], got {bits}")
        # codebooks are solved per integer bit-width 1..4 (the paper's range)
        eff_bits = (int(bits), int(bits) + 1) if not float(bits).is_integer() \
            else (int(bits), int(bits))
        if eff_bits[0] < 1 or eff_bits[1] > 4:
            raise ValueError(
                f"TurboQuant: realized bit-widths {eff_bits} exceed the "
                f"Lloyd-Max codebook range (1..4) — see codebooks.py")
        self.bits_spec = float(bits)
        if float(bits).is_integer():
            self.bits_lo = self.bits_hi = int(bits)
            frac = 0.0
        else:
            lo, hi = int(bits), int(bits) + 1
            if hi > 8:
                raise ValueError(f"TurboQuant: bits {bits} exceeds 8")
            self.bits_lo, self.bits_hi = lo, hi
            frac = float(bits) - lo
        # coordinate split: frac is the fraction at the hi bit-width
        self.group = int(group)
        if self.group < 1:
            raise ValueError(f"TurboQuant: group must be >= 1, got {group}")
        if self.d % self.group != 0:
            raise ValueError(
                f"TurboQuant({kind}): d={self.d} is not a multiple of the "
                f"channel group size {self.group}")
        self.n_channels = self.d // self.group
        self.split = self.group > 1 and not float(bits).is_integer()
        if self.split:
            k = int(round(frac * self.n_channels))
            if not (1 <= k < self.n_channels):
                raise ValueError(
                    f"TurboQuant({kind}): the outlier split needs 1 <= k < "
                    f"n_channels, got k={k} of {self.n_channels} channels "
                    f"(d={self.d}, group={self.group}, bits={bits})")
            self.k_out = k
            reg_real = (self.n_channels - k) * self.group
            out_real = k * self.group
            self.reg_d, self.out_d = _next_pow2(reg_real), _next_pow2(out_real)
            # W15: for a SPLIT quantizer n_lo/n_hi are the PADDED sub-unit
            # sizes (the codes' idx streams cover the padded sub-d's — the
            # pad carries quantization noise that the un-rotation spreads,
            # and the strip at dequant discards; measured: the padded full
            # rotation beats the exact-size segmented rotation, P1).
            self.n_lo, self.n_hi = self.reg_d, self.out_d
            # the sub-codebooks resolve lazily through _sub_quantizer
            self._cb_lo = self._cb_hi = None
            self._signs = None
        else:
            n_hi = int(round(frac * self.d))
            self.n_lo, self.n_hi = self.d - n_hi, n_hi
            self.k_out = 0
            self.reg_d = self.out_d = self.d
            self._cb_lo = codebooks.get_codebook(self.bits_lo, self.d)
            self._cb_hi = (self._cb_lo if self.bits_hi == self.bits_lo
                           else codebooks.get_codebook(self.bits_hi, self.d))
            # the D3 rotation: one sign vector per kind, generated once
            self._signs = fht.rotation_signs(self.d, self.seed)
        # W20.3.1: the lazy per-device codebook mirror cache — device str
        # -> (lo, hi) DeviceCodebook pair, built once per instance per
        # device (never per quant call; see _device_codebooks)
        self._dev_cb: Dict[str, tuple] = {}
        # W20.8.2: the lazy per-device rotation-sign mirror cache — the
        # kernel path's signs contract (the SAME (d,) fp32 D3 draw the
        # torch path feeds fht_apply, mirrored once per instance per
        # device; see _device_signs)
        self._dev_signs: Dict[str, torch.Tensor] = {}
        # W29 — the lazy UNITS-frame permutation caches (built on the first
        # quant_units/_dequant_units call, the _signs lazy discipline): a
        # (d,) int64 permutation P and its inverse, drawn ONCE from the
        # kind's seed + UNITS_PERM_OFFSET (deterministic frame — see the
        # UNITS_PERM_OFFSET note for WHY the permutation is required),
        # plus the per-device mirrors for the kernel-adjacent gathers.
        self._perm: Optional[torch.Tensor] = None
        self._perm_inv: Optional[torch.Tensor] = None
        self._dev_perm: Dict[str, torch.Tensor] = {}
        self._dev_perm_inv: Dict[str, torch.Tensor] = {}
        # W9.2 qjl flag (DEFAULT OFF — the fields below are computed
        # unconditionally but USED only on qjl paths, so flag-off output
        # is bit-identical to the pre-W9.2 behavior):
        self.qjl = bool(qjl)
        # the SECOND deterministic sign draw (kind seed + 7777, same d) —
        # the structured QJL projection S = √d·FHT(d, seed+7777)
        self._qjl_signs_vec = fht.rotation_signs(self.d,
                                                 self.seed + QJL_SEED_OFFSET)

    # ------------------------------------------------------ sub-quantizers --
    def _sub_quantizer(self, role: str) -> "TurboQuant":
        """One of the two paper-recipe sub-instances ("reg" | "out") for a
        SPLIT quantizer: uniform integer bits, power-of-two d (its own FULL
        single-segment rotation — the paper's invariant), deterministic
        frame seed (regular keeps the kind's seed; outlier = seed + 555).
        Resolved through the process registry so instances are shared."""
        assert self.split
        bits = self.bits_lo if role == "reg" else self.bits_hi
        d = self.reg_d if role == "reg" else self.out_d
        seed = self.seed if role == "reg" else self.seed + SPLIT_SEED_OFFSET
        key = f"{self.kind}:{bits}:{d}:{seed}:sub"
        reg = _REGISTRY
        if key not in reg:
            reg[key] = TurboQuant(kind="custom", bits=bits, d=d, seed=seed)
        return reg[key]

    # ---------------------------------------------------------------- API --
    def quant(self, x: torch.Tensor) -> TQCodes:
        """Quantize ONE unit: x is a (d,) tensor (any float dtype).

        qjl=True (the W9.2 flag) additionally attaches the residual sketch:
        the ORIGINAL-frame residual r = x − dequant_mse(codes), its fp32
        norm γ, and qjl = sign(FHT(r/γ, seed+7777)) (module docstring has
        the exact formulation). The MSE layer (idx/norm) is computed
        IDENTICALLY with and without the flag — the A/B is purely additive.

        group > 1 (the constructor's W15 paper-recipe flag) routes to the
        outlier-channel split: see the class docstring.

        W20.3.1: the encode is torch-native and device-parametric — the
        norm stays a 0-dim fp32 tensor on x's device (TQCodes rev-1),
        the bucketize is torch.searchsorted against norm-scaled device
        boundaries (the boundary-scaling identity — the normalized
        vector is never materialized), the packed streams are torch
        uint8. Nothing in the encode reads a value back to the host
        except the single zero-norm branch below (and, on the qjl flag,
        its exact twin — the γ == 0 branch in _attach_qjl).

        W20.8.2: kernel-first dispatch (the module docstring's REV-3
        routing table): a CUDA input with the kernel pair present and
        the geometry eligible encodes through ONE tq_quant_batched
        call — no zero-norm host branch (the chain's own sweep), no
        host round trip of any kind. Every other case (CPU input,
        kernel absent, the outlier split, the qjl flag, ineligible
        geometry) takes the torch reference body below — the
        differential twin AND the CPU path."""
        x = torch.as_tensor(x)
        if x.dim() != 1 or x.shape[0] != self.d:
            raise ValueError(
                f"TurboQuant({self.kind}): expected a ({self.d},) tensor, "
                f"got {tuple(x.shape)}")
        if not torch.is_floating_point(x):
            raise TypeError(f"TurboQuant: x must be floating point, got {x.dtype}")
        x32 = x.detach().to(torch.float32)
        if self.split:
            return self._quant_split(x32)
        if self._quant_kernel_eligible(x32):
            # CUDA + kernel pair present + geometry eligible -> the
            # kernel path, ALWAYS (§3.7; the eligibility predicate is
            # the routing table's rows 1-3 evaluated in order). qjl is
            # excluded there, so the attach below is unreachable here.
            return self._quant_kernel(x32)
        # the torch reference path — the differential twin AND the CPU
        # path (the W20.3 body, verbatim)
        # the fp norm as a 0-dim fp32 torch TENSOR (TQCodes rev-1: never
        # a python float, never a host scalar — no host round trip)
        norm = x32.norm().reshape(())
        if norm == 0.0:
            # degenerate: all-zero unit (codes of zero). This single
            # host-visible branch is the torch-twin's known cost,
            # eliminated by the W20.7 kernel design.
            codes = TQCodes(
                self.kind, self.d,
                torch.zeros((), dtype=torch.float32, device=x32.device),
                self.bits_lo, self.bits_hi, self.n_lo, self.n_hi,
                torch.zeros(_packed_len(self.n_lo, self.bits_lo),
                            dtype=torch.uint8, device=x32.device),
                torch.zeros(_packed_len(self.n_hi, self.bits_hi),
                            dtype=torch.uint8, device=x32.device),
                self.seed)
        else:
            # rotation on x32's device (backend auto — same signs, same
            # (1, d) argument shape as the legacy call). The FHT is
            # linear: fht(x) = norm · fht(x/norm), so the rotated vector
            # comes out UNSCALED and the normalization is folded into
            # the boundaries below.
            y = fht.fht_apply(x32.reshape(1, self.d),
                              self._signs).reshape(self.d)
            cb_lo, cb_hi = self._device_codebooks(x32.device)
            # boundary-scaling identity (PROPOSAL §3.3):
            #   argmin|y/n − c| ≡ argmin|y − n·c| for n>0 —
            #   monotone-exact identical bucket decisions; the
            #   normalized vector is never materialized (PROPOSAL §3.3).
            # The device fp32 boundaries (the 2.2 mirror) scale by the
            # 0-dim norm tensor ON the device; y stays UNSCALED.
            # Ties: the legacy array-era searchsorted side='left' ≡ torch
            # right=False — the W16-verified CPU/GPU parity (leftmost
            # insertion point; cell i = (boundaries[i-1], boundaries[i]]).
            idx_lo = torch.searchsorted(cb_lo.boundaries * norm,
                                        y[: self.n_lo], right=False)
            idx_hi = torch.searchsorted(cb_hi.boundaries * norm,
                                        y[self.n_lo:], right=False)
            codes = TQCodes(
                self.kind, self.d, norm,
                self.bits_lo, self.bits_hi, self.n_lo, self.n_hi,
                pack_bits_t(idx_lo.to(torch.uint8), self.bits_lo),
                pack_bits_t(idx_hi.to(torch.uint8), self.bits_hi),
                self.seed)
        if self.qjl:
            self._attach_qjl(codes, x32)
        return codes

    # ------------------------------------------- W29: the units (per-head) --
    def _units_perm(self, device: torch.device) -> torch.Tensor:
        """The UNITS frame's coordinate permutation P on `device` — the
        (d,) int64 draw from seed+UNITS_PERM_OFFSET, built once (CPU) and
        mirrored per device (the _device_signs discipline: one .to per
        device, cached). Deterministic in (d, seed) — the frame is
        seed-addressed, so units codes decode anywhere the seed is known.
        """
        key = str(device)
        if device.type == "cpu":
            if self._perm is None:
                g = torch.Generator().manual_seed(
                    int(self.seed) + UNITS_PERM_OFFSET)
                self._perm = torch.randperm(self.d, generator=g)
                self._perm_inv = torch.argsort(self._perm)
            return self._perm
        if key not in self._dev_perm:
            p = self._units_perm(torch.device("cpu")).to(device)
            self._dev_perm[key] = p
            self._dev_perm_inv[key] = torch.argsort(p)
        return self._dev_perm[key]

    def _units_perm_inv(self, device: torch.device) -> torch.Tensor:
        """P's inverse on `device` (the dequant-side scatter index)."""
        if device.type == "cpu":
            self._units_perm(torch.device("cpu"))   # builds both
            return self._perm_inv
        key = str(device)
        if key not in self._dev_perm_inv:
            self._units_perm(device)                # builds both
        return self._dev_perm_inv[key]

    def _units_alignment_ok(self) -> bool:
        """The units container's idx-stream layout law: each unit's packed
        stream must occupy WHOLE bytes (n*bits % 8 == 0 per stream) so the
        row-major concatenation is byte-separable and the (M, L) row view
        the kernel batch expects is well-formed. The production per-head
        geometry (d=16,384, 50/50 split, 3/4 bits: 8,192*3=24,576 bits =
        3,072 B; 8,192*4=32,768 bits = 4,096 B) satisfies it exactly."""
        return (self.n_lo * self.bits_lo) % 8 == 0 \
            and (self.n_hi * self.bits_hi) % 8 == 0

    def quant_units(self, x2: torch.Tensor) -> TQCodes:
        """Quantize M units of this frame in ONE vectorized pass (W29 —
        the per-head S repair, FINDINGS W28 item 9 / the box matrix's
        worst-block 7.65 measurement): x2 is an (M, d) tensor — M units,
        each row a (d,) vector quantized with its OWN fp32 norm (the
        paper's per-vector norm recipe, Theorem 1 applied per unit: the
        per-coordinate noise floor is each unit's OWN energy, so a 38x
        head-norm spread no longer crushes the small heads).

        THE FRAME: y = FHT(P x) — the kind's D3 rotation composed with
        the fixed coordinate permutation P (seed+UNITS_PERM_OFFSET; see
        the UNITS_PERM_OFFSET note for the measured Kronecker-alignment
        reason the permutation is REQUIRED at the per-head 2^14 scale).
        One P for every row — the frame is shared, the NORMS are per row.

        THE CODES: one TQCodes with partition="units", units=M, the
        (M,) fp32 norms vector, and the row-major concatenated packed
        idx streams (per-unit byte-aligned — _units_alignment_ok, else a
        LOUD refusal naming the geometry).

        Dispatch (the REV-3 routing table's units row): CUDA input +
        kernel pair present + geometry eligible (d > 16,384) -> ONE
        tq_quant_batched call over the (M, d) permuted rows (the kernel
        computes per-row norms natively — the install path's own batched
        pattern, W20.10.1); every other case (the production per-head
        d=16,384 rides HERE: one 64 KiB tile is the small-K paths' own)
        -> the vectorized torch twin: ONE norm reduction, ONE batched
        fht_apply, ONE batched searchsorted per bit-width, ONE pack per
        stream — M units in a fixed ~8-op chain, no per-unit Python
        loop, no host round trips (zero-norm rows are branchless: the
        all-zero scaled boundaries bucket to index 0 and the 0-norm
        multiply zeroes the reconstruction exactly).

        Unsupported combinations (LOUD, never silent): the W15 outlier
        split (group > 1) and the W9.2 qjl flag — both refuse; the A/B
        surfaces ride the flat single-unit quant this wave."""
        x2 = torch.as_tensor(x2)
        if x2.dim() != 2 or x2.shape[1] != self.d:
            raise ValueError(
                f"TurboQuant({self.kind}): quant_units expects an "
                f"(M, {self.d}) tensor, got {tuple(x2.shape)}")
        if x2.shape[0] < 1:
            raise ValueError(
                f"TurboQuant({self.kind}): quant_units needs M >= 1 units, "
                f"got {tuple(x2.shape)}")
        if not torch.is_floating_point(x2):
            raise TypeError(
                f"TurboQuant: x2 must be floating point, got {x2.dtype}")
        if self.split:
            raise ValueError(
                f"TurboQuant({self.kind}, bits={self.bits_spec}, "
                f"group={self.group}): quant_units does not support the "
                f"W15 outlier split — the units mode and the channel split "
                f"are separate partitions (quant a (d,) unit through "
                f"quant() for split semantics)")
        if self.qjl:
            raise ValueError(
                f"TurboQuant({self.kind}): quant_units does not support the "
                f"W9.2 qjl flag this wave (the residual sketch is "
                f"per-unit-length d — the units A/B lands with the fused "
                f"kernel chain); use qjl=False")
        if not self._units_alignment_ok():
            raise ValueError(
                f"TurboQuant({self.kind}, bits=({self.bits_lo}, "
                f"{self.bits_hi}), n=({self.n_lo}, {self.n_hi})): the units "
                f"container needs byte-aligned per-unit streams "
                f"(n_lo*bits_lo and n_hi*bits_hi multiples of 8) — got "
                f"{self.n_lo}*{self.bits_lo} and {self.n_hi}*{self.bits_hi} "
                f"bits; re-split or quantize per-unit through quant()")
        x32 = x2.detach().to(torch.float32)
        m = x32.shape[0]
        dev = x32.device
        # the frame's permutation, mirrored once per device
        perm = self._units_perm(dev)
        cb_lo, cb_hi = self._device_codebooks(dev)
        if self._quant_kernel_eligible(x32.reshape(-1)) and m > 0:
            # CUDA + kernel pair + eligible geometry (d > 16,384): ONE
            # tq_quant_batched over the PERMUTED (M, d) rows — the kernel
            # chain computes the per-row norms natively (the W20.10.1
            # install pattern); the streams come back as (M, L) rows.
            C = _load_tq_kernel()
            assert C is not None
            signs = self._device_signs(dev)
            _check_kernel_devices(dev, (
                ("x2 (the unit rows)", x32),
                ("the rotation signs mirror", signs),
                ("the boundaries_lo mirror", cb_lo.boundaries),
                ("the boundaries_hi mirror", cb_hi.boundaries),
            ))
            units = x32[:, perm].contiguous()
            codes_lo, codes_hi, norms = C.tq_quant_batched(
                units, self.d, signs,
                cb_lo.boundaries, cb_hi.boundaries,
                self.n_lo, self.bits_lo, self.n_hi, self.bits_hi)
            return TQCodes(
                self.kind, self.d, norms.reshape(m),
                self.bits_lo, self.bits_hi, self.n_lo, self.n_hi,
                codes_lo.reshape(-1), codes_hi.reshape(-1),
                self.seed, partition="units", units=m)
        # the vectorized torch twin (the production per-head path):
        # ONE norm reduction over the rows — (M,) per-unit norms, the
        # paper's per-vector recipe applied per unit.
        norms = x32.norm(dim=1)
        # the frame: permute, then rotate (y = FHT(P x), one batched
        # fht_apply over all M rows — leading batch dims supported)
        xp = x32[:, perm]
        y = fht.fht_apply(xp, self._signs)
        # per-row boundary scaling (the boundary-scaling identity, per
        # row): boundaries (Lb,) -> (M, Lb) sorted per row (norms >= 0
        # preserve order; a zero-norm row degenerates to all-zero
        # boundaries, bucketing its zero row to index 0 — deterministic,
        # the reconstruction is exactly 0 through the norm multiply)
        b_lo = cb_lo.boundaries.reshape(1, -1) * norms.reshape(m, 1)
        b_hi = cb_hi.boundaries.reshape(1, -1) * norms.reshape(m, 1)
        idx_lo = torch.searchsorted(b_lo, y[:, : self.n_lo], right=False)
        idx_hi = torch.searchsorted(b_hi, y[:, self.n_lo:], right=False)
        # pack the flattened (M*n,) index arrays — the row-major
        # concatenated streams (byte-aligned per unit, the layout law)
        return TQCodes(
            self.kind, self.d, norms,
            self.bits_lo, self.bits_hi, self.n_lo, self.n_hi,
            pack_bits_t(idx_lo.reshape(-1).to(torch.uint8), self.bits_lo),
            pack_bits_t(idx_hi.reshape(-1).to(torch.uint8), self.bits_hi),
            self.seed, partition="units", units=m)

    def _dequant_units(self, codes: TQCodes,
                       dtype: torch.dtype = torch.float32,
                       device: Optional[torch.device] = None
                       ) -> torch.Tensor:
        """Decode partition="units" codes to the FLAT (M*d,) vector — the
        layer's flat layout (every §4 consumer's geometry unchanged).

        The frame's inverse: x_hat = P^T ( FHT_adj(y_hat) * norm ) — the
        adjoint rotates, the per-unit norm scales, the inverse scatter
        lands the row back in the original coordinate order. One batched
        unpack + LUT gather + fht_adjoint over all M rows; zero-norm
        units reconstruct to the exact zero vector (0 x finite, the
        branchless discipline).

        Kernel path (CUDA codes + pair present + d > 16,384 + dtype in
        {fp32, fp16}): the streams reshape to (M, L) rows + the (M,)
        norms vector -> ONE tq_dequant_batched call (per-row norms
        native), then the inverse permutation scatters the columns —
        the torch twin is the reference and the CPU path (the production
        per-head d=16,384 rides the twin: one tile, small-K kernels)."""
        if codes.qjl_signs is not None or codes.gamma is not None:
            raise ValueError(
                f"TurboQuant({self.kind}): units codes carry a QJL sketch — "
                f"the units mode does not support qjl this wave (the sketch "
                f"is per-unit-length d; a units container with sketch "
                f"fields is a corrupted or hand-built artifact)")
        m = int(codes.units)
        l_lo = _packed_len(self.n_lo, self.bits_lo)
        l_hi = _packed_len(self.n_hi, self.bits_hi)
        if codes.idx_lo.numel() != m * l_lo or codes.idx_hi.numel() != m * l_hi:
            raise ValueError(
                f"TurboQuant({self.kind}, d={self.d}): units codes carry "
                f"{codes.idx_lo.numel()}/{codes.idx_hi.numel()} stream bytes "
                f"but partition='units' with units={m} needs "
                f"{m * l_lo}/{m * l_hi} (row-major concatenation of the "
                f"per-unit streams — corrupt codes?)")
        dev = codes.idx_lo.device
        if self._dequant_kernel_eligible(codes, dtype):
            C = _load_tq_kernel()
            assert C is not None
            signs = self._device_signs(dev)
            cb_lo, cb_hi = self._device_codebooks(dev)
            perm_inv = self._units_perm_inv(dev)
            codes_lo = codes.idx_lo.reshape(m, l_lo).contiguous()
            codes_hi = codes.idx_hi.reshape(m, l_hi).contiguous()
            rows = C.tq_dequant_batched(
                codes_lo, codes_hi, codes.norm.reshape(m), signs,
                cb_lo.centroids, cb_hi.centroids,
                self.n_lo, self.bits_lo, self.n_hi, self.bits_hi, dtype)
            out = rows[:, perm_inv].reshape(-1)
        else:
            # the vectorized torch twin (the reference AND the CPU path)
            idx_lo = unpack_bits_t(codes.idx_lo, self.bits_lo,
                                   m * self.n_lo).reshape(m, self.n_lo)
            idx_hi = unpack_bits_t(codes.idx_hi, self.bits_hi,
                                   m * self.n_hi).reshape(m, self.n_hi)
            cb_lo, cb_hi = self._device_codebooks(dev)
            # uint8 indices widened to long for torch advanced indexing
            # (the flat path's gather discipline — uint8 would be read
            # as a boolean mask)
            yhat = torch.cat(
                [cb_lo.centroids[idx_lo.to(torch.long)],
                 cb_hi.centroids[idx_hi.to(torch.long)]], dim=1)
            # the adjoint (batched over the M rows), then the per-unit
            # norm scaling, then the inverse permutation scatter
            xhat = fht.fht_adjoint(yhat, self._signs)
            xhat = xhat * codes.norm.reshape(m, 1)
            perm_inv = self._units_perm_inv(dev)
            out = xhat[:, perm_inv].reshape(-1)
        out = out.to(dtype)
        if device is not None:
            out = out.to(device)
        return out

    def _device_codebooks(self, device: torch.device) -> tuple:
        """(lo, hi) codebook pair mirrored onto `device` — the 2.2 surface
        (get_codebook(bits, d, device=...) -> DeviceCodebook; .boundaries
        are fp32 device tensors, searchsorted-ready).

        Lazy and built ONCE per instance per device (cached in
        self._dev_cb): every get_codebook call re-loads the npz cache —
        fresh tables, a new identity, and the 2.2 mirror
        registry keys on that identity — so a per-call fetch would
        re-mirror and churn the registry; the instance cache keeps the
        (bits, d, device) mirror entry hot for the process lifetime.
        The constructor's own host-side codebooks stay untouched (the
        offline-island reference); the W20.3.3 decode half now gathers
        through these mirrors too (.centroids — the encode bucketizes
        .boundaries)."""
        key = str(device)
        pair = self._dev_cb.get(key)
        if pair is None:
            lo = codebooks.get_codebook(self.bits_lo, self.d,
                                        device=device)
            hi = (lo if self.bits_hi == self.bits_lo
                  else codebooks.get_codebook(self.bits_hi, self.d,
                                              device=device))
            pair = (lo, hi)
            self._dev_cb[key] = pair
        return pair

    def _device_signs(self, device: torch.device) -> torch.Tensor:
        """The D3 rotation sign vector mirrored onto `device` (W20.8.2):
        the quantizer's OWN (d,) fp32 draw — the SAME vector the torch
        path feeds fht.fht_apply (one sign vector per kind, fixed seed,
        PROPOSAL D3) — as a contiguous fp32 tensor on `device`, the
        kernel pair's signs contract (tq_quant_batched /
        tq_dequant_batched take it IN by the caller, never generate it).

        Lazy and built ONCE per instance per device (cached in
        self._dev_signs, the _device_codebooks idiom): the constructor's
        draw is CPU-resident host state; the kernel path mirrors it once
        per device instead of paying a per-call H2D like the torch
        path's fht_apply sign move."""
        if self._signs is None:
            # a SPLIT-configured quantizer has no single rotation sign
            # vector (the sub-quantizers own theirs) — and the kernel
            # path is flat-only (the routing table's row 3)
            raise ValueError(
                f"TurboQuant({self.kind}): a split-configured quantizer "
                f"carries no single rotation sign vector (the "
                f"sub-quantizers own theirs) — the kernel path is "
                f"flat-only (see the routing table)")
        key = str(device)
        s = self._dev_signs.get(key)
        if s is None:
            s = self._signs.detach().to(device, torch.float32).contiguous()
            self._dev_signs[key] = s
        return s

    # ------------------------------------------------- kernel dispatch ----
    def _kernel_geometry_eligible(self) -> bool:
        """The TQ kernels' geometry envelope for THIS flat quantizer
        (kernel_tq.cu's check_args, mirrored — the dispatcher's copy; the
        kernels' own TORCH_CHECKs are the belt):
          * d > 16,384 (one 64 KiB butterfly tile is the small-K paths'
            own — d <= 16,384 stays on the torch twin, whose fht_apply
            still rides the small CUDA kernels),
          * d a power of two and 32-aligned (the FHT rotation the chain
            rides — pow2 past the tile implies 32-alignment; both kept
            explicit, the kernel's clause order),
          * n_lo % 16 == 0 (the 16-coordinate pack-slice ownership —
            misaligned fractional splits stay on the torch twin),
          * bits_lo / bits_hi in {3, 4} (the TQ index stream widths —
            the b1/b2 recipes and specs past 4 are torch-only).
        Consulted on the QUANTIZER's own cardinals (the codes' are
        pinned to them by _check_codes)."""
        d = self.d
        if d <= _TQ_KERNEL_TILE:
            return False
        if d % _TQ_KERNEL_ALIGN != 0 or (d & (d - 1)) != 0:
            return False
        if self.n_lo % _TQ_PACK_SLICE != 0:
            return False
        return self.bits_lo in (3, 4) and self.bits_hi in (3, 4)

    def _quant_kernel_eligible(self, x32: torch.Tensor) -> bool:
        """The dispatcher's quant-side decision (the module docstring's
        REV-3 routing table, rows 1-3 evaluated in order): CUDA input
        (device class + kernel presence in the ONE _tq_kernel_available
        probe — the fht.py kernel_available(x) signature discipline),
        flat partition (the W15 outlier split is not fused yet), qjl
        flag OFF (the W9.2 A/B rides the torch twin this wave; the
        fused qjl chain lands W20.9), geometry eligible -> the kernel
        path. Anything else -> the torch twin (the reference AND the
        CPU path)."""
        if not _tq_kernel_available(x32):
            return False
        if self.split:
            return False
        if self.qjl:
            return False
        return self._kernel_geometry_eligible()

    def _quant_kernel(self, x32: torch.Tensor) -> TQCodes:
        """The kernel-path encode (W20.8.2): ONE tq_quant_batched call
        over the unit flattened to a (1, d) fp32 CUDA row — the norm
        chain + the batched large-K butterfly + the ladder/pack + the
        zero-norm sweep, all inside the one stream-ordered chain (no
        zero-norm host branch, no host round trip of any kind: the
        wrapper never reads device memory). The signs are the
        quantizer's OWN (d,) fp32 draw mirrored to x32's device
        (_device_signs); the boundaries are the 2.2 DeviceCodebook
        mirrors, UNSCALED — the kernel applies the boundary-scaling
        identity itself (argmin|y/n − c| ≡ argmin|y − n·c|). The
        outputs pack into the SAME TQCodes rev-1 contract the torch
        twin writes: the (1, L) packed rows flatten to the 1-D idx
        streams, the (1,) norms vector reshapes to the 0-dim fp32 norm,
        and the kind/d/bits/n/seed bookkeeping is the quantizer's own
        (codes as CUDA device tensors — device tensors in, device
        tensors out)."""
        C = _load_tq_kernel()
        assert C is not None            # _quant_kernel_eligible's probe
        dev = x32.device
        units = x32.reshape(1, self.d).contiguous()
        signs = self._device_signs(dev)
        cb_lo, cb_hi = self._device_codebooks(dev)
        _check_kernel_devices(dev, (
            ("x (the unit row)", units),
            ("the rotation signs mirror", signs),
            ("the boundaries_lo mirror", cb_lo.boundaries),
            ("the boundaries_hi mirror", cb_hi.boundaries),
        ))
        codes_lo, codes_hi, norms = C.tq_quant_batched(
            units, self.d, signs,
            cb_lo.boundaries, cb_hi.boundaries,
            self.n_lo, self.bits_lo, self.n_hi, self.bits_hi)
        return TQCodes(
            self.kind, self.d, norms.reshape(()),
            self.bits_lo, self.bits_hi, self.n_lo, self.n_hi,
            codes_lo.reshape(-1), codes_hi.reshape(-1),
            self.seed)

    def _attach_qjl(self, codes: TQCodes, x32: torch.Tensor) -> None:
        """Compute + attach the structured QJL residual sketch (W9.2 flag).

        r = x32 − dequant_mse(codes) in the ORIGINAL frame (the rotated
        form is identical by linearity); γ = ‖r‖; qjl =
        sign(FHT(r/γ, seed+7777)) — a (d,) int8 ±1 vector. A zero residual
        (γ == 0: the degenerate zero-norm unit, or an exact reconstruction)
        stores zeros and γ = 0 — the compensation term is then exactly 0.

        W20.3.4: torch-native and device-parametric end-to-end. The
        residual rides the SAME adjoint/LUT decode path dequant()
        uses — _dequant_mse, device-native since 3.3, computes on the
        CODES' device, and the codes were built on x32's device by
        this very quant call, so the reconstruction lands beside x32
        with no move at all (the W16 CPU-pinned bridge is structurally
        gone). γ is the 0-dim fp32 TENSOR — quant's norm discipline,
        never a host scalar (the single γ == 0 branch below is the
        zero-norm branch's exact twin). The sketch is torch.sign of
        the residual's FHT — the scale invariance below means the
        division by γ is never materialized — and lands in the codes
        as an int8 torch TENSOR on x32's device."""
        x_mse = self._dequant_mse(codes, dtype=torch.float32)
        resid = x32 - x_mse
        # γ = ‖r‖ as a 0-dim fp32 tensor — the quant norm discipline
        gamma = resid.norm().reshape(())
        if gamma == 0.0:
            # degenerate: exact reconstruction or the zero-norm unit —
            # the flag carries, the values are zero (the compensation
            # term is then exactly 0)
            signs = torch.zeros(self.d, dtype=torch.int8,
                                device=x32.device)
        else:
            # sign(FHT(r/γ)) ≡ sign(FHT(r)) — the scale invariance
            # (PROPOSAL §3.4); the division by γ is never materialized
            proj = fht.fht_apply(resid.reshape(1, self.d),
                                 self._qjl_signs_vec).reshape(self.d)
            # ±1 exactly (torch.sign; 0 only on an exact-zero
            # projection — measure-zero for generic residuals, degrades
            # that coordinate by 1/√d of its weight)
            signs = torch.sign(proj).to(torch.int8)
        codes.qjl_signs = signs
        codes.gamma = gamma

    # ------------------------------------------- W15: the outlier split ----
    def _quant_split(self, x32: torch.Tensor) -> TQCodes:
        """The paper's §LongBench recipe at work (see the class docstring):
        top-k channels by energy → the outlier sub-set (bits_hi, own frame
        seed+555, own norm, pow2-padded own rotation); the rest → the
        regular sub-set (bits_lo, the kind's seed, own norm). Deterministic
        (stable argsort on the channel energies).

        W20.3.1: torch-native end-to-end — the channel energies, the
        stable ordering, the sub-set gathers, the pow2 padding and the
        packed channel mask all stay on x32's device; the sub-quantizers
        are flat torch-native quants through this same quant() and
        resolve their own device codebook mirrors."""
        n_ch, g = self.n_channels, self.group
        xg = x32.reshape(n_ch, g)
        energy = (xg * xg).sum(dim=-1)
        # the legacy array-era argsort(-energy, kind="stable") ≡ a
        # stable ascending argsort of the negated keys — ties keep
        # channel order (the determinism contract)
        order = torch.argsort(-energy, stable=True)
        out_ch = torch.sort(order[: self.k_out]).values
        reg_ch = torch.sort(order[self.k_out:]).values
        mask_ch = torch.zeros(n_ch, dtype=torch.bool, device=x32.device)
        mask_ch[out_ch] = True
        xreg = xg[reg_ch].reshape(-1)
        xout = xg[out_ch].reshape(-1)
        q_reg, q_out = self._sub_quantizer("reg"), self._sub_quantizer("out")

        def _pad(x: torch.Tensor, d: int) -> torch.Tensor:
            # the legacy pad-to discipline, torch-native: zero-pad UP to
            # the power-of-two sub-d; the pad adds no energy — the
            # sub-quantizer's stored norm is the real sub-set's
            if x.shape[0] < d:
                x = torch.cat(
                    [x, torch.zeros(d - x.shape[0], dtype=x.dtype,
                                    device=x.device)])
            return x

        c_reg = q_reg.quant(_pad(xreg, self.reg_d))
        c_out = q_out.quant(_pad(xout, self.out_d))
        # the packed per-channel mask, legacy layout EXACT: the legacy
        # packbits default bitorder='big' — channel 0 is the MSB of byte
        # 0, tail zero-padded to whole bytes (the dequant-side unpackbits
        # reader is MSB-first). Deliberately NOT pack_bits_t's
        # little-endian index stream — a different fixture, a different
        # layout.
        mbits = mask_ch.to(torch.uint8)
        tail = (-n_ch) % 8
        if tail:
            mbits = torch.cat(
                [mbits, torch.zeros(tail, dtype=torch.uint8,
                                    device=x32.device)])
        w = torch.tensor([128, 64, 32, 16, 8, 4, 2, 1], device=x32.device)
        packed_mask = (mbits.reshape(-1, 8) * w).sum(dim=-1).to(torch.uint8)
        codes = TQCodes(
            self.kind, self.d, c_reg.norm,
            self.bits_lo, self.bits_hi, self.n_lo, self.n_hi,
            c_reg.idx_lo, c_out.idx_lo, self.seed,
            partition="outlier", group=g,
            mask=packed_mask,
            norm_hi=c_out.norm)
        if self.qjl:
            self._attach_qjl(codes, x32)
        return codes

    def _split_channels(self, codes: TQCodes) -> tuple:
        """(reg_ch, out_ch) channel-index TENSORS from the codes' packed
        mask — validated, torch end-to-end on the mask's device."""
        if codes.mask is None or codes.group is None:
            raise ValueError(
                f"TurboQuant({self.kind}): outlier-partition codes without a "
                f"mask/group — corrupted serialization?")
        n_ch = self.d // int(codes.group)
        # W20.3.3: the MSB-first (bitorder='big') expansion — the mask
        # fixture layout, NOT pack_bits_t's little-endian index stream
        # (byte i's bit 7-j is channel 8i+j) — stays the torch shift/
        # mask from 3.2, and its 0/1 vector now feeds torch boolean
        # indexing DIRECTLY: torch.nonzero for the legacy nonzero pair,
        # the host-array bridge GONE. torch.nonzero emits ascending
        # indices — the legacy reader's exact contract, and the inverse
        # of the encode's sorted channel sets — so the decode-side
        # scatter restores the encode-side gather order.
        m_t = codes.mask.to(torch.uint8).reshape(-1)
        bits_ = ((m_t.reshape(-1, 1)
                  >> torch.arange(7, -1, -1, device=m_t.device)) & 1) \
            .reshape(-1)
        if bits_.numel() < n_ch:
            raise ValueError(
                f"TurboQuant({self.kind}): mask covers {bits_.numel()} "
                f"channels, the unit needs {n_ch}")
        mask = bits_[:n_ch].to(torch.bool)
        out_ch = torch.nonzero(mask).reshape(-1)
        reg_ch = torch.nonzero(~mask).reshape(-1)
        if out_ch.numel() != self.k_out:
            raise ValueError(
                f"TurboQuant({self.kind}): mask says {out_ch.numel()} outlier "
                f"channels, the geometry says k={self.k_out} — frame drift?")
        return reg_ch, out_ch

    def _dequant_split_mse(self, codes: TQCodes,
                           dtype: torch.dtype) -> torch.Tensor:
        """The outlier-partition MSE reconstruction: two sub-dequants
        (each through its own frame + norm), pad-strip, scatter to the
        channel layout.

        W20.3.3: torch end-to-end on the CODES' device. The channel
        index tensors from _split_channels scatter the sub-set
        reconstructions through torch advanced indexing — the
        host buffer (zeros + `xg[ch] = ...`) and the sub-dequant
        host-array hops are gone. Each sub-set's 0-dim fp32 norm
        (codes.norm / codes.norm_hi) scales INSIDE its sub-dequant —
        the legacy arithmetic exactly — so a degenerate zero norm
        needs no host read: the 0-dim multiply yields the exact zero
        vector (0 × finite) and the scatter is unconditional (the
        legacy skip left the same zeros). A None norm_hi (corrupted
        outlier codes) keeps the legacy guard: the outlier channels
        decode to zero."""
        reg_ch, out_ch = self._split_channels(codes)
        q_reg, q_out = self._sub_quantizer("reg"), self._sub_quantizer("out")
        dev = codes.device          # idx_lo's — every stream shares it
        g = int(codes.group)
        reg_real, out_real = reg_ch.numel() * g, out_ch.numel() * g
        x = torch.zeros(self.d, dtype=torch.float32, device=dev)
        xg = x.reshape(self.n_channels, g)
        # the regular sub-set: idx_lo + the whole-unit `norm` field
        # (the REGULAR sub-set's norm, the TQCodes contract)
        c_reg = TQCodes(q_reg.kind, q_reg.d, codes.norm,
                        q_reg.bits_lo, q_reg.bits_hi, q_reg.n_lo, q_reg.n_hi,
                        codes.idx_lo,
                        torch.zeros(0, dtype=torch.uint8, device=dev),
                        q_reg.seed)
        v = q_reg.dequant(c_reg, dtype=torch.float32)[: reg_real]
        xg[reg_ch] = v.reshape(-1, g)
        # the outlier sub-set: idx_hi + norm_hi (each sub-code covers its
        # PADDED pow2 sub-d; the [: out_real] strip discards the pad the
        # same way the legacy slice did)
        if codes.norm_hi is not None:
            c_out = TQCodes(q_out.kind, q_out.d, codes.norm_hi,
                            q_out.bits_lo, q_out.bits_hi, q_out.n_lo,
                            q_out.n_hi, codes.idx_hi,
                            torch.zeros(0, dtype=torch.uint8, device=dev),
                            q_out.seed)
            v = q_out.dequant(c_out, dtype=torch.float32)[: out_real]
            xg[out_ch] = v.reshape(-1, g)
        return x.to(dtype)

    def _flat_twin(self) -> "TurboQuant":
        """The legacy flat-semantics twin of a SPLIT quantizer (same kind /
        bits / d / seed, group=1) — dequantizing pre-W15 partition="half"
        codes through it keeps every legacy contract exact (the codes are
        self-describing; old snapshots load and decode unchanged)."""
        key = f"{self.kind}:{self.bits_spec}:{self.d}:{self.seed}:flat"
        if key not in _REGISTRY:
            _REGISTRY[key] = TurboQuant(
                kind=self.kind, bits=self.bits_spec, d=self.d,
                seed=self.seed, qjl=self.qjl)
        return _REGISTRY[key]

    def _split_twin(self, group: int) -> "TurboQuant":
        """The split-semantics twin at `group` coords per channel — the
        reverse of _flat_twin: a FLAT-configured quantizer (group=1, e.g.
        resolved by a generic codes-only path) dequantizing
        partition="outlier" codes routes here so the sub-quantizer frames
        resolve from the codes' own geometry."""
        key = f"{self.kind}:{self.bits_spec}:{self.d}:{self.seed}:split{group}"
        if key not in _REGISTRY:
            _REGISTRY[key] = TurboQuant(
                kind=self.kind, bits=self.bits_spec, d=self.d,
                seed=self.seed, qjl=self.qjl, group=group)
        return _REGISTRY[key]

    def _route_for(self, codes: TQCodes) -> "TurboQuant":
        """The quantizer that owns CODES' partition (self, or the twin with
        the right group semantics). Codes are self-describing: outlier
        codes need split semantics (self's group must equal the codes'
        group); half AND units codes need flat semantics (the units mode
        is a flat-frame composition — the kind's rotation + the units
        permutation — never the channel split)."""
        part = getattr(codes, "partition", "half")
        if part == "outlier":
            grp = int(codes.group) if codes.group is not None else 1
            if self.split and self.group == grp:
                return self
            return self._split_twin(grp)
        if self.split:
            return self._flat_twin()
        return self

    def dequant(self, codes: TQCodes,
                dtype: torch.dtype = torch.float32,
                device: Optional[torch.device] = None) -> torch.Tensor:
        """Dequantize ONE unit back to a (d,) tensor of `dtype`.

        Routing (W15): the CODES are self-describing — partition="outlier"
        dequantizes through the split sub-quantizers; partition="half"
        (all pre-W15 codes, and every S/M1/M2 unit) dequantizes through the
        flat frame — through the legacy twin when self is split-configured,
        so old snapshots load unchanged.

        When the codes carry the QJL sketch (qjl_signs + gamma, the W9.2
        flag — present IFF quantized with qjl=True), the reconstruction is
        the paper's Alg.-2 estimator: x̃ = x̃_mse + (√(π/2)/√d)·γ·
        FHT_adjoint(qjl)  ==  x̃_mse + (√(π/2)/d)·γ·Sᵀ·qjl with the
        structured projection S = √d·FHT(d, codes.seed+7777) (module
        docstring). Codes WITHOUT the fields: exactly the pre-W9.2 path.

        W20.3.3: the reconstruction is torch-native and DEVICE-PARAMETRIC
        end-to-end — unpack_bits_t on the packed streams, the centroid
        LUT gather through the per-instance DeviceCodebook mirrors (the
        2.2 surface, fetched once per device), fht.fht_adjoint
        (device-native, backend auto) and the 0-dim norm-tensor multiply
        all run ON THE CODES' DEVICE; the legacy CPU-reference body
        (host-array gather, host-built adjoint input,
        compute-on-CPU-then-move) is gone. CPU-resident codes decode
        on CPU, so every legacy CPU-side consumer is unchanged.

        W20.3.4: the QJL compensation term is device-native too — the
        direct _qjl_compensation_term call computes on the codes'
        device (the 3.3 CPU-detour bridge collapsed: no host hop, no
        term H2D); flag-off codes (the None pair) skip the term
        entirely.

        W18 `device`: when given, the FINAL tensor is moved there once —
        the caller's placement contract, the ONLY sanctioned transfer
        (None = the codes' device; for CPU codes that is CPU, the
        pre-W18 behavior every CPU-side consumer relies on).

        W20.8.2: kernel-first dispatch (the module docstring's REV-3
        routing table): flat codes on CUDA with the kernel pair
        present, the geometry eligible, dtype in {float32, float16}
        and NO QJL sketch decode through ONE tq_dequant_batched call
        (the decision is device class + geometry — never codes
        provenance); every other case (CPU codes, kernel absent,
        outlier-partition codes, qjl codes, other dtypes) takes the
        torch reference path below — the differential twin AND the
        CPU path.

        W29: partition="units" codes (the per-head S mode) route to
        _dequant_units — the vectorized multi-unit decode returning the
        FLAT (M*d,) vector (the layer's flat layout, so every §4
        consumer's geometry is unchanged); the cardinals are pinned by
        _check_codes and the stream lengths validated there.
        """
        part = getattr(codes, "partition", "half")
        if part == "units":
            q = self._route_for(codes)
            q._check_codes(codes)
            return q._dequant_units(codes, dtype, device=device)
        if part == "outlier":
            q = self._route_for(codes)
            q._check_codes(codes)
            out = q._dequant_split_mse(codes, dtype=torch.float32)
            # NOTE: no whole-unit zero-norm early return here — `norm` is
            # the REGULAR sub-set's norm; a zero regular set with a live
            # outlier set (or vice versa) is handled per sub-set inside
            # _dequant_split_mse (the 0-dim norm/norm_hi multiplies).
            out = self._qjl_compensation_term(codes, out)
            out = out.to(dtype)
        else:
            q = self._route_for(codes)
            out = q._dequant_flat(codes, dtype, device=device)
        if device is not None:
            out = out.to(device)
        return out

    def _qjl_compensation_term(self, codes: TQCodes,
                               out: torch.Tensor) -> torch.Tensor:
        """The Alg.-2 residual term, shared by both partitions (the sketch
        lives in the ORIGINAL frame — partition-agnostic).

        W20.3.4: torch-native and device-resident — the int8 signs
        widen to fp32 ON THE CODES' DEVICE (a dtype conversion always
        copies, so read-only / mmap-backed signs storage is never
        written — the legacy writable-copy hop deleted), the adjoint
        rides fht.fht_adjoint (device-native, backend auto), and γ
        scales as the 0-dim fp32 TENSOR — no host scalar read, no
        sync: a degenerate γ == 0 multiplies the finite adjoint to the
        exact zero term (0 × finite = 0 — the legacy γ-branch guard
        added the same zero). Flag-off codes (the None pair) return
        `out` untouched — the default path takes zero new code."""
        if codes.qjl_signs is None and codes.gamma is None:
            return out
        if codes.qjl_signs is None or codes.gamma is None:
            raise ValueError(
                f"TurboQuant({self.kind}, bits={self.bits_spec}, "
                f"d={self.d}, seed={self.seed}): codes carry a PARTIAL "
                f"QJL sketch (qjl_signs={'set' if codes.qjl_signs is not None else 'None'}, "
                f"gamma={'set' if codes.gamma is not None else 'None'}) — "
                f"corrupted serialization?")
        s = codes.qjl_signs
        if s.dtype != torch.int8 or s.dim() != 1 or s.shape[0] != codes.d:
            raise ValueError(
                f"TurboQuant({self.kind}): qjl_signs must be a "
                f"({codes.d},) int8 ±1 sketch, got shape={tuple(s.shape)}, "
                f"dtype={s.dtype}")
        # the fp32 widening copy ON THE CODES' DEVICE (a dtype
        # conversion always copies — read-only / mmap-backed signs
        # storage is never written; the (1, d) argument shape is the
        # legacy call's)
        s_t = s.to(torch.float32).reshape(1, self.d)
        # _check_codes pins codes.seed == self.seed, so the cached
        # second draw is the codes' own QJL sign vector.
        z = fht.fht_adjoint(s_t, self._qjl_signs_vec).reshape(self.d)
        return out + (QJL_C * codes.gamma / math.sqrt(self.d)) * z

    def _dequant_flat(self, codes: TQCodes,
                      dtype: torch.dtype = torch.float32,
                      device: Optional[torch.device] = None) -> torch.Tensor:
        """The flat (pre-W15 semantics) decode body: fp32 reconstruction
        on the codes' device, QJL compensation, dtype LAST, one optional
        final move.

        W20.8.2: the flat decode is where the kernel-first dispatcher
        lives (the routed q owns the frame — signs, codebook mirrors,
        geometry — and _check_codes has pinned the codes to it): CUDA
        codes + kernel pair present + eligible geometry + dtype in
        {float32, float16} + no QJL sketch -> ONE tq_dequant_batched
        call (_dequant_kernel); everything else -> the torch body
        below, verbatim.

        W20.3.3: the legacy host-read of the norm scalar is GONE — a
        degenerate zero norm yields the exact zero vector through the
        0-dim norm multiply inside _dequant_mse (0 × finite = 0; the
        encode's zero-norm codes carry all-zero index streams, so the
        gather stays finite and the product is exactly zero — and γ,
        the residual's norm, is 0 there, so the compensation term is
        exactly 0 too), with no branch and no sync; the zeros land on
        the CODES' device and `device` moves the final tensor once."""
        self._check_codes(codes)
        if self._dequant_kernel_eligible(codes, dtype):
            # the kernel path: fp32/fp16 straight from the chain's
            # final store (the qjl term is unreachable here —
            # sketch-carrying codes route to the torch body below)
            out = self._dequant_kernel(codes, dtype)
        else:
            out = self._dequant_mse(codes, dtype=torch.float32)
            out = self._qjl_compensation_term(codes, out)
            out = out.to(dtype)
        if device is not None:
            out = out.to(device)
        return out

    def _dequant_kernel_eligible(self, codes: TQCodes,
                                 dtype: torch.dtype) -> bool:
        """The dispatcher's dequant-side decision (the module docstring's
        REV-3 routing table, rows 1-3 evaluated in order): CUDA codes
        (idx_lo's device — the ONE _tq_kernel_available probe carries
        the device class + kernel presence, the fht.py
        kernel_available(x) signature discipline), flat partition, no
        QJL sketch on the codes (sketch-carrying codes ride the torch
        twin this wave; the fused qjl chain lands W20.9), dtype in
        {float32, float16} (the kernel decode's two output widths —
        every other dtype stays on the twin, which casts at the very
        end), geometry eligible -> the kernel path. Anything else ->
        the torch twin. Called AFTER _check_codes, so the quantizer's
        cardinals are the codes' own."""
        if not _tq_kernel_available(codes.idx_lo):
            return False
        if getattr(codes, "partition", "half") != "half":
            return False          # belt: _dequant_flat sees flat codes only
        if codes.qjl_signs is not None or codes.gamma is not None:
            return False
        if dtype not in (torch.float32, torch.float16):
            return False
        return self._kernel_geometry_eligible()

    def _dequant_kernel(self, codes: TQCodes,
                        dtype: torch.dtype) -> torch.Tensor:
        """The kernel-path decode (W20.8.2): ONE tq_dequant_batched call
        over the unit's packed code rows — the bfe unpack + the smem
        LUT gather + the fht_backward_large adjoint + the per-unit norm
        multiply + the single fp32/fp16 cast at the final store, one
        stream-ordered chain (the wrapper never reads device memory).
        The codes' 1-D packed idx streams stack into the entry's (1, L)
        batched rows (the dispatcher serves the per-unit public API
        this wave — M=1; the batched install/index/evals surfaces land
        in their own waves), the 0-dim fp32 norm reshapes to the (M,)
        norms vector, the centroids are the 2.2 DeviceCodebook mirrors
        (UNSCALED — the norm multiply is the chain's own), and the
        signs are the quantizer's OWN (d,) fp32 draw mirrored to the
        codes' device (_device_signs). Returns the (d,) reconstruction
        in `dtype`; the W18 `device` move is _dequant_flat's tail."""
        C = _load_tq_kernel()
        assert C is not None        # _dequant_kernel_eligible's probe
        dev = codes.device          # idx_lo's — every stream shares it
        signs = self._device_signs(dev)
        cb_lo, cb_hi = self._device_codebooks(dev)
        codes_lo = codes.idx_lo.reshape(1, -1).contiguous()
        codes_hi = codes.idx_hi.reshape(1, -1).contiguous()
        norms = codes.norm.reshape(1)
        _check_kernel_devices(dev, (
            ("codes.idx_lo", codes.idx_lo),
            ("codes.idx_hi", codes.idx_hi),
            ("codes.norm", codes.norm),
            ("the rotation signs mirror", signs),
            ("the centroids_lo mirror", cb_lo.centroids),
            ("the centroids_hi mirror", cb_hi.centroids),
        ))
        out = C.tq_dequant_batched(
            codes_lo, codes_hi, norms, signs,
            cb_lo.centroids, cb_hi.centroids,
            self.n_lo, self.bits_lo, self.n_hi, self.bits_hi,
            dtype)
        return out.reshape(self.d)

    def _dequant_mse(self, codes: TQCodes,
                     dtype: torch.dtype = torch.float32) -> torch.Tensor:
        """The MSE-layer reconstruction — the PURE TORCH TWIN (W20.8.2:
        the kernel-first dispatch lives ONE level up, in
        _dequant_flat; this body stays the reference decode the kernels
        are differentially pinned against, and the decode the qjl
        residual rides). W15: dispatches on the codes' partition
        through _route_for (the split path through the sub-quantizers;
        the flat path through the twin when self carries the other
        semantics).

        W20.3.3: torch-native and device-parametric — the packed streams
        unpack through unpack_bits_t ON THE CODES' DEVICE (3.2), the
        centroid LUT gather is a torch fancy-index into the
        per-instance DeviceCodebook mirrors (the 2.2 surface; the SAME
        _dev_cb once-per-device cache the encode half uses, so the
        codebooks live on the CODES' device — fetched once, never per
        call), the un-rotation is fht.fht_adjoint (device-native,
        backend auto: CUDA kernel when eligible, the torch butterfly
        otherwise — the host-built adjoint input is GONE), and the norm
        is the 0-dim fp32 TENSOR multiply (no host scalar read of any
        kind — a degenerate zero norm scales the finite reconstruction
        to the exact zero vector). The fp32 math runs first; the
        requested output dtype converts at the very end."""
        if getattr(codes, "partition", "half") == "outlier":
            q = self._route_for(codes)
            q._check_codes(codes)
            return q._dequant_split_mse(codes, dtype)
        q = self._route_for(codes)
        if q is not self:
            return q._dequant_mse(codes, dtype)
        dev = codes.device              # idx_lo's — every stream shares it
        idx_lo = unpack_bits_t(codes.idx_lo, codes.bits_lo, codes.n_lo)
        idx_hi = unpack_bits_t(codes.idx_hi, codes.bits_hi, codes.n_hi)
        # the tensor LUT gather: centroids_t[idx] into the device-resident
        # fp32 centroid tables (uint8 indices widened to long for torch
        # advanced indexing — no host gather, no host-array take; the
        # mirror is built from the SAME solved tables, so CPU values
        # are bit-identical to the legacy gather)
        cb_lo, cb_hi = self._device_codebooks(dev)
        y = torch.empty(self.d, dtype=torch.float32, device=dev)
        y[: codes.n_lo] = cb_lo.centroids[idx_lo.to(torch.long)]
        y[codes.n_lo:] = cb_hi.centroids[idx_hi.to(torch.long)]
        # r = y @ T^T on y's device (the signs ride fht_adjoint's own
        # device placement; the (1, d) argument shape is the legacy
        # call's)
        r = fht.fht_adjoint(y.reshape(1, self.d), self._signs) \
            .reshape(self.d)
        return (r * codes.norm).to(dtype)

    # round-trip helper (evals/Phase-1 measurement harness)
    def roundtrip(self, x: torch.Tensor) -> torch.Tensor:
        # W18: the dequant device parameter carries the placement — no
        # post-hoc move needed.
        return self.dequant(self.quant(x), dtype=x.dtype, device=x.device)

    # ------------------------------------------------------------ plumbing -
    def _check_codes(self, codes: TQCodes) -> None:
        if (codes.kind != self.kind or codes.d != self.d
                or codes.bits_lo != self.bits_lo
                or codes.bits_hi != self.bits_hi
                or codes.n_lo != self.n_lo or codes.n_hi != self.n_hi
                or codes.seed != self.seed):
            raise ValueError(
                f"TurboQuant({self.kind}, bits={self.bits_spec}, d={self.d}, "
                f"seed={self.seed}): codes are "
                f"({codes.kind}, d={codes.d}, bits=({codes.bits_lo},"
                f"{codes.bits_hi}), n=({codes.n_lo},{codes.n_hi}), "
                f"seed={codes.seed}) — mismatch (frame drift? see PROPOSAL D3)")


def _packed_len(n: int, bits: int) -> int:
    return (n * bits + 7) // 8


def _next_pow2(n: int) -> int:
    """Smallest power of two >= n (n >= 1) — the W15 split sub-units."""
    if n < 1:
        raise ValueError(f"_next_pow2: n must be >= 1, got {n}")
    return 1 << (n - 1).bit_length()


# -------------------------------------------------------------- registry ---
_REGISTRY: Dict[str, TurboQuant] = {}


def get_quantizer(kind: str, bits: float = 3.5) -> TurboQuant:
    """Process-wide registry: one TurboQuant per (kind, bits) — the shared
    rotation instance per tensor kind (D3)."""
    key = f"{kind}:{bits}"
    if key not in _REGISTRY:
        _REGISTRY[key] = TurboQuant(kind=kind, bits=bits)
    return _REGISTRY[key]
