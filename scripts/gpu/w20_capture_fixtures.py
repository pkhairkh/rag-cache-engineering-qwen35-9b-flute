#!/usr/bin/env python3
"""w20_capture_fixtures.py -- W20.1.2: the frozen-probe fixture capture.

WHAT
    Captures the W20 frozen-probe fixtures: seeded synthetic probe inputs
    plus the w20-baseline (numpy-era) TurboQuant outputs -- packed codes,
    fp32 norms, reference reconstructions, the W15 outlier-split masks and
    the QJL residual sketches -- one .npz per probe, plus a
    fixtures_manifest.json with per-file sha256 fingerprints and the
    codebook digests. These fixtures are the differential reference the
    W20 CUDA wave gates against (PROPOSAL.md §5 levels 1-2, §8 W20.K1).

WHERE -- the discipline (PROMPT.md §1)
    This script runs ONLY on the A10G GPU box, detached at tag
    w20-baseline (W20_BASELINE_SHA below). Refusal chain, in order and
    BEFORE any repo import:

      1. `git rev-parse HEAD` != W20_BASELINE_SHA  -> loud refusal, exit 2
      2. torch.cuda.is_available() is False        -> loud refusal, exit 3
      3. only then sys.path gets the repo root and the pinned numpy-era
         TurboQuant API is imported. Importing the repo's legacy path is
         legal THERE (the A10G box, pinned tree -- pinning the legacy
         semantics is the entire point). The authoring (CPU) box NEVER
         executes this script; the refusals above make sure no other box
         can either.

SYNTHETIC PROBES -- WHY THEY ARE VALID
    TurboQuant is data-oblivious: the per-(b, d) Lloyd-Max codebooks are
    solved from the ANALYTIC sphere-coordinate density (codebooks.py -- no
    calibration data), the rotation is a fixed seeded +-1 sign draw
    (fht.rotation_signs(d, seed)), and every stored quantity depends only
    on the VALUES of the input unit (its norm, its rotated coordinates,
    and -- for the W15 outlier split -- its channel energies). Provenance
    cannot change a single output bit, so a seeded torch.randn(units, d)
    unit exercises exactly the same code paths and pins exactly the same
    semantics as a captured model state.

BRIEF-TO-SOURCE API MAPPING (bound at the w20-baseline tree)
    brief term          source API (src/rag/turboquant.py @ the pin)
    ------------------  ---------------------------------------------
    "kind" mse          TurboQuant(..., qjl=False) -- the plain MSE layer
    "kind" prod         TurboQuant(..., qjl=True)  -- the paper's Alg.-2
                        "TurboQuant_prod" residual sketch (the W9.2 flag)
    3.5-bit lo/hi       bits=3.5, group=1 -- the fixed 50/50 coordinate
    split (legacy)      split at (3, 4) bits (partition="half", spec §3.3)
    the split path      bits NON-INTEGER and group > 1 -- the W15 paper
    (W15 outlier)       recipe, partition="outlier" (turboquant.py:412
                        gates it: `group > 1 and not bits.is_integer()`),
                        exercised at the production conv geometry
                        d=24576 = 6144 channels x 4 taps, kind="custom",
                        seed 202 (the turboquant.py NOTE).

    Symbols bound (line numbers at the pin):
      TurboQuant(kind, bits, d, seed, qjl, group)   turboquant.py:368
      TurboQuant.quant(x)          one (d,) unit   turboquant.py:470
      TurboQuant.dequant(codes, dtype, device)     turboquant.py:667
      TQCodes (idx_lo/idx_hi are PACKED uint8)     turboquant.py:204
      TQCodes.to_arrays (serialization vocabulary) turboquant.py:248
      pack_bits / unpack_bits                      turboquant.py:159/:186
      get_quantizer(kind, bits)    (registry; the probes need custom
                                    d/seed/group/qjl, so they build
                                    TurboQuant directly -- same class)
      codebooks.get_codebook(bits, d, cache_dir)   codebooks.py:339
      codebooks.CACHE_DIR (cb_b{bits}_d{d}.npz)    codebooks.py:70
    Import convention (house style, src/rag/_paths.py): sys.path gets
    src/rag + src/scripts + src/flute_extended, modules import top-level
    (`import turboquant`, `import codebooks`, `import fht`).

DETERMINISM
    Every probe draws its input from its own torch.Generator(device="cpu")
    seeded SEED_BASE + index (recorded in the manifest AND inside the
    npz). No global, time-based or unseeded randomness anywhere. The
    quantize/dequantize calls run the pinned tree's CPU reference path,
    so the captured artifacts are run-to-run deterministic
    (PROPOSAL.md §5 level 1) on a fixed box.

USAGE (A10G box only, at the detached w20-baseline checkout)
    python3 scripts/gpu/w20_capture_fixtures.py [--out DIR]
    default DIR: src/rag/tests/w20/fixtures
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, NamedTuple

import numpy as np  # part of the pinned tree's own stack (turboquant imports it)
import torch

# ------------------------------------------------------------- the pins ---
# The w20-baseline commit (tag w20-baseline) -- the numpy-era tree whose
# TurboQuant semantics these fixtures freeze. DO NOT update it: the whole
# point is that every capture is reproducible against THIS commit.
W20_BASELINE_SHA = "e9ea357fee64478dc9738e1c6955793c8b3a108d"
W20_BASELINE_TAG = "w20-baseline"

# Per-probe input seeds: 2026_0001 + i (the W20.1.2 brief), one
# torch.Generator(device="cpu") per probe -- fixed and recorded everywhere.
SEED_BASE = 20260001

SCHEMA_VERSION = 1                      # fixtures_manifest.json schema
MANIFEST_NAME = "fixtures_manifest.json"

# repo root = scripts/gpu/w20_capture_fixtures.py -> parents[2]
REPO_ROOT = Path(__file__).resolve().parents[2]
RAG_DIR = REPO_ROOT / "src" / "rag"
DEFAULT_OUT = RAG_DIR / "tests" / "w20" / "fixtures"


# ------------------------------------------------------------- the probes --
class Probe(NamedTuple):
    tag: str         # fixture group: G1 (S, 2^19) | G2 (conv window) | G3 (2^22)
    tq_kind: str     # turboquant KINDS key ("S") or "custom" (explicit d/seed)
    d: int           # unit dimensionality
    frame_seed: int  # the D3 rotation seed (the kind's own; fixed for custom)
    units: int       # rows per probe
    bits: float      # 3.0 / 4.0 uniform | 3.5 -> the (3, 4) lo/hi split
    qjl: bool        # the W9.2 flag: True == the brief's "prod" variant
    chan: int        # TurboQuant(group=chan): coords per channel; 1 = flat


# A covering set (~14 probes): mse/prod x b3/b4, the 3.5-bit (3,4) split,
# the W15 outlier split (non-integer bits + chan > 1), qjl on/off, at the
# three geometries G1/G2/G3. NOTE: an integer-bits "split" probe (e.g.
# b=3 + split, as sketched in the brief) is ILLEGAL per turboquant.py:412 --
# the split exists only for non-integer bits -- so those covering slots are
# realized as 3.5-bit split probes instead.
PROBES = (
    # -- G1: the S geometry (d=2^19, kind "S", D3 seed 101), 4 units -------
    Probe("G1", "S", 524_288, 101, 4, 3.0, False, 1),  # s00 mse  b3   flat
    Probe("G1", "S", 524_288, 101, 4, 4.0, False, 1),  # s01 mse  b4   flat
    Probe("G1", "S", 524_288, 101, 4, 3.0, True, 1),   # s02 prod b3   qjl ON
    Probe("G1", "S", 524_288, 101, 4, 4.0, True, 1),   # s03 prod b4   qjl ON
    Probe("G1", "S", 524_288, 101, 4, 3.5, False, 1),  # s04 mse  3.5  lo/hi split
    Probe("G1", "S", 524_288, 101, 4, 3.5, True, 1),   # s05 prod 3.5  lo/hi split
    # -- G2: the production conv full-window (d=24576 = 6144 ch x 4 taps,
    #    kind "custom", D3 seed 202 per the turboquant.py NOTE), 4 units --
    Probe("G2", "custom", 24_576, 202, 4, 3.0, False, 1),  # s06 mse  b3   flat
    Probe("G2", "custom", 24_576, 202, 4, 3.5, False, 4),  # s07 mse  3.5  outlier split
    Probe("G2", "custom", 24_576, 202, 4, 4.0, True, 1),   # s08 prod b4   qjl ON
    Probe("G2", "custom", 24_576, 202, 4, 3.5, True, 4),   # s09 prod 3.5  outlier split + qjl
    # -- G3: the 2^22 stress geometry (kind "custom", fixed frame seed 404
    #    -- M2's D3 key; any fixed seed pins the same semantics), 2 units -
    Probe("G3", "custom", 4_194_304, 404, 2, 3.0, False, 1),  # s10 mse  b3   flat
    Probe("G3", "custom", 4_194_304, 404, 2, 4.0, True, 1),   # s11 prod b4   qjl ON
    Probe("G3", "custom", 4_194_304, 404, 2, 3.5, False, 1),  # s12 mse  3.5  lo/hi split
    Probe("G3", "custom", 4_194_304, 404, 2, 3.5, True, 1),   # s13 prod 3.5  lo/hi split
)


# ------------------------------------------------------------- helpers ----
def _sha256_file(path: Path) -> str:
    """sha256 of a file's bytes, hex (hashlib, stdlib)."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def probe_filename(probe: Probe, index: int) -> str:
    """G1_mse_b3_s00.npz-style: group, kind, b, variant, index."""
    bits_tok = "b3p5" if probe.bits == 3.5 else f"b{int(probe.bits)}"
    variant = "_split" if probe.chan > 1 else ""
    kind = "prod" if probe.qjl else "mse"
    return f"{probe.tag}_{kind}_{bits_tok}{variant}_s{index:02d}.npz"


# ------------------------------------------------- step 1: the checkout ----
def check_pinned_checkout() -> None:
    """Refuse (exit 2) unless HEAD == W20_BASELINE_SHA (tag w20-baseline).

    Runs BEFORE any repo import; nothing is written on refusal.
    """
    print("[w20] step 1/5 -- pinned-checkout assert "
          "(git rev-parse HEAD vs W20_BASELINE_SHA)")
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(REPO_ROOT), capture_output=True, text=True, check=True)
        actual = proc.stdout.strip()
    except (subprocess.CalledProcessError, OSError) as exc:
        print()
        print("*" * 72)
        print("W20 CHECKOUT REFUSAL: `git rev-parse HEAD` failed at "
              f"{REPO_ROOT} ({exc!r}).")
        print(f"  expected (W20_BASELINE_SHA, tag {W20_BASELINE_TAG}): "
              f"{W20_BASELINE_SHA}")
        print("  Run me from a proper w20-baseline checkout of the repo "
              "(git checkout --detach w20-baseline).")
        print("*" * 72)
        sys.exit(2)
    if actual != W20_BASELINE_SHA:
        print()
        print("*" * 72)
        print("W20 CHECKOUT REFUSAL: this is NOT the w20-baseline checkout.")
        print(f"  expected (W20_BASELINE_SHA, tag {W20_BASELINE_TAG}): "
              f"{W20_BASELINE_SHA}")
        print(f"  actual   (git rev-parse HEAD)              : {actual}")
        print("  The fixture capture must run against the pinned numpy-era")
        print("  tree. On the A10G box: git checkout --detach w20-baseline,")
        print("  then re-run. REFUSING: no repo import, nothing written.")
        print("*" * 72)
        sys.exit(2)
    print(f"[w20]   OK -- HEAD == W20_BASELINE_SHA ({actual[:12]}...)")


# ---------------------------------------------------- step 2: the GPU -----
def check_cuda() -> Dict[str, str]:
    """Refuse (exit 3) on a CPU box; print the device report on success.

    Uses torch only -- nothing from src.rag is imported for this.
    """
    print("[w20] step 2/5 -- CUDA assert (GPU box only, PROMPT.md §1)")
    if not torch.cuda.is_available():
        print()
        print("*" * 72)
        print("W20 GPU-ONLY: A10G required — the capture runs on the GPU "
              "box (PROMPT.md §1)")
        print("  REFUSING to run on a CPU box: exit 3, no repo import, "
              "nothing written.")
        print("*" * 72)
        sys.exit(3)
    idx = torch.cuda.current_device()
    name = torch.cuda.get_device_name(idx)
    cap = torch.cuda.get_device_capability(idx)
    cap_str = f"{cap[0]}.{cap[1]}"
    print(f"[w20]   OK -- device: {name} (capability {cap_str}, "
          f"torch {torch.__version__})")
    return {"name": name, "capability": cap_str}


# ------------------------------------------------- step 4: one probe ------
def run_probe(tq: Any, probe: Probe, index: int):
    """Quantize + dequantize ONE probe with the pinned API.

    Returns (payload, seed, config): `payload` is the npz array dict (keys
    mirror TQCodes.to_arrays, plus the unpacked code twins and the
    input/reconstruction arrays); `config` is the JSON-ready scalar dict.
    Only reached from main(), i.e. AFTER both refusal checks.
    """
    seed = SEED_BASE + index
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    x = torch.randn(probe.units, probe.d, generator=gen,
                    dtype=torch.float32)

    q = tq.TurboQuant(kind=probe.tq_kind, bits=probe.bits, d=probe.d,
                      seed=probe.frame_seed, qjl=probe.qjl,
                      group=probe.chan)
    codes = [q.quant(x[u]) for u in range(probe.units)]
    recon = [q.dequant(c) for c in codes]           # the reference path

    c0 = codes[0]
    for c in codes[1:]:                             # config is per-probe
        if (c.bits_lo, c.bits_hi, c.n_lo, c.n_hi, c.partition) != \
           (c0.bits_lo, c0.bits_hi, c0.n_lo, c0.n_hi, c0.partition):
            raise RuntimeError(
                f"probe {index} ({probe_filename(probe, index)}): "
                "per-unit config drift in the pinned quantizer")

    payload: Dict[str, Any] = {
        # the frozen input (fp32, (units, d))
        "input": x.numpy(),
        # the codes: PACKED uint8 bit-streams (the source's own layout,
        # turboquant.pack_bits) per unit, plus the unpacked index twins
        # (turboquant.unpack_bits) for differential convenience
        "idx_lo": np.stack([np.ascontiguousarray(c.idx_lo, dtype=np.uint8)
                            for c in codes]),
        "idx_hi": np.stack([np.ascontiguousarray(c.idx_hi, dtype=np.uint8)
                            for c in codes]),
        "idx_lo_unpacked": np.stack(
            [tq.unpack_bits(c.idx_lo, c.bits_lo, c.n_lo) for c in codes]),
        "idx_hi_unpacked": np.stack(
            [tq.unpack_bits(c.idx_hi, c.bits_hi, c.n_hi) for c in codes]),
        # the fp32 norm (never quantized) and the reference reconstruction
        "norm": np.asarray([c.norm for c in codes], dtype=np.float32),
        "recon": np.stack([r.detach().to("cpu").to(torch.float32).numpy()
                           for r in recon]),
        # config scalars (the brief's d, units, kind, b, split, qjl, seed
        # -- plus the source-level fields the consumers will want)
        "d": np.int64(probe.d),
        "units": np.int64(probe.units),
        "kind": np.array("prod" if probe.qjl else "mse"),
        "tq_kind": np.array(probe.tq_kind),
        "b": np.float64(probe.bits),
        "split": np.bool_(probe.chan > 1),
        "partition": np.array(c0.partition),
        "chan": np.int64(probe.chan),       # TurboQuant's `group` parameter
        "qjl": np.bool_(probe.qjl),
        "seed": np.int64(seed),
        "frame_seed": np.int64(probe.frame_seed),
        "bits_lo": np.int64(c0.bits_lo),
        "bits_hi": np.int64(c0.bits_hi),
        "n_lo": np.int64(c0.n_lo),
        "n_hi": np.int64(c0.n_hi),
    }
    if c0.partition == "outlier":          # W15: the per-unit split fields
        payload["mask"] = np.stack(
            [np.ascontiguousarray(c.mask, dtype=np.uint8) for c in codes])
        payload["norm_hi"] = np.asarray([c.norm_hi for c in codes],
                                        dtype=np.float32)
    if c0.qjl_signs is not None:           # W9.2: the prod-variant sketch
        payload["qjl_signs"] = np.stack(
            [np.ascontiguousarray(c.qjl_signs, dtype=np.int8)
             for c in codes])
        payload["gamma"] = np.asarray([c.gamma for c in codes],
                                      dtype=np.float32)

    config = {
        "group": probe.tag,
        "kind": "prod" if probe.qjl else "mse",
        "tq_kind": probe.tq_kind,
        "d": probe.d,
        "units": probe.units,
        "b": probe.bits,
        "split": probe.chan > 1,
        "partition": str(c0.partition),
        "chan": probe.chan,
        "qjl": probe.qjl,
        "frame_seed": probe.frame_seed,
        "bits_lo": int(c0.bits_lo),
        "bits_hi": int(c0.bits_hi),
        "n_lo": int(c0.n_lo),
        "n_hi": int(c0.n_hi),
    }
    return payload, seed, config


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Capture the W20 frozen-probe fixtures from the pinned "
                    "w20-baseline (numpy-era) TurboQuant. A10G GPU box "
                    "only: refuses a wrong checkout (exit 2) or a missing "
                    "CUDA device (exit 3) before any repo import.")
    p.add_argument("--out", default=None, metavar="DIR",
                   help=f"output directory (default: {DEFAULT_OUT})")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    print(f"[w20] W20.1.2 fixture capture -- repo root: {REPO_ROOT}")

    # -- the refusal chain: checkout -> CUDA -> (only then) repo imports --
    check_pinned_checkout()                          # exit 2 on wrong tree
    device = check_cuda()                            # exit 3 on a CPU box

    print("[w20] step 3/5 -- importing the pinned numpy-era TurboQuant API")
    sys.path.insert(0, str(REPO_ROOT))               # the W20.1.2 letter
    for rel in ("src/rag", "src/scripts", "src/flute_extended"):
        anchored = str(REPO_ROOT / rel)              # the house convention
        if anchored not in sys.path:                 # (src/rag/_paths.py)
            sys.path.insert(0, anchored)
    import _paths  # noqa: F401  (anchors idempotently; house style)
    import codebooks
    import turboquant
    print(f"[w20]   OK -- turboquant from {turboquant.__file__}")

    # Fingerprint exactly the (bits, d) codebooks the pinned tree actually
    # loads during this capture: a thin recording shim over the public
    # codebooks.get_codebook (delegates verbatim -- behavior unchanged).
    used_codebooks: set = set()
    real_get_codebook = codebooks.get_codebook

    def recording_get_codebook(bits, d, cache_dir=None):
        used_codebooks.add((int(bits), int(d)))
        return real_get_codebook(bits, d, cache_dir)

    codebooks.get_codebook = recording_get_codebook

    out_dir = Path(args.out) if args.out else DEFAULT_OUT
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[w20] step 4/5 -- capturing {len(PROBES)} probes -> {out_dir}")

    entries = []
    for i, probe in enumerate(PROBES):
        name = probe_filename(probe, i)
        payload, seed, config = run_probe(turboquant, probe, i)
        path = out_dir / name
        np.savez(path, **payload)
        sha = _sha256_file(path)
        entries.append({"file": name, "sha256": sha, "seed": seed,
                        "config": config})
        print(f"[w20]   {i + 1:2d}/{len(PROBES)} {name}"
              f"  {path.stat().st_size / 2**20:9.2f} MiB"
              f"  sha256 {sha[:12]}"
              f"  ({config['kind']}, b={config['b']},"
              f" partition={config['partition']},"
              f" qjl={'on' if config['qjl'] else 'off'})")

    codebook_digests: Dict[str, str] = {}
    for bits, d in sorted(used_codebooks):
        cb_name = f"cb_b{bits}_d{d}.npz"
        cb_path = Path(codebooks.CACHE_DIR) / cb_name
        if not cb_path.is_file():
            raise RuntimeError(
                f"codebook {cb_name} is not on disk after the capture "
                f"({codebooks.CACHE_DIR}) -- the pinned loader lied?")
        codebook_digests[cb_name] = _sha256_file(cb_path)
    print(f"[w20]   codebooks fingerprinted: {len(codebook_digests)} "
          f"file(s) from {codebooks.CACHE_DIR}")

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "w20_baseline": W20_BASELINE_SHA,
        "torch_version": torch.__version__,
        "device": device,
        "probes": entries,
        "codebook_digests": codebook_digests,
        "notes": [
            "Synthetic probes are valid: TurboQuant is data-oblivious -- "
            "the Lloyd-Max codebooks are solved from the analytic "
            "sphere-coordinate density (no calibration data), the rotation "
            "is a fixed seeded sign draw, and the quantizer's behavior "
            "depends only on the input VALUES, never on their provenance; "
            "a seeded torch.randn unit pins exactly the same semantics as "
            "a captured model state.",
            "captured on the A10G box from the w20-baseline checkout",
            "kind mapping: 'mse' = TurboQuant qjl=False; 'prod' = qjl=True "
            "(the paper's Alg.-2 'TurboQuant_prod', the W9.2 flag)",
            "'split' = the W15 outlier split (non-integer bits + group>1, "
            "partition='outlier'); bits=3.5 with group=1 is the fixed "
            "50/50 (3,4) coordinate split (partition='half')",
            "determinism: one torch.Generator(device='cpu') per probe, "
            f"seed = {SEED_BASE} + probe index; quantize/dequantize ran "
            "the pinned tree's CPU reference path (PROPOSAL.md §5 "
            "level 1)",
            "sha256 values hash the on-disk .npz bytes; the codebook "
            "digests cover exactly the (bits, d) codebooks loaded during "
            "the capture",
        ],
    }
    manifest_path = out_dir / MANIFEST_NAME
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n",
                             encoding="utf-8")

    print("[w20] step 5/5 -- summary")
    print(f"[w20]   probes   : {len(entries)} fixture file(s) in {out_dir}")
    for e in entries:
        print(f"[w20]     {e['file']:<36} {e['sha256'][:12]}")
    print(f"[w20]   manifest : {manifest_path}")
    print(f"[w20]   codebooks: {len(codebook_digests)} digest(s)")
    print(f"[w20]   baseline : {W20_BASELINE_SHA} (tag {W20_BASELINE_TAG})")
    print("[w20] done -- the W20 frozen-probe fixture set is captured.")


if __name__ == "__main__":
    main()
