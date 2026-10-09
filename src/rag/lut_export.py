"""lut_export.py — the fine-tuned LUT artifact codec (SPECIFICATION §7/§11).

The §7 fine-tune's OUTPUT channel. After `finetune.freeze_all_luts(model)`
snaps the trained fp32 masters onto the fp16 deployment grid,
`export_luts` writes ONE `.npz` per `PalettizedLinear` plus a
`manifest.json` into an output directory (spec §11's `pretrained_luts/`
— ~5.85 GiB at production scale, bytes at test scale). `import_luts`
loads a manifest back into a live model, validating checksums, geometry
and shapes/dtypes, then overwrites the module's `lut`/`lut2` buffers IN
PLACE (`register_buffer(..., persistent=False)` — the house pattern, so
`.to(device)` keeps carrying them and size accounting sees them).

FULL LUT ARTIFACTS, NOT ADAPTERS (spec §7: the fine-tuned LUTs are served
as full LUT artifacts, `pretrained_luts/`, §11): every file carries the
COMPLETE codebook grid of its module. A consumer needs nothing else — no
base-model-plus-delta composition, no code arithmetic; the files ARE the
weights the kernel path dequantizes (the W10 idxN blobs never change,
only the codebooks do).

The export contract — the artifacts are the DEPLOYMENT grid (spec §7):
  * LOUD REFUSAL while any LUT is still a trainable fp32 master
    (`PalettizedLinear.make_trainable()` state): a master is a TRAINING
    artifact, and shipping one would silently corrupt every consumer's
    dtype gate. Call `finetune.freeze_all_luts(model)` first.
  * LOUD REFUSAL on any non-fp16 LUT buffer: the served grid is the fp16
    dequantization grid (the kernel path's dtype contract), so a
    `freeze_lut(snap_fp16=False)` residue is not exportable either.

Per-module `.npz` payload (numpy, no pickle):
  lut        (G, 2**bitwidth)   float16 — the snapped deployment LUT
  lut2       (G, 2**bitwidth2)  float16 — present only when the module
                                    has a second stream (W4 Route A)
  bitwidth / bitwidth2 / group_size / N / K   int64 — the reload geometry
  rotation_seed / rotation_k / fold_order     present only when set on
                                    the module (the Hadamard fold record)
  code_sha256  <str> — sha256 over the LUT code bytes + the canonical
                       config (recomputable from the loaded arrays)
  header       <str> — JSON {version, exported_utc, module, extra_meta}

`manifest.json` (and the return value of `export_luts`):
  {version, n_modules, files: {module_name: filename},
   sha256: {filename: whole-file digest}, code_sha256: {module_name:
   digest}, exported_utc, extra_meta}

`import_luts(model, out_dir, strict=True)` walks the manifest: per entry
the whole-file sha256, the npz header version, the in-file code digest,
the geometry (bitwidths, group_size, N, K, rotation record, stream-2
presence) against the LIVE module, and the lut shape/dtype against the
live buffers — every mismatch fails loudly. `strict=True` (default)
additionally requires the manifest and the live model to cover each
other exactly (drift in either direction raises). `strict=False` is
best-effort: mismatched entries are skipped and reported in the return
dict `{"loaded": [...], "checked": n, "mismatches": [...]}`.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import nn

import _paths  # noqa: F401  (house sys.path anchor — must precede sibling imports)

__all__ = ["LUT_EXPORT_VERSION", "export_luts", "import_luts"]

LUT_EXPORT_VERSION = 1
MANIFEST_NAME = "manifest.json"

# The canonical config-key order — the code digest's config serialization
# (never reorder: digests recorded in old artifacts must stay reproducible).
_CONFIG_KEYS = ("bitwidth", "bitwidth2", "group_size", "N", "K",
                "rotation_seed", "rotation_k", "fold_order", "has_stream2")

_INT_KEYS = ("bitwidth", "bitwidth2", "group_size", "N", "K")
_NPZ_REQUIRED = _INT_KEYS + ("lut", "code_sha256", "header")


class _EntryError(ValueError):
    """One manifest entry failed validation (ValueError subclass: strict
    re-raises it verbatim; non-strict records it as a mismatch)."""


# ------------------------------------------------------------- helpers ---- #

def _module_config(mod) -> Dict:
    """The reload geometry of a PalettizedLinear (JSON-safe dict)."""
    return {
        "bitwidth": int(mod.bitwidth),
        "bitwidth2": int(mod.bitwidth2),
        "group_size": int(mod.group_size),
        "N": int(mod.N),
        "K": int(mod.K),
        "rotation_seed": (None if mod.rotation_seed is None
                          else int(mod.rotation_seed)),
        "rotation_k": (None if mod.rotation_k is None
                       else int(mod.rotation_k)),
        "fold_order": mod.fold_order,
        "has_stream2": bool(mod.has_stream2),
    }


def _canonical_config(cfg: Dict) -> str:
    """Deterministic one-line serialization of a config dict."""
    return "|".join(f"{k}={cfg[k]!r}" for k in _CONFIG_KEYS)


def _code_digest(lut: np.ndarray, lut2: Optional[np.ndarray],
                 cfg: Dict) -> str:
    """sha256 over the LUT code bytes + the canonical config — the content
    digest of a module's artifact (independent of the npz container)."""
    h = hashlib.sha256()
    h.update(np.ascontiguousarray(lut, dtype=np.float16).tobytes())
    if lut2 is not None:
        h.update(np.ascontiguousarray(lut2, dtype=np.float16).tobytes())
    h.update(_canonical_config(cfg).encode("utf-8"))
    return h.hexdigest()


def _file_digest(path: str) -> str:
    """sha256 over a file's raw bytes (the manifest's per-file digest)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def _sanitize(name: str) -> str:
    """Filesystem-safe form of a dotted module name ('a.b.c' -> 'a__b__c')."""
    s = re.sub(r"[^A-Za-z0-9_.-]+", "_", name).replace(".", "__")
    if not s:
        raise ValueError(f"lut_export: cannot sanitize empty module name {name!r}")
    return s


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# -------------------------------------------------------------- export ---- #

def export_luts(model, out_dir: str,
                extra_meta: Optional[dict] = None) -> dict:
    """Write the fp16 deployment grid of every PalettizedLinear under
    `model` to `<out_dir>/*.npz` + `<out_dir>/manifest.json` (spec §11's
    `pretrained_luts/` — FULL LUT artifacts, spec §7, not adapters).

    Refuses loudly while any LUT is a trainable fp32 master (call
    `finetune.freeze_all_luts(model)` first — the artifacts are the
    deployment grid) and on any non-fp16 LUT buffer. Returns the manifest
    dict (also written to disk).
    """
    import palettized_modules as pm  # lazy (house pattern; _paths anchors it)

    if extra_meta is not None and not isinstance(extra_meta, dict):
        raise TypeError(f"export_luts: extra_meta must be a dict or None, "
                        f"got {type(extra_meta).__name__}")
    try:
        json.dumps(extra_meta if extra_meta is not None else {})
    except (TypeError, ValueError) as e:
        raise ValueError(f"export_luts: extra_meta must be JSON-serializable "
                         f"(it lands in manifest.json): {e}") from e

    mods = list(pm.iter_palettized_linears(model))
    if not mods:
        raise ValueError(
            "export_luts: no PalettizedLinear modules under this model — "
            "nothing to export (spec §11 pretrained_luts/ holds the "
            "fine-tuned LUT set; a model without palettized linears has none)")

    # -- the deployment-grid contract (loud, never silent) ---------------- #
    trainable = [
        n for n, m in mods
        if isinstance(getattr(m, "lut", None), nn.Parameter)
        or (m.lut2 is not None and isinstance(m.lut2, nn.Parameter))]
    if trainable:
        raise RuntimeError(
            f"export_luts: refusing to export while a LUT is a trainable "
            f"fp32 master: {trainable}. The artifacts are the DEPLOYMENT "
            f"grid (spec §7/§11 — full LUT artifacts, not adapters); call "
            f"finetune.freeze_all_luts(model) (snap_fp16=True) first.")
    bad_dtype = [(n, str(m.lut.dtype)) for n, m in mods
                 if m.lut.dtype != torch.float16]
    bad_dtype += [(n, str(m.lut2.dtype)) for n, m in mods
                  if m.lut2 is not None and m.lut2.dtype != torch.float16]
    if bad_dtype:
        raise RuntimeError(
            f"export_luts: LUT buffers must be the fp16 deployment grid "
            f"(spec §7): {bad_dtype} — re-freeze with snap_fp16=True")

    os.makedirs(out_dir, exist_ok=True)
    exported_utc = _utc_now()
    files: Dict[str, str] = {}
    sha: Dict[str, str] = {}
    code: Dict[str, str] = {}
    claimed: Dict[str, str] = {}

    for name, mod in mods:
        fname = _sanitize(name) + ".npz"
        if fname in claimed:
            raise ValueError(
                f"export_luts: sanitized filename collision: {fname!r} for "
                f"module {name!r} and {claimed[fname]!r} — refusing an "
                f"ambiguous artifact set")
        claimed[fname] = name

        cfg = _module_config(mod)
        lut_np = np.ascontiguousarray(
            mod.snapped_lut().detach().cpu().numpy(), dtype=np.float16)
        lut2_np = None
        if mod.lut2 is not None:
            lut2_np = np.ascontiguousarray(
                mod.snapped_lut2().detach().cpu().numpy(), dtype=np.float16)

        digest = _code_digest(lut_np, lut2_np, cfg)
        header = {"version": LUT_EXPORT_VERSION, "exported_utc": exported_utc,
                  "module": name, "extra_meta": extra_meta or {}}

        payload = {k: np.int64(cfg[k]) for k in _INT_KEYS}
        payload["lut"] = lut_np
        if lut2_np is not None:
            payload["lut2"] = lut2_np
        if cfg["rotation_seed"] is not None:
            payload["rotation_seed"] = np.int64(cfg["rotation_seed"])
        if cfg["rotation_k"] is not None:
            payload["rotation_k"] = np.int64(cfg["rotation_k"])
        if cfg["fold_order"] is not None:
            payload["fold_order"] = np.array(cfg["fold_order"])
        payload["code_sha256"] = np.array(digest)
        payload["header"] = np.array(json.dumps(header, sort_keys=True))

        path = os.path.join(out_dir, fname)
        np.savez(path, **payload)

        files[name] = fname
        sha[fname] = _file_digest(path)
        code[name] = digest

    manifest = {
        "version": LUT_EXPORT_VERSION,
        "n_modules": len(mods),
        "files": files,
        "sha256": sha,
        "code_sha256": code,
        "exported_utc": exported_utc,
        "extra_meta": extra_meta or {},
    }
    with open(os.path.join(out_dir, MANIFEST_NAME), "w",
              encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
    return manifest


# -------------------------------------------------------------- import ---- #

def _load_entry_npz(path: str, fname: str) -> Tuple[np.ndarray,
                                                    Optional[np.ndarray],
                                                    Dict, str, Dict]:
    """Load + structurally validate one artifact npz. Returns
    (lut, lut2, cfg, code_sha, header)."""
    with np.load(path) as z:  # allow_pickle=False (default): plain arrays only
        missing = [k for k in _NPZ_REQUIRED if k not in z.files]
        if missing:
            raise _EntryError(
                f"import_luts: {fname}: npz is missing keys {missing} — "
                f"not a lut_export artifact")
        lut_np = np.asarray(z["lut"])
        lut2_np = np.asarray(z["lut2"]) if "lut2" in z.files else None
        cfg = {k: int(z[k].item()) for k in _INT_KEYS}
        cfg["rotation_seed"] = (int(z["rotation_seed"].item())
                                if "rotation_seed" in z.files else None)
        cfg["rotation_k"] = (int(z["rotation_k"].item())
                             if "rotation_k" in z.files else None)
        cfg["fold_order"] = (str(z["fold_order"].item())
                             if "fold_order" in z.files else None)
        cfg["has_stream2"] = lut2_np is not None
        try:
            code_sha = str(z["code_sha256"].item())
            header = json.loads(str(z["header"].item()))
        except (ValueError, TypeError) as e:
            raise _EntryError(
                f"import_luts: {fname}: unreadable code_sha256/header "
                f"({e}) — corrupt artifact?") from e
    return lut_np, lut2_np, cfg, code_sha, header


def _import_entry(live: Dict[str, "object"], name: str, fname: str,
                  out_dir: str, manifest: Dict) -> None:
    """Validate + in-place-overwrite ONE manifest entry. Raises
    _EntryError on every mismatch (loud)."""
    mod = live.get(name)
    if mod is None:
        raise _EntryError(
            f"import_luts: manifest entry {name!r} has no live "
            f"PalettizedLinear — the artifact set and this model have "
            f"drifted (strict import requires exact coverage)")
    path = os.path.join(out_dir, fname)
    if not os.path.isfile(path):
        raise _EntryError(
            f"import_luts: {path} is missing (the manifest lists it)")
    want = (manifest.get("sha256") or {}).get(fname)
    got = _file_digest(path)
    if want is None or got != want:
        raise _EntryError(
            f"import_luts: sha256 mismatch for {fname}: manifest "
            f"{want}, got {got} — the file changed after export")

    lut_np, lut2_np, cfg, code_sha, header = _load_entry_npz(path, fname)
    if int(header.get("version", -1)) != LUT_EXPORT_VERSION:
        raise _EntryError(
            f"import_luts: {fname}: npz header version "
            f"{header.get('version')!r} != LUT_EXPORT_VERSION "
            f"{LUT_EXPORT_VERSION} — regenerate the export")
    if code_sha != _code_digest(lut_np, lut2_np, cfg):
        raise _EntryError(
            f"import_luts: {fname}: code sha256 mismatch — the LUT content "
            f"does not match its recorded digest (corrupt artifact?)")

    live_cfg = _module_config(mod)
    if cfg != live_cfg:
        raise _EntryError(
            f"import_luts: {name!r}: exported geometry {cfg} != live module "
            f"geometry {live_cfg} — the artifact set and the model "
            f"disagree (bitwidths/group_size/N/K/rotation record)")

    # the buffers are overwritten IN PLACE — a master would be destroyed
    if isinstance(mod.lut, nn.Parameter) or (
            mod.lut2 is not None and isinstance(mod.lut2, nn.Parameter)):
        raise _EntryError(
            f"import_luts: {name!r}: the live LUT is a trainable fp32 "
            f"master — call finetune.freeze_all_luts(model) before "
            f"installing artifacts (import overwrites the buffers in place)")

    # explicit shape/dtype validation vs the live buffers
    if lut_np.dtype != np.float16 or tuple(lut_np.shape) != tuple(mod.lut.shape):
        raise _EntryError(
            f"import_luts: {name!r}: exported lut shape/dtype "
            f"{tuple(lut_np.shape)}/{lut_np.dtype} != live buffer "
            f"{tuple(mod.lut.shape)}/float16")
    if lut2_np is not None:
        live2 = mod.lut2
        if live2 is None or tuple(lut2_np.shape) != tuple(live2.shape) \
                or lut2_np.dtype != np.float16:
            raise _EntryError(
                f"import_luts: {name!r}: exported lut2 shape/dtype "
                f"{tuple(lut2_np.shape)}/{lut2_np.dtype} does not match "
                f"the live module (no stream 2 or "
                f"{tuple(live2.shape) if live2 is not None else None})")
    elif mod.lut2 is not None:
        raise _EntryError(
            f"import_luts: {name!r}: the live module has a stream-2 LUT "
            f"but the artifact has none (lut2)")

    # in-place overwrite (house pattern: non-persistent buffer)
    dev = mod.lut.device
    lut_t = torch.from_numpy(np.ascontiguousarray(lut_np)).to(
        dtype=torch.float16, device=dev)
    mod.register_buffer("lut", lut_t, persistent=False)
    if lut2_np is not None:
        lut2_t = torch.from_numpy(np.ascontiguousarray(lut2_np)).to(
            dtype=torch.float16, device=dev)
        mod.register_buffer("lut2", lut2_t, persistent=False)


def import_luts(model, out_dir: str, strict: bool = True) -> dict:
    """Install a `pretrained_luts/` export into `model` (spec §7/§11: the
    fine-tuned FULL LUT artifacts — every lut/lut2 buffer is overwritten
    in place after checksum + geometry + shape validation).

    strict=True (default): every manifest entry must match a live
    PalettizedLinear and vice versa — drift fails loudly. strict=False:
    best-effort (mismatched entries skipped, reported). Returns
    {"loaded": [module names], "checked": n_manifest_entries,
    "mismatches": [messages]}.
    """
    import palettized_modules as pm  # lazy (house pattern; _paths anchors it)

    mpath = os.path.join(out_dir, MANIFEST_NAME)
    if not os.path.isfile(mpath):
        raise ValueError(
            f"import_luts: no {MANIFEST_NAME} at {mpath} — not a "
            f"pretrained_luts/ export directory (spec §11)")
    try:
        with open(mpath, encoding="utf-8") as f:
            manifest = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(
            f"import_luts: {mpath} is not valid JSON ({e}) — corrupt "
            f"manifest") from e
    if int(manifest.get("version", -1)) != LUT_EXPORT_VERSION:
        raise ValueError(
            f"import_luts: manifest version {manifest.get('version')!r} != "
            f"LUT_EXPORT_VERSION {LUT_EXPORT_VERSION} — regenerate the export")
    files = manifest.get("files")
    if not isinstance(files, dict) or not files:
        raise ValueError(
            f"import_luts: {mpath}: manifest has no 'files' mapping — "
            f"corrupt manifest")

    live = {name: mod for name, mod in pm.iter_palettized_linears(model)}
    loaded: List[str] = []
    mismatches: List[str] = []
    checked = 0
    for name, fname in files.items():
        checked += 1
        try:
            _import_entry(live, name, str(fname), str(out_dir), manifest)
            loaded.append(name)
        except _EntryError as e:
            if strict:
                raise ValueError(str(e)) from e
            mismatches.append(str(e))

    extra = sorted(n for n in live if n not in files)
    if extra:
        msg = (f"import_luts: live palettized modules not covered by the "
               f"manifest: {extra} — the artifact set and this model have "
               f"drifted (strict import requires exact coverage)")
        if strict:
            raise ValueError(msg)
        mismatches.append(msg)
    return {"loaded": loaded, "checked": checked, "mismatches": mismatches}
