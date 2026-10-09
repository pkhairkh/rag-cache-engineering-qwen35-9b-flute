#!/usr/bin/env python3
"""scripts/capture.py — the boundary-state capture store + CLI.

P0 of the layerwise pipeline (PROPOSAL §2.1-2.2): one batched forward of
the pristine dense teacher persists, per captured token, every layer's
INPUT hidden state (h_0..h_{N-1}, h_0 = the embedding output), the
post-norm final_hidden.npy, and the exact input_ids.npy — the boundary
store, manifest format 3, packed rows, no stored targets (the layerwise
trainer computes targets on the fly from the resident teacher layer).

CaptureStore is the uniform reader: row-exact (i0, i1) spans and
ascending row lists at amplification 1.0 (one memmap slice + one H2D
per contiguous run; the reader_amp registry measures it), plus the
readable v2 (x1 + stored targets) and legacy v1 (padded) layouts, which
refuse misuse loudly. run_boundary_capture is the testable core (any
model with the pmod.get_layers structure drives it); the `capture`
subcommand is the CLI of record. The engine imports this module until
its own deletion (TASKS W2-T10).
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import palettized_modules as pmod          # noqa: E402  (local, same dir)
from eval_common import atomic_json_dump as _atomic_json_dump  # noqa: E402

def _load_model_for_causal_lm(model_ref, device="cpu"):
    """Load the model using transformers AutoModelForCausalLM with trust_remote_code.
    
    The vendored Qwen3_5ForCausalLM cannot load the HF checkpoint correctly
    because the checkpoint has multimodal architecture (model.language_model.layers)
    while the vendored class expects text-only (model.layers). Use transformers'
    AutoModelForCausalLM which handles the architecture mismatch.
    """
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(
        model_ref,
        trust_remote_code=True,
        dtype=torch.float16,
        low_cpu_mem_usage=True,
        attn_implementation="sdpa",
    )
    if device != "cpu":
        model = model.to(device)
    return model

def _text_config(model):
    # `model` may be a real model (has .config) or a raw (composite) config
    cfg = getattr(model, "config", model)
    if getattr(cfg, "text_config", None) is not None:
        return cfg.text_config
    return cfg

def _memmap_rw(path: str, shape, dtype=np.float16):
    """Open (creating if needed) an .npy memmap for read/write."""
    return np.lib.format.open_memmap(path, mode="r+" if os.path.exists(path)
                                     else "w+", shape=shape, dtype=dtype)

# ---------------------------------------------------------------------------
# Capture store (boundary + v2 packed + legacy reader)
# ---------------------------------------------------------------------------

# --- row-exact reader primitives (W1.T1 / PROPOSAL T1) --------------------

def _split_runs(rows):
    """Split an ASCENDING int row list into maximal contiguous runs
    [(i0, i1_inclusive), ...]: [1, 2, 3, 7, 9, 10] -> [(1, 3), (7, 7),
    (9, 10)]. The row-exact readers turn each run into ONE memmap slice
    + ONE H2D transfer, so the run count is the read's I/O op count.

    Non-ascending (unsorted or duplicate) input raises ValueError: every
    engine call site hands the reader SORTED rows (batch rows are sorted
    at the single point they are drawn, eval rows are ascending slices,
    target-cache batches are ranges), so an out-of-order list is a caller
    bug — silently sorting HERE would reorder the caller's rows against
    their targets, so the reader refuses loudly instead."""
    rows = [int(r) for r in rows]
    for a, b in zip(rows, rows[1:]):
        if not b > a:
            raise ValueError(
                f"_split_runs: rows must be strictly ascending, got "
                f"{rows!r} — sort the batch before calling the reader "
                f"(e.g. `rows = sorted(int(r) for r in rows)`) and drop "
                f"duplicates; the engine sorts batch rows at every call "
                f"site, so an unsorted list here is a caller bug")
    runs = []
    i = 0
    while i < len(rows):
        j = i
        while j + 1 < len(rows) and rows[j + 1] == rows[j] + 1:
            j += 1
        runs.append((rows[i], rows[j]))
        i = j + 1
    return runs


# --- reader-I/O amplification registry (W1.T1 s3; the G-T1 metric) --------
# bytes_needed = bytes the caller actually asked for; bytes_moved = bytes
# the reader physically read off the store; runs = contiguous block reads
# performed (one _record_reader_io call == one block read). Row bytes are
# seq * hidden * itemsize of the STORE dtype (amplification is a ratio, so
# store-dtype units are the honest numerator/denominator pair).
_READER_COUNTERS = {"bytes_needed": 0, "bytes_moved": 0, "runs": 0}


def _reader_counters_reset():
    """Zero the reader-I/O counters. Tests bracket their reads with this;
    the layer job resets per job when it starts logging the ratio."""
    _READER_COUNTERS["bytes_needed"] = 0
    _READER_COUNTERS["bytes_moved"] = 0
    _READER_COUNTERS["runs"] = 0


def _reader_counters_snapshot() -> dict:
    """Copy of the current counters: {"bytes_needed", "bytes_moved",
    "runs"}."""
    return dict(_READER_COUNTERS)


def _record_reader_io(rows_needed, rows_moved, row_nbytes):
    """Accumulate ONE contiguous block read: `rows_needed` rows were
    requested from this block, `rows_moved` rows were physically read (a
    span/block reader moves the whole min..max block — the I/O
    amplification this registry exists to expose), `row_nbytes` bytes per
    row. h_rows() records one call per run (needed == moved == run
    length); h_span_rows() records one call for the whole span."""
    _READER_COUNTERS["bytes_needed"] += int(rows_needed) * int(row_nbytes)
    _READER_COUNTERS["bytes_moved"] += int(rows_moved) * int(row_nbytes)
    _READER_COUNTERS["runs"] += 1


def _reader_amp() -> float:
    """W5.T1 (PROPOSAL §3 T7): the reader-I/O amplification ratio —
    bytes_moved / bytes_needed, cumulative since the last
    _reader_counters_reset() (the layer job resets per job at start).
    Exactly 1.0 on row-exact reads (W1.T1's G-T1 metric); 1.0 when
    nothing was recorded yet (the honest no-data value — a division
    guard, never a nan on a log line)."""
    snap = _reader_counters_snapshot()
    if snap["bytes_needed"] <= 0:
        return 1.0
    return snap["bytes_moved"] / snap["bytes_needed"]

class CaptureStore:
    """Uniform reader over the teacher capture.

    boundary format (this script's `capture` subcommand since W3 — the
    Stage 0 store; packed rows, no padding, pristine teacher):
        <dir>/manifest.json   {"format": 3, "kind": "boundary", "rows": R,
                               "seq_length": S, "hidden_size": H,
                               "num_layers": N, ...}
        <dir>/h_{L}.npy               (R, S, H) fp16 — the INPUT hidden
                                      state of layer L (h_0 = the embedding
                                      output); served by h(L, i0, i1)
        <dir>/final_hidden.npy        (R, S, H) fp16 (post-norm — ALWAYS
                                      written; Stage 2's target)
        <dir>/input_ids.npy           (R, S) int32 (ALWAYS written — Stage
                                      2 must feed the model the exact
                                      tokens that produced the hiddens)
        Boundary stores carry NO stored targets: the Stage-1 target is
        computed on the fly by the resident teacher layer (W4).

    v2 packed format (the pre-W3 `capture`; no longer written, still
    readable — the LUT escalation path consumes it via an existing v2
    capture dir):
        <dir>/manifest.json   {"format": 2, ...}
        <dir>/x1.npy                  (R, S, H) fp16
        <dir>/targets/layer_L.npy     (R, S, H) fp16   (down_proj outputs)
        <dir>/final_hidden.npy        (R, S, H) fp16   (optional)

    legacy format (capture_down_proj_outputs.py — padded rows, teacher
    contaminated with norm edits when --awq-scale was used):
        <hidden-dir>/x1_set{i}.npy            (rows_i, S, H) fp16
        <activations-dir>/layer_L_set{i}.npy  (rows_i, S, H) fp16
    """

    def __init__(self, capture_dir: str = None,
                 legacy_hidden_dir: str = None,
                 legacy_activations_dir: str = None,
                 legacy_include_padding: bool = False):
        self.kind = None
        self.boundary = False
        self._final = None
        if capture_dir:
            man_path = os.path.join(capture_dir, "manifest.json")
            if not os.path.exists(man_path):
                raise FileNotFoundError(
                    f"{capture_dir}/manifest.json missing — run "
                    f"`capture.py capture` first (the v1 padded "
                    f"capture is consumed via --legacy-* flags)")
            with open(man_path) as f:
                self.manifest = json.load(f)
            fmt = int(self.manifest.get("format", 0))
            if fmt not in (2, 3):
                raise ValueError(
                    f"unsupported capture format {fmt} in {capture_dir} "
                    f"(expected format: 2 or 3)")
            self.dir = capture_dir
            self.rows = int(self.manifest["rows"])
            self.seq = int(self.manifest["seq_length"])
            self.hidden = int(self.manifest["hidden_size"])
            self.num_layers = int(self.manifest["num_layers"])
            if fmt == 3:
                if self.manifest.get("kind") != "boundary":
                    raise ValueError(
                        f"format-3 capture at {capture_dir} declares kind "
                        f"{self.manifest.get('kind')!r} (expected "
                        f"'boundary') — refusing to guess the file layout")
                self.kind = "boundary"
                self.boundary = True
                self.has_final_hidden = os.path.exists(
                    os.path.join(capture_dir, "final_hidden.npy"))
                # manifest-vs-file contract, checked once at open so a
                # broken store fails loudly HERE, not silently at
                # row-serve time
                h0_path = os.path.join(capture_dir, "h_0.npy")
                if not os.path.exists(h0_path):
                    raise FileNotFoundError(
                        f"boundary store {capture_dir} is missing h_0.npy")
                self._check_boundary_array(
                    np.load(h0_path, mmap_mode="r"), "h_0.npy")
                self._h = {}
                self._ids = None
            else:
                self.kind = "v2"
                self.has_final_hidden = os.path.exists(
                    os.path.join(capture_dir, "final_hidden.npy"))
                self._x1 = None
                self._targets = {}
        elif legacy_hidden_dir and legacy_activations_dir:
            if not legacy_include_padding:
                raise SystemExit(
                    "legacy (v1) capture selected but --legacy-include-"
                    "padding not set.\n\nThe v1 capture padded every row to "
                    "seq_length WITHOUT storing attention masks, so pad "
                    "positions cannot be excluded from the loss; it also "
                    "applied norm_gain_edits to the dense teacher (targets "
                    "from a modified model) whenever --awq-scale had been "
                    "used.\n\nRecommended: re-run the `capture` subcommand "
                    "(packed rows, pristine teacher, ~1-2 h on the A10G for "
                    "64 batches). Pass --legacy-include-padding to proceed "
                    "anyway (loss will include pad positions).")
            self.kind = "legacy"
            self.hidden_dir = legacy_hidden_dir
            self.activations_dir = legacy_activations_dir
            self._x1_sets = sorted(
                [f for f in os.listdir(legacy_hidden_dir)
                 if f.startswith("x1_set") and f.endswith(".npy")],
                key=lambda s: int(s.split("set")[1].split(".")[0]))
            if not self._x1_sets:
                raise FileNotFoundError(f"no x1_set*.npy under {legacy_hidden_dir}")
            x1_0 = np.load(os.path.join(legacy_hidden_dir, self._x1_sets[0]),
                           mmap_mode="r")
            self.seq = int(x1_0.shape[1])
            self.hidden = int(x1_0.shape[2])
            self.rows = 0
            self._set_rows = []
            for f in self._x1_sets:
                r = int(np.load(os.path.join(legacy_hidden_dir, f),
                                mmap_mode="r").shape[0])
                self._set_rows.append(r)
                self.rows += r
            self.num_layers = len({f.split("_set")[0] for f in
                                   os.listdir(legacy_activations_dir)
                                   if f.startswith("layer_")})
            self.has_final_hidden = False
            self._x1 = {}
            self._targets = {}
            print(f"[capture] legacy v1 store: {self.rows} rows x "
                  f"{self.seq} x {self.hidden}, {self.num_layers} layers "
                  f"(padding INCLUDED in loss)", flush=True)
        else:
            raise ValueError(
                "no capture source given (--capture-dir or --legacy-*)")

    # -- row access --------------------------------------------------------

    def h(self, layer: int, i0: int, i1: int, device,
          dtype=torch.float32):
        """(i1 - i0, S, H) rows of layer `layer`'s INPUT hidden state from
        a boundary store — h(0) is the embedding output (the tensor the v2
        layout called x1). Boundary stores only; v2/legacy stores raise
        with a pointer to re-run capture."""
        if self.kind != "boundary":
            raise RuntimeError(
                f"h() requires a boundary store (format 3, kind "
                f"'boundary'); this store is {self.kind!r} — re-run "
                f"`capture.py capture` to write the boundary "
                f"layout (h_0..h_{self.num_layers - 1} + final_hidden + "
                f"input_ids), or read the old layout via x1()/target()")
        if not (0 <= int(layer) < self.num_layers):
            raise IndexError(
                f"layer {layer} outside 0..{self.num_layers - 1} (boundary "
                f"store at {self.dir!r} has num_layers={self.num_layers})")
        if layer not in self._h:
            path = os.path.join(self.dir, f"h_{layer}.npy")
            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"boundary store {self.dir!r} is missing h_{layer}.npy "
                    f"(manifest promises num_layers={self.num_layers}) — "
                    f"incomplete capture, re-run into a fresh dir")
            mm = np.load(path, mmap_mode="r")
            self._check_boundary_array(mm, f"h_{layer}.npy")
            self._h[layer] = mm
        a = np.ascontiguousarray(self._h[layer][i0:i1].copy())
        return torch.from_numpy(a).to(device=device, dtype=dtype)

    def _h_memmap(self, layer: int):
        """The h_{layer}.npy memmap from the shared self._h cache (load +
        contract-check on first use) — the common access path of
        h()/h_rows()/h_span_rows()."""
        if layer not in self._h:
            path = os.path.join(self.dir, f"h_{layer}.npy")
            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"boundary store {self.dir!r} is missing h_{layer}.npy "
                    f"(manifest promises num_layers={self.num_layers}) — "
                    f"incomplete capture, re-run into a fresh dir")
            mm = np.load(path, mmap_mode="r")
            self._check_boundary_array(mm, f"h_{layer}.npy")
            self._h[layer] = mm
        return self._h[layer]

    def h_rows(self, layer: int, rows, device=None,
               dtype=torch.float32):
        """Row-EXACT read of layer `layer`'s INPUT hidden state from a
        boundary store: a (len(rows), S, H) tensor on `device`/`dtype`
        assembled from ONLY the requested rows. The row list is split
        into maximal contiguous runs (_split_runs); each run is ONE
        memmap slice + ONE H2D transfer; the runs are torch.cat-ed in run
        order. NEVER computes min(rows)..max(rows): the old span reader
        moved the whole block to gather a scattered batch (a 4-row batch
        spanning ~452 of 768 rows copied 7.6 GB to consume 67 MB — 113x
        I/O amplification, the P1 root cause). `rows` must be ascending
        (every engine call site sorts; non-ascending raises loudly via
        _split_runs); row ids are bounds-checked because a numpy slice
        would silently clip (or wrap for negatives). Empty rows ->
        torch.empty(0, S, H). Boundary stores only; validations mirror
        h()."""
        if self.kind != "boundary":
            raise RuntimeError(
                f"h_rows() requires a boundary store (format 3, kind "
                f"'boundary'); this store is {self.kind!r} — re-run "
                f"`capture.py capture` to write the boundary "
                f"layout (h_0..h_{self.num_layers - 1} + final_hidden + "
                f"input_ids), or read the old layout via x1()/target()")
        if not (0 <= int(layer) < self.num_layers):
            raise IndexError(
                f"layer {layer} outside 0..{self.num_layers - 1} (boundary "
                f"store at {self.dir!r} has num_layers={self.num_layers})")
        rows = [int(r) for r in rows]
        if not rows:
            return torch.empty(0, self.seq, self.hidden, device=device,
                               dtype=dtype)
        for r in rows:
            if not 0 <= r < self.rows:
                raise IndexError(
                    f"row {r} outside 0..{self.rows - 1} (boundary store "
                    f"at {self.dir!r} has rows={self.rows}) — a slice read "
                    f"would silently clip (or wrap, for negatives), so the "
                    f"row-exact reader refuses it loudly")
        runs = _split_runs(rows)
        mm = self._h_memmap(layer)
        row_nbytes = self.seq * self.hidden * mm.dtype.itemsize
        parts = []
        for i0, i1 in runs:
            # ONE slice per run -> ONE H2D per run; never a span read.
            # The runs partition the ascending row list exactly, so the
            # per-run records sum to bytes_needed == bytes_moved ==
            # len(rows) * row_nbytes (amplification 1.0).
            a = np.ascontiguousarray(mm[i0:i1 + 1].copy())
            _record_reader_io(i1 - i0 + 1, i1 - i0 + 1, row_nbytes)
            parts.append(torch.from_numpy(a).to(device=device, dtype=dtype))
        return torch.cat(parts, dim=0)

    def h_span_rows(self, layer: int, rows, device, dtype=torch.float32):
        """The OLD min..max span reader, kept ONLY as the regression
        canary for the reader-amplification metric (tests assert it still
        reports its honest blow-up): reads the WHOLE min(rows)..max(rows)
        block via h() onto the device and gathers the requested rows,
        recording rows_moved = span length. Production code must call
        h_rows() instead — never this method."""
        if self.kind != "boundary":
            raise RuntimeError(
                f"h_span_rows() requires a boundary store (format 3, kind "
                f"'boundary'); this store is {self.kind!r} — re-run "
                f"`capture.py capture` to write the boundary "
                f"layout (h_0..h_{self.num_layers - 1} + final_hidden + "
                f"input_ids), or read the old layout via x1()/target()")
        if not (0 <= int(layer) < self.num_layers):
            raise IndexError(
                f"layer {layer} outside 0..{self.num_layers - 1} (boundary "
                f"store at {self.dir!r} has num_layers={self.num_layers})")
        rows = [int(r) for r in rows]
        if not rows:
            return torch.empty(0, self.seq, self.hidden, device=device,
                               dtype=dtype)
        mm = self._h_memmap(layer)
        row_nbytes = self.seq * self.hidden * mm.dtype.itemsize
        i0, i1 = min(rows), max(rows) + 1
        _record_reader_io(len(rows), i1 - i0, row_nbytes)
        block = self.h(layer, i0, i1, device, dtype)
        indices = torch.tensor([i - i0 for i in rows], device=device)
        return block[indices]

    def input_ids(self, i0: int, i1: int, device=None,
                  dtype=torch.long):
        """(i1 - i0, S) captured token ids (stored int32; torch.long by
        default). Stage 2 must feed the model exactly these ids — they are
        the tokens that produced every stored hidden state. Boundary
        stores only."""
        if self.kind != "boundary":
            raise RuntimeError(
                f"input_ids() requires a boundary store (format 3); this "
                f"store is {self.kind!r} — re-run "
                f"`capture.py capture` (the boundary capture "
                f"always records input_ids.npy)")
        if self._ids is None:
            path = os.path.join(self.dir, "input_ids.npy")
            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"boundary store {self.dir!r} is missing input_ids.npy "
                    f"— the capture always writes it; re-run into a fresh "
                    f"dir")
            mm = np.load(path, mmap_mode="r")
            self._check_boundary_array(mm, "input_ids.npy",
                                       shape=(self.rows, self.seq),
                                       dtype="int32")
            self._ids = mm
        a = np.ascontiguousarray(self._ids[i0:i1].copy())
        return torch.from_numpy(a).to(device=device, dtype=dtype)

    def x1(self, i0: int, i1: int, device, dtype=torch.float32):
        if self.kind == "boundary":
            # h(0) IS the v2 x1 (the embedding output)
            return self.h(0, i0, i1, device, dtype)
        if self.kind == "v2":
            if self._x1 is None:
                self._x1 = np.load(os.path.join(self.dir, "x1.npy"),
                                   mmap_mode="r")
            a = np.ascontiguousarray(self._x1[i0:i1].copy())
        else:
            a = self._legacy_slice(self._x1_sets, self._x1, "x1", i0, i1)
        return torch.from_numpy(a).to(device=device, dtype=dtype)

    def target(self, layer: int, i0: int, i1: int, device,
               dtype=torch.float32):
        if self.kind == "boundary":
            raise RuntimeError(
                "boundary stores (format 3) carry no stored targets: the "
                "Stage-1 target is computed on the fly by the resident "
                "teacher layer, and the LUT escalation taps down_proj from "
                "the resident teacher layer (W4). Stored down_proj targets "
                "exist only in the old v2 layout — consume an existing v2 "
                "capture dir (that layout is no longer written)")
        if self.kind == "v2":
            path = os.path.join(self.dir, "targets", f"layer_{layer}.npy")
            if layer not in self._targets:
                if not os.path.exists(path):
                    raise FileNotFoundError(
                        f"teacher target file missing: {path}")
                self._targets[layer] = np.load(path, mmap_mode="r")
            a = np.ascontiguousarray(self._targets[layer][i0:i1].copy())
        else:
            files = sorted(
                [f for f in os.listdir(self.activations_dir)
                 if f.startswith(f"layer_{layer}_set") and f.endswith(".npy")],
                key=lambda s: int(s.split("set")[1].split(".")[0]))
            a = self._legacy_slice(files, self._targets.get(layer),
                                   f"layer_{layer}", i0, i1)
        return torch.from_numpy(a).to(device=device, dtype=dtype)

    def final_hidden(self, i0: int, i1: int, device,
                     dtype=torch.float32):
        if self.kind == "boundary":
            if not self.has_final_hidden:
                raise FileNotFoundError(
                    f"final_hidden.npy missing from the boundary (format "
                    f"3) store {self.dir!r} — the capture always writes it "
                    f"(Stage 2's target); the store is incomplete, re-run "
                    f"capture into a fresh dir")
        elif not self.has_final_hidden:
            raise FileNotFoundError("final_hidden.npy not captured")
        if self._final is None:
            self._final = np.load(os.path.join(self.dir, "final_hidden.npy"),
                                  mmap_mode="r")
            if self.kind == "boundary":
                self._check_boundary_array(self._final, "final_hidden.npy")
        a = np.ascontiguousarray(self._final[i0:i1].copy())
        return torch.from_numpy(a).to(device=device, dtype=dtype)

    def _check_boundary_array(self, arr, name: str, shape=None,
                              dtype: str = "float16"):
        """Boundary-store contract: a file and the manifest must agree —
        a mismatch is a loud ValueError, never silently-wrong rows."""
        if shape is None:
            shape = (self.rows, self.seq, self.hidden)
        if tuple(arr.shape) != tuple(shape) or str(arr.dtype) != dtype:
            raise ValueError(
                f"boundary store {self.dir!r}: {name} is "
                f"{tuple(arr.shape)}/{arr.dtype}, the manifest promises "
                f"{tuple(shape)}/{dtype} — store and manifest disagree")

    def _legacy_slice(self, files, cache, tag, i0, i1):
        if cache is None:
            cache = {}
            if tag == "x1":
                self._x1 = cache
            else:
                self._targets[tag] = cache
        out = []
        pos = 0
        for fi, f in enumerate(files):
            r = self._set_rows[fi]
            if i1 <= pos or i0 >= pos + r:
                pos += r
                continue
            if f not in cache:
                base = (self.hidden_dir if tag == "x1"
                        else self.activations_dir)
                cache[f] = np.load(os.path.join(base, f), mmap_mode="r")
            lo = max(i0, pos) - pos
            hi = min(i1, pos + r) - pos
            out.append(np.ascontiguousarray(cache[f][lo:hi]))
            pos += r
        return np.concatenate(out, axis=0) if out else np.zeros(
            (0, self.seq, self.hidden), np.float16)

def _token_stream(args, tokenizer, rng):
    """Infinite generator of token ids (python ints) for packing.

    Sources: fineweb (streaming), wikitext2, file:<path> (one doc per
    line), synthetic (random ids over the vocab — testing only). Documents
    are separated by EOS; sequences are exact seq_length blocks cut from
    the stream (the calibrate_real_text.py 'concatenate then re-chunk'
    convention), so no padding exists anywhere in the capture."""
    if args.source == "synthetic":
        vocab = args.synthetic_vocab
        while True:
            block = rng.integers(0, vocab, size=args.seq_length)
            yield from block.tolist()
    if args.source.startswith("file:"):
        path = args.source[len("file:"):]
        with open(path) as f:
            texts = [ln.strip() for ln in f if ln.strip()]
    elif args.source == "wikitext2":
        from datasets import load_dataset
        ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1",
                          split="train")
        texts = [t for t in ds["text"] if len(t.strip()) > 0]
    elif args.source == "fineweb":
        from datasets import load_dataset
        ds = iter(load_dataset("HuggingFaceFW/fineweb-edu",
                               name="sample-10BT", split="train",
                               streaming=True))
        texts = []
        while True:
            texts = [next(ds)["text"] for _ in range(256)]
            ids = []
            for t in texts:
                ids.extend(tokenizer(t, add_special_tokens=False)
                           ["input_ids"])
                ids.append(tokenizer.eos_token_id)
            yield from ids
    else:
        raise ValueError(f"unknown --source {args.source!r}")

    # file:/wikitext2: tokenize the whole corpus once, then loop it
    ids = []
    for t in texts:
        ids.extend(tokenizer(t, add_special_tokens=False)["input_ids"])
        ids.append(tokenizer.eos_token_id)
    if len(ids) < args.seq_length:
        raise ValueError(
            f"corpus yielded {len(ids)} tokens < seq_length "
            f"{args.seq_length}")
    while True:
        yield from ids


# ---------------------------------------------------------------------------
# capture subcommand — boundary-state store (PROPOSAL §4 Stage 0)
# ---------------------------------------------------------------------------

def _final_norm_module(model):
    """Locate the post-layers final norm the same way pmod.get_layers
    locates the layer stack: the `.norm` sibling of the `.layers` list —
    model.model.norm for text-only layouts, model.model.language_model.norm
    for the multimodal HF layout (the loaded checkpoint's actual shape) —
    with model.model.norm as a last-resort fallback. Loud AttributeError
    when nothing is found: the capture needs the norm for final_hidden.npy
    and guessing wrong would corrupt the store."""
    m = getattr(model, "model", None)
    candidates = []
    if m is not None:
        if hasattr(m, "layers"):
            candidates.append(m)
        lm = getattr(m, "language_model", None)
        if lm is not None and hasattr(lm, "layers"):
            candidates.append(lm)
        candidates.append(m)
    for c in candidates:
        norm = getattr(c, "norm", None)
        if norm is not None:
            return norm
    raise AttributeError(
        "cannot locate the final norm for the boundary capture (looked "
        "for model.model.norm / model.model.language_model.norm) — the "
        "post-layers RMSNorm is required for final_hidden.npy")


def run_boundary_capture(model, layers=None, out_dir=None, rows_total=None,
                         seq_length=None, hidden_size=None, token_iter=None,
                         batch_size=None, log_every=8, source_desc=None,
                         model_desc=None, seed=None, device=None):
    """Capture CORE (testable without the real 9B model): one batched
    streaming forward of `model` persisting every layer's input hidden
    state — the boundary-state store, manifest format 3.

    The model is used ONLY through: model(input_ids=..., attention_mask=
    None, use_cache=False) under no_grad; pmod.get_layers(model); forward
    PRE-hooks on those layers (args[0] = the layer's input hidden state);
    and a forward hook on the post-layers final norm. Any model with that
    structure works — a tiny toy stack (embedding -> layers -> norm, the
    hidden state as each layer's first positional argument) drives the
    tests (tests/test_capture_store.py).

    Args:
      model:       the teacher (eval mode is the caller's job).
      layers:      the FULL decoder stack; None -> pmod.get_layers(model).
                   A partial list is refused loudly (h_L files would be
                   mislabeled).
      out_dir:     destination directory (created; must not already contain
                   boundary files — overwrite is refused).
      rows_total:  total rows R to capture.
      seq_length:  tokens per row S.
      hidden_size: hidden dimension H.
      token_iter:  iterator yielding one ROW per call — each row a
                   sequence (list/tuple/1-D array) of exactly `seq_length`
                   token ids; they are stored bit-exactly (int32) and fed
                   to the model unchanged.
      batch_size:  rows per forward pass (the last batch may be smaller).
      log_every:   batch-progress print period.
      source_desc / model_desc / seed: manifest provenance fields.
      device:      where the forward runs (None -> the model's own device).

    Writes <out_dir>/h_{L}.npy (fp16, (R, S, H)) for L in 0..num_layers-1,
    final_hidden.npy (post-norm, ALWAYS), input_ids.npy (int32, ALWAYS)
    and manifest.json (format 3, kind "boundary", written last). Per-batch
    memmap slice writes with one flush at the end — the batched streaming
    layout of the old v2 writer, unchanged. Returns a summary dict.
    """
    if out_dir is None or rows_total is None or seq_length is None or \
            hidden_size is None or token_iter is None or batch_size is None:
        raise TypeError(
            "run_boundary_capture requires out_dir, rows_total, "
            "seq_length, hidden_size, token_iter and batch_size")
    if rows_total <= 0 or seq_length <= 0 or hidden_size <= 0 or \
            batch_size <= 0:
        raise ValueError(
            f"rows_total/seq_length/hidden_size/batch_size must be "
            f"positive (got {rows_total}/{seq_length}/{hidden_size}/"
            f"{batch_size})")

    if layers is None:
        layers = pmod.get_layers(model)
    else:
        if list(layers) != list(pmod.get_layers(model)):
            raise ValueError(
                "`layers` must be the FULL decoder stack "
                "(pmod.get_layers(model)) — the boundary store records "
                "every layer's input hidden state; a partial list would "
                "mislabel the h_L files")
    num_layers = len(layers)
    if num_layers == 0:
        raise ValueError("the model has no decoder layers (pmod.get_layers)")
    norm = _final_norm_module(model)
    if device is None:
        device = next(model.parameters()).device
    else:
        model = model.to(device)

    os.makedirs(out_dir, exist_ok=True)
    # overwrite guard: refuse if ANY file this capture would write already
    # exists (manifest.json and h_0.npy are the canonical sentinels;
    # checking them all also refuses half-written dirs from a crashed run)
    would_write = ["manifest.json", "input_ids.npy", "final_hidden.npy"] + \
        [f"h_{L}.npy" for L in range(num_layers)]
    existing = [f for f in would_write
                if os.path.exists(os.path.join(out_dir, f))]
    if existing:
        first = os.path.join(out_dir, existing[0])
        raise SystemExit(f"{first} already exists — refusing to overwrite; "
                         f"remove the directory or pass a fresh --out-dir")

    h_mms = {L: _memmap_rw(os.path.join(out_dir, f"h_{L}.npy"),
                           (rows_total, seq_length, hidden_size))
             for L in range(num_layers)}
    final_mm = _memmap_rw(os.path.join(out_dir, "final_hidden.npy"),
                          (rows_total, seq_length, hidden_size))
    ids_mm = _memmap_rw(os.path.join(out_dir, "input_ids.npy"),
                        (rows_total, seq_length), dtype=np.int32)

    # hooks: every layer's INPUT hidden state (pre-hook on args[0] — the
    # old x1 pre-hook generalized from layers[0] to ALL layers) + the
    # post-norm final hidden. The old down_proj target taps are gone:
    # Stage 1 computes targets on the fly from the resident teacher layer
    # (W4) and the LUT escalation taps down_proj from that same layer.
    sink = {}

    def make_pre_hook(L):
        def pre_hook(module, args_):
            x = args_[0]
            sink[("h", L)] = x.detach().to(torch.float16).cpu().numpy()
        return pre_hook

    def final_hook(mod, inp, out):
        o = out[0] if isinstance(out, tuple) else out
        sink[("final",)] = o.detach().to(torch.float16).cpu().numpy()

    handles = [layer.register_forward_pre_hook(make_pre_hook(L))
               for L, layer in enumerate(layers)]
    handles.append(norm.register_forward_hook(final_hook))

    num_batches = (rows_total + batch_size - 1) // batch_size
    print(f"[capture] boundary store: h_0..h_{num_layers - 1} + "
          f"final_hidden + input_ids — {rows_total} rows x {seq_length} "
          f"tokens, hidden {hidden_size}, {num_batches} batch(es) of "
          f"{batch_size}", flush=True)
    t0 = time.time()
    row = 0
    try:
        for b in range(num_batches):
            n = min(batch_size, rows_total - row)
            try:
                rows = [next(token_iter) for _ in range(n)]
            except StopIteration as e:
                raise RuntimeError(
                    f"token iterator exhausted after {row} of {rows_total} "
                    f"rows — the capture is incomplete") from e
            arr = np.asarray(rows)
            if arr.dtype.kind not in "iu" or arr.shape != (n, seq_length):
                raise ValueError(
                    f"token iterator yielded a wrong-shaped batch "
                    f"{arr.shape}/{arr.dtype} (expected {(n, seq_length)} "
                    f"integer ids) — every row must carry exactly "
                    f"seq_length ids")
            ids = arr.astype(np.int64, copy=False)
            if int(ids.min()) < 0 or int(ids.max()) >= 2 ** 31:
                raise ValueError(
                    "token ids outside the storable int32 range — "
                    "input_ids.npy cannot represent them")
            input_ids = torch.from_numpy(ids).to(device)
            sink.clear()
            with torch.no_grad():
                model(input_ids=input_ids, attention_mask=None,
                      use_cache=False)
            expected = (n, seq_length, hidden_size)
            for L in range(num_layers):
                if ("h", L) not in sink:
                    raise RuntimeError(
                        f"layer {L}'s pre-hook never fired — the model did "
                        f"not call every decoder layer in this forward; "
                        f"refusing to write unverified rows")
                if sink[("h", L)].shape != expected:
                    raise ValueError(
                        f"h_{L}: captured shape {sink[('h', L)].shape} != "
                        f"{expected} (seq_length/hidden_size disagree with "
                        f"the model)")
            if ("final",) not in sink:
                raise RuntimeError(
                    "the final-norm hook never fired — no final_hidden to "
                    "store")
            if sink[("final",)].shape != expected:
                raise ValueError(
                    f"final_hidden: captured shape "
                    f"{sink[('final',)].shape} != {expected}")
            sl = slice(row, row + n)
            for L in range(num_layers):
                h_mms[L][sl] = sink[("h", L)]
            final_mm[sl] = sink[("final",)]
            ids_mm[sl] = ids.astype(np.int32)
            row += n
            if (b + 1) % log_every == 0 or b == num_batches - 1:
                el = time.time() - t0
                print(f"  batch {b + 1}/{num_batches} "
                      f"({row} rows, {el:.0f}s, {el / (b + 1):.2f}s/batch)",
                      flush=True)
    finally:
        for h in handles:
            h.remove()
        sink.clear()
        del sink
        gc.collect()

    for mm in [ids_mm, final_mm, *h_mms.values()]:
        mm.flush()

    manifest = {
        "format": 3,
        "kind": "boundary",
        "model": model_desc,
        "source": source_desc,
        "rows": int(rows_total),
        "seq_length": int(seq_length),
        "hidden_size": int(hidden_size),
        "num_layers": int(num_layers),
        "batch_size": int(batch_size),
        "num_batches": int(num_batches),
        "dtype": "float16",
        "packed": True,
        "padding": "none",
        "teacher_pristine": True,
        "norm_edits_applied_to_teacher": False,
        "seed": seed,
        "final_hidden": True,
        "input_ids": True,
        "boundary": True,
    }
    _atomic_json_dump(manifest, os.path.join(out_dir, "manifest.json"))
    files = ["input_ids.npy", "final_hidden.npy", "manifest.json"] + \
        [f"h_{L}.npy" for L in range(num_layers)]
    return {
        "format": 3,
        "kind": "boundary",
        "out_dir": out_dir,
        "rows": int(rows_total),
        "seq_length": int(seq_length),
        "hidden_size": int(hidden_size),
        "num_layers": int(num_layers),
        "num_batches": int(num_batches),
        "batch_size": int(batch_size),
        "files": files,
        "bytes": sum(os.path.getsize(os.path.join(out_dir, f))
                     for f in files),
        "wall_s": time.time() - t0,
        "manifest": manifest,
    }


def cmd_capture(args):
    if args.source not in ("fineweb", "wikitext2", "synthetic") and \
            not args.source.startswith("file:"):
        raise SystemExit(f"unknown --source {args.source!r}")
    if not args.capture_final_hidden:
        raise SystemExit(
            "--no-final-hidden is no longer supported: the boundary-state "
            "capture ALWAYS writes final_hidden.npy (Stage 2's target). "
            "Re-run without the flag")
    device = args.device
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    need_tok = args.source != "synthetic"
    tokenizer = None
    if need_tok:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            args.model, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

    print("=" * 78)
    print("Teacher boundary-state capture (pristine dense model, packed")
    print("rows, streaming writes: every layer's input hidden state)")
    print("=" * 78)
    model = _load_model_for_causal_lm(args.model, device=device).eval()
    layers = pmod.get_layers(model)
    num_layers = len(layers)
    hidden = _text_config(model).hidden_size
    rows_total = args.num_batches * args.batch_size

    stream = _token_stream(args, tokenizer, rng)

    def token_rows():
        while True:
            yield [next(stream) for _ in range(args.seq_length)]

    summary = run_boundary_capture(
        model, layers, out_dir=args.out_dir, rows_total=rows_total,
        seq_length=args.seq_length, hidden_size=hidden,
        token_iter=token_rows(), batch_size=args.batch_size,
        log_every=args.log_every, source_desc=args.source,
        model_desc=args.model, seed=args.seed, device=device)

    print(f"\nDONE: {summary['rows']} rows "
          f"({summary['rows'] * args.seq_length:,} tokens) -> "
          f"{args.out_dir} ({summary['bytes'] / 1e9:.2f} GB, "
          f"{num_layers + 1} hidden-state files + input_ids)")
    
    # Explicit cleanup to avoid PyGILState error on exit
    del model
    del layers
    gc.collect()
    torch.cuda.empty_cache()


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Boundary-state teacher capture (format 3) for the "
                    "FLUTE-palettized Qwen3.5-9B layerwise pipeline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp):
        sp.add_argument("--model", default="Qwen/Qwen3.5-9B",
                        help="HF id or LOCAL checkpoint dir (required for "
                             "--target-mode teacher-on-student)")
        sp.add_argument("--device", default=(
            "cuda:0" if torch.cuda.is_available() else "cpu"))

    # ---- capture ----------------------------------------------------------
    c = sub.add_parser(
        "capture",
        help="boundary-state teacher capture (format 3): h_0..h_31 = every "
             "layer's input hidden state + post-norm final_hidden + "
             "input_ids (int32, always written)",
        description=(
            "Boundary-state capture (Stage 0, format 3): one batched "
            "forward of the pristine dense teacher persists, for every "
            "captured token, h_0..h_31 — each layer's INPUT hidden state "
            "(h_0 = the embedding output) — plus the post-norm "
            "final_hidden.npy and input_ids.npy (int32, ALWAYS written: "
            "Stage 2 must feed the model the exact tokens that produced "
            "the captured hiddens). Packed rows (documents EOS-concatenated "
            "and cut into exact seq_length blocks — no padding anywhere); "
            "batched streaming memmap writes; targets are NOT stored "
            "(Stage 1 computes them on the fly from the resident teacher "
            "layer). The old v2 layout (x1 + down_proj targets) is no "
            "longer written but stays readable."))
    common(c)
    c.add_argument("--out-dir", required=True)
    c.add_argument("--source", default="fineweb",
                   help="'fineweb' | 'wikitext2' | 'file:<path>' (one doc "
                        "per line) | 'synthetic' (random tokens, testing)")
    c.add_argument("--num-batches", type=int, default=64)
    c.add_argument("--batch-size", type=int, default=4)
    c.add_argument("--seq-length", type=int, default=512)
    c.add_argument("--capture-final-hidden", action="store_true",
                   default=True,
                   help="inert (kept for compatibility): the boundary "
                        "capture ALWAYS writes final_hidden.npy — Stage "
                        "2's target")
    c.add_argument("--no-final-hidden", dest="capture_final_hidden",
                   action="store_false",
                   help="REFUSED: final_hidden.npy is always written by "
                        "the boundary capture (Stage 2's target); passing "
                        "this flag exits with an explanation")
    c.add_argument("--seed", type=int, default=42)
    c.add_argument("--log-every", type=int, default=8)
    c.add_argument("--synthetic-vocab", type=int, default=32000,
                   help="vocab size for --source synthetic (testing)")


    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.cmd == "capture":
        cmd_capture(args)
    else:  # pragma: no cover
        raise SystemExit(f"unknown subcommand {args.cmd}")


if __name__ == "__main__":
    main()
