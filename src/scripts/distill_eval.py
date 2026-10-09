#!/usr/bin/env python3
"""scripts/distill_eval.py — the verification orchestrator (the
evaluators' acceptance harness; see PROPOSAL.md §2.9 for the program
gates this table feeds).

One ``verify`` subcommand: runs the O-1 probe and the greedy matcher
STRICTLY SEQUENTIALLY — one model on the GPU at a time — reads the
trainer's run report from disk, then assembles the success-criteria
table with measured values, verdicts and provenance into
``reports/distill_eval_<tag>.json``. This file orchestrates only: it
imports NO torch and does NO tensor math (the invariant is structural);
every number in the report comes from a parsed producer JSON, and the
only arithmetic here is the paired per-doc mean that mirrors
``o1_baseline_check.paired_diff``'s ``mean_delta`` semantics.

Producers and the invocation seam:

  1. O-1 paired same-doc probe — ``scripts/o1_baseline_check.py``.
     The probe's own CLI pairs whichever optional stage is requested
     with the always-on base stage, so the orchestrator issues TWO probe
     runs (each its own process; process exit frees the GPU before the
     next run starts; inside each run the probe itself frees the dense
     model with del + gc + empty_cache before loading the palettized
     student):
       run 1 "dense+base":   ``--dense``            -> dense + base+0
                             + deltas.quantization_gap
       run 2 "base+adapters": ``--adapters-dir A``  -> base+0 + base+N
                             + deltas.training_movement
     Parsed fields: ``results.{dense_fp16, base_adapter0,
     base_adapterN}.per_doc``, ``deltas.quantization_gap.mean_delta``,
     ``deltas.training_movement``, ``verdicts``, ``args``.
     ``--dense-report <path>`` reuses an existing probe report's dense
     stage (its ``results.dense_fp16.per_doc``) so the ~18 GiB dense
     reference is loaded once, ever; pairing is refused loudly unless
     the doc-replication args (dataset / max-samples / seq-len /
     eval-holdout / seed — the probe's document list is a pure function
     of those) and n_docs match the fresh run.
     Why subprocess and not ``import o1_baseline_check.main``: that main
     takes no argv (a sys.argv-only CLI), and a process boundary is the
     only residency guarantee this orchestrator can make on its own.
  2. The trainer's run report — read DIRECTLY from the train output dir
     (``--run``): ``finetune_provenance.json`` (the run identity + the
     per-layer file map), each done layer's
     ``qlora_layers/layer_<L>/metrics.json`` (the per-layer before /
     banked (``best_step``) / after lines — ``after`` is the eval of the
     restored banked best), and, when the export ran,
     ``export_report.json`` (the G-J3 roundtrip lines: rel_mse(train)
     vs rel_mse(snap) per exported layer). No subprocess: the run dir is
     a directory of JSON files, not a GPU job (PROPOSAL §5.2: the
     per-layer producer re-pointed from the engine's report to the
     trainer's run report). Loud on anything malformed.
  3. Greedy-decode exact match — ``scripts/eval_greedy_match.py
     --qlora-adapters <dir>`` (also sys.argv-only; subprocess again) with
     ``--n-prompts``/``--max-new-tokens`` mirroring the matcher's own
     32 x 96 defaults. Parsed fields: its JSON report's
     ``aggregate.exact_match_fraction`` (cross-checked against
     ``per_prompt[].exact_match``), ``n_prompts``,
     ``max_new_tokens``, the first-divergence stats.

Success-criteria semantics (this module's table is the contract; boundary
behavior matters): cosine > 0.999 (strict) — measured as the mean
per-layer tok_cos (after, the banked best) over the run's done layers
(the per-layer down_proj-tap cosine of hidden states; the end-to-end
final-hidden cosine has no producer in this chain), gap < 0.02
nats/doc (strict — a gap of exactly 0.02 FAILS),
greedy >= 0.90 target with the 100% stretch noted. Rows whose value no
producer supplies get verdict "unknown" with the reason recorded — a
measured value is NEVER fabricated. The nMSE and KL rows of the
original table are DROPPED (PROPOSAL §5.2: drop the two
permanently-"unknown" criteria rows or wire a producer): no producer
computes the end-to-end hidden-state nMSE or the logits KL — the
paired O-1 gap and the greedy match are their end-to-end proxies
(PROPOSAL §2.9).

Base-only mode: ``--adapters`` may be omitted for a base-only
verification. The probe still measures the adapter-free quantization gap
(recorded in ``o1_probe.gap_base`` and in the gap-baseline line), but the
adapter-dependent criteria rows — gap vs the < 0.02 target (G2), greedy
match (G3) — are marked unknown: those targets apply to the distilled
student. The trainer's run report always scores the student defined by
the run's ``--artifacts`` (the layer metrics' before/after lines), so
the cosine row measures that student.
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from datetime import datetime, timezone

from eval_common import atomic_json_dump as _atomic_json_dump
from eval_common import paired_mean

__all__ = ["main", "parse_args", "cmd_verify", "DistillEvalError"]

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROBE = os.path.join(_HERE, "o1_baseline_check.py")
_GREEDY = os.path.join(_HERE, "eval_greedy_match.py")
_PROBE_NAME = os.path.basename(_PROBE)
_GREEDY_NAME = os.path.basename(_GREEDY)
_TRAINER_NAME = "trainer train run dir"

# The success-criteria table (metric / target / current-baseline
# strings), plus the machine-readable comparison.
_SPEC_GAP_BASELINE = 0.0793  # nats/doc, reports/o1_baseline_fineweb.json
_SPEC_ROWS = [
    {"metric": "Cosine similarity (hidden states)", "target": "> 0.999",
     "op": ">", "threshold": 0.999,
     "spec_baseline": "unknown (measure in Stage 3)"},
    {"metric": "Quantization gap (paired O-1 probe)",
     "target": "< 0.02 nats/doc", "op": "<", "threshold": 0.02,
     "spec_baseline": "0.0793 (reports/o1_baseline_fineweb.json)"},
    {"metric": "Greedy decode match (32×96)",
     "target": "100% exact (≥ 90% target)", "op": ">=",
     "threshold": 0.90, "stretch": 1.0,
     "spec_baseline": "Diverges"},
]

# the probe's document-replication identity (o1_baseline_check.py: the
# doc list is a pure function of these five args — the pairing contract)
_DOC_FIELDS = ("dataset", "max_samples", "seq_len", "eval_holdout", "seed")
# the probe report's "args" fields this orchestrator itself issues
_PROBE_ARG_FIELDS = ("artifacts_dir", "model", "device", "adapters_dir",
                     "dense", "output")

# the probe's doc-replication flags this orchestrator passes through (the
# defaults mirror o1_baseline_check.py's own defaults, so a verification
# against the recorded dense report pairs unchanged; non-default doc
# identities — a toy rehearsal, a different corpus — stay pairable)
def _probe_doc_flags(args) -> list:
    return ["--dataset", args.probe_dataset,
            "--max-samples", str(args.probe_max_samples),
            "--seq-len", str(args.probe_seq_len),
            "--eval-holdout", str(args.probe_eval_holdout),
            "--seed", str(args.probe_seed)]

_LAYER_MEAN_TOL = 1e-6     # the mean is recomputed from the layer lines
_GREEDY_FRAC_TOL = 1e-9    # fraction must equal n_exact / n_prompts


class DistillEvalError(RuntimeError):
    """Loud orchestrator failure: the message names the producer and the
    expected artifact — never a silently empty table."""


# ---------------------------------------------------------------------------
# Small utilities (repo style)
# ---------------------------------------------------------------------------

def _require(cond, msg: str) -> None:
    if not cond:
        raise DistillEvalError(msg)


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def _is_num(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _fmt(v) -> str:
    return "—" if v is None else f"{v:.6g}"


def _verdict(measured, op: str, threshold: float) -> str:
    """Boundary-exact verdict: '>'/'<' are STRICT (a value exactly at the
    target FAILS), '>=' passes at the target. None -> unknown."""
    if measured is None:
        return "unknown"
    if op == ">":
        ok = measured > threshold
    elif op == "<":
        ok = measured < threshold
    elif op == ">=":
        ok = measured >= threshold
    else:  # pragma: no cover - guarded by the spec table itself
        raise DistillEvalError(f"unknown comparison op {op!r}")
    return "pass" if ok else "fail"


def _paired_mean(a, b, what: str):
    """The paired per-doc mean, projected from eval_common.paired_mean
    (the shared mean_delta semantics; negative = b better) with this
    orchestrator's loud doc-for-doc validation."""
    _require(isinstance(a, list) and isinstance(b, list) and a and b
             and len(a) == len(b),
             f"refusing to pair {what}: per-doc vector length mismatch "
             f"({len(a) if isinstance(a, list) else '?'} vs "
             f"{len(b) if isinstance(b, list) else '?'}) — the paired-delta "
             f"noise floor only holds doc-for-doc")
    return {"n": len(a), "mean_delta": paired_mean(a, b)}


def _load_json(path: str, producer: str):
    try:
        with open(path) as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        raise DistillEvalError(
            f"producer {producer} wrote malformed JSON to {path}: {e}"
        ) from e


def _run_producer(commands, cmd, producer: str, stage: str,
                  report_path: str) -> dict:
    """Run one producer as its own subprocess (the invocation seam:
    subprocess.run — patched in tests). Sequential by construction: this
    call blocks until the producer exits, so two producers can never be
    resident at once. Nonzero exit or a missing report artifact raises
    DistillEvalError naming both."""
    cmd = [str(c) for c in cmd]
    cmd_str = " ".join(shlex.quote(c) for c in cmd)
    print(f"[{producer} / {stage}] {cmd_str}", flush=True)
    t0 = time.time()
    try:
        proc = subprocess.run(cmd, stderr=subprocess.PIPE, text=True)
    except OSError as e:
        raise DistillEvalError(
            f"producer {producer} ({stage}) could not be launched "
            f"({e}); expected report artifact: {report_path}") from e
    wall = round(time.time() - t0, 1)
    rec = {"producer": producer, "stage": stage, "command": cmd,
           "command_str": cmd_str, "returncode": proc.returncode,
           "report": report_path, "wall_s": wall}
    commands.append(rec)
    if proc.returncode != 0:
        tail = (proc.stderr or "").strip()[-1500:]
        raise DistillEvalError(
            f"producer {producer} ({stage}) EXITED {proc.returncode} — "
            f"expected report artifact: {report_path} (not usable)"
            + (f"; stderr tail: {tail}" if tail else ""))
    _require(os.path.isfile(report_path),
             f"producer {producer} ({stage}) exited 0 but its expected "
             f"report artifact {report_path} is MISSING")
    return rec


# ---------------------------------------------------------------------------
# Producer report parsing + validation (loud on anything malformed)
# ---------------------------------------------------------------------------

def _parse_probe_report(path: str, rep, *, need_dense: bool,
                        need_adapterN: bool, need_base0: bool = True,
                        producer: str = _PROBE_NAME) -> dict:
    """Validate + project an o1_baseline_check.py JSON report. Fields
    read: results.{dense_fp16, base_adapter0, base_adapterN}.per_doc,
    n_docs, args, deltas.{quantization_gap, training_movement},
    verdicts. base_adapter0 is required for a report from a run this
    orchestrator issued (the probe always scores the base stage); a
    reused --dense-report file only needs its dense stage. The per_doc
    vectors are the ONLY probe numbers used for pairing; deltas are
    taken verbatim when present."""
    _require(isinstance(rep, dict),
             f"{producer} report {path}: top level is not a JSON object")
    _require(isinstance(rep.get("results"), dict),
             f"{producer} report {path}: 'results' missing or not an object")
    n_docs = rep.get("n_docs")
    _require(_is_num(n_docs) and n_docs > 0,
             f"{producer} report {path}: n_docs missing or non-positive")
    n = int(n_docs)
    need = {"base_adapter0": need_base0, "dense_fp16": need_dense,
            "base_adapterN": need_adapterN}
    per_docs = {}
    for key, wanted in need.items():
        if not wanted:
            continue
        st = rep["results"].get(key)
        _require(isinstance(st, dict),
                 f"{producer} report {path}: results.{key} missing — the "
                 f"stage was requested but the report does not carry it "
                 f"(expected an object with per_doc)")
        per = st.get("per_doc")
        _require(isinstance(per, list) and len(per) > 0
                 and all(_is_num(x) for x in per),
                 f"{producer} report {path}: results.{key}.per_doc must be "
                 f"a nonempty list of numbers")
        _require(len(per) == n,
                 f"{producer} report {path}: n_docs={n} but "
                 f"results.{key}.per_doc has {len(per)} entries — "
                 f"inconsistent report")
        per_docs[key] = [float(x) for x in per]
    _require(isinstance(rep.get("args"), dict),
             f"{producer} report {path}: 'args' missing or not an object")
    deltas = rep.get("deltas")
    _require(isinstance(deltas, dict),
             f"{producer} report {path}: 'deltas' missing or not an object")
    verdicts = rep.get("verdicts")
    return {"n_docs": n,
            "dense_per_doc": per_docs.get("dense_fp16"),
            "base0_per_doc": per_docs.get("base_adapter0"),
            "baseN_per_doc": per_docs.get("base_adapterN"),
            "args": rep["args"], "deltas": deltas,
            "verdicts": list(verdicts) if isinstance(verdicts, list) else []}


def _probe_delta(deltas: dict, path: str, key: str) -> dict:
    """One of the probe's own paired stats, verbatim (mean_delta
    validated numeric)."""
    d = deltas.get(key)
    _require(isinstance(d, dict) and _is_num(d.get("mean_delta")),
             f"{_PROBE_NAME} report {path}: deltas.{key}.mean_delta missing "
             f"or non-numeric — the stage ran, its paired stat must be "
             f"there")
    return d


def _validate_probe_args(args_rec: dict, path: str, expected: dict) -> None:
    """Loud guard against a stale/wrong probe report: the 'args' it
    records must match the invocation this orchestrator issued."""
    for f in _PROBE_ARG_FIELDS:
        got, want = args_rec.get(f), expected[f]
        _require(got == want,
                 f"{_PROBE_NAME} report {path}: args.{f} records {got!r} "
                 f"but the invocation issued {want!r} — stale or wrong "
                 f"report")


def _check_doc_pairing(path_a: str, args_a: dict, n_a: int,
                       path_b: str, args_b: dict, n_b: int) -> None:
    """The probe's document-replication contract: paired per-doc deltas
    only hold doc-for-doc, so the two reports must agree on the doc list
    identity (n_docs + the five replication args)."""
    _require(n_a == n_b,
             f"refusing to pair the O-1 probe reports {path_a} "
             f"(n_docs={n_a}) and {path_b} (n_docs={n_b}): paired deltas "
             f"only hold doc-for-doc")
    for f in _DOC_FIELDS:
        va, vb = args_a.get(f), args_b.get(f)
        _require(va == vb,
                 f"refusing to pair the O-1 probe reports {path_a} and "
                 f"{path_b}: doc-replication arg {f!r} differs "
                 f"({va!r} vs {vb!r}) — the document lists would not be "
                 f"identical")


def _require_metrics_block(run_dir: str, rel_path: str, met,
                           layer_key: str) -> dict:
    """Validate one layer's metrics.json block and project it to the
    per-layer line the report embeds: before / banked (best_step) /
    after (the restored banked best's eval) + the run bookkeeping."""
    where = f"{_TRAINER_NAME} {run_dir} ({rel_path})"
    _require(isinstance(met, dict),
             f"{where}: top level is not a JSON object")
    _require(_is_num(met.get("layer")) and int(met["layer"]) == int(layer_key),
             f"{where}: 'layer' missing or does not match the provenance "
             f"key {layer_key!r}")
    block_type = met.get("block_type")
    _require(isinstance(block_type, str) and block_type,
             f"{where}: 'block_type' missing or not a string")
    for field in ("before", "after"):
        blk = met.get(field)
        _require(isinstance(blk, dict),
                 f"{where}: {field!r} missing or not an object — expected "
                 f"{{tok_cos, flat_cos, rel_mse}}")
        for k in ("tok_cos", "flat_cos", "rel_mse"):
            _require(_is_num(blk.get(k)),
                     f"{where}: {field}.{k} missing or non-numeric")
    best = met.get("best_step")
    _require(best is None or _is_num(best),
             f"{where}: 'best_step' present but non-numeric")
    steps = met.get("steps")
    _require(_is_num(steps),
             f"{where}: 'steps' missing or non-numeric")
    return {
        "layer": int(met["layer"]), "type": block_type,
        "tok_cos": float(met["after"]["tok_cos"]),
        "flat_cos": float(met["after"]["flat_cos"]),
        "rel_mse": float(met["after"]["rel_mse"]),
        "before": {"tok_cos": float(met["before"]["tok_cos"]),
                   "flat_cos": float(met["before"]["flat_cos"]),
                   "rel_mse": float(met["before"]["rel_mse"])},
        "after": {"tok_cos": float(met["after"]["tok_cos"]),
                  "flat_cos": float(met["after"]["flat_cos"]),
                  "rel_mse": float(met["after"]["rel_mse"])},
        "banked_step": (int(best) if _is_num(best) else None),
        "steps": int(steps),
        "stop_reason": met.get("stop_reason"),
    }


def _read_export_report(run_dir: str) -> dict:
    """Read + validate <run>/export_report.json when the export ran:
    per-layer G-J3 roundtrip lines (rel_mse(train) vs rel_mse(snap))
    plus the export's own all-pass flag. A written report whose
    gj3_all_pass is not True is refused loudly (the export path only
    writes after every layer passed its bound; anything else was
    patched or truncated)."""
    path = os.path.join(run_dir, "export_report.json")
    if not os.path.isfile(path):
        return {"export_report": None, "per_layer": {},
                "note": "no export_report.json in the run dir — the "
                        "export step has not run (the G-J3 roundtrip "
                        "lines are absent, not failed)"}
    rep = _load_json(path, _TRAINER_NAME)
    _require(isinstance(rep, dict),
             f"{_TRAINER_NAME} {run_dir}: export_report.json top level "
             f"is not a JSON object")
    layers = rep.get("layers")
    _require(isinstance(layers, dict),
             f"{_TRAINER_NAME} {run_dir}: export_report.json 'layers' "
             f"missing or not an object")
    per_layer = {}
    for key, entry in layers.items():
        _require(isinstance(entry, dict),
                 f"{_TRAINER_NAME} {run_dir}: export_report.json "
                 f"layers[{key!r}] is not an object")
        for f in ("rel_mse_train", "rel_mse_snap"):
            _require(_is_num(entry.get(f)),
                     f"{_TRAINER_NAME} {run_dir}: export_report.json "
                     f"layers[{key!r}].{f} missing or non-numeric")
        rt = entry.get("roundtrip_modules")
        _require(rt is None or _is_num(rt),
                 f"{_TRAINER_NAME} {run_dir}: export_report.json "
                 f"layers[{key!r}].roundtrip_modules non-numeric")
        train = float(entry["rel_mse_train"])
        snap = float(entry["rel_mse_snap"])
        ratio = (snap / train) if train > 0 else None
        per_layer[key] = {
            "rel_mse_train": train, "rel_mse_snap": snap,
            "ratio": ratio,
            "roundtrip_modules": (int(rt) if _is_num(rt) else None),
            "pass": bool(snap <= 1.05 * train),
        }
    all_pass = rep.get("gj3_all_pass")
    _require(all_pass is True,
             f"{_TRAINER_NAME} {run_dir}: export_report.json carries "
             f"gj3_all_pass={all_pass!r} — the export did not pass its "
             f"own G-J3 bound (PROPOSAL 2.9); this run is not verified")
    n_ok = sum(1 for v in per_layer.values() if v["pass"])
    return {"export_report": path, "per_layer": per_layer,
            "n_roundtrip": len(per_layer),
            "n_pass": n_ok, "all_pass": True,
            "note": "G-J3 per-layer roundtrip lines from "
                    "export_report.json (rel_mse(snap) <= 1.05 x "
                    "rel_mse(train) per layer)"}


def _read_trainer_run(run_dir: str) -> dict:
    """Read + validate the trainer train run's report (the re-pointed
    layer producer): finetune_provenance.json (the run identity + the
    per-layer file map), each done layer's metrics.json (the
    before / banked / after lines), and export_report.json when the
    export ran (the G-J3 roundtrip lines). Loud on anything malformed
    — a missing or inconsistent run report never yields an empty
    table."""
    _require(isinstance(run_dir, str) and os.path.isdir(run_dir),
             f"--run {run_dir!r} is not an existing directory — expected "
             f"a trainer train output dir (finetune_provenance.json + "
             f"qlora_layers/)")
    prov_path = os.path.join(run_dir, "finetune_provenance.json")
    _require(os.path.isfile(prov_path),
             f"{_TRAINER_NAME} {run_dir}: finetune_provenance.json is "
             f"MISSING — not a trainer train output dir")
    prov = _load_json(prov_path, _TRAINER_NAME)
    _require(isinstance(prov, dict),
             f"{_TRAINER_NAME} {run_dir}: finetune_provenance.json top "
             f"level is not a JSON object")
    prov_layers = prov.get("layers")
    _require(isinstance(prov_layers, dict) and len(prov_layers) > 0,
             f"{_TRAINER_NAME} {run_dir}: provenance 'layers' missing or "
             f"empty — the run trained no layers")
    lines = []
    for key in sorted(prov_layers, key=lambda k: int(k) if str(k).lstrip
                      ("-").isdigit() else 1 << 30):
        rec = prov_layers[key]
        _require(isinstance(rec, dict),
                 f"{_TRAINER_NAME} {run_dir}: provenance layers[{key!r}] "
                 f"is not an object")
        if not rec.get("done"):
            continue
        rel_metrics = rec.get("metrics")
        _require(isinstance(rel_metrics, str) and rel_metrics,
                 f"{_TRAINER_NAME} {run_dir}: provenance layers[{key!r}] "
                 f"is done but carries no 'metrics' path")
        met_path = os.path.join(run_dir, rel_metrics)
        _require(os.path.isfile(met_path),
                 f"{_TRAINER_NAME} {run_dir}: done layer {key!r} points "
                 f"at {rel_metrics!r} which does not exist — the run dir "
                 f"is incomplete")
        met = _load_json(met_path, _TRAINER_NAME)
        line = _require_metrics_block(run_dir, rel_metrics, met, key)
        line["gj3"] = None
        lines.append(line)
    _require(lines,
             f"{_TRAINER_NAME} {run_dir}: provenance lists layers but "
             f"none is done")
    export = _read_export_report(run_dir)
    for line in lines:
        gj3 = export["per_layer"].get(str(line["layer"]))
        if gj3 is not None:
            line["gj3"] = gj3
    mean = sum(l["tok_cos"] for l in lines) / len(lines)
    worst = min(lines, key=lambda l: l["tok_cos"])
    return {
        "source_run": os.path.abspath(run_dir),
        "layers": lines,
        "mean_tok_cos": mean,
        "worst_layer": worst["layer"],
        "worst_tok_cos": worst["tok_cos"],
        "gj3": export,
        "note": "per-layer before/banked/after lines from the trainer "
                "train run's metrics.json (after = the restored banked "
                "best's holdout eval); gj3 per line when the export ran",
    }


def _parse_greedy_report(path: str, rep, *, model: str, artifacts: str,
                         qlora_adapters: str, n_prompts: int,
                         max_new_tokens: int) -> dict:
    """Validate + project an eval_greedy_match.py JSON report. Fields
    read: aggregate.exact_match_fraction (cross-checked against
    per_prompt[].exact_match), n_prompts, max_new_tokens, the
    first-divergence stats, and the identity fields (model /
    artifacts_dir / qlora_adapters) which must match the invocation."""
    _require(isinstance(rep, dict),
             f"{_GREEDY_NAME} report {path}: top level is not a JSON object")
    agg = rep.get("aggregate")
    _require(isinstance(agg, dict),
             f"{_GREEDY_NAME} report {path}: 'aggregate' missing or not an "
             f"object")
    frac = agg.get("exact_match_fraction")
    _require(_is_num(frac) and 0.0 <= float(frac) <= 1.0,
             f"{_GREEDY_NAME} report {path}: "
             f"aggregate.exact_match_fraction missing or outside [0, 1]")
    frac = float(frac)
    n_rep = rep.get("n_prompts")
    _require(_is_num(n_rep) and int(n_rep) > 0,
             f"{_GREEDY_NAME} report {path}: n_prompts missing or "
             f"non-positive")
    _require(int(n_rep) == n_prompts,
             f"{_GREEDY_NAME} report {path}: records n_prompts={int(n_rep)} "
             f"but the invocation requested {n_prompts} — stale report?")
    mnt = rep.get("max_new_tokens")
    _require(_is_num(mnt),
             f"{_GREEDY_NAME} report {path}: max_new_tokens missing or "
             f"non-numeric")
    _require(int(mnt) == max_new_tokens,
             f"{_GREEDY_NAME} report {path}: records max_new_tokens="
             f"{int(mnt)} but the invocation requested {max_new_tokens}")
    per = rep.get("per_prompt")
    _require(isinstance(per, list) and len(per) == n_prompts,
             f"{_GREEDY_NAME} report {path}: per_prompt must be a list of "
             f"{n_prompts} records")
    n_exact = 0
    for i, entry in enumerate(per):
        _require(isinstance(entry, dict) and isinstance(
            entry.get("exact_match"), bool),
            f"{_GREEDY_NAME} report {path}: per_prompt[{i}].exact_match "
            f"missing or not a boolean")
        n_exact += int(entry["exact_match"])
    _require(abs(frac - n_exact / n_prompts) <= _GREEDY_FRAC_TOL,
             f"{_GREEDY_NAME} report {path}: "
             f"aggregate.exact_match_fraction={frac!r} is inconsistent "
             f"with per_prompt ({n_exact}/{n_prompts} exact) — malformed "
             f"report")
    for field, want in (("model", model), ("artifacts_dir", artifacts),
                        ("qlora_adapters", qlora_adapters)):
        got = rep.get(field)
        _require(got == want,
                 f"{_GREEDY_NAME} report {path}: records {field}={got!r} "
                 f"but the invocation issued {want!r} — stale or wrong "
                 f"report")
    fd = {}
    for k in ("mean_first_divergence", "median_first_divergence",
              "min_first_divergence", "max_first_divergence"):
        v = agg.get(k)
        if _is_num(v):
            fd[k] = v
    return {"exact_match_fraction": frac, "n_exact": n_exact,
            "n_prompts": int(n_rep), "max_new_tokens": int(mnt),
            "first_divergence": fd or None}


# ---------------------------------------------------------------------------
# The verify subcommand
# ---------------------------------------------------------------------------

def cmd_verify(args) -> dict:
    wall_t0 = time.time()
    started_utc = _utcnow()
    tag = args.tag or time.strftime("run_%Y%m%d_%H%M%S")
    out_path = args.out or os.path.join("reports", f"distill_eval_{tag}.json")
    out_dir = os.path.dirname(out_path) or "."
    prov_dir = os.path.join(out_dir, f"distill_eval_{tag}_producers")
    probe_dense_path = os.path.join(prov_dir, "o1_probe_dense.json")
    probe_adapters_path = os.path.join(prov_dir, "o1_probe_adapters.json")
    probe_base_path = os.path.join(prov_dir, "o1_probe_base.json")
    greedy_path = os.path.join(prov_dir, "greedy.json")

    print("=" * 78, flush=True)
    print(f"distill_eval verify — Stage-3 verification (tag={tag})",
          flush=True)
    print(f"  artifacts={args.artifacts}", flush=True)
    print(f"  adapters={args.adapters or '(none — BASE-ONLY verification)'}",
          flush=True)
    print(f"  run={args.run or '(none — per-layer lines skipped)'}",
          flush=True)
    print(f"  dense reference="
          f"{'reused: ' + args.dense_report if args.dense_report else 'fresh probe run (--dense)'}",
          flush=True)
    print(f"  model={args.model}  device={args.device} "
          f"(producers only — this orchestrator does no tensor math)",
          flush=True)
    print(f"  out={out_path}", flush=True)
    print("=" * 78, flush=True)

    # ---- upfront validation (fail fast, before any GPU minute is spent) --
    if not 1 <= args.greedy_prompts <= 32:
        raise DistillEvalError(
            f"--greedy-prompts must be 1..32 (eval_greedy_match.py's "
            f"built-in pool has exactly 32 prompts); got "
            f"{args.greedy_prompts}")
    if args.greedy_tokens < 1:
        raise DistillEvalError(
            f"--greedy-tokens must be >= 1; got {args.greedy_tokens}")
    for flag, path in (("--artifacts", args.artifacts),
                       ("--adapters", args.adapters),
                       ("--run", args.run)):
        if path is not None and not os.path.isdir(path):
            raise DistillEvalError(
                f"{flag} {path!r} is not an existing directory")
    dense_parsed = None
    if args.dense_report:
        if not os.path.isfile(args.dense_report):
            raise DistillEvalError(
                f"--dense-report {args.dense_report!r} does not exist — "
                f"expected an o1_baseline_check.py JSON report from a run "
                f"with --dense")
        rep = _load_json(args.dense_report,
                         f"{_PROBE_NAME} (--dense-report)")
        dense_parsed = _parse_probe_report(
            args.dense_report, rep, need_dense=True, need_adapterN=False,
            need_base0=False,
            producer=f"{_PROBE_NAME} (--dense-report)")
        print(f"[dense] reusing {args.dense_report} "
              f"(n_docs={dense_parsed['n_docs']})", flush=True)

    os.makedirs(prov_dir, exist_ok=True)
    commands = []

    # ---- producer 1: the O-1 paired same-doc probe ------------------------
    # Sequential residency, in stage order: dense (+base) run first — its
    # process exits before the base+adapters run starts; inside each run
    # the probe frees the dense model before loading the palettized one.
    gap_base = None   # dense vs base+0
    gap_final = None  # dense vs base+adapters (the distilled student)
    movement = None   # base+0 vs base+N (the probe's own stat)
    probe_verdicts = []
    if args.dense_report:
        dense_per_doc = dense_parsed["dense_per_doc"]
        dense_path_used = args.dense_report
        dense_source = (f"reused --dense-report "
                        f"{args.dense_report}")
        dense_args, dense_n = dense_parsed["args"], dense_parsed["n_docs"]
    else:
        cmd = [sys.executable, _PROBE, "--artifacts-dir", args.artifacts,
               "--model", args.model, "--device", args.device,
               "--forward", args.forward, *_probe_doc_flags(args),
               "--dense", "--output", probe_dense_path]
        _run_producer(commands, cmd, _PROBE_NAME, "dense+base",
                      probe_dense_path)
        rep = _load_json(probe_dense_path, _PROBE_NAME)
        pr = _parse_probe_report(probe_dense_path, rep, need_dense=True,
                                 need_adapterN=False)
        _validate_probe_args(pr["args"], probe_dense_path, dict(
            artifacts_dir=args.artifacts, model=args.model,
            device=args.device, adapters_dir=None, dense=True,
            output=probe_dense_path))
        dense_per_doc = pr["dense_per_doc"]
        dense_path_used = probe_dense_path
        dense_source = "fresh probe run (--dense)"
        dense_args, dense_n = pr["args"], pr["n_docs"]
        qg = _probe_delta(pr["deltas"], probe_dense_path,
                          "quantization_gap")
        gap_base = {"n": int(qg.get("n", dense_n)),
                    "mean_delta": float(qg["mean_delta"]),
                    "origin": "probe deltas.quantization_gap (the "
                              "producer's own paired stat: dense vs "
                              "base+0)"}
        probe_verdicts.extend(pr["verdicts"])

    # student run: base+adapters when verifying a distilled student; a
    # bare base run when reusing a dense report in base-only mode; not
    # needed at all for a fresh-dense base-only verification (the dense
    # run already measured the base).
    if args.adapters:
        student_path = probe_adapters_path
        student_stage = "base+adapters"
        student_flags = ["--adapters-dir", args.adapters]
    elif args.dense_report:
        student_path = probe_base_path
        student_stage = "base"
        student_flags = []
    else:
        student_path = None
        student_stage = None
        student_flags = None

    if student_path is not None:
        cmd = [sys.executable, _PROBE, "--artifacts-dir", args.artifacts,
               "--model", args.model, "--device", args.device,
               "--forward", args.forward, *_probe_doc_flags(args),
               *student_flags, "--output", student_path]
        _run_producer(commands, cmd, _PROBE_NAME, student_stage,
                      student_path)
        rep = _load_json(student_path, _PROBE_NAME)
        spr = _parse_probe_report(student_path, rep, need_dense=False,
                                  need_adapterN=bool(args.adapters))
        _validate_probe_args(spr["args"], student_path, dict(
            artifacts_dir=args.artifacts, model=args.model,
            device=args.device, adapters_dir=args.adapters,
            dense=False, output=student_path))
        _check_doc_pairing(dense_path_used, dense_args, dense_n,
                           student_path, spr["args"], spr["n_docs"])
        if args.dense_report:
            gap_base = _paired_mean(
                dense_per_doc, spr["base0_per_doc"],
                f"{dense_path_used} dense vs {student_path} base+0")
            gap_base["origin"] = ("paired per-doc mean computed by "
                                  "distill_eval (paired_diff semantics) "
                                  "from the reused dense report vs the "
                                  "fresh base+0 run")
        if args.adapters:
            gap_final = _paired_mean(
                dense_per_doc, spr["baseN_per_doc"],
                f"{dense_path_used} dense vs {student_path} base+N")
            gap_final["origin"] = ("paired per-doc mean computed by "
                                   "distill_eval (paired_diff semantics: "
                                   "mean(base+N_i - dense_i)) from the "
                                   "probe reports' per_doc vectors")
            tm = spr["deltas"].get("training_movement")
            if isinstance(tm, dict) and _is_num(tm.get("mean_delta")):
                movement = tm
        probe_verdicts.extend(spr["verdicts"])

    # ---- producer 2: the trainer's run report (read from disk) -----------
    if args.run:
        layer = _read_trainer_run(args.run)
        layer_doc = {"source_run": layer["source_run"], **layer}
        gj3 = layer["gj3"]
        gj3_note = (f"{gj3['n_roundtrip']} roundtrip line(s), "
                    f"{gj3['n_pass']} pass"
                    if gj3["export_report"] else
                    "export not run (no roundtrip lines)")
        print(f"[layers] {len(layer['layers'])} done layer(s)  "
              f"mean tok_cos (after) {layer['mean_tok_cos']:.6f}  "
              f"worst L{layer['worst_layer']} "
              f"({layer['worst_tok_cos']:.6f})  G-J3: {gj3_note}",
              flush=True)
    else:
        layer_doc = {"source_run": None, "layers": None,
                     "mean_tok_cos": None,
                     "skipped_reason": (
                         "--run not given: the per-layer before/banked/"
                         "after lines and the export roundtrip lines come "
                         "from the trainer train output dir "
                         "(finetune_provenance.json + per-layer "
                         "metrics.json + export_report.json)")}

    # ---- producer 3: greedy-decode exact match -----------------------------
    if not args.adapters:
        greedy_doc = {"source_path": None, "exact_match_fraction": None,
                      "n_exact": None, "n_prompts": None,
                      "max_new_tokens": None, "first_divergence": None,
                      "skipped_reason": (
                          "base-only verification (--adapters absent): the "
                          "G3 criterion applies to the distilled student; "
                          "eval_greedy_match.py without --qlora-adapters "
                          "would measure the pre-distillation 'Diverges' "
                          "state, not the criterion — the greedy producer "
                          "is not invoked")}
    elif args.skip_greedy:
        greedy_doc = {"source_path": None, "exact_match_fraction": None,
                      "n_exact": None, "n_prompts": None,
                      "max_new_tokens": None, "first_divergence": None,
                      "skipped_reason": "--skip-greedy: the greedy-decode "
                                        "producer was not run"}
    else:
        cmd = [sys.executable, _GREEDY, "--model", args.model,
               "--artifacts-dir", args.artifacts,
               "--qlora-adapters", args.adapters,
               "--n-prompts", str(args.greedy_prompts),
               "--max-new-tokens", str(args.greedy_tokens),
               "--device", args.device, "--forward", args.forward,
               "--output", greedy_path]
        _run_producer(commands, cmd, _GREEDY_NAME, "greedy", greedy_path)
        rep = _load_json(greedy_path, _GREEDY_NAME)
        g = _parse_greedy_report(greedy_path, rep, model=args.model,
                                 artifacts=args.artifacts,
                                 qlora_adapters=args.adapters,
                                 n_prompts=args.greedy_prompts,
                                 max_new_tokens=args.greedy_tokens)
        greedy_doc = {"source_path": greedy_path, **g}
        print(f"[greedy] exact match {g['n_exact']}/{g['n_prompts']} "
              f"({g['exact_match_fraction']:.4f})", flush=True)

    # ---- the success-criteria table (the surviving criteria rows) ---------
    rows = []
    # 1. cosine — the mean per-layer tok_cos (after, banked best) from
    #    the trainer's run report (no producer computes the end-to-end
    #    final-hidden cosine)
    if layer_doc.get("mean_tok_cos") is not None:
        rows.append(_row(
            _SPEC_ROWS[0], layer_doc["mean_tok_cos"],
            source=f"trainer train run report (--run) → the mean of the "
                   f"done layers' metrics.json after.tok_cos",
            note="the per-layer down_proj-tap cosine of hidden states "
                 "after banking (the after line is the restored banked "
                 "best's holdout eval); the end-to-end final-hidden "
                 "cosine has no producer in this chain"))
    else:
        rows.append(_row(_SPEC_ROWS[0], None, reason=(
            "--run not given: the per-layer lines (and their mean "
            "tok_cos) come from the trainer train output dir")))
    # 2. quantization gap — the metric of record (G1/G2)
    if gap_final is not None:
        rows.append(_row(_SPEC_ROWS[1], gap_final["mean_delta"],
                         source=f"{_PROBE_NAME} — paired same-doc per-doc "
                                f"mean, dense vs base+adapters (per_doc "
                                f"vectors from the probe reports; the "
                                f"pairing computed by distill_eval with "
                                f"paired_diff mean semantics)"))
    else:
        rows.append(_row(_SPEC_ROWS[1], None, reason=(
            "base-only verification (--adapters absent): the < 0.02 "
            "nats/doc target (G2) applies to the distilled student "
            "(base+adapters); the adapter-free base gap IS measured — "
            "see o1_probe.gap_base and success_criteria.gap_baseline")))
    # 3. greedy decode match — G3
    if greedy_doc.get("exact_match_fraction") is not None:
        frac = greedy_doc["exact_match_fraction"]
        rows.append(_row(
            _SPEC_ROWS[2], frac,
            source=f"{_GREEDY_NAME} --qlora-adapters → "
                   f"aggregate.exact_match_fraction",
            stretch={"target": "100% exact", "threshold": 1.0,
                     "met": frac >= 1.0}))
    else:
        rows.append(_row(_SPEC_ROWS[2], None,
                         reason=greedy_doc.get("skipped_reason")))

    # the explicit gap-delta line vs the 0.0793 pre-distillation baseline
    measured_gap = None
    gap_kind = None
    if gap_final is not None:
        measured_gap, gap_kind = gap_final["mean_delta"], \
            "final (dense vs base+adapters)"
    elif gap_base is not None:
        measured_gap, gap_kind = gap_base["mean_delta"], \
            "base (dense vs base+0, adapter-free)"
    gap_baseline = {
        "baseline_nats_per_doc": _SPEC_GAP_BASELINE,
        "baseline_source": "reports/o1_baseline_fineweb.json (PROPOSAL §1.1, the metric of record)",
        "measured_nats_per_doc": measured_gap,
        "measured_gap_kind": gap_kind,
        "delta_vs_baseline": (measured_gap - _SPEC_GAP_BASELINE
                              if measured_gap is not None else None),
        "note": "delta = measured − baseline; negative = the gap closed "
                "vs the pre-distillation 0.0793 baseline",
    }

    summary = {"pass": sum(1 for r in rows if r["verdict"] == "pass"),
               "fail": sum(1 for r in rows if r["verdict"] == "fail"),
               "unknown": sum(1 for r in rows if r["verdict"] == "unknown")}

    probe_doc = {
        "dense_report_path": dense_path_used,
        "probe_report_path": student_path or dense_path_used,
        "dense_source": dense_source,
        "n_docs": dense_n,
        "gap_base": gap_base,
        "gap_final": gap_final,
        "training_movement": movement,
        "verdicts": probe_verdicts,
    }

    finished_utc = _utcnow()
    doc = {
        "schema": "distill_eval/1",
        "purpose": "verification (PROPOSAL §2.9 criteria, the §7.3 "
                   "chain) — orchestrates the O-1 probe and the greedy "
                   "matcher, reads the trainer's run report from --run; "
                   "this orchestrator performs no tensor math (no torch "
                   "import): every number comes from a parsed producer "
                   "JSON",
        "tag": tag,
        "created_utc": finished_utc,
        "started_utc": started_utc,
        "finished_utc": finished_utc,
        "wall_s": round(time.time() - wall_t0, 1),
        "success_criteria": {"rows": rows, "gap_baseline": gap_baseline,
                             "summary": summary},
        "o1_probe": probe_doc,
        "layer_alignment": layer_doc,
        "greedy": greedy_doc,
        "run": {
            "argv": list(sys.argv),
            "args": {k: v for k, v in vars(args).items() if k != "func"},
            "resolved": {"tag": tag, "out": out_path,
                         "producers_dir": prov_dir},
            "paths": {"artifacts": args.artifacts,
                      "adapters": args.adapters,
                      "run": args.run,
                      "dense_report": args.dense_report,
                      "out": out_path,
                      "producers_dir": prov_dir},
            "producer_seam": "the O-1 probe and the greedy matcher run "
                             "as subprocesses (sequential GPU residency: "
                             "process exit frees the device; their mains "
                             "are sys.argv-only CLIs); the trainer's run "
                             "report is read directly from --run's files "
                             "— a directory of JSONs, not a GPU job",
            "device_note": "--device only configures the producer "
                           "invocations (passed through); distill_eval "
                           "itself does no tensor math",
            "commands": commands,
        },
    }

    _atomic_json_dump(doc, out_path)
    _print_summary(doc, out_path)
    return doc


def _row(spec: dict, measured, source=None, reason=None, note=None,
         **extra) -> dict:
    row = {"metric": spec["metric"], "target": spec["target"],
           "op": spec["op"], "threshold": spec["threshold"],
           "measured": measured,
           "verdict": _verdict(measured, spec["op"], spec["threshold"]),
           "source": source, "reason": reason,
           "spec_baseline": spec["spec_baseline"]}
    if note is not None:
        row["note"] = note
    row.update(extra)
    return row


def _print_summary(doc, out_path: str) -> None:
    print("-" * 78, flush=True)
    print(f"Success criteria (PROPOSAL §2.9) — tag {doc['tag']}", flush=True)
    for row in doc["success_criteria"]["rows"]:
        print(f"  {row['metric']:<36} {row['target']:<26} "
              f"measured {_fmt(row['measured']):>10}  "
              f"{row['verdict'].upper()}", flush=True)
        if row.get("reason"):
            print(f"      reason: {row['reason']}", flush=True)
        if row.get("note"):
            print(f"      note: {row['note']}", flush=True)
        if row.get("stretch"):
            print(f"      stretch: {row['stretch']['target']} "
                  f"({'met' if row['stretch']['met'] else 'not met'})",
                  flush=True)
    gb = doc["success_criteria"]["gap_baseline"]
    if gb["measured_nats_per_doc"] is not None:
        print(f"  gap delta vs {gb['baseline_nats_per_doc']} baseline: "
              f"{gb['delta_vs_baseline']:+.4f} nats/doc "
              f"({gb['measured_gap_kind']}; negative = gap closed)",
              flush=True)
    print(f"  summary: {doc['success_criteria']['summary']}", flush=True)
    print("-" * 78, flush=True)
    print(f"wrote {out_path}", flush=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Stage-3 verification orchestrator for the "
                    "FLUTE-palettized Qwen3.5-9B distillation program: "
                    "runs the O-1 probe and the greedy match (one "
                    "producer at a time — never two models resident), "
                    "reads the trainer's run report from disk, and "
                    "writes the success-criteria JSON "
                    "(PROPOSAL §2.9).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    v = sub.add_parser(
        "verify",
        help="Stage-3 verification: O-1 gap probe (dense -> base -> "
             "base+adapters, sequentially), the trainer run report's "
             "per-layer lines, greedy-decode match, success-criteria "
             "JSON",
        description=(
            "Stage-3 verification (PROPOSAL §2.9 gates, §7.3 chain). "
            "Two producers run in their own subprocesses — strictly "
            "sequentially, one model on the GPU at a time: (1) "
            "scripts/o1_baseline_check.py, first its dense+base run "
            "(which exits before the next run starts), then the "
            "base+adapters run; (2) scripts/eval_greedy_match.py "
            "--qlora-adapters. The third producer is read from disk, no "
            "subprocess: --run <dir> (the trainer train output dir) "
            "gives the per-layer before/banked/after lines "
            "(finetune_provenance.json + qlora_layers/layer_<L>/"
            "metrics.json) and the export roundtrip lines "
            "(export_report.json, the G-J3 records). Assembles "
            "reports/distill_eval_<tag>.json with the success-criteria "
            "table (measured values, pass/fail/unknown verdicts with "
            "reasons — never a fabricated number), the gap delta vs the "
            "0.0793 baseline, the per-layer lines + mean, the greedy "
            "result, and the exact commands issued to every subprocess "
            "producer. Missing producer output, malformed JSON or a "
            "nonzero exit fails LOUDLY, naming the producer and the "
            "expected artifact. This orchestrator imports no torch and "
            "does no tensor math; --device only configures the producer "
            "invocations. The nMSE and KL rows of the original table "
            "are dropped (no producer computes them; the O-1 gap and "
            "the greedy match are their end-to-end proxies)."))
    v.add_argument("--artifacts", required=True,
                   help="palettized STUDENT artifacts dir (metadata.json) — "
                        "passed to every producer as --artifacts-dir")
    v.add_argument("--adapters", default=None,
                   help="Stage-1/2 adapter dir (qlora_adapters.pt + "
                        "qlora_config.json) — passed to the probe as "
                        "--adapters-dir and to the greedy matcher as "
                        "--qlora-adapters. Omit for a BASE-ONLY "
                        "verification: the probe still measures the "
                        "adapter-free quantization gap (recorded in "
                        "o1_probe.gap_base and the gap-baseline line) but "
                        "the adapter-dependent criteria rows (gap vs "
                        "<0.02, greedy match) are marked unknown — those "
                        "targets (G2/G3) apply to the distilled student")
    v.add_argument("--run", default=None,
                   help="the trainer train output dir — the per-layer "
                        "before/banked/after lines (finetune_provenance."
                        "json + per-layer metrics.json) and the export "
                        "roundtrip lines (export_report.json when the "
                        "export ran); omit to skip the layer section "
                        "(the cosine row is then unknown with the "
                        "reason recorded)")
    v.add_argument("--model", default="Qwen/Qwen3.5-9B",
                   help="HF id or LOCAL checkpoint of the dense base "
                        "(passed through to every producer; mirrors the "
                        "producers' own default)")
    v.add_argument("--dense-report", default=None,
                   help="REUSE an existing o1_baseline_check.py JSON "
                        "report's dense stage (results.dense_fp16."
                        "per_doc) instead of re-running the expensive "
                        "dense reference (e.g. reports/"
                        "o1_baseline_fineweb.json). Pairing is refused "
                        "loudly unless the doc-replication args "
                        "(dataset/max-samples/seq-len/eval-holdout/seed) "
                        "and n_docs match the fresh probe run")
    v.add_argument("--out", default=None,
                   help="output JSON (default: reports/"
                        "distill_eval_<tag>.json)")
    v.add_argument("--greedy-prompts", type=int, default=32,
                   help="greedy-decode prompts (mirrors eval_greedy_"
                        "match.py --n-prompts; the built-in pool has "
                        "exactly 32 prompts, so 1..32)")
    v.add_argument("--greedy-tokens", type=int, default=96,
                   help="greedy-decode new tokens per prompt (mirrors "
                        "eval_greedy_match.py --max-new-tokens)")
    v.add_argument("--tag", default=None,
                   help="run tag (default: run_YYYYmmdd_HHMMSS) — names "
                        "the output file and the producers dir")
    v.add_argument("--forward", choices=["kernel", "reference"],
                   default="kernel",
                   help="the producers' quant forward route, passed "
                        "THROUGH to every probe and greedy invocation "
                        "(kernel = the FLUTE fused path, the box default; "
                        "reference = the torch dequant path, the only "
                        "CPU-legal route)")
    v.add_argument("--probe-dataset", default="FineWeb-Edu",
                   help="the probe's --dataset, passed through (the doc "
                        "list is a pure function of the doc-replication "
                        "args — the reused --dense-report must match)")
    v.add_argument("--probe-max-samples", type=int, default=10000,
                   help="the probe's --max-samples, passed through")
    v.add_argument("--probe-seq-len", type=int, default=256,
                   help="the probe's --seq-len, passed through")
    v.add_argument("--probe-eval-holdout", type=int, default=64,
                   help="the probe's --eval-holdout, passed through")
    v.add_argument("--probe-seed", type=int, default=0,
                   help="the probe's --seed, passed through")
    v.add_argument("--device", default="cuda",
                   help="device passed THROUGH to every producer "
                        "invocation (cuda on the box); this orchestrator "
                        "does no tensor math")
    v.add_argument("--skip-greedy", action="store_true",
                   help="skip the greedy-decode producer (the slowest "
                        "stage; the greedy row becomes unknown with the "
                        "reason recorded). There is deliberately NO "
                        "--skip-probe: the paired O-1 gap is the metric "
                        "of record (G1/G2) — a verification without it "
                        "is not a verification. Skipping the per-layer "
                        "lines needs no flag: omit --run")
    v.set_defaults(func=cmd_verify)
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    main()
