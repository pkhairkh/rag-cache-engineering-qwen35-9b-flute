#!/usr/bin/env python3
"""scripts/doctor.py — the session-settling pass (kernel/numerics report).

The FIRST command of every GPU-box session (PROPOSAL §2.1 P0
pre-flight): settles the kernel import states (flute_extended,
flute_train_kernels, attn_sm86 — the error verbatim + remediation when
absent), materializes the requested layers exactly like a layer job
(shared shard read, palettized swap, adapter attach, attention pin, the
[qlora-guard] startup guard, the frozen dequant-path table), then the
numerics triple-check, the attention parity+timing probe, step timing,
and the optional reader-amplification probe. Every doctor line is
prefixed [doctor] except the per-module path table ([L%02d]). CUDA is
NOT required: a CPU run reports import states + the path table and
prints an explicit skip line for everything needing the GPU — never a
silent no-op; any real failure exits nonzero with remediation.

Import design (one-way): from-imports the engine's trainer plane; the
engine dispatches lazily (never imports this module at module level).
Unwinds at W2-T07/T08 and W2-T10.
"""
from __future__ import annotations

import argparse
import gc
import os
import sys
import time

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import palettized_modules as pmod          # noqa: E402  (local, same dir)
from qlora import (                       # noqa: E402  (local, same dir)
    QLoRALinear,
    iter_qlora_modules,
    report_frozen_paths,
)
from loss import distill_loss              # noqa: E402  (local, same dir)
from capture import (                      # noqa: E402  (local, same dir)
    CaptureStore,
    _reader_counters_reset,
    _reader_counters_snapshot,
    _split_runs,
    _text_config,
)
from trainer import (                     # noqa: E402  (the trainer plane)
    _PreloadedLayerSource,
    _QLORA_DEFAULT_ALPHA,
    _QLORA_DEFAULT_RANK,
    _attach_layer_adapters,
    _forward_layer,
    _guard_frozen_paths_cuda,
    _layer_type,
    _load_pinned_text_config,
    _load_rank_map,
    _load_spectrum_predictions,
    _parse_layers_ordered,
    _pin_student_attn,
    _pos_batch,
    _position_embeddings,
    _report_dequant_paths,
    StudentLayerShell,
    TeacherLayerSource,
    materialize_student_layer,
)

# ---------------------------------------------------------------------------
# doctor subcommand (W2.T3 — PROPOSAL §3 T3 doctor half + §6 output contract)
# ---------------------------------------------------------------------------

_DOCTOR_FP16_TOL = 20e-3      # §6: fp16 tolerance of the numerics check (relaxed for Tensor Core vs cuBLAS accumulation order differences)
_DOCTOR_TOKENS = 256          # synthetic B·S tokens (numerics + timing batch)
_DOCTOR_TIMING_BATCH = 2      # timing batch B (B·S = 256 tokens)
_DOCTOR_TIMING_SEQ = 128      # timing batch S


def _doctor_import_flute_extended():
    """(ok, error) — the flute_extended import-check for the doctor.

    The package exports NO import_error() helper (its __init__ raises at
    import time when the _C extension is missing), so the CAUGHT error is
    returned verbatim — exactly what the doctor prints when absent."""
    try:
        p = os.path.join(_HERE, "..", "flute_extended")
        if p not in sys.path:
            sys.path.insert(0, p)
        import flute_extended
        if hasattr(flute_extended, "qgemm_per_group_lut"):
            return True, None
        return False, ("flute_extended imported but qgemm_per_group_lut "
                       "is absent (a partial/broken build)")
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


def _doctor_import_flute_train_kernels():
    """(importable, cuda_ok, error) — the flute_train_kernels check.

    This package DOES export import_error() (its __init__ catches the _C
    import internally), so the absent-build error is fetched from it
    verbatim; a build that loads on a CPU box reports importable-but-no-
    CUDA (available() requires CUDA) rather than an error."""
    try:
        p = os.path.join(_HERE, "..", "flute_train_kernels")
        if p not in sys.path:
            sys.path.insert(0, p)
        import flute_train_kernels as ftk
    except Exception as e:
        return False, False, f"{type(e).__name__}: {e}"
    if ftk.available():
        return True, True, None
    err = ftk.import_error()
    if err is None:
        return True, False, None        # loads; CUDA is simply down
    return True, False, err


def _doctor_report_imports():
    """Doctor step (1): settle the kernel import states.

    An import failure is REPORTED (the error verbatim + the build command)
    but does not by itself fail the run: a CPU box has no use for the CUDA
    kernels (every dependent check below prints its own explicit skip
    line), and a CUDA run already fails loudly at the attach-time T10
    guard / the [qlora-guard] startup guard — never a silent no-op either
    way. The attention (attn_sm86) import/CUDA state settles here like
    the other kernels — W4 delivered the kernel; the parity+timing probe
    itself is step (4) below (_doctor_attention_check)."""
    ok, err = _doctor_import_flute_extended()
    if ok:
        print("[doctor] flute_extended: importable "
              "(qgemm_per_group_lut present)", flush=True)
    else:
        print(f"[doctor] flute_extended: IMPORT FAILED — {err}", flush=True)
        print("[doctor]   remediation: cd flute_extended && python "
              "setup.py build_ext --inplace", flush=True)
    _ftk_import, ftk_cuda, err2 = _doctor_import_flute_train_kernels()
    if ftk_cuda:
        print("[doctor] flute_train_kernels: available (fused backward "
              "kernel importable + CUDA up)", flush=True)
    elif err2:
        # the build is absent (import_error() verbatim when the package
        # exposes it; the caught ImportError otherwise — see above)
        print(f"[doctor] flute_train_kernels: NOT BUILT — {err2}",
              flush=True)
        print("[doctor]   remediation: cd flute_train_kernels && python "
              "setup.py build_ext --inplace", flush=True)
    else:
        # the package imports and _C loads — CUDA is simply down
        print("[doctor] flute_train_kernels: importable but CUDA "
              "unavailable — the fused backward is not exercisable on "
              "this device", flush=True)
    # attn (W4 attn_sm86): the import/CUDA state settles like the other
    # kernels — the parity+timing probe itself is step (4) below.
    try:
        import attn_sm86 as _attn_mod
    except ImportError as e:
        _attn_ok, _attn_why = False, f"attn_sm86 import failed: {e}"
    else:
        _attn_ok, _attn_why = _attn_mod.kernel_available()
    if _attn_ok:
        print("[doctor] attention: attn_sm86 kernel available "
              "(triton importable + CUDA up)", flush=True)
    else:
        print(f"[doctor] attention: attn kernel unavailable ({_attn_why})",
              flush=True)


def _doctor_module_representatives(shells):
    """One (layer, name, QLoRALinear) per DISTINCT (N, K) shape class over
    the doctor's resident layers — the per-class probe set of the numerics
    check (PROPOSAL T3: one module per shape class)."""
    reps = {}
    for L, shell, _has_adapter in shells:
        for name, mod in iter_qlora_modules(shell):
            if not isinstance(mod, QLoRALinear):
                continue   # a QLoRASplitQKV: its q/k/v children are
                           # yielded below in their own right
            key = (int(mod.base.N), int(mod.base.K))
            if key not in reps:
                reps[key] = (L, name, mod)
    return reps


def _doctor_module_forward(base, x2, path):
    """One frozen-branch forward of a PalettizedLinear `base` on the input
    (tokens, K) through the given path — the EXACT branch code of
    QLoRALinear.forward (bias + residual included), factored out so the
    numerics triple-check compares like with like.

    path: "fused" (qlora_gemm.fused_qlora_gemm) | "torch-gpu"
    (qlora_fallback.materialize_weight + x @ W16.t()) | "reference"
    (base(x2) with base.reference=True — base.forward itself adds the bias
    and the resA/resB branch, so the caller must not add them again)."""
    if path == "fused":
        import qlora_gemm
        y = qlora_gemm.fused_qlora_gemm(
            x2, base.indices, base.lut, base.bitwidth, base.group_size,
            base.N, base.K)
        if y is None:
            raise RuntimeError(
                "[doctor] fused_qlora_gemm returned None although the "
                "module probed eligible — the FLUTE forward kernel went "
                "away mid-run (T10: never a silent fallback)")
        if base.bias is not None:
            y = y + base.bias
        if base.resA is not None and base.resB is not None:
            y = y + ((x2.float() @ base.resB.t().float())
                     @ base.resA.t().float()).to(y.dtype)
        return y
    if path == "torch-gpu":
        import qlora_fallback
        W16 = qlora_fallback.materialize_weight(
            base.indices, base.lut, base.N, base.K, base.group_size,
            torch.float16)
        xh = x2 if x2.dtype == W16.dtype else x2.to(W16.dtype)
        y = xh @ W16.t()
        if x2.dtype != y.dtype:
            y = y.to(x2.dtype)
        if base.bias is not None:
            y = y + base.bias
        if base.resA is not None and base.resB is not None:
            y = y + ((x2.float() @ base.resB.t().float())
                     @ base.resA.t().float()).to(y.dtype)
        return y
    # "reference": the base's own reference forward (reference=True)
    was_ref = base.reference
    base.reference = True
    try:
        return base(x2)
    finally:
        base.reference = was_ref


def _doctor_numerics(shells, device, failures):
    """Doctor step (3): the numerics triple-check — fused vs torch-gpu-
    cached vs reference on the SAME blob/LUT, one random synthetic batch
    (B·S=256 tokens) per distinct (N, K) shape class, fp16 tolerance 6e-3
    (PROPOSAL §6).

    On CPU there is exactly ONE legal path (reference-cpu — the frozen
    policy resolves it for every module), so the check prints the explicit
    skip line and stays exit-0. On CUDA the fused eligibility is probed
    with qlora_gemm.fused_gemm_eligible(base); a class that is not
    eligible still gets its TWO exercisable paths compared (torch-gpu vs
    reference) plus an explicit fused-skip reason — never a silent no-op."""
    dev = torch.device(device)
    if dev.type != "cuda":
        print("[doctor] numerics: skipped (single path on CPU)", flush=True)
        return
    reps = _doctor_module_representatives(shells)
    if not reps:
        print("[doctor] numerics: skipped (no QLoRALinear module resident "
              "— every --layers slice is all-r=0 or unwrapped)", flush=True)
        return
    import qlora_gemm
    eligible = {}
    for key, rep in reps.items():
        try:
            ok = bool(qlora_gemm.fused_gemm_eligible(rep[2].base))
        except Exception:
            ok = False
        if ok:
            eligible[key] = rep
    if not eligible:
        ok, err = _doctor_import_flute_extended()
        why = err if not ok else (
            "fused_gemm_eligible(base) is False for every resident module "
            "(geometry N%128==0/K%64==0, frozen fp16 LUT, CUDA-resident "
            "indices/lut)")
        print(f"[doctor] numerics: fused path unavailable — {why}",
              flush=True)
    for (N, K), (L, name, mod) in sorted(reps.items()):
        base = mod.base
        rel = (name[len(f"model.layers.{L}."):]
               if name.startswith(f"model.layers.{L}.") else name)
        tag = f"(L{L:02d} {rel} N={N} K={K})"
        if (N, K) not in eligible and eligible:
            print(f"[doctor] numerics: fused skipped for {tag} — "
                  f"fused_gemm_eligible(base) is False (geometry "
                  f"N%128==0/K%64==0, frozen fp16 LUT, CUDA-resident "
                  f"indices/lut)", flush=True)
        x2 = torch.randn(_DOCTOR_TOKENS, K, device=dev,
                         dtype=torch.float16)
        outs = {}
        try:
            if (N, K) in eligible:
                with torch.no_grad():
                    outs["fused"] = _doctor_module_forward(
                        base, x2, "fused")
            with torch.no_grad():
                outs["torch-gpu"] = _doctor_module_forward(
                    base, x2, "torch-gpu")
                outs["reference"] = _doctor_module_forward(
                    base, x2, "reference")
        except (OSError, RuntimeError, ValueError) as e:
            # e.g. dequant_idx4_torch's idx4 geometry contract on a module
            # the kernel path would never accept — a probe failure, never
            # a traceback and never a silent skip
            print(f"[doctor] numerics: probe FAILED — {type(e).__name__}: "
                  f"{e} {tag}", flush=True)
            failures.append(
                f"numerics probe {tag}: {type(e).__name__}: {e} — a path "
                f"forward raised on this module; inspect the geometry "
                f"(idx4 requires N%128==0 and K%64==0)")
            continue
        pairs = []
        if "fused" in outs:
            # the §6 contract line (gated): fused vs torch-gpu
            pairs.append(("fused vs torch-gpu", outs["fused"],
                          outs["torch-gpu"]))
            pairs.append(("fused vs reference", outs["fused"],
                          outs["reference"]))
        pairs.append(("torch-gpu vs reference", outs["torch-gpu"],
                      outs["reference"]))
        for label, a, b in pairs:
            d = float((a.float() - b.float()).abs().max())
            ok = d <= _DOCTOR_FP16_TOL
            print(f"[doctor] numerics: {label} max|d|={d:.2e} "
                  f"(fp16 tol {_DOCTOR_FP16_TOL:.0e}) {'PASS' if ok else 'FAIL'} {tag}",
                  flush=True)
            if not ok:
                failures.append(
                    f"numerics {label} {tag}: max|d|={d:.3e} exceeds the "
                    f"fp16 tolerance 6e-3 on the same blob/LUT — the "
                    f"paths disagree; rebuild flute_extended (cd "
                    f"flute_extended && python setup.py build_ext "
                    f"--inplace) and re-run, or take the explicit "
                    f"FLUTE_FROZEN_PATH=torch opt-out and compare "
                    f"torch-gpu vs reference only")


def _doctor_step_timing(shells, device, failures):
    """Doctor step (5): ONE forward+backward per AVAILABLE path on the
    resident layer (the last --layers entry with attached modules), a
    synthetic B·S=256-token batch, torch.cuda.synchronize + perf_counter.
    On CPU: the explicit skip line (stay exit-0).

    The path forcing is a TIMING PROBE ONLY: each QLoRALinear's frozen
    path (resolved once at attach, W2.T1) is saved, switched to the probed
    path (the reference probe also flips base.reference — the sanctioned
    eval_reference() switch), and RESTORED afterwards; the [qlora-guard]
    startup guard has already asserted the REAL frozen paths (T10). The
    fused probe switches only the fused-eligible modules (an ineligible
    module would be rejected by the kernel — it keeps its real frozen
    path and still contributes to the timing)."""
    dev = torch.device(device)
    if dev.type != "cuda":
        print("[doctor] step timing: skipped (CPU device)", flush=True)
        return
    resident = None
    for L, shell, _has_adapter in reversed(shells):
        if any(isinstance(m, QLoRALinear)
               for _, m in iter_qlora_modules(shell)):
            resident = (L, shell)
            break
    if resident is None:
        print("[doctor] step timing: skipped (no QLoRALinear module "
              "resident — every --layers slice is all-r=0)", flush=True)
        return
    L, shell = resident
    student = shell.layer
    B, S = _DOCTOR_TIMING_BATCH, _DOCTOR_TIMING_SEQ
    hidden = int(_text_config(shell).hidden_size)
    pos16 = _position_embeddings(shell, S, torch.float16, dev)
    pos = _pos_batch(pos16, B)
    x16 = torch.randn(B, S, hidden, device=dev, dtype=torch.float16)
    tgt = torch.randn(B, S, hidden, device=dev, dtype=torch.float32)

    # available paths on CUDA: fused-flute needs the FLUTE kernel AND an
    # eligible module; torch-gpu-cached and reference-cpu are always
    # exercisable (the probe switches below drive them explicitly).
    fused_mods = []
    try:
        import qlora_gemm
        fused_mods = [m for _, m in iter_qlora_modules(shell)
                      if isinstance(m, QLoRALinear)
                      and qlora_gemm.fused_gemm_eligible(m.base)]
    except Exception:
        fused_mods = []
    paths = ["torch-gpu-cached", "reference-cpu"]
    if fused_mods:
        paths.insert(0, "fused-flute")
    else:
        ok, err = _doctor_import_flute_extended()
        why = err if not ok else \
            "no fused-eligible module in the resident layer"
        print(f"[doctor] step timing: fused skipped ({why})", flush=True)
    labels = {"fused-flute": "fused fwd+bwd",
              "torch-gpu-cached": "torch-gpu",
              "reference-cpu": "reference"}
    segments = []
    for path in paths:
        saved = []
        for _name, mod in iter_qlora_modules(shell):
            if isinstance(mod, QLoRALinear):
                saved.append((mod, mod._frozen_path,
                              bool(mod.base.reference)))
                if path == "fused-flute" and mod not in fused_mods:
                    continue     # an ineligible module keeps its REAL
                                 # frozen path (the kernel would reject it;
                                 # the probe is per-path, not per-module)
                mod._frozen_path = path
                mod.base.reference = (path == "reference-cpu")
        try:
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            # x requires grad so the backward actually flows through the
            # frozen branch (the fused backward kernel / the cached-W
            # cuBLAS / the reference matmul), not just the LoRA branch
            x = x16.detach().clone().requires_grad_(True)
            y = _forward_layer(student, x, pos, shell)
            loss, _ = distill_loss(y.float(), tgt, mse_weight=1.0,
                                   cos_weight=0.05)
            loss.backward()
            torch.cuda.synchronize()
            dt = time.perf_counter() - t0
        except Exception as e:
            print(f"[doctor] step timing: {labels[path]} FAILED — "
                  f"{type(e).__name__}: {e}", flush=True)
            failures.append(
                f"step timing {labels[path]} (L{L:02d}) raised "
                f"{type(e).__name__}: {e} — the path is not runnable; "
                f"rebuild the kernels (cd flute_extended && python "
                f"setup.py build_ext --inplace; cd flute_train_kernels && "
                f"python setup.py build_ext --inplace)")
            continue
        finally:
            for mod, p, ref in saved:
                mod._frozen_path = p
                mod.base.reference = ref
        if dt >= 1.0:
            segments.append(f"{labels[path]} {dt:.1f}s")
        else:
            segments.append(f"{labels[path]} {dt * 1000.0:.0f}ms")
    if segments:
        print(f"[doctor] step timing: {' | '.join(segments)}  "
              f"(L{L:02d}, B·S={B * S})", flush=True)


def _doctor_attention_check(device, failures):
    """Doctor step (4): the SM86 Triton attention parity + timing probe
    (the W2.T3 deferred half, delivered by W4).

    Gate: torch.cuda.is_available() AND attn_sm86.kernel_available()[0]
    (the guard every kernel-path caller uses). On the gate: synthetic
    REAL-GEOMETRY tensors (B=1, H=16, H_kv=4, S=2048, D=256, fp16, fixed
    seed) on CUDA; sm86_attention_forward vs reference_attention_forward
    under the binding parity standard (reports/decisions.md W4.T2.s2:
    L2rel <= 3e-3 AND max|d|/max|b| <= 3e-2 — the same gates
    tests/test_attn_kernel.py records); then ONE timed fwd+bwd
    (torch.cuda.synchronize + perf_counter — the G-A1 instrument; one
    untimed warmup pass so the number measures the kernels, not
    triton's JIT compile). A parity FAIL is a doctor failure (the
    remediation text names the standard and the reference suite).

    Off the gate (a CPU box, or triton's import failed): the explicit
    skip line and CONTINUE (exit 0) — the parity+timing gates live on
    the GPU box (tests/test_attn_kernel.py CUDA path)."""
    import attn_sm86 as attn
    ok, reason = attn.kernel_available()
    if not (torch.cuda.is_available() and ok):
        print("[doctor] attention: skipped (no CUDA triton on this box — "
              "parity+timing run on the GPU box; see "
              "tests/test_attn_kernel.py CUDA path)", flush=True)
        return
    gen = torch.Generator(device="cuda").manual_seed(0)
    B, H, H_KV, S, D = 1, 16, 4, 2048, 256

    def _rnd(*shape):
        return torch.randn(*shape, device="cuda", dtype=torch.float16,
                           generator=gen)

    q, k, v = _rnd(B, H, S, D), _rnd(B, H_KV, S, D), _rnd(B, H_KV, S, D)
    sm_scale = D ** -0.5
    # parity: the kernel forward vs the fp32-accumulating reference
    with torch.no_grad():
        out, _lse = attn.sm86_attention_forward(q, k, v, sm_scale)
        ref, _ = attn.reference_attention_forward(q, k, v, sm_scale)
    d = out.float() - ref.float()
    l2 = float(d.norm() / ref.float().norm())
    mr = float(d.abs().max() / ref.float().abs().max())
    passed = l2 <= 3e-3 and mr <= 3e-2
    print(f"[doctor] attention parity: L2rel={l2:.2e} max_ratio={mr:.2e} "
          f"(tol 3e-3/3e-2) {'PASS' if passed else 'FAIL'}", flush=True)
    if not passed:
        failures.append(
            f"attention parity L2rel={l2:.3e} max_ratio={mr:.3e} exceeds "
            f"the 3e-3/3e-2 standard (reports/decisions.md W4.T2.s2) at "
            f"B=1,H=16,H_kv=4,S=2048,D=256 fp16 — the Triton kernel "
            f"disagrees with reference_attention_forward; see "
            f"tests/test_attn_kernel.py (the CUDA-gated parity class) and "
            f"reports/design_attn_sm86.md")
    # one warmup fwd+bwd (JIT compile exclusion — the parity probe above
    # already compiled the forward; this warms the backward), then the
    # ONE timed pass
    qw, kw, vw = (t.detach().requires_grad_(True) for t in (q, k, v))
    yw = attn.flute_sm86_attention(qw, kw, vw, sm_scale)
    yw.backward(torch.randn(yw.shape, device="cuda",
                            dtype=torch.float16, generator=gen))
    qt, kt, vt = (t.detach().requires_grad_(True) for t in (q, k, v))
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    y = attn.flute_sm86_attention(qt, kt, vt, sm_scale)
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    y.backward(torch.randn(y.shape, device="cuda",
                           dtype=torch.float16, generator=gen))
    torch.cuda.synchronize()
    t2 = time.perf_counter()
    print(f"[doctor] attention timing: fwd {(t1 - t0) * 1e3:.1f} ms | "
          f"bwd {(t2 - t1) * 1e3:.1f} ms (B=1,H=16,S=2048,D=256)",
          flush=True)


def _doctor_reader_probe(store, layer, device, failures):
    """Doctor step (6): one scattered 4-row batch through the row-exact
    reader (store.h_rows) — the reader-I/O amplification registry must
    report EXACTLY 1.0 (W1.T1's G-T1 metric; the old span reader's 113x
    blow-up is the P1 root cause this probes for regressions)."""
    rows_total = store.rows
    if rows_total >= 4:
        rows = sorted({0, 1, max(2, rows_total // 2), rows_total - 1})
    else:
        rows = list(range(rows_total))
        print(f"[doctor] reader probe: store has {rows_total} row(s) — "
              f"probing a {len(rows)}-row batch (the scattered 4-row "
              f"probe needs >= 4 rows)", flush=True)
    _reader_counters_reset()
    try:
        store.h_rows(layer, rows, device, torch.float16)
    except RuntimeError as e:
        print(f"[doctor] reader_amp: FAIL — {e}", flush=True)
        failures.append(f"reader probe: {e}")
        return
    snap = _reader_counters_snapshot()
    amp = (snap["bytes_moved"] / snap["bytes_needed"]
           if snap["bytes_needed"] > 0 else float("nan"))
    ok = (amp == 1.0)
    print(f"[doctor] reader_amp={amp:.2f} ({len(rows)}-row scattered "
          f"batch) {'PASS' if ok else 'FAIL'}", flush=True)
    if not ok:
        failures.append(
            f"reader amplification {amp:.3f} != 1.0 on the scattered "
            f"{len(rows)}-row batch (moved {snap['bytes_moved']} B for "
            f"{snap['bytes_needed']} B needed, {snap['runs']} run(s)) — "
            f"the row-exact reader regressed (P1); inspect h_rows / "
            f"_split_runs")


def cmd_doctor(args):
    """W2.T3 (PROPOSAL §3 T3 doctor half, §6 output contract): the
    session-settling pass — the FIRST command of every GPU-box session.

    One pass settles, in order:
      (1) the kernel import states — flute_extended and
          flute_train_kernels (the error verbatim + the build commands
          when absent) and the attn_sm86 kernel_available() state;
      (2) per --layers layer (default the pilot pair 0,3): the SAME
          materialize+attach machinery as cmd_finetune (shared shard read,
          StudentLayerShell, rank-map slice / uniform default, optional
          warm start), the W4.T3 student attn pin (_pin_student_attn —
          the path report then shows the true training attn_impl), the
          [qlora-guard] T10 startup guard, and the W2.T2 frozen dequant-
          path table (its own [L%02d] prefixes, attn=... in the summary);
      (3) the numerics triple-check (fused vs torch-gpu-cached vs
          reference, one module per distinct (N, K) class, fp16 tol 6e-3)
          on CUDA; on CPU an explicit skipped line (single path), exit 0;
      (4) the attention parity + timing probe (_doctor_attention_check:
          real-geometry sm86_attention_forward vs reference under the
          3e-3/3e-2 standard + one fwd+bwd timing on CUDA; a CPU /
          triton-less box prints the explicit skip line, exit 0);
      (5) one forward+backward timing per available path on the resident
          layer (synthetic B·S=256 batch, cuda-synchronized; CPU skips
          loudly);
      (6) the optional reader probe when --capture-dir is given: one
          scattered 4-row batch, reader_amp == 1.0 asserted via the
          registry;
      (7) exit nonzero with remediation text on any failure (a failed
          hard-guard, a numerics FAIL, a missing store).

    Every doctor line is prefixed [doctor] except the per-module path
    table, which keeps its [L%02d] prefix (the shared W2.T2 report's
    format). CUDA is NOT required: a CPU run reports the import states and
    the path table and prints an explicit skip line for everything that
    needs the GPU — never a silent no-op."""
    device = args.device
    dev = torch.device(device)
    if dev.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit(
            f"[doctor] FAIL: --device {device} requested but "
            f"torch.cuda.is_available() is False — run on the GPU box or "
            f"drop --device (the doctor runs on CPU with explicit skip "
            f"lines)")
    torch.manual_seed(args.seed)
    failures = []

    print("=" * 78)
    print("doctor — session settling (imports, paths, numerics, timing)")
    print("=" * 78)
    print(f"[doctor] model: {args.model}", flush=True)
    print(f"[doctor] artifacts: {args.artifacts_dir}  device: {device} "
          f"(torch.cuda.is_available()="
          f"{torch.cuda.is_available()})", flush=True)
    print(f"[doctor] layers: {args.layers}  capture: "
          f"{args.capture_dir or 'none (reader probe off)'}", flush=True)

    # ---- (1) import states + attention availability ----------------------
    _doctor_report_imports()

    # ---- (2) per-layer materialize + attach + guard + path report --------
    if not os.path.isdir(args.model):
        raise SystemExit(
            f"[doctor] FAIL: --model {args.model!r} is not a LOCAL "
            f"checkpoint directory — the doctor streams one layer's dense "
            f"parts from its safetensors shards exactly like `finetune "
            f"--train-target qlora` (pre-download the model and pass the "
            f"local path)")
    try:
        metadata = pmod.load_metadata(args.artifacts_dir)
    except (OSError, ValueError, RuntimeError) as e:
        raise SystemExit(
            f"[doctor] FAIL: --artifacts-dir {args.artifacts_dir!r}: {e} — "
            f"expected the palettization output (metadata.json + idx4/"
            f"lut_scalar files); fix the path or re-run the palettizer")
    cfg = _load_pinned_text_config(args.model)
    num_layers = int(cfg.num_hidden_layers)
    rank_map = _load_rank_map(args.rank_map, num_layers) \
        if args.rank_map else None
    if rank_map is None:
        print(f"[doctor] no --rank-map given — attaching the uniform "
              f"default r={_QLORA_DEFAULT_RANK}/"
              f"alpha={_QLORA_DEFAULT_ALPHA} (kappa 0.25), exactly like a "
              f"rank-map-less finetune run", flush=True)
    if args.warm_start and rank_map is None:
        raise SystemExit(
            "--warm-start requires --rank-map (the spectrum's rank map — "
            "a uniform default has no matching warm-start geometry)")
    if args.spectrum:
        spectrum = _load_spectrum_predictions(args.spectrum)
        print(f"[doctor] spectrum: {len(spectrum)} per-module predictions "
              f"loaded (validated) — doctor records but does not consume "
              f"them (the per-layer stop target is a training-time input)",
              flush=True)
    wanted = _parse_layers_ordered(args.layers, num_layers)

    teacher_source = TeacherLayerSource(args.model, device=device)
    shells = []            # (L, shell, has_adapter) — resident for (3)/(5)
    for L in wanted:
        # one shared shard read (F7/F16.3): the same seam the layer job
        # uses — the state feeds the student's dense parts only (the
        # doctor builds no teacher layer: it has no targets to compute)
        try:
            sd = teacher_source.layer_state(L)
            student, n_swapped = materialize_student_layer(
                L, cfg, _PreloadedLayerSource(sd, L), args.artifacts_dir,
                metadata, device=device, dtype=torch.float16)
        except (OSError, RuntimeError, ValueError, KeyError) as e:
            # a layer the model/artifacts pair cannot materialize is a
            # doctor failure, never a traceback (a missing store is one of
            # the contract's failure classes)
            raise SystemExit(
                f"[doctor] FAIL: layer {L} materialization: "
                f"{type(e).__name__}: {e} — the checkpoint has no tensors "
                f"for this layer or the artifacts set does not cover it "
                f"(missing idx4/lut_scalar files)")
        shell = StudentLayerShell(student, L, cfg, device=device)
        # W4.T3 / W2.T3: pin the doctor's resident students EXACTLY like
        # a layer job (_pin_student_attn: per-module config COPY, the
        # model-level cfg stays sdpa — the doctor builds no teacher, so
        # the pin's teacher-safety precondition holds) — the path report
        # below then shows the TRUE training attn_impl (attn=...); on a
        # box where the kernel is unavailable, the modeling branch's
        # one-shot fallback banner is the honest signal if anything
        # forwards. The doctor settles and reports; it never changes
        # policy.
        _pin_student_attn(student, L, _layer_type(cfg, L), cfg, device,
                          ctx=f"doctor L{L}")
        try:
            acfg, _rank_slice = _attach_layer_adapters(
                shell, L, rank_map, args)
        except RuntimeError as e:
            # the T10 attach-time error already names the build commands
            raise SystemExit(f"[doctor] FAIL: {e}")
        has_adapter = acfg is not None
        if not has_adapter:
            print(f"[doctor] L{L:02d}: rank slice is all r=0 — layer "
                  f"already aligned, no adapter attached ({n_swapped} "
                  f"modules checked); the path table below is empty by "
                  f"construction", flush=True)
        # W2.T1 s3 startup guard: a CUDA doctor run obeys T10 exactly like
        # a layer job (never degraded, never silent); CPU passes silently.
        try:
            _guard_frozen_paths_cuda(student, L, device)
        except RuntimeError as e:
            raise SystemExit(f"[doctor] FAIL: {e}")
        # W2.T2: the frozen dequant-path table (per-module [L%02d] lines +
        # summary — the shared report's own prefixes)
        _report_dequant_paths(student, L)
        if has_adapter and args.warm_start:
            # after the report, exactly like the layer job's F9 anchor
            import spectrum  # noqa: E402  (lazy: spectrum imports the engine)
            spectrum._apply_warm_starts(
                shell, args.warm_start, ctx=f"L{L}",
                gauge=getattr(args, "warm_start_gauge", "balanced"))
        shells.append((L, shell, has_adapter))
        del sd
        gc.collect()

    # the §6 aggregate path summary across the doctor's resident layers
    total = {"fused-flute": 0, "torch-gpu-cached": 0, "reference-cpu": 0}
    for _L, shell, _has_adapter in shells:
        rep = report_frozen_paths(shell)
        for k in total:
            total[k] += rep["counts"].get(k, 0)
    print(f"[doctor] fused-flute: {total['fused-flute']} modules | "
          f"torch-gpu-cached: {total['torch-gpu-cached']} | "
          f"reference-cpu: {total['reference-cpu']}", flush=True)

    # ---- (3) numerics triple-check ----------------------------------------
    _doctor_numerics(shells, device, failures)

    # ---- (4) attention parity + timing (W4 delivered the W2.T3
    # deferred half): real-geometry parity vs reference + one fwd+bwd
    # timing on CUDA; the explicit skip line (exit 0) otherwise
    _doctor_attention_check(device, failures)

    # ---- (5) one fwd+bwd timing per available path -----------------------
    _doctor_step_timing(shells, device, failures)

    # ---- (6) optional reader probe ----------------------------------------
    if args.capture_dir:
        try:
            store = CaptureStore(capture_dir=args.capture_dir)
        except SystemExit as e:
            raise SystemExit(f"[doctor] FAIL: --capture-dir: {e}")
        except (FileNotFoundError, ValueError, RuntimeError) as e:
            raise SystemExit(
                f"[doctor] FAIL: --capture-dir {args.capture_dir!r}: {e} — "
                f"a missing/broken store is a failure (re-run "
                f"`python scripts/capture.py capture` into a fresh dir)")
        if store.num_layers != num_layers:
            raise SystemExit(
                f"[doctor] FAIL: layer count mismatch: model {num_layers} "
                f"vs capture {store.num_layers} — the capture does not "
                f"belong to this model")
        _doctor_reader_probe(store, wanted[0], device, failures)

    # ---- (7) exit code ----------------------------------------------------
    if failures:
        for f in failures:
            print(f"[doctor] FAIL: {f}", flush=True)
        raise SystemExit(
            f"[doctor] {len(failures)} failure(s) — see the [doctor] FAIL "
            f"lines above. Remediation: build the kernels (cd "
            f"flute_extended && python setup.py build_ext --inplace; cd "
            f"flute_train_kernels && python setup.py build_ext --inplace) "
            f"or take the explicit opt-outs (FLUTE_FROZEN_PATH=torch, "
            f"FLUTE_FUSED_BWD=0) — never a silent downgrade")
    print("[doctor] PASS — imports, paths, numerics, timing settled "
          "(every skip above carries its reason)", flush=True)


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Session settling for the FLUTE-palettized "
                    "Qwen3.5-9B layerwise pipeline: kernel imports, "
                    "per-layer frozen dequant paths, numerics, timing.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp):
        sp.add_argument("--model", default="Qwen/Qwen3.5-9B",
                        help="HF id or LOCAL checkpoint dir (required for "
                             "--target-mode teacher-on-student)")
        sp.add_argument("--device", default=(
            "cuda:0" if torch.cuda.is_available() else "cpu"))

    # ---- doctor ------------------------------------------------------------
    d = sub.add_parser(
        "doctor",
        help="session settling: kernel imports, per-layer frozen dequant "
             "paths, numerics triple-check, one-step timing — the first "
             "command of every GPU-box session",
        description=(
            "doctor (W2.T3 / PROPOSAL §3 T3 + §6): settles import states, "
            "paths, numerics and timing in ONE pass — run it before any "
            "GPU-box session. (1) import-checks flute_extended and "
            "flute_train_kernels (the error verbatim + the build commands "
            "when absent); (2) materializes the --layers student layers "
            "EXACTLY like `finetune --train-target qlora` (shared shard "
            "read, rank-map slice or the uniform default, optional warm "
            "start), runs the [qlora-guard] T10 startup guard and prints "
            "the frozen dequant-path table per layer; (3) numerics "
            "triple-check — fused vs torch-gpu-cached vs reference on the "
            "same blob/LUT, one module per distinct (N,K) shape class, "
            "fp16 tolerance 6e-3 — on CUDA (on CPU: an explicit skipped "
            "line, exit 0); (4) attention parity + timing (attn_sm86, "
            "real geometry B=1/H=16/S=2048, the 3e-3/3e-2 parity standard "
            "+ one fwd+bwd timing — CUDA; a CPU/triton-less box prints "
            "the explicit skip line, exit 0); "
            "(5) one forward+backward timing per available "
            "path on the resident layer (CUDA only; CPU skips loudly); "
            "(6) optional reader probe (--capture-dir): one scattered "
            "4-row batch, reader_amp == 1.0; (7) exits nonzero with "
            "remediation text on any failure. CUDA is NOT required."))
    common(d)
    d.add_argument("--artifacts-dir", required=True,
                   help="original palettization output (metadata.json) — "
                        "READ-ONLY (the doctor never writes)")
    d.add_argument("--capture-dir", default=None,
                   help="optional teacher capture dir (boundary store, "
                        "format 3) — enables the reader probe (one "
                        "scattered 4-row batch, reader_amp == 1.0)")
    d.add_argument("--layers", default="0,3",
                   help="layer subset to materialize (the pilot pair by "
                        "default; e.g. '0-7,12' — the given order is kept)")
    d.add_argument("--rank-map", default=None,
                   help="rank_map.json of the distill_rank_alloc.py schema "
                        "({\"rank_map\": {module path: r}}) — the same "
                        "args surface as finetune; absent => the uniform "
                        "default r=64/alpha=16")
    d.add_argument("--warm-start", default=None,
                   help="warm_starts.pt (or its dir) from the spectrum "
                        "subcommand — applied to the attached adapters "
                        "after the path report (requires --rank-map)")
    d.add_argument("--warm-start-gauge", default="balanced",
                   choices=("balanced", "stored"),
                   help="the gauge the doctor's warm-start application "
                        "uses (same semantics as finetune's; balanced "
                        "default: verify exactly what training will load)")
    d.add_argument("--spectrum", default=None,
                   help="functional_spectrum.json — validated and recorded "
                        "(the doctor does not consume the stop target)")
    d.add_argument("--seed", type=int, default=42,
                   help="seed of the synthetic numerics/timing batches and "
                        "the adapter init")


    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.cmd == "doctor":
        cmd_doctor(args)
    else:  # pragma: no cover
        raise SystemExit(f"unknown subcommand {args.cmd}")


if __name__ == "__main__":
    main()
