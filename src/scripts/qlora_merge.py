#!/usr/bin/env python3
"""qlora_merge.py — merge a trained LoRA back into the frozen LUT+indices.

Artifact conventions:
  * metadata tensor names carry the parameter suffix (…weight); adapter
    state-dict keys use module paths (no suffix) — _module_path bridges
    the two (a name appended verbatim would match nothing, making the
    merge a silent no-op).
  * QKV components carry no dense_shape of their own; their N follows
    from packed_len_bytes.

LoRA scale: per-module, not global. attach_qlora records each module's
geometry in the adapter dir's qlora_config.json
(cfg.tensors[module]["scale"] for plain modules;
cfg.tensors[module]["components"]["q"/"k"/"v"]["scale"] for SplitQKV
components) — the merge reads that, so a mixed-rank adapter dir merges
each module at ITS scale, not the (wrong, silently-mismatching) global
cfg.alpha/cfg.r. Adapter dirs whose tensors entries predate those
fields fall back to the global alpha/r, and a resolved scale of 0.0 for
a module that does carry adapter tensors is a loud RuntimeError (a
scale-0 merge is a silent no-op).

Re-palettization engine (`--assign`, default gptvq):
  lloyd   unweighted Lloyd on CPU/GPU
  gptq    GPTQ error-compensated sweep against the stored Gram
  gptvq   exact BCD engine (LUT solve + ICM) against the stored Gram
The calibrated engines require `grams/<name>.gram.npy` in the artifacts
(written by the palettizer's --persist-grams); without them they fail
loudly rather than silently degrading to unweighted Lloyd.
"""
from __future__ import annotations
import argparse, hashlib, json, os, shutil, sys, time
from typing import Optional
import numpy as np, torch
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path: sys.path.insert(0, _HERE)
import palettized_modules as pmod
import qlora

_PARAM_SUFFIX = ".weight"


def _module_path(tensor_name: str) -> str:
    """Adapter state-dict key prefix for a metadata tensor name."""
    if tensor_name.endswith(_PARAM_SUFFIX):
        return tensor_name[: -len(_PARAM_SUFFIX)]
    return tensor_name


def _component_shape(cm: dict, K: int) -> int:
    """Component N from the packed blob length (components carry no
    dense_shape; the parent tensor does)."""
    return int(cm["packed_len_bytes"]) * (8 // int(cm["bitwidth"])) // K


def _adapter_scale(cfg, module_path, component=None):
    """LoRA scale for one module (or SplitQKV component) that HAS adapter
tensors in the state dict.

Adapter dirs carry per-module geometry recorded by attach_qlora: plain
modules at cfg.tensors[path]["scale"], SplitQKV components at
cfg.tensors[path]["components"]["q"/"k"/"v"]["scale"] (mixed-rank QKVs
have no top-level scalar — only the components do). Dirs whose entries
carry no "scale" key fall back to the global cfg.alpha / cfg.r.

A scale of 0.0 here means the merge would silently produce a no-op;
r=0 modules never reach this function (they have no lora_A/lora_B
keys), so raise instead."""
    entry = (cfg.tensors or {}).get(module_path)
    scale = None
    if isinstance(entry, dict):
        holder = entry
        if component is not None:
            comps = entry.get("components")
            holder = comps.get(component) if isinstance(comps, dict) else None
        if isinstance(holder, dict) and "scale" in holder:
            scale = float(holder["scale"])
    if scale is None:
        scale = cfg.alpha / cfg.r  # old dir: no per-module scale recorded
    if scale == 0.0:
        where = module_path if component is None \
            else f"{module_path}.{component}"
        raise RuntimeError(
            f"merge_qlora: {where}: LoRA scale resolved to 0.0 although "
            f"lora_A/lora_B exist — the merge would silently no-op. "
            f"Refusing (r=0 modules carry no adapter keys and never reach "
            f"the merge; check the adapter dir's qlora_config.json)")
    return scale


def _pinned_awq_scales(artifacts_dir):
    """{consumer tensor name: (K,) s} from the norm doc's PINNED scale
    records (W14 export entries carry awq_scale_file). Entries without
    a pin are absent from the map (their diff-based recovery needs the
    pristine model, which the merge does not have — the callers refuse
    loudly when a fold tensor actually needs the frame transform)."""
    path = os.path.join(artifacts_dir, "norm_gain_edits.json")
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        doc = json.load(f)
    out = {}
    for pname, entry in (doc.get("edits") or {}).items():
        rec = entry.get("awq_scale_file")
        if not rec:
            continue
        spath = os.path.join(artifacts_dir, str(rec))
        s = torch.from_numpy(
            np.ascontiguousarray(np.load(spath))).float().reshape(-1)
        for consumer in entry.get("consumers") or []:
            out[str(consumer)] = s
    return out


def _tensor_fold_frame(pm, pinned):
    """(signs, s, fold_order) of one tensor meta, for the Gram frame
    transform. `signs` is None when the tensor is unrotated; `s` is None
    when the fold carries no AWQ scale OR the pin is missing (the caller
    decides whether that is fatal)."""
    rot = pm.get("rotation") or {}
    if not rot:
        return None, None, None
    seed, k = int(rot["seed"]), int(rot["k"])
    signs = pmod._rotation_signs_for(k, seed)
    s = pinned.get(pm.get("var") or pm.get("name"))
    fold_order = (pm.get("awq") or {}).get("fold_order") \
        or pm.get("fold_order")
    if fold_order not in ("rotate_then_awq", "awq_then_rotate"):
        fold_order = "rotate_then_awq"    # legacy default (W13 loader rule)
    return signs, s, fold_order


def _materialize_weight(module, lora_A=None, lora_B=None, scale=0.0):
    W = pmod.reference_dequant(module.indices, module.lut, module.N, module.K,
                                module.group_size, module.bitwidth).float().cpu()
    if module.resA is not None and module.resB is not None:
        W = W + (module.resA.float().cpu() @ module.resB.float().cpu())
    if lora_A is not None and lora_B is not None:
        W = W + scale * (lora_B.float().cpu() @ lora_A.float().cpu())
    return W.contiguous()


def _resolve_gram(pm: dict, artifacts_dir: str, san: str) -> Optional[str]:
    """Path of the persisted calibration Gram for one tensor, if any."""
    rel = pm.get("gram_file")
    if rel:
        path = os.path.join(artifacts_dir, rel)
        if os.path.exists(path):
            return path
    fallback = os.path.join(artifacts_dir, "grams", f"{san}.gram.npy")
    return fallback if os.path.exists(fallback) else None


def _repalettize(W: torch.Tensor, gs: int, bw: int, assign: str,
                 gram_path: Optional[str], ctx: str, frame=None):
    """Pack merged weights back into idx4 + LUT with the chosen engine.

    W14 (frame): `frame` = (signs, s, fold_order) of the tensor's fold.
    The persisted Gram is the PRISTINE-input (pre-fold) Gram while the
    re-palettization weight is FOLD-space, so the Gram rides the fold
    congruence (pmod.fold_input_gram) before any engine — Lloyd's
    diag(H) column weights included — consumes it."""
    import palettize_qwen3_5_9b as palettizer
    dev = "cuda:0" if torch.cuda.is_available() else "cpu"
    H = None
    if gram_path is not None:
        H = torch.from_numpy(np.load(gram_path).astype(np.float32))
        signs, s, fold_order = frame if frame is not None \
            else (None, None, None)
        if signs is not None:
            H = pmod.fold_input_gram(
                signs, s, fold_order, H.cpu())
        H = H.to(dev)
    if assign == "lloyd":
        h = torch.diag(H) if H is not None else torch.ones(W.shape[1], dtype=torch.float32)
        idx, lut, _ = palettizer.palettize_groups_gpu(
            W.to(dev), h.to(dev), bw, gs, device=dev)
    else:
        if H is None:
            raise ValueError(
                f"{ctx}: --assign {assign} requires the calibration Gram "
                f"(grams/<name>.gram.npy; palettize with --persist-grams or "
                f"re-capture) — refusing to silently degrade to unweighted "
                f"Lloyd")
        h = torch.diag(H)
        idx, lut, _, _ = palettizer.palettize_assign(
            W.to(dev), h, bw, gs, assign=assign, H=H, device=dev)
    blob = palettizer._pack_idx4_blob(idx.detach().cpu().numpy())
    lut_np = lut.detach().reshape(-1).to(torch.float16).cpu().numpy()
    return blob, lut_np


def _merge_module(module, name, lora_A, lora_B, scale, out_dir, verbose,
                  assign, gram_path, frame=None):
    t0 = time.time()
    if verbose:
        print(f"  [merge] {name} ({module.N},{module.K}) gs={module.group_size}",
              flush=True)
    W = _materialize_weight(module, lora_A, lora_B, scale)
    blob, lut_np = _repalettize(W, module.group_size, module.bitwidth,
                                assign, gram_path, ctx=name, frame=frame)
    blob.tofile(os.path.join(out_dir, f"{name}.idx4"))
    lut_np.tofile(os.path.join(out_dir, f"{name}.lut_scalar"))
    if verbose:
        print(f"    done ({time.time()-t0:.1f}s)", flush=True)


def _copy_no_lora_tensor(pm, artifacts_dir, out_dir):
    """Copy one r=0/adapter-free tensor's artifact set (idx + lut + the
    residual pair) into the merge output, record intact."""
    targets = [pm] if "components" not in pm else \
        [pm["components"][c] for c in ("Q", "K", "V")]
    for entry in targets:
        files = [entry.get("index_file"), entry.get("lut_file")]
        res = entry.get("residual") or {}
        files += [res.get("resA_file"), res.get("resB_file")]
        for fn in files:
            if not fn:
                continue
            src = os.path.join(artifacts_dir, fn)
            if os.path.exists(src):
                shutil.copy2(src, os.path.join(out_dir, fn))


def _strip_residual_records(pm):
    """The merged entry with every residual record dropped (the factors
    were ABSORBED into the re-palettized weight — a stale record would
    point at files the merge never copies and double-apply on reload)."""
    out = dict(pm)
    if "residual" in out:
        out.pop("residual", None)
    if isinstance(out.get("components"), dict):
        comps = {}
        for c, cm in out["components"].items():
            cm2 = dict(cm)
            cm2.pop("residual", None)
            comps[c] = cm2
        out["components"] = comps
    return out


def merge_qlora(artifacts_dir, adapters_dir, out_dir, verbose=True,
                assign="gptvq"):
    t0 = time.time()
    print(f"  [merge] {artifacts_dir} + {adapters_dir} -> {out_dir}", flush=True)
    print(f"  [merge] engine: {assign}", flush=True)
    metadata = pmod.load_metadata(artifacts_dir)
    cfg = qlora.QLoRAConfig.from_json(os.path.join(adapters_dir, "qlora_config.json"))
    sd = torch.load(os.path.join(adapters_dir, "qlora_adapters.pt"), map_location="cpu")
    os.makedirs(out_dir, exist_ok=True)
    for fn in os.listdir(artifacts_dir):
        src, dst = os.path.join(artifacts_dir, fn), os.path.join(out_dir, fn)
        if os.path.isdir(src):
            if not os.path.exists(dst): shutil.copytree(src, dst)
        elif fn.endswith((".idx4", ".lut_scalar", ".resA", ".resB")) or fn == "metadata.json":
            # idx/lut/resA/resB are copied PER-TENSOR below: merged
            # tensors rewrite idx+lut and ABSORB their residual (the
            # records are stripped — a top-level copy would leave
            # orphans); adapter-free tensors copy their whole set
            # (_copy_no_lora_tensor, residual pair included — the
            # pre-W14 merge dropped the files and kept the record, a
            # load-time hard failure).
            continue
        else: shutil.copy2(src, dst)
    pinned = _pinned_awq_scales(artifacts_dir)
    n_m = n_s = 0; new_meta = {}
    for name, pm in metadata["tensors"].items():
        san = name.replace(".", "_").replace("/", "_")
        base = _module_path(name)
        is_qkv = "components" in pm
        has_lora = False
        if is_qkv:
            for c in ("q", "k", "v"):
                if f"{base}.{c}.lora_A" in sd: has_lora = True; break
        else:
            has_lora = f"{base}.lora_A" in sd
        if not has_lora:
            _copy_no_lora_tensor(pm, artifacts_dir, out_dir)
            new_meta[name] = pm; n_s += 1; continue
        gram_path = _resolve_gram(pm, artifacts_dir, san)
        # W14: the fold frame for the re-palettization Gram. A rotated
        # tensor whose fold carries an AWQ scale needs the s vector for
        # the congruence; the merge has no pristine model to diff
        # against, so the pin (the export's awq_scale_file) is the ONLY
        # source — without it the calibrated engines (and Lloyd's
        # diag(H) weights) would mis-weight the fold columns silently.
        frame = _tensor_fold_frame(pm, pinned)
        _signs, _s, _fold = frame
        awq_applied = bool((pm.get("awq") or {}).get("applied"))
        if gram_path is not None and _signs is not None \
                and awq_applied and _s is None:
            raise RuntimeError(
                f"merge_qlora: {name}: the fold's Gram frame needs the AWQ "
                f"scale, but norm_gain_edits.json carries no pinned "
                f"awq_scale_file for it — re-run the trainer's export "
                f"(W14+ pins awq_scales/<norm>.npy alongside the trained "
                f"gains) or merge with --assign lloyd and no grams. "
                f"Refusing to silently mis-weight the fold columns.")
        if is_qkv:
            K = pm["dense_shape"][1]
            for cn in ("Q", "K", "V"):
                cm = pm["components"][cn]
                comp_N = _component_shape(cm, K)
                idx, lut_t, bw, gs, N_, K_ = pmod._read_indices_and_lut(
                    cm, artifacts_dir, ctx=f"{name}:{cn}", shape=(comp_N, K))
                resA, resB = pmod._read_residual(
                    cm, artifacts_dir, f"{name}:{cn}", N_, K_)
                module = pmod.PalettizedLinear(
                    idx, lut_t, bw, gs, N_, K_, reference=True,
                    resA=resA, resB=resB)
                a = sd.get(f"{base}.{cn.lower()}.lora_A"); b = sd.get(f"{base}.{cn.lower()}.lora_B")
                sc = _adapter_scale(cfg, base, cn.lower()) \
                    if (a is not None and b is not None) else 0.0
                _merge_module(module, f"{san}_{cn}", a, b, sc, out_dir, verbose,
                              assign, gram_path, frame=frame)
        else:
            idx, lut_t, bw, gs, N_, K_ = pmod._read_indices_and_lut(pm, artifacts_dir, ctx=name)
            # W14: the residual is part of the deployed weight — read it
            # here so _materialize_weight ABSORBS it (the pre-W14 merge
            # built a bare module and silently dropped the residual while
            # the copied metadata still pointed at the uncopied files)
            resA, resB = pmod._read_residual(pm, artifacts_dir, name, N_, K_)
            module = pmod.PalettizedLinear(
                idx, lut_t, bw, gs, N_, K_, reference=True,
                resA=resA, resB=resB)
            a = sd.get(f"{base}.lora_A"); b = sd.get(f"{base}.lora_B")
            sc = _adapter_scale(cfg, base) \
                if (a is not None and b is not None) else 0.0
            _merge_module(module, san, a, b, sc, out_dir, verbose,
                          assign, gram_path, frame=frame)
        new_meta[name] = _strip_residual_records(pm); n_m += 1
    for name, pm in new_meta.items():
        targets = [(pm["components"][c], f"{name}:{c}") for c in ("Q", "K", "V")] if "components" in pm else [(pm, name)]
        for entry, _ in targets:
            for ek in ("index_file", "lut_file"):
                p = os.path.join(out_dir, entry[ek])
                if os.path.exists(p):
                    with open(p, "rb") as f:
                        entry["sha256_" + ("idx" if ek == "index_file" else "lut")] = hashlib.sha256(f.read()).hexdigest()
    metadata["tensors"] = new_meta
    with open(os.path.join(out_dir, "metadata.json"), "w") as f: json.dump(metadata, f, indent=2)
    print(f"  [merge] done. {n_m} merged, {n_s} copied, {time.time()-t0:.0f}s.", flush=True)
    if n_m == 0 and n_s > 0 and sd:
        raise RuntimeError(
            "merge_qlora merged nothing although adapters were loaded — "
            "adapter/metadata key mismatch (T6 regression)")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--artifacts-dir", required=True)
    p.add_argument("--adapters-dir", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--assign", choices=["lloyd", "gptq", "gptvq"], default="gptvq",
                   help="re-palettization engine (default gptvq); gptq/gptvq "
                        "require the persisted calibration Gram")
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args()
    merge_qlora(args.artifacts_dir, args.adapters_dir, args.output_dir,
                not args.quiet, assign=args.assign)

if __name__ == "__main__": main()
