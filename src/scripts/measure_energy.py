#!/usr/bin/env python3
"""
measure_energy.py — dense-vs-palettized energy and performance harness
(strict measurement protocol).

Measurement-protocol rules:
  *   real decode: prefill with use_cache=True + autoregressive KV-cache
      decode (a fixed 1-token zeros forward with use_cache=False measures
      host dispatch, not decode)
  *   analytic HBM estimator includes the palettized branch; both branches
      labeled as upper bounds
  *   per-component QKV N derived from metadata packed_len_bytes (not N//3)
  *   NVML power sampling with timestamps (fallback: subprocess), energy via
      the NVML cumulative energy counter when available (exact integration)
  *   optional clock locking, idle-baseline subtraction, per-run timing -> CIs,
      randomized ratio order, dense/palettized order flip via --rounds
  *   --dense-dtype fp16|bf16 (dtype-symmetric comparisons)
  *   real prompts (fineweb-edu streaming or --prompts-source file:PATH)
  *   indices_layout="idx4" wired end-to-end, flat blob, extension + SHA256
      verification, loud refusal of legacy-era artifacts
  *   indices/LUT registered as buffers -> size accounting includes them;
      dense-remainder decomposition printed at load time

Usage:
  python scripts/measure_energy.py                           # full defaults
  python scripts/measure_energy.py --lock-clocks 1530  # pin SM clock (sudo)
  python scripts/measure_energy.py --rounds 2        # flip A/B order, detect drift
  python scripts/measure_energy.py --dense-dtype fp16  # symmetric dtype A/B
  python scripts/measure_energy.py --batch-sizes 1 --ratios 20:80 --num-runs 30
"""

import argparse
import atexit
import hashlib
import json
import os
import random
import subprocess
import sys
import threading
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="Qwen/Qwen3.5-9B")
    p.add_argument("--palettized-dir", required=True,
                   help="the palettized artifacts dir (usage example: "
                        "/home/ubuntu/qwen3_5_9b_palettized)")
    p.add_argument("--flute-dir", default=None,
                   help="path to the flute_extended package "
                        "(default: ../flute_extended relative to this script, "
                        "then /home/ubuntu/flute_extended)")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--dense-dtype", choices=["bf16", "fp16"], default="bf16",
                   help="dtype for the dense baseline (fp16 gives a "
                        "dtype-symmetric comparison against the FLUTE kernel)")
    p.add_argument("--batch-sizes", default="1,2,4,8,16")
    p.add_argument("--ratios", default="20:80,40:60,60:40,80:20",
                   help="prefill:decode percent pairs")
    p.add_argument("--total-tokens", type=int, default=512)
    p.add_argument("--warmup-runs", type=int, default=3)
    p.add_argument("--num-runs", type=int, default=10)
    p.add_argument("--rounds", type=int, default=1,
                   help="1 = dense-then-palettized (fixed order); 2 adds a "
                        "second pass in flipped order to detect thermal/dvfs drift")
    p.add_argument("--idle-seconds", type=float, default=60.0)
    p.add_argument("--lock-clocks", type=int, default=0, metavar="SM_MHZ",
                   help="lock SM clock (e.g. 1530 on A10G); requires root; "
                        "memory clock pinned at its max")
    p.add_argument("--residual", action="store_true",
                   help="attach the whitened-SVD residual branch when the "
                        "artifacts carry one (measured A/B configuration)")
    p.add_argument("--verify-sha", action="store_true", default=True)
    p.add_argument("--no-verify-sha", dest="verify_sha", action="store_false")
    p.add_argument("--prompts-source", default="fineweb",
                   help="'fineweb' (streaming), or 'file:/path/one-per-line.txt'")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--outdir", required=True,
                   help="measurement output dir (usage example: "
                        "/home/ubuntu/energy_measurements)")
    p.add_argument("--power-sample-ms", type=float, default=5.0)
    return p.parse_args()


# Configuration, populated by main() through _init_config(). Importing
# this module must not parse argv, mutate torch global state, create
# directories, or require the built CUDA extension - all of that is
# main-time work. The kernel package import happens lazily inside
# palettized_modules when its kernel path is first taken.
ARGS = None
DEVICE = None
DENSE_DTYPE = None
BATCH_SIZES = None
PREFILL_DECODE_RATIOS = None
TOTAL_TOKENS = None
WARMUP_RUNS = None
NUM_RUNS = None
PALETTIZED_DIR = None


def _init_config(args):
    """Populate the module configuration (main only): derived constants,
    the flute_extended package path, the output directory, and the
    measurement protocol's torch settings."""
    global ARGS, DEVICE, DENSE_DTYPE, BATCH_SIZES, PREFILL_DECODE_RATIOS
    global TOTAL_TOKENS, WARMUP_RUNS, NUM_RUNS, PALETTIZED_DIR
    ARGS = args
    DEVICE = args.device
    DENSE_DTYPE = {"bf16": torch.bfloat16, "fp16": torch.float16}[args.dense_dtype]
    BATCH_SIZES = [int(x) for x in args.batch_sizes.split(",")]
    PREFILL_DECODE_RATIOS = [tuple(int(y) for y in r.split(":"))
                             for r in args.ratios.split(",")]
    TOTAL_TOKENS = args.total_tokens
    WARMUP_RUNS = args.warmup_runs
    NUM_RUNS = args.num_runs
    PALETTIZED_DIR = args.palettized_dir
    # FLUTE kernel package: checkout-relative first, deployment fallback
    if args.flute_dir is None:
        _here = os.path.dirname(os.path.abspath(__file__))
        for _cand in (os.path.join(_here, "..", "flute_extended"),
                      "/home/ubuntu/flute_extended"):
            if os.path.isdir(os.path.join(_cand, "flute_extended")):
                args.flute_dir = _cand
                break
    sys.path.insert(0, args.flute_dir)
    os.makedirs(args.outdir, exist_ok=True)
    torch.backends.cudnn.enabled = False


# Shared module layer: PalettizedLinear / SplitQKV /
# loaders live in palettized_modules.py, imported by the capture script,
# the evaluators, and this harness — one implementation, no drift.
import palettized_modules as pmod  # noqa: E402
from palettized_modules import (  # noqa: E402,F401
    PalettizedLinear, SplitQKV, validate_indices_layout,
    load_palettized_weight, replace_linear_with_palettized)
from eval_common import load_dense_fp16  # noqa: E402

from transformers import AutoConfig, AutoTokenizer  # noqa: E402

# --------------------------------------------------------------------------- #
# Power monitoring: NVML first, subprocess fallback; timestamps; energy
# counter when available (exact integration).
# --------------------------------------------------------------------------- #

class PowerMonitor:
    def __init__(self, device_id: int = 0, sample_interval_ms: float = 5.0):
        self.device_id = device_id
        self.sample_interval = sample_interval_ms / 1000.0
        self.samples: List[Tuple[float, float]] = []   # (t_s, watts)
        self.monitoring = False
        self._thread: Optional[threading.Thread] = None
        self.backend = "none"
        self._handle = None
        self.energy_counter_available = False
        try:
            import pynvml  # type: ignore
            pynvml.nvmlInit()
            self._pynvml = pynvml
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(device_id)
            try:
                pynvml.nvmlDeviceGetTotalEnergyConsumption(self._handle)
                self.energy_counter_available = True
            except Exception:
                pass
            self.backend = "pynvml"
        except Exception as e:  # noqa: BLE001
            print(f"[power] pynvml unavailable ({e}); falling back to "
                  f"nvidia-smi subprocess polling (~25 Hz effective)", flush=True)
            self._pynvml = None
            self.backend = "subprocess"

    # -- sampling thread ---------------------------------------------------- #
    def _read_power_w(self) -> float:
        if self.backend == "pynvml":
            return self._pynvml.nvmlDeviceGetPowerUsage(self._handle) / 1000.0
        out = subprocess.run(
            f"nvidia-smi --id={self.device_id} --query-gpu=power.draw "
            f"--format=csv,noheader,nounits",
            shell=True, capture_output=True, text=True, timeout=2).stdout
        return float(out.strip())

    def _monitor_loop(self):
        while self.monitoring:
            try:
                self.samples.append((time.perf_counter(), self._read_power_w()))
            except Exception:  # noqa: BLE001
                pass
            time.sleep(self.sample_interval)

    def start(self):
        self.samples = []
        self.monitoring = True
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()

    def stop(self) -> Dict:
        self.monitoring = False
        if self._thread:
            self._thread.join(timeout=5)
        n = len(self.samples)
        span = (self.samples[-1][0] - self.samples[0][0]) if n > 1 else 0.0
        return {
            "n_samples": n,
            "effective_hz": (n / span) if span > 0 else None,
            "mean_w": float(np.mean([w for _, w in self.samples])) if n else None,
            "samples": self.samples,
        }

    # -- exact energy (preferred) -------------------------------------------- #
    def read_energy_counter_mj(self) -> Optional[float]:
        """NVML cumulative energy counter in mJ (Volta+), or None."""
        if not (self.backend == "pynvml" and self.energy_counter_available):
            return None
        return self._pynvml.nvmlDeviceGetTotalEnergyConsumption(self._handle)

    @staticmethod
    def trapezoid_energy_j(samples: List[Tuple[float, float]],
                           t0: float, t1: float) -> Optional[float]:
        """Integrate timestamped power over [t0, t1] (fallback path)."""
        pts = [(t, w) for (t, w) in samples if t0 - 1e-6 <= t <= t1 + 1e-6]
        if len(pts) < 2:
            return None
        if pts[0][0] > t0:
            pts.insert(0, (t0, pts[0][1]))
        if pts[-1][0] < t1:
            pts.append((t1, pts[-1][1]))
        ts = np.array([p[0] for p in pts])
        ws = np.array([p[1] for p in pts])
        trapz = getattr(np, "trapezoid", None) or np.trapz
        return float(trapz(ws, ts))


# --------------------------------------------------------------------------- #
# Clock control
# --------------------------------------------------------------------------- #

_CLOCKS_LOCKED = False


def _smi(*flags: str) -> subprocess.CompletedProcess:
    dev = ARGS.device.split(":")[-1]
    return subprocess.run(["nvidia-smi", f"--id={int(dev)}", *flags],
                          capture_output=True, text=True)


def lock_clocks(sm_mhz: int) -> bool:
    global _CLOCKS_LOCKED
    dev = ARGS.device.split(":")[-1]
    mem_max = subprocess.run(
        f"nvidia-smi --id={dev} --query-gpu=clocks.max.mem "
        f"--format=csv,noheader,nounits",
        shell=True, capture_output=True, text=True).stdout.strip()
    r1 = _smi("-lgc", str(sm_mhz))
    r2 = _smi("-lmc", mem_max) if mem_max else None
    ok = r1.returncode == 0 and (r2 is None or r2.returncode == 0)
    if ok:
        _CLOCKS_LOCKED = True
        print(f"[clocks] locked sm={sm_mhz} MHz, mem={mem_max} MHz "
              f"(reset on exit)", flush=True)
    else:
        print(f"[clocks] WARNING: could not lock clocks. Run with sudo, or "
              f"use tools/lock_clocks.sh. Results will carry DVFS noise.",
              flush=True)
    return ok


def reset_clocks():
    global _CLOCKS_LOCKED
    if _CLOCKS_LOCKED:
        _smi("-rgc"); _smi("-rmc")
        _CLOCKS_LOCKED = False
        print("[clocks] reset to defaults", flush=True)


atexit.register(reset_clocks)


def gpu_status() -> Dict:
    dev = ARGS.device.split(":")[-1]
    out = subprocess.run(
        f"nvidia-smi --id={dev} --query-gpu=name,clocks.sm,power.limit,"
        f"temperature.gpu,clocks_throttle_reasons.active "
        f"--format=csv,noheader", shell=True, capture_output=True, text=True)
    parts = [x.strip() for x in out.stdout.split(",")]
    return {"gpu": parts[0] if parts else "?",
            "sm_clock_mhz": parts[1] if len(parts) > 1 else "?",
            "power_limit_w": parts[2] if len(parts) > 2 else "?",
            "temp_c": parts[3] if len(parts) > 3 else "?",
            "throttle": parts[4] if len(parts) > 4 else "?"}

# --------------------------------------------------------------------------- #
# Real prompts
# --------------------------------------------------------------------------- #

_FALLBACK_PROMPTS = [
    "The history of computing hardware is marked by alternating waves of "
    "specialization and generalization: early machines were built for a single "
    "purpose, then general-purpose processors dominated for decades, and now "
    "accelerators are pulling specialized workloads back out of the CPU.",
    "Photosynthesis converts light energy into chemical energy stored in "
    "glucose, a process that underpins nearly every food chain on Earth and "
    "accounts for the oxygen in today's atmosphere.",
    "In financial markets, liquidity is a measure of how quickly an asset can "
    "be bought or sold without moving its price, and it can evaporate exactly "
    "when it is most needed, which is why regulators monitor it closely.",
    "The Silk Road was a network of trade routes connecting East and West for "
    "over fifteen centuries, carrying silk, spices, paper, and eventually "
    "ideas, religions, and diseases across continents.",
    "Machine learning models are only as good as the data they learn from, "
    "and subtle biases in collection or labeling can propagate into "
    "decisions that affect millions of people in invisible ways.",
    "Volcanic eruptions inject sulfate aerosols into the stratosphere, where "
    "they reflect sunlight and can cool global temperatures by a fraction of "
    "a degree for a year or two after a major event.",
    "The rules of chess were standardized only in the nineteenth century, "
    "long after the game had spread from India through Persia to Europe, "
    "evolving piece movements along the way.",
    "Urban planning debates often reduce to a tension between density and "
    "liveability, yet the evidence suggests well-designed density reduces "
    "traffic, energy use, and cost of services simultaneously.",
    "Deep-sea hydrothermal vents host ecosystems that thrive without "
    "sunlight, relying on chemosynthesis by bacteria that oxidize hydrogen "
    "sulfide released from the vents.",
    "The invention of movable type in Korea preceded Gutenberg by decades, "
    "but the alphabetic structure of European languages made the technology "
    "dramatically cheaper to apply there.",
    "Antibiotic resistance develops faster than new antibiotics are "
    "discovered, which makes stewardship of existing drugs a public health "
    "priority comparable to developing new ones.",
    "Ocean currents redistribute heat from the tropics toward the poles, and "
    "small changes in their strength can shift regional climates by "
    "centimeters of rainfall per year.",
    "A compiler's optimizer walks a line between doing what the programmer "
    "wrote and doing something observably equivalent but faster, and every "
    "aggressive optimization is a bet that equivalence holds.",
    "The Antikythera mechanism, a geared bronze device recovered from a "
    "shipwreck, computed astronomical positions two thousand years before "
    "anything comparable appears in the historical record.",
    "Sleep is not a uniform state but a structured cycle of stages, each "
    "with distinct neural signatures and apparently distinct functions for "
    "memory consolidation and metabolic maintenance.",
    "Grain silos fail in peculiar ways because stored grain behaves partly "
    "like a fluid and partly like a solid, a duality that still challenges "
    "structural engineers.",
]


def load_prompt_pool(n_needed: int, min_chars: int = 200) -> List[str]:
    src = ARGS.prompts_source
    pool: List[str] = []
    try:
        if src.startswith("file:"):
            with open(src[5:]) as f:
                pool = [ln.strip() for ln in f if len(ln.strip()) >= min_chars]
        elif src == "fineweb":
            from datasets import load_dataset
            ds = iter(load_dataset("HuggingFaceFW/fineweb-edu",
                                   name="sample-10BT", split="train",
                                   streaming=True))
            while len(pool) < n_needed:
                try:
                    t = next(ds)["text"]
                except StopIteration:
                    break
                if len(t) >= min_chars:
                    pool.append(t)
        else:
            raise ValueError(f"unknown --prompts-source {src}")
    except Exception as e:  # noqa: BLE001
        print(f"[prompts] {src} unavailable ({type(e).__name__}: {e}); "
              f"using built-in diverse prompt pool", flush=True)
    if len(pool) < n_needed:
        reps = (n_needed - len(pool)) // len(_FALLBACK_PROMPTS) + 1
        pool = pool + (_FALLBACK_PROMPTS * reps)
    rng = random.Random(ARGS.seed)
    rng.shuffle(pool)
    return pool[:n_needed]

class VRAMMonitor:
    """nvidia-smi VRAM + torch allocator peaks."""

    def get_memory_usage(self) -> Dict[str, float]:
        dev = ARGS.device.split(":")[-1]
        out = subprocess.run(
            f"nvidia-smi --id={dev} "
            f"--query-gpu=memory.used,memory.free,memory.total "
            f"--format=csv,noheader,nounits",
            shell=True, capture_output=True, text=True).stdout
        try:
            used, free, total = [float(x.strip()) for x in out.split(",")]
            return {"used_gb": used / 1024, "free_gb": free / 1024,
                    "total_gb": total / 1024}
        except Exception:  # noqa: BLE001
            return {"used_gb": 0.0, "free_gb": 0.0, "total_gb": 0.0}

    @staticmethod
    def get_model_size(model) -> Dict[str, float]:
        param = sum(p.numel() * p.element_size() for p in model.parameters())
        buf = sum(b.numel() * b.element_size() for b in model.buffers())
        return {"param_gb": param / 1024**3, "buffer_gb": buf / 1024**3,
                "total_gb": (param + buf) / 1024**3}


def classify_dense_remainder(state_dict_keys) -> Dict[str, int]:
    """Byte-classify the dense weights loaded for the palettized model."""
    classes = {"embed": 0, "lm_head": 0, "norm": 0, "other": 0}
    for name, t in state_dict_keys.items():
        b = t.numel() * t.element_size()
        if "embed_tokens" in name:
            classes["embed"] += b
        elif "lm_head" in name:
            classes["lm_head"] += b
        elif "norm" in name or ".bias" in name:
            classes["norm"] += b
        else:
            classes["other"] += b
    return {k: round(v / 1024**3, 3) for k, v in classes.items()}


def load_dense_model():
    print("Loading dense model...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(ARGS.model, trust_remote_code=True)
    mem_before = VRAMMonitor().get_memory_usage()
    torch.cuda.reset_peak_memory_stats()
    model = load_dense_fp16(ARGS.model, DEVICE, dtype=DENSE_DTYPE)
    mem_after = VRAMMonitor().get_memory_usage()
    sizes = VRAMMonitor.get_model_size(model)
    print(f"  dense dtype={ARGS.dense_dtype} size={sizes['total_gb']:.2f} GB "
          f"(params {sizes['param_gb']:.2f} + buffers {sizes['buffer_gb']:.2f})"
          f"  VRAM delta={mem_after['used_gb']-mem_before['used_gb']:.2f} GB "
          f"torch-peak={torch.cuda.max_memory_allocated()/1024**3:.2f} GB",
          flush=True)
    return model, tokenizer, {
        **sizes,
        "vram_allocated_gb": mem_after["used_gb"] - mem_before["used_gb"],
        "torch_peak_gb": torch.cuda.max_memory_allocated() / 1024**3,
    }


def load_palettized_model():
    print("Loading palettized model (FLUTE, layout-aware)...", flush=True)
    with open(os.path.join(PALETTIZED_DIR, "metadata.json")) as f:
        metadata = json.load(f)

    # --- artifact audit printout ------------------------------------------ #
    n_single = sum(1 for m in metadata["tensors"].values()
                   if "components" not in m)
    n_qkv = len(metadata["tensors"]) - n_single
    print(f"  metadata: {len(metadata['tensors'])} tensors "
          f"({n_single} single, {n_qkv} QKV-split x3)", flush=True)
    idx4_files = [fn for fn in os.listdir(PALETTIZED_DIR)
                  if fn.endswith(".idx4")]
    stale_files = [fn for fn in os.listdir(PALETTIZED_DIR)
                   if fn.endswith(".fd") or fn.endswith(".idx2")]
    print(f"  on-disk .idx4 artifacts: {len(idx4_files)}"
          f"{'; WARNING stale legacy files present: ' + str(len(stale_files)) if stale_files else ''}",
          flush=True)

    tokenizer = AutoTokenizer.from_pretrained(ARGS.model, trust_remote_code=True)

    # dense-vs-palettized byte accounting
    dense_weight_size = 0
    palettized_weight_size = 0
    for m in metadata["tensors"].values():
        N, K = m["dense_shape"]
        dense_weight_size += N * K * 2
        metas = (list(m["components"].values())
                 if "components" in m else [m])
        for cm in metas:
            lut_bytes = cm.get("n_groups", 0) * (1 << cm["bitwidth"]) * 2 \
                if "n_groups" in cm else os.path.getsize(
                    os.path.join(PALETTIZED_DIR, cm["lut_file"]))
            palettized_weight_size += cm["packed_len_bytes"] + lut_bytes

    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file as load_safetensors
    import gc

    config = AutoConfig.from_pretrained(ARGS.model, trust_remote_code=True)
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)

    index_path = hf_hub_download(ARGS.model, "model.safetensors.index.json")
    with open(index_path) as f:
        weight_map = json.load(f)["weight_map"]

    # NOTE: the filter below excludes a weight only if it is a *matrix*
    # (endswith '.weight') — biases of palettized linears are still loaded
    # so the replacement modules can adopt them (dropping them would leave
    # uninitialized meta-tensor garbage and lose the QKV biases).
    pat = ["in_proj_qkv", "in_proj_z", "out_proj", "q_proj", "k_proj",
           "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
           "in_proj_a", "in_proj_b"]
    weights_to_load = {}
    for weight_name, file_name in weight_map.items():
        if not (weight_name.endswith(".weight")
                and any(t in weight_name for t in pat)):
            weights_to_load[weight_name] = file_name

    state_dict = {}
    for file_name in set(weights_to_load.values()):
        file_state = load_safetensors(hf_hub_download(ARGS.model, file_name))
        for name, tensor in file_state.items():
            if name in weights_to_load:
                state_dict[name] = tensor.to(DENSE_DTYPE)
        del file_state

    # H10: decompose the dense remainder
    decomp = classify_dense_remainder(state_dict)
    print(f"  dense remainder loaded from checkpoint: {decomp} "
          f"(GB by class: embed/lm_head/norm/other)", flush=True)

    model = model.to_empty(device="cpu")
    model.load_state_dict(state_dict, strict=False)
    del state_dict
    gc.collect()

    pmod.LAYOUT_AUDIT.update(idx4=0)
    mem_before = VRAMMonitor().get_memory_usage()   # before the .to(DEVICE) move
    torch.cuda.reset_peak_memory_stats()
    model = replace_linear_with_palettized(
        model, metadata, PALETTIZED_DIR,
        residual=ARGS.residual, verify_sha=ARGS.verify_sha)
    model = model.to(DEVICE)
    model.eval()

    print(f"  layout audit: {pmod.LAYOUT_AUDIT['idx4']} tensors loaded via the idx4 "
          f"path (any mismatch would have raised)", flush=True)

    sizes = VRAMMonitor.get_model_size(model)
    compression = dense_weight_size / max(palettized_weight_size, 1)
    print(f"  weights: dense {dense_weight_size/1024**3:.2f} GB -> "
          f"palettized {palettized_weight_size/1024**3:.2f} GB "
          f"({compression:.4f}x over palettized tensors)", flush=True)
    print(f"  model size (params+buffers, honest): {sizes['total_gb']:.2f} GB "
          f"(params {sizes['param_gb']:.2f} + idx/LUT buffers "
          f"{sizes['buffer_gb']:.2f})", flush=True)
    return model, tokenizer, {
        **sizes,
        "dense_weight_size_gb": dense_weight_size / 1024**3,
        "palettized_weight_size_gb": palettized_weight_size / 1024**3,
        "compression_ratio": compression,
        "dense_remainder_by_class_gb": decomp,
        "layout_summary": dict(pmod.LAYOUT_AUDIT),
    }

# --------------------------------------------------------------------------- #
# Analytic HBM estimator — an upper bound, labeled as such
# --------------------------------------------------------------------------- #

def estimate_hbm_traffic(metadata, batch_size, seq_length, num_iters) -> Dict:
    M = batch_size * seq_length
    dense = {"weight_read_gb": 0.0, "activation_read_gb": 0.0,
             "output_write_gb": 0.0}
    palet = {"indices_read_gb": 0.0, "lut_read_gb": 0.0,
             "activation_read_gb": 0.0, "output_write_gb": 0.0}

    def add_dense(N, K):
        dense["weight_read_gb"] += N * K * 2 / 1024**3
        dense["activation_read_gb"] += M * K * 2 / 1024**3
        dense["output_write_gb"] += M * N * 2 / 1024**3

    def add_palet(N, K, packed_bytes, lut_bytes):
        palet["indices_read_gb"] += packed_bytes / 1024**3
        palet["lut_read_gb"] += lut_bytes / 1024**3
        palet["activation_read_gb"] += M * K * 2 / 1024**3
        palet["output_write_gb"] += M * N * 2 / 1024**3

    for name, m in metadata["tensors"].items():
        K = m["dense_shape"][1]
        if "components" in m:
            for comp_name, cm in m["components"].items():
                bw = cm["bitwidth"]
                comp_N = cm["packed_len_bytes"] * (8 // bw) // K  # H3
                lut_bytes = cm.get("n_groups", 0) * (1 << bw) * 2
                add_dense(comp_N, K)
                add_palet(comp_N, K, cm["packed_len_bytes"], lut_bytes)
        else:
            N = m["dense_shape"][0]
            lut_bytes = m.get("n_groups", 0) * (1 << m["bitwidth"]) * 2
            add_dense(N, K)
            add_palet(N, K, m["packed_len_bytes"], lut_bytes)

    for d in (dense, palet):
        for k in d:
            d[k] *= num_iters
        d["total_gb"] = sum(d.values())
    return {"dense": dense, "palettized": palet,
            "analytic_upper_bound": True, "note":
            "closed-form byte counts, no cache model; for measured DRAM "
            "bytes use ncu dram__bytes_{read,write}.sum or DCGM"}

# --------------------------------------------------------------------------- #
# Measurement (real prefill + KV-cache decode; per-run CIs)
# --------------------------------------------------------------------------- #

def _cache_len(past) -> int:
    if past is None:
        return 0
    if hasattr(past, "get_seq_length"):
        try:
            return int(past.get_seq_length())
        except Exception:  # noqa: BLE001
            pass
    if isinstance(past, tuple) and len(past) > 0 and isinstance(past[0], tuple):
        return int(past[0][0].shape[2])
    if isinstance(past, tuple) and len(past) > 0:
        return int(past[0].shape[2])
    return 0


def stats(values: List[float]) -> Dict:
    a = np.asarray(values, dtype=np.float64)
    out = {"mean": float(a.mean()), "min": float(a.min()),
           "max": float(a.max()), "n": int(a.size)}
    if a.size >= 5:
        out["ci95"] = float(1.96 * a.std(ddof=1) / np.sqrt(a.size))
        out["std"] = float(a.std(ddof=1))
    return out


class EnergyMeasurement:
    def __init__(self, model, tokenizer, metadata, model_name, vram_info):
        self.model = model
        self.tokenizer = tokenizer
        self.metadata = metadata
        self.model_name = model_name
        self.vram_info = vram_info
        self.monitor = PowerMonitor(sample_interval_ms=ARGS.power_sample_ms)
        self.is_palettized = model_name.startswith("palettized")
        if getattr(self.tokenizer, "pad_token", None) is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    # -- energy helpers ------------------------------------------------------ #
    def _energy_j_between(self, t0: float, t1: float) -> Optional[float]:
        e = self.monitor.read_energy_counter_mj()
        if e is not None:
            return e / 1000.0
        return PowerMonitor.trapezoid_energy_j(
            self.monitor.samples, t0, t1)

    def measure_inference(self, batch_size, prefill_pct, decode_pct,
                          prompts: List[str]) -> Dict:
        prefill_tokens = max(1, int(TOTAL_TOKENS * prefill_pct / 100))
        decode_tokens = max(1, int(TOTAL_TOKENS * decode_pct / 100))
        texts = (prompts * batch_size)[:batch_size]

        inputs = self.tokenizer(
            texts, return_tensors="pt", padding=True, truncation=True,
            max_length=prefill_tokens).to(DEVICE)
        actual_prefill_len = int(inputs["input_ids"].shape[1])

        # ---------------- prefill (per-run timing) -------------------------- #
        with torch.no_grad():
            for _ in range(WARMUP_RUNS):
                _ = self.model(**inputs)
            torch.cuda.synchronize()

        prefill_ms, prefill_j = [], []
        for _ in range(NUM_RUNS):
            ev0, ev1 = torch.cuda.Event(True), torch.cuda.Event(True)
            torch.cuda.synchronize()
            self.monitor.start()
            e0 = self.monitor.read_energy_counter_mj()
            t0 = time.perf_counter()
            ev0.record()
            _ = self.model(**inputs)
            ev1.record()
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            e1 = self.monitor.read_energy_counter_mj()
            mono = self.monitor.stop()
            prefill_ms.append(ev0.elapsed_time(ev1))
            j = ((e1 - e0) / 1000.0 if e1 is not None
                 else PowerMonitor.trapezoid_energy_j(mono["samples"],
                                                        t0, t1))
            prefill_j.append(j if j is not None else float("nan"))

        prefill_total_tokens = batch_size * actual_prefill_len
        prefill_tok_per_s = [prefill_total_tokens / (ms / 1000.0)
                             for ms in prefill_ms]

        # ---------------- decode: real KV-cache loop ------------------------- #
        decode_step_ms, decode_run_j, decode_e2e_ms = [], [], []
        for _ in range(NUM_RUNS):
            with torch.no_grad():
                out = self.model(**inputs, use_cache=True)
                past = out.past_key_values
                next_tok = out.logits[:, -1:, :].argmax(dim=-1)
                torch.cuda.synchronize()

                self.monitor.start()
                e0 = self.monitor.read_energy_counter_mj()
                t0 = time.perf_counter()
                events = []
                for _step in range(decode_tokens):
                    cur_len = _cache_len(past)
                    attn = torch.ones((batch_size, cur_len + 1),
                                      dtype=torch.long, device=DEVICE)
                    pos = torch.full((batch_size, 1), cur_len,
                                     dtype=torch.long, device=DEVICE)
                    ev0, ev1 = torch.cuda.Event(True), torch.cuda.Event(True)
                    ev0.record()
                    out = self.model(input_ids=next_tok,
                                     attention_mask=attn,
                                     position_ids=pos,
                                     past_key_values=past,
                                     use_cache=True)
                    ev1.record()
                    events.append((ev0, ev1))
                    past = out.past_key_values
                    next_tok = out.logits[:, -1:, :].argmax(dim=-1)
                torch.cuda.synchronize()
                t1 = time.perf_counter()
                e1 = self.monitor.read_energy_counter_mj()
                mono = self.monitor.stop()

            decode_step_ms.extend(ev0.elapsed_time(ev1)
                                  for ev0, ev1 in events)
            decode_e2e_ms.append((t1 - t0) * 1000.0 / decode_tokens)
            j = ((e1 - e0) / 1000.0 if e1 is not None
                 else PowerMonitor.trapezoid_energy_j(mono["samples"],
                                                        t0, t1))
            decode_run_j.append(j if j is not None else float("nan"))
            del past, out
            torch.cuda.empty_cache()

        decode_total_tokens = batch_size * decode_tokens
        # per-step device stats over all runs*steps
        step_stats = stats(decode_step_ms)
        per_run_step_ms = [np.mean(decode_step_ms[i * decode_tokens:
                                                  (i + 1) * decode_tokens])
                           for i in range(NUM_RUNS)]
        decode_tok_per_s = stats([decode_total_tokens / (s / 1000.0)
                                  for s in per_run_step_ms])
        prefill_ms_stats = stats(prefill_ms)
        prefill_j_stats = stats(prefill_j)
        decode_j_stats = stats(decode_run_j)

        # ---------------- combine ------------------------------------------- #
        prefill_s = prefill_ms_stats["mean"] / 1000.0 * NUM_RUNS
        decode_s = step_stats["mean"] / 1000.0 * decode_tokens * NUM_RUNS
        total_tokens = (prefill_total_tokens + decode_total_tokens) * NUM_RUNS
        raw_j = (prefill_j_stats["mean"] * NUM_RUNS
                 + decode_j_stats["mean"] * NUM_RUNS)
        idle_w = IDLE["mean_w"] or 0.0
        active_s = prefill_s + decode_s
        net_j = raw_j - idle_w * active_s
        energy_per_tok_j = net_j / total_tokens

        hbm = estimate_hbm_traffic(self.metadata, batch_size,
                                   actual_prefill_len, NUM_RUNS)
        hbm_dec = estimate_hbm_traffic(self.metadata, batch_size, 1,
                                       decode_tokens * NUM_RUNS)
        side = "palettized" if self.is_palettized else "dense"

        return {
            "batch_size": batch_size,
            "prefill_pct": prefill_pct, "decode_pct": decode_pct,
            "prefill_tokens": actual_prefill_len,
            "decode_tokens": decode_tokens,
            "prefill": {
                "latency_ms": prefill_ms_stats,
                "throughput_tok_s": stats(prefill_tok_per_s),
                "energy_j_per_run": prefill_j_stats,
            },
            "decode": {
                "step_device_ms": step_stats,
                "step_e2e_ms": stats(decode_e2e_ms),
                "throughput_tok_s": decode_tok_per_s,
                "energy_j_per_run": decode_j_stats,
                "kv_cache": True, "greedy": True,
            },
            "overall": {
                "throughput_tok_s": total_tokens / (prefill_s + decode_s),
                "total_energy_j_raw": raw_j,
                "idle_subtracted_j": idle_w * active_s,
                "total_energy_j_net": net_j,
                "energy_per_token_j": energy_per_token_j,
            },
            "vram_used_gb": VRAMMonitor().get_memory_usage()["used_gb"],
            "torch_peak_gb": torch.cuda.max_memory_allocated() / 1024**3,
            "hbm_analytic_gb": {k: hbm[k]["total_gb"] + hbm_dec[k]["total_gb"]
                                for k in ("dense", "palettized")},
            "hbm_analytic_note": "upper bound, both models, "
                                 f"reported side uses '{side}'",
        }

    def sweep(self, batch_sizes, ratios, prompt_pool) -> Dict:
        results = {}
        rng = random.Random(ARGS.seed)
        for bs in batch_sizes:
            print(f"\n  Batch size {bs}:", flush=True)
            order = list(ratios)
            rng.shuffle(order)
            for prefill_pct, decode_pct in order:
                key = f"{bs}_{prefill_pct}_{decode_pct}"
                print(f"    {prefill_pct}% prefill / {decode_pct}% decode "
                      f"...", flush=True)
                try:
                    results[key] = self.measure_inference(
                        bs, prefill_pct, decode_pct, prompt_pool)
                except RuntimeError as e:
                    if "OOM" in str(e) or "out of memory" in str(e).lower():
                        print(f"      OOM, skipping rest of bs={bs}",
                              flush=True)
                        break
                    raise
        return results


IDLE: Dict = {"mean_w": None}

# --------------------------------------------------------------------------- #
# Main comparison
# --------------------------------------------------------------------------- #

def run_pass(order: List[str], results: Dict, prompt_pool: List[str]) -> None:
    for kind in order:
        print(f"\n[{kind}] {gpu_status()}", flush=True)
        if kind == "dense":
            model, tok, vram = load_dense_model()
        else:
            model, tok, vram = load_palettized_model()
        with open(os.path.join(PALETTIZED_DIR, "metadata.json")) as f:
            metadata = json.load(f)
        m = EnergyMeasurement(model, tok, metadata, kind, vram)
        res = m.sweep(BATCH_SIZES, PREFILL_DECODE_RATIOS, prompt_pool)
        for k, v in res.items():
            results.setdefault(kind, {}).setdefault(k, []).append(v)
        results.setdefault("vram", {})[kind] = vram
        del model, m
        import gc
        gc.collect(); torch.cuda.empty_cache(); torch.cuda.synchronize()
        time.sleep(5)
        print(f"  GPU after free: {VRAMMonitor().get_memory_usage()['used_gb']:.2f}"
              f" GB used; cooldown 30 s", flush=True)
        time.sleep(30)


def main():
    _init_config(parse_args())
    print("=" * 80)
    print("ENERGY AND PERFORMANCE COMPARISON — STRICT PROTOCOL")
    print("=" * 80)
    print(f"model={ARGS.model}  device={DEVICE}  dense_dtype={ARGS.dense_dtype}")
    print(f"total_tokens={TOTAL_TOKENS}  batch_sizes={BATCH_SIZES}  "
          f"ratios={PREFILL_DECODE_RATIOS}")
    print(f"runs: warmup={WARMUP_RUNS} measure={NUM_RUNS}  rounds={ARGS.rounds}")
    print(f"indices layout: idx4 (fixed)   sha256 verify: {ARGS.verify_sha}")
    print(f"status: {gpu_status()}")
    print("=" * 80)

    if ARGS.lock_clocks:
        lock_clocks(ARGS.lock_clocks)

    # idle baseline (H5)
    print(f"\nmeasuring idle power for {ARGS.idle_seconds:.0f} s ...",
          flush=True)
    mon = PowerMonitor(sample_interval_ms=ARGS.power_sample_ms)
    mon.start(); time.sleep(ARGS.idle_seconds); idle = mon.stop()
    IDLE["mean_w"] = idle["mean_w"] or 0.0
    print(f"  idle: {IDLE['mean_w']:.1f} W "
          f"({idle['n_samples']} samples, {idle['effective_hz'] or 0:.1f} Hz "
          f"effective, backend={mon.backend}, "
          f"energy_counter={mon.energy_counter_available})", flush=True)

    n_prompts = max(BATCH_SIZES)
    prompt_pool = load_prompt_pool(n_prompts)
    print(f"prompt pool: {len(prompt_pool)} real texts "
          f"(source={ARGS.prompts_source})", flush=True)

    results: Dict = {"meta": {
        "protocol": "energy-strict",
        "model": ARGS.model,
        "dense_dtype": ARGS.dense_dtype,
        "total_tokens": TOTAL_TOKENS,
        "batch_sizes": BATCH_SIZES,
        "ratios": PREFILL_DECODE_RATIOS,
        "warmup_runs": WARMUP_RUNS, "num_runs": NUM_RUNS,
        "seed": ARGS.seed,
        "clocks_locked_sm_mhz": ARGS.lock_clocks or None,
        "idle_watts": IDLE["mean_w"],
        "power_backend": mon.backend,
        "energy_counter_used": mon.energy_counter_available,
        "indices_layout": "idx4",
        "sha_verified": ARGS.verify_sha,
        "gpu_status_start": gpu_status(),
    }}

    orders = [["dense", "palettized"]]
    if ARGS.rounds >= 2:
        orders.append(["palettized", "dense"])
    for i, order in enumerate(orders[:ARGS.rounds]):
        print(f"\n{'='*80}\nPASS {i+1}/{ARGS.rounds} order={order}\n{'='*80}",
              flush=True)
        run_pass(order, results, prompt_pool)

    # ---------------- summary ---------------------------------------------- #
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY (strict protocol)")
    print("=" * 80)
    for prefill_pct, decode_pct in PREFILL_DECODE_RATIOS:
        print(f"\nPREFILL/DECODE {prefill_pct}%/{decode_pct}%")
        print(f"{'BS':>3} | {'model':<11} | {'prefill tok/s':>14} | "
              f"{'decode step ms':>14} | {'decode tok/s':>12} | "
              f"{'J/tok (net)':>12} | {'VRAM GB':>7}")
        print("-" * 96)
        for bs in BATCH_SIZES:
            key = f"{bs}_{prefill_pct}_{decode_pct}"
            row = {}
            for kind in ("dense", "palettized"):
                entries = results.get(kind, {}).get(key)
                if entries:
                    row[kind] = entries
            if not row:
                print(f"{bs:>3} | (skipped)")
                continue
            for kind in ("dense", "palettized"):
                if kind not in row:
                    continue
                es = row[kind]
                pt = np.mean([e["prefill"]["throughput_tok_s"]["mean"]
                              for e in es])
                dm = np.mean([e["decode"]["step_device_ms"]["mean"]
                              for e in es])
                dt = np.mean([e["decode"]["throughput_tok_s"]["mean"]
                              for e in es])
                jt = np.mean([e["overall"]["energy_per_token_j"] for e in es])
                vr = np.mean([e["vram_used_gb"] for e in es])
                print(f"{bs:>3} | {kind:<11} | {pt:>14.2f} | {dm:>14.3f} | "
                      f"{dt:>12.2f} | {jt:>12.4f} | {vr:>7.2f}")
            if "dense" in row and "palettized" in row:
                d, p = row["dense"][0], row["palettized"][0]
                sp = (d["decode"]["step_device_ms"]["mean"]
                      / p["decode"]["step_device_ms"]["mean"])
                er = (p["overall"]["energy_per_token_j"]
                      / d["overall"]["energy_per_token_j"])
                print(f"{'':>3} | {'ratio':<11} | {'':>14} | "
                      f"{sp:>13.2f}x | {'':>12} | {er:>11.2f}x |")

    out_file = os.path.join(
        ARGS.outdir, f"energy_strict_{int(time.time())}.json")
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2, default=float)
    print(f"\nDetailed results saved to: {out_file}")


if __name__ == "__main__":
    main()

