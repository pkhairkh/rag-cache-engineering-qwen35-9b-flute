#!/usr/bin/env python3
"""
eval_greedy_match.py — greedy-decode equivalence probe.

Decodes 32 real prompts (built-in deterministic pool) with the dense FP16
model and the palettized model (the auto recipe's mixed-radix artifacts,
optionally with the whitened-SVD residual branch), both with KV-cache
greedy decoding (the protocol fixed for defect H1 — never a fixed 1-token
forward), and reports:

  * exact-match fraction  — full generated token sequences identical
  * first-divergence position — index of the first differing NEW token
    (= max_new_tokens when the sequences are identical)

W21 instrumentation (the timing/PPL/VRAM round — the probe previously
reported ONLY the match statistics):
  * timing — per arm: model load seconds, greedy-decode seconds, new
    tokens, tokens/second, and the quant-vs-fp16 speedup ratio;
  * PPL — WikiText-2 test perplexity for BOTH arms in the SAME run
    (the eval_ppl protocol: non-overlapping 2048-token windows,
    PPL = exp(total NLL / total tokens)); one model load per arm —
    decode + PPL run back-to-back on the same resident instance;
  * VRAM — per-arm peak allocated/reserved GiB (the CUDA peak stats
    reset at each phase boundary) plus a [MEM] line after each release,
    so the residency of each arm is on the record.

W22 (the chunked-CE round): the dense arm's PPL OOM'd at its first
window on the A10G — routing labels through the model made
ForCausalLMLoss upcast the full (1, 2047, 248320) logits to fp32
(2034237440 B, the exact observed ask) and cross_entropy then
materialize the log-softmax twin, 2 x 1.89 GiB of temps next to the
~18.1 GiB fp16 resident. eval_ppl.evaluate_nll now scores chunk-wise
(see its module docstring); _ppl_arm additionally releases the
decode phase's cached blocks before scoring, and both entry scripts
default PYTORCH_CUDA_ALLOC_CONF to expandable_segments:True (the
allocator's own advice from the OOM banner; an explicit user setting
wins).

W23 (the INSPECTION round — INSPECTION.md at the repo root): the 2026-10-07 run
measured quant decode at 9.499 tok/s = 0.462x dense (the acceptance
bar is >= 2x FASTER). The kernels were exonerated — the cost is
host dispatch: ~2,000 small CUDA ops per token (per-module FHT,
two-stream qgemm pairs, residual GEMMs, AWQ compensation) at ~50 us
of eager launch overhead each. This round adds the prescribed fix:
a CUDA-Graph greedy decode path (--decode-backend, default auto =
graphs when CUDA is present). One graph is captured per ARM (the
static KV cache keeps every shape constant; the cache's device-side
cumulative_length drives positions, masks and cache writes, so the
graph is prompt-agnostic and re-used across prompts) and replayed
once per token — the ~2,000-op dispatch collapses to one replay.
Safety: the capture is verified against an eager ground-truth token
BEFORE first use, the arm's first --graphs-verify-prompts prompt(s)
are re-decoded with model.generate() and compared token-for-token
(early divergence -> the whole arm falls back to generate, loudly;
late divergence >= 32 tokens is tolerated as float-order noise and
noted), and ANY capture failure falls back to generate without
losing the run. A second W23 fix: the two-stream qgemm add is now
in-place on the no-grad path (saves one full (M, N) fp16 transient
per two-stream module — the quant PPL batch-4 OOM's dominant ask,
lm_head at M=8192: 3 x 4.07 GiB -> 2).

W24 (the box-feedback round): the A10G's first graphs run failed the
capture-verify gate (captured token 279 vs eager 8160 on the DENSE
arm — the CPU double could not reproduce it: a re-invoking replay
re-runs Python, a real replay re-launches recorded kernels). Three
fixes, all verified by the existing gates: (1) the pre-capture
warmup's KV-slot pollution is now rolled back too (the snapshot
covers the exact [cursor, cursor+warmup_n) region the warmup writes,
so the captured step re-runs from the exact post-prefill state the
ground truth saw — the gate compares like with like regardless of
the attention route); (2) the whole graphed arm runs under a
math-forced SDPA context (_sdpa_math_ctx) — the StaticCache decode
mask is a [1,1,1,max_cache_len] boolean the math backend applies by
construction, so a kernel dispatch that lets beyond-cursor slots
through (the observed signature) is impossible; (3) the first
warmup_n-1 replays are verified IN LOCKSTEP against the warmup
ground truth (a distinct "replay verify" gate — a frozen
cursor/mask/position at step 2+ is caught immediately, not at the
whole-arm verification). Plus: the W22 expandable-segments allocator
default is now skipped when the graph path will run (set in main()
after argparse instead of at import), and the loaders pass `dtype=`
(transformers 5.x renamed torch_dtype). W24 completion: every
[GRAPHS] gate line and the report's environment section carry the
exact torch/cuda/transformers pair — a still-failing gate on the box
is a dispatch property of that pair, and it belongs on the record.

W25 (the stale-capture-read round): the box ran W24 and STILL failed
the capture gate with the same token pair — which localized the
defect to the GATE itself, not the graph. torch.cuda.graph RECORDS
the captured work without executing it (graph.__enter__ =
capture_begin, graph.__exit__ = capture_end, no replay anywhere —
verified against the torch source), so the gate's static_out read
held the LAST WARMUP token (279 = the third warmup token), while the
eager ground truth was the FIRST (8160) — the captured step had never
run, so the comparison was warmup-vs-warmup, apples to oranges. The
graph itself was healthy the whole time (no frozen mask, no rogue
SDPA dispatch, no allocator interaction). _CudaGraphRunner.capture()
now ends with self.replay(), implementing its own documented contract
("capture records AND executes the step once" — the exact contract
the CPU test double implements by re-invoking step(), which is why
every CPU gate passed while the box failed): the recorded kernels
really run with the current static-buffer state, static_out holds the
captured step's true token, and the capture gate finally tests what
it was designed to test. The W24 hardenings (KV-slot rollback,
math-forced SDPA, lockstep replay verification, conditional
allocator) stand as defense-in-depth for any REAL replay defect.

Output: reports/greedy_equivalence_idx4.json (written by this script only).
"""

import argparse
import contextlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from functools import partial

import torch

# W24: the W22 expandable-segments allocator default moved from import
# time into main() and is now CONDITIONAL on the decode backend: the
# box's first W23 graphs attempt captured the graph with
# expandable_segments active and failed the capture-verify gate — the
# allocator disables expandable segments while capturing on most
# builds, but the combination is a documented capture hazard and the
# W22 OOM pressure it addressed is gone (the W23 two-stream fix
# removed the third live (M,N) tensor; the PPL OOM ladder remains as
# the backstop). main() re-applies the setdefault (before the first
# CUDA allocation) only when the graph path will not run; an explicit
# user env always wins in either direction.

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
from eval_common import (  # noqa: E402
    git_head,
    load_dense_fp16,
    load_quant_model,
    release_model_memory,
)

# 32 real, diverse prompts — deterministic (no dataset dependency), so the
# probe reproduces bit-exactly across machines at the same seed.
PROMPTS = [
    "Explain the difference between weather and climate in simple terms.",
    "Write a Python function that reverses a string without using slicing.",
    "Summarize the plot of Romeo and Juliet in three sentences.",
    "What are the primary causes of the Industrial Revolution?",
    "Translate 'The weather is beautiful today' into French.",
    "List five healthy breakfast ideas with roughly 300 calories each.",
    "How does photosynthesis work, step by step?",
    "Write a short poem about autumn leaves.",
    "What is the difference between a virus and a bacterium?",
    "Explain recursion to a ten-year-old.",
    "Describe the water cycle.",
    "What were the main causes of World War I?",
    "Write a SQL query to find the second-highest salary in an employees table.",
    "Why is the sky blue?",
    "Give three tips for improving sleep quality.",
    "Explain what compound interest is with an example.",
    "What is the Turing test?",
    "Write a haiku about the ocean.",
    "How do vaccines work?",
    "What is the difference between HTTP and HTTPS?",
    "Summarize the theory of evolution by natural selection.",
    "Explain the concept of opportunity cost.",
    "What causes earthquakes?",
    "Write a paragraph describing a busy city market.",
    "What is machine learning, in plain language?",
    "How does a refrigerator keep food cold?",
    "List the planets of the solar system in order.",
    "What is the significance of the Rosetta Stone?",
    "Explain why the seasons change.",
    "Write a short dialogue between a customer and a barista.",
    "What is inflation and why does it happen?",
    "Describe how a bicycle stays upright when moving.",
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3.5-9B")
    p.add_argument("--artifacts-dir", required=True,
                   help="the palettized artifacts dir (usage example: "
                        "/home/ubuntu/qwen3_5_9b_palettized)")
    p.add_argument("--heads-dir", default=None,
                   help="optional separate heads artifacts dir (embed_tokens + "
                        "lm_head); layers are loaded from --artifacts-dir, heads "
                        "from this dir if specified")
    p.add_argument("--residual", action="store_true",
                   help="attach the whitened-SVD residual branch when the "
                        "artifacts carry one (W4 configuration)")
    p.add_argument("--qlora-adapters", default=None,
                   help="QLoRA adapters directory")
    p.add_argument("--n-prompts", type=int, default=32)
    p.add_argument("--max-new-tokens", type=int, default=96)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--forward", choices=["kernel", "reference"],
                   default="kernel",
                   help="quant forward route: kernel = FLUTE fused path "
                        "(the box default; requires CUDA) or reference "
                        "= torch dequant path (the CPU-legal route)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--decode-backend", choices=["auto", "generate", "graphs"],
                   default="auto",
                   help="greedy decode route (W23): auto = CUDA graphs when "
                        "CUDA is available (generate otherwise), graphs = "
                        "force the graph path (falls back loudly on capture "
                        "failure), generate = the pre-W23 eager loop. The "
                        "graph path captures ONE decode step per arm — the "
                        "INSPECTION §3.5 headline fix for the 0.462x "
                        "host-dispatch-bound decode")
    p.add_argument("--graphs-verify-prompts", type=int, default=1,
                   help="per arm: prompts re-decoded with model.generate() "
                        "and compared token-for-token after the graphed "
                        "decode (0 disables; divergence before token 32 "
                        "falls the whole arm back to generate — the eager "
                        "loop is never lost to a graph defect)")
    p.add_argument("--no-awq-compensation", action="store_true",
                   help="differential debugging arm: serve the legacy "
                        "rotate-then-AWQ fold WITHOUT the W13 exact "
                        "compensation (reproduces the reported "
                        "bad-decode composition deliberately)")
    p.add_argument("--no-ppl", action="store_true",
                   help="skip the WikiText-2 perplexity arm (W21; on by "
                        "default — the report then carries ppl: null)")
    p.add_argument("--ppl-seq-len", type=int, default=2048,
                   help="PPL window length (GPTQ convention: 2048)")
    p.add_argument("--ppl-max-windows", type=int, default=0,
                   help="cap on PPL windows (0 = the full test split)")
    p.add_argument("--ppl-batch", type=int, default=4,
                   help="PPL forward batch for the palettized arm")
    p.add_argument("--ppl-batch-dense", type=int, default=1,
                   help="PPL forward batch for the dense arm (default 1: "
                        "the ~18.1 GiB fp16 9B resident leaves ~3 GiB for "
                        "the full-vocab logits — 0.95 GiB fp16 per window "
                        "at batch 1 — plus the W22 chunked fp32 CE, "
                        "<= 254 MiB live; the old labels path asked for "
                        "2 x 1.89 GiB fp32 twins and OOM'd at ANY batch. "
                        "Batch 2 now fits; higher still risks the card)")
    p.add_argument("--output", default="reports/greedy_equivalence_idx4.json")
    return p.parse_args()


@torch.no_grad()
def greedy_decode(model, tokenizer, prompts, device, max_new_tokens):
    """KV-cache greedy decode (H1 protocol): one token per step, cache on."""
    outs = []
    for i, prompt in enumerate(prompts):
        ids = tokenizer(prompt, return_tensors="pt").to(device)
        generated = model.generate(
            **ids, max_new_tokens=max_new_tokens, do_sample=False,
            temperature=None, top_p=None, top_k=None,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id)
        new_tokens = generated[0][ids["input_ids"].shape[1]:]
        outs.append(new_tokens.cpu())
        if (i + 1) % 8 == 0:
            print(f"    {i + 1}/{len(prompts)} prompts", flush=True)
    return outs


# --------------------------------------------------------------------------- #
# W23: the CUDA-graph decode path (INSPECTION §3.5 recommendation 1)
# --------------------------------------------------------------------------- #

# A divergence between the graphed decode and the eager generate() decode
# before this token position is treated as an implementation bug (mask
# semantics, a frozen index, a stale pointer) -> the arm falls back to
# generate(). Divergence at or after this position is tolerated as benign
# float-order noise (generate's DynamicCache path and the static-cache
# path pick different reduction orders; near-tie argmax flips are rare and
# late) and noted in the report.
_GRAPHS_VERIFY_TOLERANCE = 32


class GraphDecodeError(RuntimeError):
    """The graphed decode failed a correctness gate (never silent: the
    dispatcher catches it and re-runs the arm with model.generate())."""


class _CudaGraphRunner:
    """Capture/replay over torch.cuda.CUDAGraph (the box path).

    Interface (also implemented by the test/_ReplayRunner): warmup(step, n)
    runs the step n times on a side stream (lazy kernel/autotune init off
    the default stream, per the torch.cuda.graph guidance), capture(step)
    records AND executes the step once, replay() re-launches the recorded
    work (the step's static buffers see the new values), close() frees the
    graph and its private memory pool."""

    def __init__(self):
        if not torch.cuda.is_available():
            raise GraphDecodeError(
                "_CudaGraphRunner: no CUDA device (use --decode-backend "
                "generate or run with the CPU-legal reference route)")
        self._graph = torch.cuda.CUDAGraph()

    def warmup(self, step, n=3):
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(n):
                step()
        torch.cuda.current_stream().wait_stream(s)

    def capture(self, step):
        torch.cuda.synchronize()
        with torch.cuda.graph(self._graph):
            step()
        # W25 — the stale-capture-read fix: CUDA stream capture RECORDS
        # the work without executing it (torch.cuda.graph.__enter__ =
        # capture_begin, __exit__ = capture_end — no replay anywhere;
        # verified against the torch source). The runner contract is
        # "capture records AND executes the step once" (the CPU test
        # double implements it by re-invoking step()), so the recorded
        # kernels must actually RUN here, with the current static-buffer
        # state (input t1, cache cursor at the post-prefill position):
        # static_out then holds the captured step's true token. Without
        # this replay, the capture gate read the STALE last-warmup token
        # out of static_out — the box's "token 2 = 279, eager ground
        # truth = 8160" compared the THIRD warmup token against the
        # FIRST; the graph itself was healthy, and the first loop
        # replay would have double-emitted the captured token.
        self.replay()

    def replay(self):
        self._graph.replay()

    def close(self):
        g = getattr(self, "_graph", None)
        if g is not None:
            try:
                g.reset()
            except Exception:  # noqa: BLE001 — best-effort pool release
                pass
        self._graph = None


def _eos_token_ids(model) -> set:
    """The generation config's EOS ids (int or list — Qwen ships a list)."""
    gc = getattr(model, "generation_config", None)
    raw = getattr(gc, "eos_token_id", None) if gc is not None else None
    if raw is None:
        return set()
    if isinstance(raw, (list, tuple, set)):
        return {int(x) for x in raw}
    return {int(raw)}


def _sdpa_math_ctx():
    """Force torch SDPA to the MATH backend for the graphed arm (W24).

    Returns (context_manager, mode_string). The box's first W23 graphs
    run failed the capture-verify gate with a wildly wrong token — the
    signature of the dispatched SDPA kernel not honoring the
    [1,1,1,max_cache_len] boolean mask the StaticCache path builds
    (masking_utils keeps positions/offsets device-derived off
    cumulative_length, so the MASK is correct; whether the kernel
    dispatch that reaches the box applies it is a torch/cuDNN-build
    property we cannot assume). The math backend applies attn_mask by
    construction (plain torch ops). Decode is q_len=1 — the math
    kernel's shapes are tiny, and every extra op lives INSIDE the
    replayed graph where launch cost is already amortized. The eager
    generate() fallback arm is untouched (its DynamicCache path never
    needs a mask). Very old torch without torch.nn.attention: returns
    a no-op context (mode "default") — the verification gates still
    protect the run.
    """
    try:
        from torch.nn.attention import SDPBackend, sdpa_kernel
    except ImportError:  # pragma: no cover — torch < 2.3
        return contextlib.nullcontext(), "default"
    return sdpa_kernel([SDPBackend.MATH]), "math"


def _transformers_version() -> str:
    """transformers version for the report (best effort)."""
    try:
        import transformers
        return transformers.__version__
    except Exception:  # noqa: BLE001  # pragma: no cover
        return "?"


def _graphs_env_line() -> str:
    """torch/cuda/transformers pair for the [GRAPHS] gate lines (W24
    completion): the box's capture-verify failure is a dispatch property
    of the box's exact library pair (torch 2.14 / CUDA 13.0 was named in
    the operator's analysis) — whenever a gate fires, that pair belongs
    on the record so the next round can diff it against the audited
    source (transformers 5.17.0 here: StaticLayer is in-place and
    device-derived by construction, so a still-failing gate on that
    exact pair points at the torch/cuDNN dispatch, not the cache)."""
    parts = [f"torch {torch.__version__}"]
    if torch.cuda.is_available() and torch.version.cuda:
        parts.append(f"cuda {torch.version.cuda}")
    parts.append(f"transformers {_transformers_version()}")
    return "; ".join(parts)


def _cache_snapshot(cache, kv_slots=0):
    """Clone the per-layer mutable decode state so the pre-capture warmup
    can be rolled back: cumulative_length (the static layers' device-side
    write cursor), the linear-attention conv/recurrent states (they are
    mutated IN-PLACE by the decode step), and — W24 — the NEXT
    `kv_slots` full-attention KV slots beyond each layer's cursor, the
    exact region a `kv_slots`-step warmup writes. The W23 code left
    those slots dirty: the warmup's KV is masked by the
    cache-state-derived causal mask, so the eager path stayed correct,
    but the captured step then ran on a cache whose tail differed from
    the warmup's — the capture-verify gate compared a dirty-cache step
    against a clean-cache step, which is apples-to-oranges the moment
    any attention route lets beyond-cursor slots through (the box's
    first graphs run). Rolled back here, the captured step re-runs
    from the EXACT post-prefill state the warmup ground truth saw.

    NOTE: the linear-attention state containers are DICTS keyed by
    state_idx (transformers 5.x LinearAttentionCacheLayerMixin:
    conv_states/recurrent_states/is_*_initialized are dict[int, ...]),
    not lists — iterate by key, never enumerate().
    """
    snaps = []
    for layer in cache.layers:
        entry = {}
        cum = getattr(layer, "cumulative_length", None)
        if torch.is_tensor(cum):
            entry["cum"] = cum.detach().clone()
            keys = getattr(layer, "keys", None)
            values = getattr(layer, "values", None)
            if kv_slots > 0 and keys is not None and values is not None \
                    and getattr(layer, "is_initialized", False):
                start = int(cum)  # host sync — fine: never inside capture
                end = min(start + kv_slots, keys.shape[2])
                if end > start:
                    entry["kv"] = (
                        start,
                        keys[:, :, start:end, :].detach().clone(),
                        values[:, :, start:end, :].detach().clone(),
                    )
        conv, rec = [], []
        n_states = getattr(layer, "number_of_states", None)
        if n_states:
            for i in range(n_states):
                if layer.is_conv_states_initialized.get(i) \
                        and layer.conv_states.get(i) is not None:
                    conv.append((i, layer.conv_states[i].detach().clone()))
                if layer.is_recurrent_states_initialized.get(i) \
                        and layer.recurrent_states.get(i) is not None:
                    rec.append((i, layer.recurrent_states[i].detach().clone()))
        entry["conv"], entry["rec"] = conv, rec
        snaps.append(entry)
    return snaps


def _cache_restore(cache, snaps) -> None:
    """Roll the cache back to a _cache_snapshot state (copy_ into the
    ORIGINAL buffers — the static addresses the captured graph uses);
    W24: the full-attention KV slots the warmup dirtied are rolled back
    too (entry["kv"] = (start, k_slice, v_slice))."""
    for layer, entry in zip(cache.layers, snaps):
        if "cum" in entry:
            layer.cumulative_length.copy_(entry["cum"])
        if "kv" in entry:
            start, k_snap, v_snap = entry["kv"]
            end = start + k_snap.shape[2]
            layer.keys[:, :, start:end, :].copy_(k_snap)
            layer.values[:, :, start:end, :].copy_(v_snap)
        for i, t in entry["conv"]:
            layer.conv_states[i].copy_(t)
        for i, t in entry["rec"]:
            layer.recurrent_states[i].copy_(t)


@torch.no_grad()
def greedy_decode_graphs(model, tokenizer, prompts, device, max_new_tokens,
                         runner_factory=None, verify_prompts=0,
                         warmup_steps=3):
    """Greedy decode with the per-token forward captured into ONE CUDA
    graph per arm (W23 — the INSPECTION throughput fix).

    Mechanism: a transformers StaticCache is pre-allocated once per arm at
    the worst-case length, so EVERY decode step has identical shapes. The
    cache's device-side cumulative_length drives everything per-step (the
    KV write cursor, the position ids, the [1,1,1,L] causal mask — all
    rebuilt from it by device ops), so the captured graph is
    prompt-agnostic: prefill stays eager, then each token is one replay.

    Per arm: prompt 0 prefill -> snapshot (incl. the KV slots the warmup
    will write — W24) -> warmup_n warmup steps (eager ground truth) ->
    complete snapshot restore -> capture (executes step 1 of prompt 0) ->
    the captured output token must EQUAL the warmup ground truth (the
    capture gate) -> the first warmup_n-1 replays are checked against the
    REMAINING ground-truth tokens IN LOCKSTEP (the replay gate, W24 —
    they are real decode steps of prompt 0, so a frozen cursor/mask/
    position is caught at step 2+ instead of at the whole-arm
    verification) -> free replay loop. Subsequent prompts: prefill ->
    replay loop (one graph for the whole arm).

    W24: the WHOLE arm (prefill, warmup, capture, replays and the eager
    generate() verification) runs under a math-forced SDPA context —
    see _sdpa_math_ctx. This keeps warmup, capture and verification on
    one mask-honoring attention route (the eager ground truth, the
    captured kernels and the generate() comparison all see the same
    numerics instead of three different dispatches).

    Returns (outs, info); raises GraphDecodeError on any gate failure —
    the dispatcher catches it and re-runs the arm with model.generate().
    """
    if runner_factory is None:
        runner_factory = _CudaGraphRunner
    from transformers import StaticCache

    eos_ids = _eos_token_ids(model)
    enc_ids = [tokenizer(p, return_tensors="pt")["input_ids"][0]
               for p in prompts]
    max_p = max(int(x.numel()) for x in enc_ids) if enc_ids else 0
    warmup_n = max(1, min(warmup_steps, max(1, max_new_tokens - 1)))
    cache_len = max_p + max_new_tokens + warmup_n + 1
    cache = StaticCache(config=model.config, max_cache_len=cache_len)

    static_input = torch.zeros(1, 1, dtype=torch.long, device=device)
    static_out = torch.zeros(1, dtype=torch.long, device=device)

    def step():
        out = model(input_ids=static_input, past_key_values=cache,
                    use_cache=True)
        nxt = out.logits[:, -1, :].argmax(dim=-1)      # (1,)
        static_out.copy_(nxt)
        static_input.copy_(nxt.view(1, 1))             # recorded feedback

    runner = runner_factory()
    math_ctx, sdpa_mode = _sdpa_math_ctx()
    outs, captured, notes = [], False, []
    ground_truth = []      # prompt 0 only — the warmup's eager tokens
    try:
        with math_ctx:
            for i, ids0 in enumerate(enc_ids):
                ids = ids0.unsqueeze(0).to(device)
                cache.reset()
                logits = model(input_ids=ids, past_key_values=cache,
                               use_cache=True).logits[:, -1, :]
                t1 = int(logits.argmax(dim=-1))
                toks = [t1]
                if max_new_tokens > 1 and t1 not in eos_ids:
                    if not captured:
                        # the warmup must consume t1 — the exact input the
                        # captured step will consume (the pre-fix code left
                        # static_input at zeros for prompt 0, so the ground
                        # truth and the capture diverged at the gate — the
                        # gate caught it, as designed)
                        static_input.fill_(t1)
                        # W24: kv_slots=warmup_n — the warmup writes the
                        # KV slots [cursor, cursor+warmup_n); rolling
                        # them back makes the captured step re-run from
                        # the exact post-prefill state the ground truth
                        # saw, so the capture gate compares like with
                        # like regardless of the attention route
                        snap = _cache_snapshot(cache, kv_slots=warmup_n)

                        def _warm_step():
                            step()
                            ground_truth.append(int(static_out[0].item()))

                        runner.warmup(_warm_step, warmup_n)
                        _cache_restore(cache, snap)
                        static_input.fill_(t1)
                        runner.capture(step)
                        captured = True
                        t_cap = int(static_out[0].item())
                        if ground_truth and t_cap != ground_truth[0]:
                            raise GraphDecodeError(
                                f"capture verify: prompt 0 token 2 = {t_cap}, "
                                f"eager ground truth = {ground_truth[0]} — the "
                                f"captured step disagrees with the eager step "
                                f"(frozen pointer/index?)")
                        toks.append(t_cap)
                    else:
                        static_input.fill_(t1)
                    # W24 lockstep: only prompt 0 carries ground truth.
                    # Decode step j (1-based) must equal ground_truth[j-1]
                    # while j <= warmup_n — the capture consumed step 1,
                    # so the first warmup_n-1 replays are verified here,
                    # INSIDE the normal loop (they are real tokens —
                    # greedy is deterministic, no extra compute).
                    lockstep = (i == 0)
                    while len(toks) < max_new_tokens:
                        runner.replay()
                        t = int(static_out[0].item())
                        toks.append(t)
                        if lockstep:
                            gi = len(toks) - 2   # ground_truth index
                            if 0 <= gi < len(ground_truth) \
                                    and t != ground_truth[gi]:
                                raise GraphDecodeError(
                                    f"replay verify: prompt 0 token "
                                    f"{len(toks)} = {t}, eager ground truth "
                                    f"= {ground_truth[gi]} — the replayed "
                                    f"step disagrees with the eager step "
                                    f"(state not advancing inside the "
                                    f"graph? — a frozen cursor, mask or "
                                    f"position)")
                        if t in eos_ids:
                            break
                outs.append(torch.tensor(toks, dtype=torch.long))
                if (i + 1) % 8 == 0:
                    print(f"    {i + 1}/{len(prompts)} prompts", flush=True)

            verify = None
            if verify_prompts and verify_prompts > 0 and prompts:
                n = min(verify_prompts, len(prompts))
                eager = greedy_decode(model, tokenizer, prompts[:n], device,
                                      max_new_tokens)
                for j in range(n):
                    exact, fd = compare(outs[j], eager[j], max_new_tokens)
                    if exact:
                        verify = verify or "exact"
                        continue
                    if fd < _GRAPHS_VERIFY_TOLERANCE:
                        raise GraphDecodeError(
                            f"generate() verification: prompt {j} diverged at "
                            f"token {fd} (< {_GRAPHS_VERIFY_TOLERANCE}) — "
                            f"treating as an implementation defect")
                    notes.append(
                        f"verify: prompt {j} diverged at token {fd} "
                        f"(>= {_GRAPHS_VERIFY_TOLERANCE}: float-order noise "
                        f"tolerated; --decode-backend generate is the strict "
                        f"path)")
                    verify = verify or "lenient"
        return outs, {
            "backend": "graphs",
            "captured": captured,
            "verify": verify,
            "notes": notes,
            "cache_len": cache_len,
            "warmup_steps": warmup_n,
            "sdpa": sdpa_mode,
        }
    finally:
        runner.close()


def _graphs_will_run(backend: str, device: str) -> bool:
    """Whether the CUDA-graph decode path will be attempted for this
    combination (W24 — shared by greedy_decode_dispatch and main()'s
    allocator-default decision so the two can never drift apart)."""
    return backend == "graphs" or (
        backend == "auto" and torch.cuda.is_available()
        and str(device).startswith("cuda"))


def greedy_decode_dispatch(model, tokenizer, prompts, device, max_new_tokens,
                           backend="auto", verify_prompts=0,
                           runner_factory=None):
    """Route the arm's decode (W23): graphs when asked/possible, with a
    LOUD fallback to the eager generate() loop on any failure — the eval
    never loses a run to the graph machinery.

    Returns (outs, info) where info["backend"] is "graphs" or "generate".
    """
    reason = None
    use_graphs = _graphs_will_run(backend, device)
    if use_graphs:
        try:
            outs, info = greedy_decode_graphs(
                model, tokenizer, prompts, device, max_new_tokens,
                runner_factory=runner_factory, verify_prompts=verify_prompts)
            info["requested"] = backend
            return outs, info
        except Exception as e:  # noqa: BLE001 — loud, total fallback
            reason = f"{type(e).__name__}: {e}"
        # W24 completion: the library pair rides on the gate line — a
        # still-failing box gate is a dispatch property of that pair
        print(f"  [GRAPHS] {reason} [{_graphs_env_line()}]", flush=True)
        print("  [GRAPHS] falling back to model.generate() for this arm "
              "(rerun with --decode-backend generate to skip the attempt; "
              "if the error mentions capture/allocator, an unset "
              "PYTORCH_CUDA_ALLOC_CONF can also help — expandable_segments "
              "interacts with graph capture on some torch builds)",
              flush=True)
    outs = greedy_decode(model, tokenizer, prompts, device, max_new_tokens)
    return outs, {
        "backend": "generate",
        "requested": backend,
        "fallback_reason": reason if use_graphs else None,
    }


# --------------------------------------------------------------------------- #
# W21 instrumentation: the per-phase timing / VRAM / PPL helpers
# --------------------------------------------------------------------------- #

def _sync(device: str) -> None:
    """Synchronize the device before a wall-clock reading (no-op on CPU)."""
    if torch.cuda.is_available() and str(device).startswith("cuda"):
        torch.cuda.synchronize()


def _reset_vram_peak() -> None:
    """Reset the CUDA peak stats at a phase boundary (CPU-safe)."""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def _vram_peak() -> dict:
    """The phase's peak allocated/reserved GiB (None without CUDA)."""
    if not torch.cuda.is_available():
        return None
    return {
        "peak_alloc_gib": round(
            torch.cuda.max_memory_allocated() / 2 ** 30, 3),
        "peak_reserved_gib": round(
            torch.cuda.max_memory_reserved() / 2 ** 30, 3),
    }


def _mem_line(tag: str) -> None:
    """The project-style [MEM] line: live alloc/reserved after a release."""
    if not torch.cuda.is_available():
        return
    a = torch.cuda.memory_allocated() / 2 ** 30
    r = torch.cuda.memory_reserved() / 2 ** 30
    free = torch.cuda.get_device_properties(0).total_memory / 2 ** 30 - r
    print(f"  [MEM] {tag}: alloc={a:.2f}GiB reserved={r:.2f}GiB "
          f"free~{free:.2f}GiB", flush=True)


def _time_phase(fn, device: str) -> tuple:
    """Run fn() between two synchronized wall-clock readings (W21).
    Returns (result, seconds)."""
    _sync(device)
    t0 = time.perf_counter()
    out = fn()
    _sync(device)
    return out, time.perf_counter() - t0


def _timing_record(load_s: float, decode_s: float, n_new_tokens: int) -> dict:
    """The per-arm timing record (W21): load, decode, throughput."""
    return {
        "load_seconds": round(load_s, 3),
        "decode_seconds": round(decode_s, 3),
        "new_tokens": int(n_new_tokens),
        "tokens_per_second": (
            round(n_new_tokens / decode_s, 3) if decode_s > 0 else None),
    }


def _ppl_arm(model, windows, device: str, batch_size: int) -> dict:
    """One arm's WikiText-2 PPL (the eval_ppl protocol, W21 reuse):
    evaluate_nll over the SAME windows, PPL = exp(NLL / tokens).

    W22: the decode phase's cached blocks (the KV-cache arena the
    allocator still holds) are released to the driver BEFORE the
    first window, so the full-vocab fp16 logits ask meets a
    contiguous free region; evaluate_nll itself scores chunk-wise —
    the labels route's 2 x 1.89 GiB fp32 twins are what OOM'd the
    dense arm on the 22 GiB card.
    """
    from eval_ppl import evaluate_nll
    release_model_memory()   # gc + empty_cache; CPU-safe no-op
    nll, tok = evaluate_nll(model, windows, device, batch_size=batch_size)
    return {"total_nll": nll, "n_tokens": int(tok),
            "ppl": float(torch.exp(torch.tensor(nll / tok)))}


def compare(a_ids, b_ids, max_new):
    n = min(a_ids.numel(), b_ids.numel())
    exact = a_ids.numel() == b_ids.numel() and bool(
        torch.equal(a_ids, b_ids))
    first_div = max_new
    for j in range(n):
        if a_ids[j] != b_ids[j]:
            first_div = j
            break
    if exact:
        first_div = max_new
    return exact, int(first_div)


def main():
    args = parse_args()

    # W24: the W22 expandable-segments default, now conditional (see the
    # comment block above `import torch`). torch reads PYTORCH_CUDA_ALLOC_CONF
    # when the CACHING ALLOCATOR initializes — at the first CUDA
    # allocation, i.e. at the first model load below — so setting it here
    # (after argparse, before any tensor hits the device) is still in
    # time. torch.cuda.is_available() initializes the driver but does
    # NOT touch the allocator, so the probe below is order-safe. An
    # explicit user env is never overridden: if the operator pinned
    # expandable_segments:True themselves, graphs capture under it and
    # the [GRAPHS] fallback text explains the interaction.
    graphs_will_run = _graphs_will_run(args.decode_backend, args.device)
    if not graphs_will_run:
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF",
                              "expandable_segments:True")

    torch.manual_seed(args.seed)
    from transformers import AutoTokenizer

    prompts = PROMPTS[: args.n_prompts]
    assert len(prompts) == args.n_prompts, "built-in pool has 32 prompts"

    # W21: the PPL windows are loaded ONCE (tokenizer is shared); each
    # arm then decodes AND scores on the same resident instance.
    windows = None
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if not args.no_ppl:
        from eval_ppl import load_eval_windows
        windows, n_corpus_tokens = load_eval_windows(
            tokenizer, args.ppl_seq_len, args.ppl_max_windows)
        print(f"ppl split: wikitext2 test — {windows.shape[0]} windows x "
              f"{args.ppl_seq_len} tokens (corpus {n_corpus_tokens})",
              flush=True)

    print(f"[1/3] Dense FP16 reference ({len(prompts)} prompts, "
          f"{args.max_new_tokens} new tokens)", flush=True)
    _reset_vram_peak()
    dense, dense_load_s = _time_phase(
        lambda: load_dense_fp16(args.model, args.device), args.device)
    # partial (not a closure): the model binds EAGERLY here, before the
    # later `del dense` — pyflakes' scope analysis and the runtime agree
    (dense_out, dense_decode_info), dense_decode_s = _time_phase(
        partial(greedy_decode_dispatch, dense, tokenizer, prompts,
                args.device, args.max_new_tokens,
                backend=args.decode_backend,
                verify_prompts=(0 if args.decode_backend == "generate"
                                else args.graphs_verify_prompts)),
        args.device)
    dense_timing = _timing_record(
        dense_load_s, dense_decode_s,
        sum(t.numel() for t in dense_out))
    dense_timing["decode_backend"] = dense_decode_info["backend"]
    print(f"  [TIME] fp16: decode {dense_timing['decode_seconds']}s "
          f"({dense_timing['new_tokens']} new tokens, "
          f"{dense_timing['tokens_per_second']} tok/s; load "
          f"{dense_timing['load_seconds']}s; decode-backend "
          f"{dense_decode_info['backend']})", flush=True)
    dense_ppl = None
    if windows is not None:
        dense_ppl = _ppl_arm(dense, windows, args.device,
                             args.ppl_batch_dense)
        print(f"  [PPL] fp16 wikitext2: {dense_ppl['ppl']:.4f} "
              f"({dense_ppl['n_tokens']} tokens, batch "
              f"{args.ppl_batch_dense})", flush=True)
    dense_vram = _vram_peak()
    if dense_vram:
        print(f"  [VRAM] fp16 phase peak: alloc "
              f"{dense_vram['peak_alloc_gib']} GiB / reserved "
              f"{dense_vram['peak_reserved_gib']} GiB", flush=True)
    del dense
    release_model_memory()
    _mem_line("after dense release")

    # W23: the honest artifact label — the auto recipe is mixed-radix
    # (mean ~2.31 bits/weight per INSPECTION §5), not a pure idx4 set
    print(f"[2/3] Palettized model (auto mixed-radix)"
          f"{' + residual' if args.residual else ''}"
          f"{' + qlora(' + args.qlora_adapters + ')' if args.qlora_adapters else ''}",
          flush=True)
    _reset_vram_peak()
    # load_quant_model returns (model, metadata); _time_phase wraps it as
    # ((model, metadata), seconds) — unpack BOTH layers (the pre-W21 code
    # kept `metadata` for the report; the probes below need the MODEL)
    (quant, _metadata), quant_load_s = _time_phase(
        lambda: load_quant_model(
            args.artifacts_dir, args.model, args.device,
            qlora_adapters=args.qlora_adapters, residual=args.residual,
            dtype=torch.float16, forward=args.forward,
            awq_compensation=not args.no_awq_compensation,
            heads_dir=args.heads_dir), args.device)
    # partial — same eager-binding note as the dense arm
    (quant_out, quant_decode_info), quant_decode_s = _time_phase(
        partial(greedy_decode_dispatch, quant, tokenizer, prompts,
                args.device, args.max_new_tokens,
                backend=args.decode_backend,
                verify_prompts=(0 if args.decode_backend == "generate"
                                else args.graphs_verify_prompts)),
        args.device)
    quant_timing = _timing_record(
        quant_load_s, quant_decode_s,
        sum(t.numel() for t in quant_out))
    quant_timing["decode_backend"] = quant_decode_info["backend"]
    print(f"  [TIME] quant: decode {quant_timing['decode_seconds']}s "
          f"({quant_timing['new_tokens']} new tokens, "
          f"{quant_timing['tokens_per_second']} tok/s; load "
          f"{quant_timing['load_seconds']}s; decode-backend "
          f"{quant_decode_info['backend']})", flush=True)
    quant_ppl = None
    if windows is not None:
        quant_ppl = _ppl_arm(quant, windows, args.device, args.ppl_batch)
        print(f"  [PPL] quant wikitext2: {quant_ppl['ppl']:.4f} "
              f"({quant_ppl['n_tokens']} tokens, batch "
              f"{args.ppl_batch})", flush=True)
    quant_vram = _vram_peak()
    if quant_vram:
        print(f"  [VRAM] quant phase peak: alloc "
              f"{quant_vram['peak_alloc_gib']} GiB / reserved "
              f"{quant_vram['peak_reserved_gib']} GiB", flush=True)
    del quant
    release_model_memory()
    _mem_line("after quant release")

    print("[3/3] Comparing", flush=True)
    records = []
    n_exact = 0
    first_divs = []
    for i, (a, b) in enumerate(zip(dense_out, quant_out)):
        exact, fd = compare(a, b, args.max_new_tokens)
        n_exact += int(exact)
        first_divs.append(fd)
        records.append({
            "id": i,
            "prompt": prompts[i],
            "exact_match": exact,
            "first_divergence": fd,
            "n_generated_fp16": int(a.numel()),
            "n_generated_quant": int(b.numel()),
            "fp16_text": tokenizer.decode(a, skip_special_tokens=True),
            "quant_text": tokenizer.decode(b, skip_special_tokens=True),
        })

    # W21: the timing summary (speedup only when both arms produced a
    # throughput — a 0-second arm, e.g. 1 prompt on CPU, is skipped).
    tps_f = dense_timing["tokens_per_second"]
    tps_q = quant_timing["tokens_per_second"]
    speedup = (round(tps_q / tps_f, 3)
               if tps_f and tps_q else None)

    # W21: the PPL summary (None when --no-ppl).
    ppl_report = None
    if dense_ppl is not None and quant_ppl is not None:
        ppl_report = {
            "protocol": ("wikitext-2-raw-v1 test split, non-overlapping "
                         "windows, PPL = exp(total NLL / total tokens) "
                         "(the eval_ppl convention)"),
            "seq_len": args.ppl_seq_len,
            "n_windows": int(windows.shape[0]),
            "n_eval_tokens": int(windows.numel()),
            "batch_dense": args.ppl_batch_dense,
            "batch_quant": args.ppl_batch,
            "fp16": dense_ppl,
            "quant": quant_ppl,
            "delta": quant_ppl["ppl"] - dense_ppl["ppl"],
            "delta_pct": 100.0 * (quant_ppl["ppl"] - dense_ppl["ppl"])
                         / dense_ppl["ppl"],
        }

    report = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_head": git_head(),
        "model": args.model,
        "artifacts_dir": args.artifacts_dir,
        "heads_dir": args.heads_dir,
        "residual": bool(args.residual),
        "qlora_adapters": args.qlora_adapters,
        "seed": args.seed,
        "n_prompts": len(prompts),
        "max_new_tokens": args.max_new_tokens,
        "aggregate": {
            "exact_match_fraction": n_exact / len(prompts),
            "mean_first_divergence": sum(first_divs) / len(first_divs),
            "median_first_divergence": sorted(first_divs)[len(first_divs) // 2],
            "min_first_divergence": min(first_divs),
            "max_first_divergence": max(first_divs),
        },
        "timing": {
            "fp16": dense_timing,
            "quant": quant_timing,
            "speedup_tokens_per_second": speedup,
        },
        "decode": {
            "requested_backend": args.decode_backend,
            "graphs_verify_prompts": (
                0 if args.decode_backend == "generate"
                else args.graphs_verify_prompts),
            "fp16": dense_decode_info,
            "quant": quant_decode_info,
        },
        "ppl": ppl_report,
        "vram": {
            "fp16": dense_vram,
            "quant": quant_vram,
        },
        "per_prompt": records,
        "environment": {
            "device": args.device,
            "torch": torch.__version__,
            "cuda": (str(torch.version.cuda)
                     if torch.cuda.is_available() else None),
            "transformers": _transformers_version(),
            "cuda_available": torch.cuda.is_available(),
            "gpu": (torch.cuda.get_device_name(0)
                    if torch.cuda.is_available() else "none"),
        },
    }

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(report, f, indent=2)
    agg = report["aggregate"]
    print(f"exact-match: {agg['exact_match_fraction']:.4f}  "
          f"first-divergence: mean {agg['mean_first_divergence']:.1f} / "
          f"median {agg['median_first_divergence']} / "
          f"min {agg['min_first_divergence']}", flush=True)
    if speedup is not None:
        print(f"[TIME] speedup: quant {tps_q} tok/s vs fp16 {tps_f} tok/s "
              f"= {speedup}x", flush=True)
    if ppl_report is not None:
        print(f"[PPL] wikitext2: fp16 {dense_ppl['ppl']:.4f} / quant "
              f"{quant_ppl['ppl']:.4f} (delta {ppl_report['delta']:+.4f}, "
              f"{ppl_report['delta_pct']:+.2f}%)", flush=True)
    print(f"wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
