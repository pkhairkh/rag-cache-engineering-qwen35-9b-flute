# PROPOSAL — PoC: LUT-Model Cache Engineering, K3-Style, OfficeQA on A10G

> **Status:** proposed — proof-of-concept — v2 (focused)
> **Scope:** the FLUTE idxN W4+r32 Qwen3.5-9B LUT model, **only**. No dense FP16/bf16 baseline. No Databricks. No SharePoint. No `ai_parse_document`. No Unity Catalog. The PoC is: the LUT model + a per-session linear-attention state cache + a global cross-session state cache + a W10 LUT fine-tune, measured on `databricks/officeqa` on one A10G.
> **Style:** Kimi K3 — the fixed per-head recurrent state is the cache asset; it is paged, content-addressed, namespace-keyed, LRU-evicted; the global pool shares state across sessions with identical prefixes.
> **Correction from v1:** v1 claimed the cache asset was 576 KiB per session. That was wrong. Reading `scripts/modeling.py::Qwen3_5GatedDeltaNet.forward` (lines 710, 738-739) shows the recurrent state is `cache_params.layers[layer_idx].recurrent_states[0]`, shape `(batch, num_v_heads=32, head_k_dim=128, head_k_dim=128)` per layer — a matrix, not a vector. The correct per-session size is **~25.5 MiB** (24 layers × 1 MiB recurrent + 1.5 MiB conv1d). v1 was off by ~44×. This document uses the corrected number throughout.

---

## 0. The question, the answer

> *"Databricks is out of scope. No FP16/bf16 baseline. Only the LUT model. Add a global cache — snapshot the linear-attention state, add a global cache too, fine-tune. KIMI K3 style."*

**Answer:** build a K3-style cache-engineering PoC in four layers, all on one A10G, all on the FLUTE idxN LUT model:

1. **The LUT model.** Qwen3.5-9B with FLUTE idxN W4+r32 (`--recipe auto --auto-cos 0.9995`), loaded via `scripts/eval_common.py::load_quant_model`. The only model in the PoC. No dense arm, no bf16 arm.

2. **The per-session state cache.** Snapshot the linear-attention recurrent state (`cache_params.layers[L].recurrent_states[0]`) and the conv1d state (`cache_params.layers[L].conv_states[0]`) at each of the 8 full-attention block boundaries (layers 3, 7, 11, 15, 19, 23, 27, 31). ~25.5 MiB per session. On a cache hit at a boundary: restore the state, skip the 3 preceding linear-attention layers' forward, continue from the full-attention block. This is the K3 per-head state snapshot.

3. **The global cache.** A cross-session, content-addressed pool of state snapshots, keyed by `(namespace, boundary_idx, state_content_hash)`. Sessions with identical prefixes share state. LRU-evicted. The global cache is what makes the K3 discipline real: a new session installing the system prompt + policy block hits the global cache at boundary 0 (the system prompt is the same for all sessions in a context class); a session re-running a query on an unchanged corpus block hits at deeper boundaries. The global cache is the cross-session reuse layer.

4. **The fine-tune.** Train the LUTs on OfficeQA Q&A pairs using the W10 two-stream path (`scripts/qlora_gemm.py::FusedQLoRAGEMMTrainLUTTwoStreams` + `scripts/qlora.py::attach_qlora` + `scripts/trainer.py`). The fine-tune recovers the accuracy lost to W4 quantization (the existing +3.52% PPL gap from `reports/greedy_equivalence_idx4.json`) and adapts the LUT model to the OfficeQA domain. Optionally: train with state-snapshot/restore in the loop (K3-style data augmentation — the model learns to produce state that survives re-installation).

**The PoC's single measurement:** on the same 50 OfficeQA questions, same LUT model, same A10G, measure the delta between (a) no cache, (b) per-session state cache only, (c) per-session + global cache. The delta is: accuracy (OfficeQA gold), throughput (tok/s), TTFT (ms), ITL (ms), VRAM (GiB), cache hit rate (%). The fine-tune is measured as the accuracy recovery before and after.

---

## 1. The K3-style design, concretely

### 1.1 The cache asset (corrected)

From `scripts/modeling.py::Qwen3_5GatedDeltaNet.forward` (lines 636-748):

```python
# line 710: the recurrent state is read from the cache
recurrent_state = cache_params.layers[self.layer_idx].recurrent_states[0] if use_precomputed_states else None

# lines 711-723: at decode (seq_len==1), the delta rule updates the state
core_attn_out, last_recurrent_state = torch_recurrent_gated_delta_rule(
    query, key, value, g=g, beta=beta,
    initial_state=recurrent_state,
    output_final_state=cache_params is not None,
    use_qk_l2norm_in_kernel=True,
)

# line 738-739: the updated state is written back to the cache
if cache_params is not None:
    cache_params.update_recurrent_state(last_recurrent_state, self.layer_idx)
```

The recurrent state shape (from `fla.ops.gated_delta_rule`): `(batch, num_v_heads, head_k_dim, head_k_dim)` = `(1, 32, 128, 128)` per layer.

The conv1d state (line 661): `cache_params.layers[self.layer_idx].conv_states[0]`, shape `(batch, conv_dim, conv_kernel_dim)` = `(1, 8192, 4)` per layer.

**Per-layer state sizes (fp16):**

| Component | Shape | Elements | Bytes (fp16) |
|---|---|---|---|
| Recurrent state | `(1, 32, 128, 128)` | 524,288 | 1,048,576 = **1.0 MiB** |
| Conv1d state | `(1, 8192, 4)` | 32,768 | 65,536 = **64 KiB** |
| **Per layer total** | | | **~1.06 MiB** |

**Per-session totals (24 linear-attention layers):**

| Component | Per layer | × 24 layers | Total |
|---|---|---|---|
| Recurrent state | 1.0 MiB | 24 MiB | **24 MiB** |
| Conv1d state | 64 KiB | 1.5 MiB | **1.5 MiB** |
| **Per-session total** | | | **~25.5 MiB** |

This is the cache asset. Not 576 KiB (v1's error — it confused the K/V projection shape with the recurrent state matrix shape). **25.5 MiB per session.**

**A10G capacity (W4+r32, ~5.85 GiB weights, ~13 GiB cache pool):**
- Per-session state: 25.5 MiB
- Concurrent session states in pool: 13 GiB / 25.5 MiB ≈ **530 concurrent sessions**
- At 8k context, the full-attention KV adds ~256 MiB per session (8 layers × 4 KiB/token × 8k tokens) — this is the working-set that grows, not the cache asset
- With full-attention KV: ~50 concurrent sessions at 8k context per replica

### 1.2 The per-session state cache

The per-session cache snapshots the linear-attention state at each of the 8 full-attention block boundaries. The boundaries are the 8 full-attention layers (layers 3, 7, 11, 15, 19, 23, 27, 31 per `full_attention_interval: 4`).

**Why boundaries matter:** after each full-attention layer, the linear-attention state from the 3 preceding layers is complete for that 4-layer block. A snapshot at boundary N captures the state of all linear-attention layers that have run so far (layers 0, 1, 2 at boundary 1; layers 0, 1, 2, 4, 5, 6 at boundary 2; etc.). The state at boundary N INCLUDES the state at all earlier boundaries (the recurrent state is cumulative — each layer's state evolves over the whole sequence).

**The snapshot/restore hooks:**

```python
# poc/state_cache.py
BOUNDARY_LAYERS = [3, 7, 11, 15, 19, 23, 27, 31]  # the 8 full-attention layers

class PerSessionStateCache:
    """Snapshots the linear-attention recurrent + conv1d state at each
    full-attention block boundary. On a hit at boundary N: restore all
    linear-attention layers' states, skip the forward through those layers,
    continue from full-attention layer N's output."""

    def snapshot(self, model, cache_params, boundary_idx: int) -> StateSnapshot:
        """Called after full-attention layer BOUNDARY_LAYERS[boundary_idx].
        Copies recurrent_states[0] and conv_states[0] for every
        linear-attention layer < BOUNDARY_LAYERS[boundary_idx] to host.
        Returns a StateSnapshot (content-hashed, namespace-keyed)."""

    def restore(self, model, cache_params, snapshot: StateSnapshot):
        """Copies the snapshot's states back to cache_params.layers[L].
        recurrent_states[0] and conv_states[0] for each linear-attention
        layer in the snapshot. The model resumes forward from the boundary."""

    def lookup(self, namespace, boundary_idx, content_hash) -> Optional[StateSnapshot]:
        """Content-addressed lookup in the per-session LRU pool."""
```

**The content hash** is a SHA-256 of the concatenation of all state tensors at the boundary, computed on the host after the snapshot copy. This is the K3 "content hashes at 512-token granularity" discipline, applied to the linear-attention state instead of the full-attention KV.

### 1.3 The global cache

The global cache is the cross-session reuse layer. It stores state snapshots keyed by `(namespace, boundary_idx, content_hash)`. Multiple sessions can install/lookup the same snapshot.

**When the global cache hits:**

| Boundary | What's shared | Hit rate (expected) |
|---|---|---|
| Boundary 0 (before any tokens) | Nothing — the initial state is zeros | N/A (trivial) |
| Boundary 1 (after system prompt + policy block) | The system prompt is the same for all sessions in a context class | **High** — every session hits |
| Boundary 2 (after system prompt + policy + first corpus chunk) | The first retrieved chunk varies per query | **Low** — depends on retrieval overlap |
| Boundaries 3-7 | Session-specific conversation turns + retrieved chunks | **Very low** — per-session only |

The global cache primarily benefits **boundary 1** (and to a lesser extent boundary 2) — the shared prefix. The per-session cache benefits boundaries 3-7 (session-specific state, reusable within the session on re-prefill after a doc edit or re-query).

```python
# poc/global_cache.py
class GlobalStateCache:
    """Cross-session, content-addressed pool of linear-attention state
    snapshots. Keyed by (namespace, boundary_idx, state_content_hash).
    LRU-evicted. Thread-safe (the A10G serves one request at a time, but
    the cache is shared across sequential requests)."""

    NAMESPACE_KEY = ("model_checkpoint_id", "quant_recipe_signature",
                     "context_class", "corpus_version")

    def install(self, namespace, boundary_idx, snapshot: StateSnapshot) -> bool:
        """Store a snapshot. Returns True if new, False if already present
        (the content hash matched an existing entry — the snapshot is
        deduplicated)."""

    def lookup(self, namespace, boundary_idx, content_hash) -> Optional[StateSnapshot]:
        """Content-addressed lookup. Returns the snapshot if present, None
        if miss. On a hit, the caller restores the state and skips the
        forward through the linear-attention layers."""

    def evict_lru(self, target_bytes: int) -> int:
        """Evict least-recently-used entries until the pool is under
        target_bytes. Returns the number of entries evicted."""
```

**The namespace key** (from the production proposal v1.1, made concrete):

```python
@dataclass
class CacheNamespace:
    model_checkpoint_id: str        # SHA-256 of the model weights
    quant_recipe_signature: str     # SHA-256 of metadata.json's auto_decision ledger
    context_class: str              # hash of (system_prompt, policy_block)
    corpus_version: str             # hash of the OfficeQA corpus version
```

A fine-tuned model is a different `model_checkpoint_id`. A re-palettized model is a different `quant_recipe_signature`. A changed system prompt is a different `context_class`. A corpus update is a different `corpus_version`. Cross-namespace reuse is refused (KVShareArena finding: unrepaired reuse is worse than no cache).

### 1.4 The fine-tune

The fine-tune trains the LUTs on OfficeQA Q&A pairs. The W10 two-stream training path is already shipped in the repo:

- `scripts/qlora.py::attach_qlora` — wraps the palettized model with trainable LUT masters (fp32 Parameters, straight-through estimator)
- `scripts/qlora_gemm.py::FusedQLoRAGEMMTrainLUTTwoStreams` — the W10 autograd Function (forward: two `qgemm_per_group_lut` calls + ordered add; backward: `lut_grad_scatter_sub4_kernel` per stream)
- `scripts/trainer.py` — the layerwise distillation trainer (two-layer residency: one teacher layer + one student layer resident at a time; the full student model is never built — this is what makes it fit on A10G)
- `scripts/muon_optimizer.py` — the Muon optimizer for the LoRA branch

**The fine-tune's two objectives:**

1. **Accuracy recovery.** The existing W4 model has +3.52% PPL on WikiText-2 (`reports/greedy_equivalence_idx4.json`). The fine-tune on OfficeQA's Q&A pairs recovers task-specific accuracy. This is standard QLoRA on the LUTs — the W10 path makes it trainable without materializing the (N, K) `dW` transient to DRAM (the `lut_grad_scatter_sub4_kernel`'s whole purpose, per `docs/KERNEL_SPEC_DLDLUT.md`).

2. **K3-style cache-aware fine-tune (optional, the PoC's research bet).** Train with state-snapshot/restore in the loop as data augmentation: occasionally restore a state from an earlier boundary and continue generation from there, forcing the model to produce recurrent state that survives re-installation. This is the K3 principle — K3 was trained with the cache mechanics in mind; our LUT model was not. The fine-tune is where we adapt the LUT model to the cache discipline.

**The fine-tune's data:**

```python
# poc/officeqa_sft.py
def load_officeqa_sft(
    hf_dataset: str = "databricks/officeqa",
    split: str = "test",
    tokenizer,
    max_samples: int = 200,
    seq_len: int = 2048,
) -> List[Dict]:
    """Loads OfficeQA Q&A pairs as SFT examples.
    Format: [system: "You are a financial analyst. Answer the question
    based on the provided context."] + [context: retrieved chunks] +
    [question] + [answer].
    Returns [{input_ids, labels, attention_mask}] like scripts/data.py."""
```

### 1.5 The K3 discipline (the standing rules)

From Kimi K3 §5.5, applied to the linear-attention state:

| K3 rule | PoC implementation |
|---|---|
| Hash ≠ physical granularity | Content hash at boundary granularity (8 boundaries per session); physical storage at per-layer granularity (24 layers × 1 MiB) |
| State checkpoints at sparse boundaries | Only at the 8 full-attention block boundaries, not at every token |
| Atomic invalidation across cache groups | A hit is installed only if ALL linear-attention layers' states at the boundary are present and consistent; partial installs are refused |
| Two-stage prefix matching | Stage 1: whole-boundary chained-hash match (the global cache). Stage 2: hash-endpoint fallback inside the first missing boundary (recompute the 3 linear-attention layers between the last hit boundary and the current position) |
| Cache namespaces | `(model_checkpoint_id, quant_recipe_signature, context_class, corpus_version)` — cross-namespace reuse refused |
| Edit-local repair | On a doc edit, restore the state at the boundary before the edit, recompute the 3 linear-attention layers between that boundary and the query (Contiguity finding: 13-21× cheaper than re-prefill) |
| LRU eviction | Plain LRU per replica; no learned policies (14 sophisticated policies fail to beat LRU) |
| Session-affinity scheduling | Consistent hashing to the replica that already holds the session's prefix |

---

## 2. The literature (focused)

Only the papers that directly shape the K3-style PoC. The OfficeQA Pro paper, the chunking taxonomy, and the enterprise RAG guides are dropped (no Databricks, no parser comparison, no corpus lane).

| Paper | arXiv | What the PoC uses |
|---|---|---|
| **Kimi Linear** | [2510.26692](https://arxiv.org/abs/2510.26692) | The KDA mechanics: the fixed per-head recurrent state is the cache asset; the state is snapshot-able, paged, content-addressed. The 6× decode figure is at 1M context — the PoC measures at OfficeQA's 4k-16k context. |
| **Kimi K3 §5.5** | [K3 platform docs](https://platform.kimi.ai/docs/guide/kimi-k3-quickstart) + [Semianalysis](https://inferencex.semianalysis.com/model/kimi-k3) | The cache discipline: hash ≠ physical granularity, atomic invalidation, namespace admission, edit-local repair, session-affinity. The PoC's §1.5 standing rules are K3 §5.5 verbatim. |
| **Contiguity** | [2609.17983](https://arxiv.org/abs/2609.17983) | Edit-local repair: 13-21× cheaper than re-prefill; recovers ≥0.94 of the post-edit answer margin. The PoC's doc-edit repair policy. |
| **Prefix cache eviction** | [2609.28870](https://arxiv.org/abs/2609.28870) | LRU beats 14 sophisticated policies under agentic load; recency is unusually predictive. The PoC's eviction policy. The two-oracle diagnostic (Belady vs BeladyCompute) decides if anything beyond LRU is needed. |
| **BoxOffice** | [2609.31415](https://arxiv.org/abs/2609.31415) | 42% of reported F1 gains from KV reuse are metric artifacts; F1 flips 0.98 ↔ 0.00 on staleness. The PoC's meaningful-query filter is mandatory before any accuracy claim. |
| **KVShareArena** | [2609.10266](https://arxiv.org/abs/2609.10266) | Free position rotation recovers only 50-66% of the gap; unrepaired cross-context reuse worse than no cache. The PoC refuses cross-namespace reuse. |
| **PatchKV** | [2609.26219](https://arxiv.org/html/2609.26219v1) | Transport-based KV recovery. The PoC's Round 3 comparison baseline for the doc-edit repair axis. |

---

## 3. What works (the existing infrastructure)

The PoC is built on top of infrastructure that already works. Every component is a file in `qwen3_5_9B_flute_qlora_v1.3`:

| Component | File | PoC role |
|---|---|---|
| **FLUTE idxN forward kernel** | `flute_extended/src/kernel_cutlass_streaming.cu` (`flute_kernel_streaming_fd_sub4<Cfg, B>`) | The dequant+GEMM hot path for every linear layer. ~58.5 TFLOPS (93% peak A10G). |
| **FLUTE idxN backward kernel** | `flute_train_kernels/src/kernel_lut_grad.cu` (`lut_grad_scatter_sub4_kernel`) | The dL/dLUT scatter. Makes W4 trainable on A10G without OOM (the (N,K) dW transient never touches DRAM). |
| **Two-stream training (W10)** | `scripts/qlora_gemm.py::FusedQLoRAGEMMTrainLUTTwoStreams` | The fine-tune's autograd Function for palette > 16 (e.g., hybrid422). |
| **The QLoRA wrapper** | `scripts/qlora.py::attach_qlora` + `QLoRAConfig` | Wraps the palettized model with trainable LUT masters. |
| **The trainer** | `scripts/trainer.py` | The layerwise distillation trainer (two-layer residency — fits on A10G). |
| **The Muon optimizer** | `scripts/muon_optimizer.py` | The optimizer for the LoRA branch. |
| **The fla wiring** | `scripts/modeling.py::_fla_resolve` (W29) + `Qwen3_5GatedDeltaNet.forward` | The linear-attention state mechanics: `fla.ops.gated_delta_rule.fused_recurrent_gated_delta_rule` at decode, `torch_chunk_gated_delta_rule` at prefill. The recurrent state IS the cache asset. |
| **The sm86 flash attention** | `scripts/attn_sm86.py` | The 8 full-attention layers' KV (the secondary cache). |
| **The palettizer** | `scripts/palettize_qwen3_5_9b.py` | `--recipe auto --auto-cos 0.9995` produces the W4+r32 artifacts + the `auto_decision` ledger. |
| **The palettized module** | `scripts/palettized_modules.py::PalettizedLinear` | The runtime module: kernel path on CUDA, reference path on CPU. |
| **The model loader** | `scripts/eval_common.py::load_quant_model` | Loads the W4+r32 model with all the AWQ compensation, norm-gain edits, heads-dir handling. |
| **Greedy decode with KV cache** | `scripts/eval_greedy_match.py::greedy_decode` + `greedy_decode_dispatch` | KV-cache greedy decode with CUDA graphs. |
| **The energy harness** | `scripts/measure_energy.py::EnergyMeasurement` | NVML energy, throughput, VRAM, idle baseline subtraction, clock locking. |
| **The paired probe** | `scripts/o1_baseline_check.py::score_docs` + `paired_diff` | Paired per-document measurement (for the fine-tune's accuracy recovery). |
| **The VRAM ledger** | `scripts/vram_ledger.py` | Every allocation recorded. Extended with a `cache_pool` tier. |
| **The kernel parity suite** | `tests/test_kernel_status.py`, `tests/test_lut_gradients.py`, `tests/test_two_stream_training.py`, `tests/test_attn_kernel.py` | The CI precondition for any cache work. |
| **The existing artifacts** | `/home/ubuntu/qwen3_5_9B_palettized` + `_heads` (from the handover) | Round 1 starts immediately — no re-palettization needed if the existing recipe signature is accepted. |

---

## 4. What's added (the new code)

Four scripts under `poc/` in the new repo. Each is small (<300 lines), CPU-tested first, built on top of the existing eval plane.

### 4.1 `poc/state_cache.py` — the per-session state cache

```python
# poc/state_cache.py
"""The per-session linear-attention state cache. Snapshots the recurrent
state + conv1d state at each of the 8 full-attention block boundaries.
On a hit: restore the state, skip the linear-attention forward, continue."""

import torch
import hashlib
from dataclasses import dataclass
from typing import Optional, Dict, List

# The 8 full-attention layers (every 4th, 0-indexed)
BOUNDARY_LAYERS = [3, 7, 11, 15, 19, 23, 27, 31]
# The linear-attention layers (the other 24)
LINEAR_LAYERS = [i for i in range(32) if i not in BOUNDARY_LAYERS]

@dataclass
class StateSnapshot:
    namespace: tuple           # (checkpoint, recipe, context_class, corpus_version)
    boundary_idx: int          # 0..7
    content_hash: str          # SHA-256 of the concatenated state tensors
    recurrent_states: Dict[int, torch.Tensor]  # {layer_idx: (32, 128, 128) fp16 on CPU}
    conv_states: Dict[int, torch.Tensor]       # {layer_idx: (8192, 4) fp16 on CPU}
    size_bytes: int

class PerSessionStateCache:
    def __init__(self, max_entries: int = 64):
        self._lru: List[StateSnapshot] = []  # most-recent at end
        self._index: Dict[tuple, StateSnapshot] = {}  # (boundary_idx, content_hash) → snapshot
        self._max = max_entries

    def snapshot(self, model, cache_params, namespace, boundary_idx) -> StateSnapshot:
        """Snapshot all linear-attention layers' states up to boundary_idx.
        Copies to CPU, hashes, stores in the LRU pool."""
        recurrent = {}
        conv = {}
        for layer_idx in LINEAR_LAYERS:
            if layer_idx < BOUNDARY_LAYERS[boundary_idx]:
                recurrent[layer_idx] = cache_params.layers[layer_idx].recurrent_states[0].cpu().clone()
                conv[layer_idx] = cache_params.layers[layer_idx].conv_states[0].cpu().clone()
        # content hash
        hasher = hashlib.sha256()
        for idx in sorted(recurrent.keys()):
            hasher.update(recurrent[idx].numpy().tobytes())
            hasher.update(conv[idx].numpy().tobytes())
        content_hash = hasher.hexdigest()
        snap = StateSnapshot(namespace, boundary_idx, content_hash,
                             recurrent, conv, sum(t.nelement()*2 for t in recurrent.values()) + sum(t.nelement()*2 for t in conv.values()))
        self._install(snap)
        return snap

    def lookup(self, namespace, boundary_idx, content_hash) -> Optional[StateSnapshot]:
        key = (boundary_idx, content_hash)
        snap = self._index.get(key)
        if snap and snap.namespace == namespace:
            self._touch(snap)
            return snap
        return None

    def restore(self, model, cache_params, snapshot: StateSnapshot):
        """Copy the snapshot's states back to the GPU cache_params."""
        for layer_idx, state in snapshot.recurrent_states.items():
            cache_params.layers[layer_idx].recurrent_states[0].copy_(state.to(cache_params.layers[layer_idx].recurrent_states[0].device))
        for layer_idx, state in snapshot.conv_states.items():
            cache_params.layers[layer_idx].conv_states[0].copy_(state.to(cache_params.layers[layer_idx].conv_states[0].device))

    def _install(self, snap): ...
    def _touch(self, snap): ...
```

### 4.2 `poc/global_cache.py` — the cross-session global cache

```python
# poc/global_cache.py
"""The global cross-session state cache. Content-addressed, namespace-keyed,
LRU-evicted. Sessions with identical prefixes share state."""

class GlobalStateCache:
    def __init__(self, max_pool_bytes: int = 10 * 1024**3):  # 10 GiB default
        self._pool: Dict[tuple, StateSnapshot] = {}  # (namespace, boundary_idx, content_hash) → snapshot
        self._lru: List[tuple] = []
        self._max_bytes = max_pool_bytes
        self._current_bytes = 0

    def install(self, snapshot: StateSnapshot) -> bool:
        """Store a snapshot. Deduplicates by content hash.
        Returns True if new, False if already present."""
        key = (snapshot.namespace, snapshot.boundary_idx, snapshot.content_hash)
        if key in self._pool:
            self._touch(key)
            return False
        # evict if over capacity
        while self._current_bytes + snapshot.size_bytes > self._max_bytes and self._lru:
            self._evict_oldest()
        self._pool[key] = snapshot
        self._lru.append(key)
        self._current_bytes += snapshot.size_bytes
        return True

    def lookup(self, namespace, boundary_idx, content_hash) -> Optional[StateSnapshot]:
        key = (namespace, boundary_idx, content_hash)
        snap = self._pool.get(key)
        if snap:
            self._touch(key)
        return snap
```

### 4.3 `poc/finetune.py` — the W10 LUT fine-tune on OfficeQA

```python
# poc/finetune.py
"""Fine-tune the LUT model on OfficeQA Q&A pairs. Uses the existing W10
two-stream training path (scripts/qlora_gemm.py::FusedQLoRAGEMMTrainLUTTwoStreams)
+ the layerwise trainer (scripts/trainer.py)."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "qwen3_5_9B_flute_qlora_v1.3", "scripts"))

from qlora import attach_qlora, QLoRAConfig
from eval_common import load_quant_model, release_model_memory

def finetune_on_officeqa(
    artifacts_dir: str,
    heads_dir: str,
    output_dir: str,
    hf_dataset: str = "databricks/officeqa",
    n_samples: int = 200,
    r: int = 64,
    alpha: int = 16,
    lr_lut: float = 1e-4,
    lr_lora: float = 2e-4,
    n_steps: int = 500,
    device: str = "cuda:0",
):
    """1. Load the W4+r32 LUT model.
    2. Attach QLoRA (trainable LUT masters + LoRA A/B).
    3. Load OfficeQA Q&A pairs as SFT examples.
    4. Run the layerwise trainer (two-layer residency — fits on A10G).
    5. Export the fine-tuned LUTs + the recipe signature (SHA-256 of the
       updated metadata.json auto_decision ledger — the new namespace).
    6. The fine-tuned model is a different checkpoint_id → different cache
       namespace (KVShareArena: refuse cross-checkpoint reuse)."""
    model, metadata = load_quant_model(
        artifacts_dir, "Qwen/Qwen3.5-9B", device,
        residual=True, forward="kernel", heads_dir=heads_dir)
    config = QLoRAConfig(r=r, alpha=alpha, scope="all",
                         base_model="Qwen/Qwen3.5-9B", artifacts_dir=artifacts_dir)
    attach_qlora(model, metadata, **config.__dict__)
    # ... load OfficeQA SFT data, run trainer.py's layerwise loop, export
```

### 4.4 `poc/run_poc.py` — the orchestrator

```python
# poc/run_poc.py
"""Run the K3-style cache-engineering PoC on OfficeQA on A10G.
Four phases: env, fine-tune, cache measurement, staleness envelope."""

def main():
    # P0: env + kernel parity
    # P1: fine-tune the LUT model on OfficeQA
    # P2: measure (no-cache vs per-session vs per-session+global)
    # P3: staleness envelope (doc-edit, checkpoint-change, role-flip)
    # P4: verdict
    ...
```

---

## 5. The build path

Five phases, strictly sequential. Total: 5 weeks of box time on one A10G.

| Phase | Builds | Box time | Exit criteria |
|---|---|---|---|
| **P0. Environment + kernel parity** | Confirm A10G on the allowlist. Build FLUTE idxN wheels. Run the kernel parity suite (6 test files). Confirm the existing `/home/ubuntu/qwen3_5_9B_palettized` artifacts load. Record the `auto_decision` ledger; SHA-256 it as the `quant_recipe_signature`. | 3 days | Kernel parity green; LUT model loads and generates; the 32-prompt sanity check matches the existing `reports/greedy_equivalence_idx4.json` numbers (within tolerance). |
| **P1. Fine-tune the LUT model** | `poc/finetune.py`. Load OfficeQA. Attach QLoRA. Run the layerwise trainer (500 steps). Export the fine-tuned LUTs. Measure: PPL on WikiText-2 (target: < 9.5749, the pre-fine-tune figure), OfficeQA accuracy on 50 questions (target: > pre-fine-tune accuracy — the fine-tune recovers the quantization gap). | 1 week | Fine-tuned model exported; new `quant_recipe_signature` recorded (a different namespace); PPL and OfficeQA accuracy measured before and after. |
| **P2. Cache measurement** | `poc/state_cache.py` + `poc/global_cache.py`. Run 50 OfficeQA questions through three cache configurations: (a) no cache, (b) per-session state cache only, (c) per-session + global cache. Measure: accuracy, throughput (tok/s), TTFT, ITL, VRAM, cache hit rate per boundary. Run the two-oracle diagnostic. | 1.5 weeks | The delta between (a), (b), (c) measured. The global cache's hit rate at boundary 1 (shared system prompt) measured. The two-oracle gap documented (if <5%, plain LRU is the policy). |
| **P3. Staleness envelope** | Three axes: (1) doc-edit — replace one chunk in the corpus block, measure repair cost (edit-local vs re-prefill vs PatchKV); (2) checkpoint-change — the pre-fine-tune vs post-fine-tune models are different namespaces; refuse cross-checkpoint reuse, measure the cost; (3) role-flip — change the system prompt, measure the namespace switch cost. | 1 week | Envelope curves on 3 axes. The edit-local repair ratio measured (target: 13-21× per the Contiguity paper). The cross-checkpoint refusal verified. |
| **P4. Verdict** | `poc/reports/poc_verdict_<ts>.json`. The single document with all measured numbers. | 3 days | The verdict is `go`, `no_go`, or `conditional_go` with measured numbers for every claim. |

**Rollback.** If P1 fails (the fine-tune does not recover accuracy, or the trainer OOMs), the PoC uses the pre-fine-tune model for P2-P3. The fine-tune is an enhancement, not a blocker. If P2 fails (the per-session + global cache does not beat no-cache on throughput), the PoC continues to P3 to document the failure mode.

---

## 6. The decision record

| # | Decision | Rejected alternative | Why |
|---|---|---|---|
| K3-1 | **LUT model only** (FLUTE idxN W4+r32 Qwen3.5-9B) | Dense FP16/bf16 baseline; multi-model comparison | User directive. The PoC is about the LUT model's cache engineering, not about quantization vs dense. |
| K3-2 | **No Databricks, no SharePoint, no ai_parse_document** | The production proposal's four-lane architecture | User directive. This is a single-machine PoC. |
| K3-3 | **Per-session linear-attention state cache** (25.5 MiB per session, 8 boundaries) | Full-attention KV cache only (the vLLM standard) | The linear-attention state is fixed-size (does not grow with context), NoPE (re-installable at any position), and is the K3 cache asset. The full-attention KV grows with context and is the working-set, not the cache asset. |
| K3-4 | **Global cross-session state cache** (content-addressed, namespace-keyed, LRU) | Per-session cache only | The global cache enables cross-session reuse at the shared-prefix boundaries (system prompt, policy block). Without it, every session re-computes the same state from scratch. This is the K3 cross-session paged pool. |
| K3-5 | **Fine-tune the LUTs on OfficeQA** (W10 two-stream, layerwise trainer) | Use the pre-fine-tune W4 model as-is | The existing W4 model has +3.52% PPL on WikiText-2 and exact_match 0.0 on the greedy match. The fine-tune recovers task-specific accuracy, making the cache engineering meaningful (if accuracy is too low, cache hit rate is irrelevant). |
| K3-6 | **K3-style cache-aware fine-tune** (optional: train with state snapshot/restore in the loop) | Standard QLoRA only | K3 was trained with the cache mechanics in mind. Our LUT model was not. The cache-aware fine-tune adapts the model to produce state that survives re-installation. This is the PoC's research bet; the standard fine-tune is the fallback. |
| K3-7 | **Content-addressed state hashing** (SHA-256 of the concatenated state tensors) | Position-based addressing | The state's content, not its position, determines reusability. Two sessions with the same prefix produce the same state. Content addressing enables cross-session deduplication. |
| K3-8 | **Namespace-keyed admission** (refuse cross-namespace reuse) | Free rotation, cross-checkpoint reuse | KVShareArena: free rotation recovers only 50-66% of the gap; unrepaired reuse worse than no cache. The fine-tuned model is a different namespace; refuse cross-checkpoint reuse. |
| K3-9 | **Plain LRU eviction** | Learned, frequency-based, analytic policies | The prefix-cache eviction paper: 14 sophisticated policies fail to beat LRU under agentic load. Run the two-oracle diagnostic first; if the gap is <5%, ship LRU. |
| K3-10 | **Edit-local contiguous repair** on doc-edit | Full re-prefill; importance-based repair; PatchKV transport-based | Contiguity: edit-local recovers ≥0.94 of the answer margin at 13-21× below re-prefill. PatchKV is the comparison baseline (P3 measures both). |
| K3-11 | **OfficeQA as the workload** | WikiText-2 PPL only; synthetic prompts | OfficeQA has gold answers (measurable accuracy), real enterprise documents (Treasury Bulletins — dense tables, the "not in the best state" condition), and is CC-BY-SA-4.0 (no license friction). WikiText-2 PPL is a sanity check (target: < 9.5749), not the primary metric. |
| K3-12 | **50 questions, meaningful-query filtered** | All questions; raw accuracy | BoxOffice: 42% of F1 gains are metric artifacts. The meaningful-query filter removes questions the cache-free run fails anyway (model capability, not cache), questions answerable without context (world knowledge), and low-information yes/no questions. The filter's cut list is itself a finding. |

---

## 7. Risks and standing controls

| Risk | Evidence | Standing control |
|---|---|---|
| **The per-session state cache (25.5 MiB) is too large to be a "paged object"** | v1 claimed 576 KiB; the corrected number is 25.5 MiB — 44× larger. At 530 concurrent sessions per replica, the cache pool is full. | P2 measures the actual hit rate at each boundary. If boundary 1 (the shared system prompt) has a high global-cache hit rate, the effective per-session cost is much lower (the shared state is stored once in the global pool, not per-session). The 25.5 MiB is the worst case; the global cache is the mitigation. |
| **The linear-attention state's advantage at 4k-16k context is unmeasured** | Kimi Linear's 6× decode speedup is at 1M context. At OfficeQA's context, the full-attention KV (which grows with context) may still be cheaper than the linear-attention state (which is fixed). | P2 measures the delta directly. If the linear-attention state cache does not beat no-cache, the PoC documents it. The K3 thesis may hold only at long context. |
| **The fine-tune OOMs on A10G** | The handover log records OOMs in the palettizer; the trainer's two-layer residency is designed to avoid this, but it's untested on OfficeQA. | P1 uses the existing `scripts/trainer.py` with the two-layer residency (one teacher + one student layer resident at a time — the full model is never built). If the trainer OOMs, reduce `n_samples` to 100 and `seq_len` to 1024. |
| **The global cache's hit rate at boundary 1 is low** | The system prompt is the same for all sessions in a context class, but if the context class changes frequently (different system prompts), the global cache misses. | P2 measures the hit rate per boundary. If boundary 1's hit rate is <80%, the global cache's value is limited to deeper boundaries (which have lower hit rates by construction). |
| **The content hash computation is too expensive** | SHA-256 of 25.5 MiB per snapshot, per boundary, per session. | The hash is computed on the CPU after the GPU→CPU copy (which is the dominant cost anyway). The hash itself is ~1 ms for 25 MiB on modern CPUs. The GPU→CPU copy (~25 MiB at 12 GB/s PCIe) is ~2 ms. Total snapshot overhead: ~3 ms per boundary. |
| **The state restore produces different outputs than re-computation** | The recurrent state is fp16; the fla kernel accumulates in fp32 internally but stores fp16. Restoring fp16 state and continuing may introduce drift. | P2 validates: run the same prompt with (a) no cache (full re-computation) and (b) state cache (snapshot + restore), compare token-by-token. If the outputs diverge, the cache is lossy — document the divergence rate. |
| **The fla wiring is not active at decode time** | `scripts/modeling.py::_fla_resolve` (W29) settles during the pre-capture warmup. | P0's sanity check (the 32 deterministic prompts) validates the wiring. If throughput is below 10 tok/s, `FLUTE_NO_FLA=1` restores the pure-torch fallback for differential diagnosis. |
| **The GEMV kernel caps throughput at 5-6.5× below the bandwidth wall** | `docs/A10G_DECODE_INVESTIGATION.md`: the decode-GEMV is the binding constraint, not the cache. | Out of scope. The PoC measures the cache delta *on top of* the existing kernel. If the GEMV is the bottleneck, the cache engineering's throughput improvement is masked — the PoC documents this and notes that the GEMV fix (in `flute_extended/src/kernel_cutlass_streaming.cu`) is a prerequisite for the throughput claim. |

---

## 8. Traceability

| PoC element | Source |
|---|---|
| The LUT model (FLUTE idxN W4+r32) | `scripts/palettize_qwen3_5_9b.py --recipe auto --auto-cos 0.9995`; `docs/QUANTIZATION_FORMAT.md` §2; `docs/AUTO_SELECTION_GUIDE.md` |
| The 3:1 hybrid is native | `docs/MODEL_GEOMETRY.md` §1 (`layer_types`, `full_attention_interval: 4`) |
| The recurrent state (the cache asset) | `scripts/modeling.py::Qwen3_5GatedDeltaNet.forward` lines 710, 738-739; `cache_params.layers[layer_idx].recurrent_states[0]` |
| The conv1d state | `scripts/modeling.py` line 661; `cache_params.layers[layer_idx].conv_states[0]` |
| The fla wiring | `scripts/modeling.py::_fla_resolve` (W29); `fla.ops.gated_delta_rule.fused_recurrent_gated_delta_rule` |
| The forward kernel | `flute_extended/src/kernel_cutlass_streaming.cu` (`flute_kernel_streaming_fd_sub4`) |
| The backward kernel | `flute_train_kernels/src/kernel_lut_grad.cu` (`lut_grad_scatter_sub4_kernel`); `docs/KERNEL_SPEC_DLDLUT.md` |
| The two-stream training (W10) | `scripts/qlora_gemm.py::FusedQLoRAGEMMTrainLUTTwoStreams`; `docs/TWO_STREAM_ANALYSIS.md` |
| The QLoRA wrapper | `scripts/qlora.py::attach_qlora` + `QLoRAConfig` |
| The trainer | `scripts/trainer.py` (two-layer residency) |
| The model loader | `scripts/eval_common.py::load_quant_model` |
| The energy harness | `scripts/measure_energy.py::EnergyMeasurement` |
| The kernel parity suite | `tests/test_kernel_status.py`, `tests/test_lut_gradients.py`, `tests/test_two_stream_training.py`, `tests/test_attn_kernel.py`, `tests/test_dequant_reference.py`, `tests/test_idxn_pack_cpu.py` |
| The existing measurement | `reports/greedy_equivalence_idx4.json` (exact_match 0.0, +3.52% PPL, 0.571× dense throughput) |
| Kimi Linear (KDA mechanics) | [arXiv:2510.26692](https://arxiv.org/abs/2510.26692) |
| Kimi K3 (cache discipline) | [K3 platform docs](https://platform.kimi.ai/docs/guide/kimi-k3-quickstart); [Semianalysis](https://inferencex.semianalysis.com/model/kimi-k3) |
| Contiguity (edit-local repair) | [arXiv:2609.17983](https://arxiv.org/abs/2609.17983) |
| LRU eviction | [arXiv:2609.28870](https://arxiv.org/abs/2609.28870) |
| BoxOffice (measurement discipline) | [arXiv:2609.31415](https://arxiv.org/abs/2609.31415) |
| KVShareArena (refuse cross-namespace) | [arXiv:2609.10266](https://arxiv.org/abs/2609.10266) |
| PatchKV (comparison baseline) | [arXiv:2609.26219](https://arxiv.org/html/2609.26219v1) |
| OfficeQA dataset | [databricks/officeqa on HuggingFace](https://huggingface.co/datasets/databricks/officeqa) |
| OfficeQA Pro (34.1% frontier average) | [arXiv:2603.08655](https://arxiv.org/abs/2603.08655) |
| The A10G spec | `docs/GPU_SPEC.md` (24 GiB, sm_86, 600 GB/s) |
| The decode investigation | `docs/A10G_DECODE_INVESTIGATION.md` |
| The handover OOM log | `scripts/HANDOVER.issue-2026-10-04.md` |

---

## 9. Immediate next actions

1. **Phase P0 (3 days).** Clone the new repo. Confirm A10G on the GPU contract allowlist. Build the FLUTE idxN wheels. Run the kernel parity suite. Load the existing `/home/ubuntu/qwen3_5_9B_palettized` artifacts. Run the 32-prompt sanity check. Record the `quant_recipe_signature`.
2. **Phase P1 (1 week).** Implement `poc/finetune.py`. Load OfficeQA. Attach QLoRA. Run the layerwise trainer (500 steps, two-layer residency). Export the fine-tuned LUTs. Measure PPL and OfficeQA accuracy before and after.
3. **Phase P2 (1.5 weeks).** Implement `poc/state_cache.py` + `poc/global_cache.py`. Run 50 OfficeQA questions through three cache configurations. Measure the delta. Run the two-oracle diagnostic.
4. **Phase P3 (1 week).** Run the staleness envelope on three axes (doc-edit, checkpoint-change, role-flip). Compare edit-local repair vs PatchKV.
5. **Phase P4 (3 days).** Produce `poc_verdict_<ts>.json`. The verdict with measured numbers.

**The PoC's single success criterion:** the K3-style cache (per-session linear-attention state + global cross-session pool) is *measured* to beat no-cache on throughput at equal accuracy, on the LUT model, on A10G, on OfficeQA. Whatever the measurement says, the PoC has done its job.

---

## Appendix A — The corrected cache arithmetic

This appendix is the worked arithmetic for every number in this proposal, so any reviewer can recompute for a different GPU or model.

### A.1 The recurrent state per layer

From `docs/MODEL_GEOMETRY.md` §1 and `scripts/modeling.py::Qwen3_5GatedDeltaNet`:

- `linear_num_value_heads = 32`
- `linear_key_head_dim = 128`
- The delta rule's recurrent state shape: `(batch, num_v_heads, head_k_dim, head_k_dim)` = `(1, 32, 128, 128)`
- Elements: 32 × 128 × 128 = 524,288
- Bytes (fp16): 524,288 × 2 = 1,048,576 = **1.0 MiB per layer**

### A.2 The conv1d state per layer

- `conv_dim = key_dim * 2 + value_dim = 2048*2 + 4096 = 8192`
- `conv_kernel_dim = 4`
- State shape: `(batch, conv_dim, conv_kernel_dim)` = `(1, 8192, 4)`
- Elements: 8192 × 4 = 32,768
- Bytes (fp16): 32,768 × 2 = 65,536 = **64 KiB per layer**

### A.3 Per-session totals (24 linear-attention layers)

| Component | Per layer | × 24 | Total |
|---|---|---|---|
| Recurrent state | 1.0 MiB | 24 MiB | **24 MiB** |
| Conv1d state | 64 KiB | 1.5 MiB | **1.5 MiB** |
| **Total** | | | **25.5 MiB** |

### A.4 A10G capacity (W4+r32, ~5.85 GiB weights)

- Total HBM: 24 GiB (23,028 MiB usable per `nvidia-smi -q`)
- Weights (W4+r32): ~5.85 GiB
- Framework + CUDA context: ~1.5 GiB
- Activations + intermediates: ~1.5 GiB
- Safety margin: ~2.0 GiB
- **Cache pool: ~13 GiB**
- Concurrent session states (25.5 MiB each): 13 GiB / 25.5 MiB ≈ **530 sessions**
- With full-attention KV at 8k context (~256 MiB per session): ~50 sessions at 8k context

### A.5 v1's error (corrected)

v1 claimed 576 KiB per session. v1 confused the K/V projection shape `(num_heads, head_dim)` = `(16, 128)` with the recurrent state matrix shape `(num_v_heads, head_k_dim, head_k_dim)` = `(32, 128, 128)`. The recurrent state is a matrix (the delta rule's outer-product accumulator), not a vector. The correct per-session size is **25.5 MiB**, not 576 KiB.

This changes the capacity arithmetic: 530 concurrent sessions (not 23,000), and 50 at 8k context (not 26). The PoC's P2 measures whether the global cache (cross-session deduplication at shared-prefix boundaries) recovers the effective capacity by sharing state across sessions.

---

## Appendix B — The diff from v1 (the PoC I wrote before)

v1 was wrong in three ways. This appendix records the corrections so any reviewer can verify.

### B.1 Removed

- **The dense FP16/bf16 baseline** (v1 §2.2.3 Round 1). Removed per user directive. The PoC is LUT-model-only.
- **Databricks, SharePoint, `ai_parse_document`, Unity Catalog, the corpus lane, the permission lane** (v1 §2.1, §2.2). Removed per user directive. Single-machine PoC.
- **docling, BAAI/bge-m3, the chunker, the embedder** (v1 §2.2.1). Removed. OfficeQA's corpus is loaded as-is; the PoC does not build a corpus pipeline.
- **The 3-round structure** (v1 §0: feasibility, cache delta, staleness). Replaced with the 5-phase structure (P0 env, P1 fine-tune, P2 cache measurement, P3 staleness, P4 verdict).

### B.2 Corrected

- **The cache asset size: 576 KiB → 25.5 MiB** (v1 §2.3.4). v1 confused the K/V projection shape with the recurrent state matrix shape. See Appendix A.5.
- **The capacity arithmetic: 23,000 sessions → 530 sessions** (v1 §2.3.8). Consequence of the corrected cache asset size.
- **The cache asset identity: "(K_state, V_state) at boundaries" → "the recurrent state matrix at boundaries"** (v1 §2.3.4). v1 described it abstractly; this version names the exact tensor (`cache_params.layers[L].recurrent_states[0]`) and its shape.

### B.3 Added

- **The global cache** (§1.3). v1 mentioned it in passing; this version makes it a first-class component with its own implementation (`poc/global_cache.py`), its own hit-rate targets (boundary 1: shared system prompt), and its own measurement (P2 measures the global cache's hit rate separately from the per-session cache).
- **The fine-tune** (§1.4). v1 did not include a fine-tune step. This version adds the W10 LUT fine-tune on OfficeQA as Phase P1, with the K3-style cache-aware fine-tune as an optional research bet.
- **The K3 discipline table** (§1.5). v1 referenced K3 but did not enumerate the standing rules. This version lists all 8 K3 rules and their PoC implementations.
