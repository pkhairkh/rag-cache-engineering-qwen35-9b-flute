# SPECIFICATION v4 — Cache-Engineered RAG (definitive, corrected)

> **The architecture:**
> - The model is Qwen3.5-9B (FLUTE idxN W4+r32). 32 layers: 24 linear-attn (`Qwen3_5GatedDeltaNet`) + 8 full-attn (`Qwen3_5Attention`), pattern `[L,L,L,F,L,L,L,F,...]`.
> - The recurrent state S is **per-layer** — each of the 24 linear-attn layers has its own S (`cache_params.layers[layer_idx].recurrent_states[0]`). 24 separate tensors.
> - The conv_state is **per-layer** — 24 separate tensors (`cache_params.layers[layer_idx].conv_states[0]`).
> - We capture S at **9 hook points** during the forward pass: after layer 0 (the first linear), and after each full-attn layer (3, 7, 11, 15, 19, 23, 27, 31). At each hook, we snapshot the S of the linear layers that ran since the last hook. **Result: all 24 S tensors are captured**, just at 9 different points.
> - We **ADD two global caches M1, M2** (Kimi-style key-memory + value-memory) to the model — new parameters, new forward logic, new training. These are TWO for the WHOLE model, shared across all 24 linear layers. They extend geometrical expressivity and capacity.
> - The full-attn layers are **NOT snapshotted**. They keep their state after the user query is prefilled, then converge by the next forward passes. No KV snapshot, no special handling.
> - We **monkey-patch** the cache API (`cache_params.update_recurrent_state`, `cache_params.layers[L].recurrent_states[0]`) for our purposes — direct assignment, not the stock API.
> - The retrieval vector = flattened **S (24 per-layer) + M1 + M2 (2 global)**.
> - Augmentation = sum the 24 S deltas + restore M1, M2. Install via monkey-patch. Answer from installed caches.

---

## 1. The model (from `scripts/modeling.py` + `docs/qwen3_5_9b_config.json`)

### 1.1 The layer structure

```
Layer  0: linear_attn  ← HOOK 0: after layer 0, capture S_0
Layer  1: linear_attn
Layer  2: linear_attn  ← HOOK 1: after layer 3 (full-attn), capture S_1, S_2
Layer  3: full_attn
Layer  4: linear_attn
Layer  5: linear_attn
Layer  6: linear_attn  ← HOOK 2: after layer 7, capture S_4, S_5, S_6
Layer  7: full_attn
Layer  8: linear_attn
Layer  9: linear_attn
Layer 10: linear_attn  ← HOOK 3: after layer 11, capture S_8, S_9, S_10
Layer 11: full_attn
Layer 12: linear_attn
Layer 13: linear_attn
Layer 14: linear_attn  ← HOOK 4: after layer 15, capture S_12, S_13, S_14
Layer 15: full_attn
Layer 16: linear_attn
Layer 17: linear_attn
Layer 18: linear_attn  ← HOOK 5: after layer 19, capture S_16, S_17, S_18
Layer 19: full_attn
Layer 20: linear_attn
Layer 21: linear_attn
Layer 22: linear_attn  ← HOOK 6: after layer 23, capture S_20, S_21, S_22
Layer 23: full_attn
Layer 24: linear_attn
Layer 25: linear_attn
Layer 26: linear_attn  ← HOOK 7: after layer 27, capture S_24, S_25, S_26
Layer 27: full_attn
Layer 28: linear_attn
Layer 29: linear_attn
Layer 30: linear_attn  ← HOOK 8: after layer 31, capture S_28, S_29, S_30
Layer 31: full_attn
```

**9 hooks capture all 24 S tensors.** Hook 0 captures S_0 (1 layer). Hooks 1-8 capture 3 layers each (the linear layers since the last hook). Total: 1 + 8×3 = 25... wait, that's 25. Let me recount.

Linear layers: 0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18, 20, 21, 22, 24, 25, 26, 28, 29, 30 = 24 layers.

- Hook 0 (after layer 0): S_0
- Hook 1 (after layer 3): S_1, S_2
- Hook 2 (after layer 7): S_4, S_5, S_6
- Hook 3 (after layer 11): S_8, S_9, S_10
- Hook 4 (after layer 15): S_12, S_13, S_14
- Hook 5 (after layer 19): S_16, S_17, S_18
- Hook 6 (after layer 23): S_20, S_21, S_22
- Hook 7 (after layer 27): S_24, S_25, S_26
- Hook 8 (after layer 31): S_28, S_29, S_30

Total: 1 + 2 + 3×7 = 24. ✓ All 24 linear layers captured.

### 1.2 The per-layer state shapes

| Object | Per linear layer | Shape | Size (fp16) |
|---|---|---|---|
| S (recurrent state) | 24 layers | `(1, 32, 128, 128)` | 1.0 MiB |
| conv_state | 24 layers | `(1, 8192, 4)` | 64 KiB |
| **Total S** | 24 | | **24 MiB** |
| **Total conv** | 24 | | **1.5 MiB** |

### 1.3 The two global caches M1, M2 (ADDED — new parameters)

The model does NOT have M1/M2. We ADD them as new parameters on the model, shared across all 24 linear layers.

**Implementation:** modify `Qwen3_5GatedDeltaNet.__init__` to accept a reference to the shared M1, M2 (or add them to the model and pass them in the forward). The forward adds a read/write mechanism:

```python
# the ADDED code (monkey-patched or a new subclass)
class Qwen3_5GatedDeltaNetWithKimiCaches(Qwen3_5GatedDeltaNet):
    def __init__(self, config, layer_idx, global_M1, global_M2):
        super().__init__(config, layer_idx)
        self.global_M1 = global_M1  # shared reference — ONE tensor for all layers
        self.global_M2 = global_M2  # shared reference — ONE tensor for all layers
        # the read/write gates (per-layer, but the memory is shared)
        self.mem_write_gate = nn.Linear(config.hidden_size, self.num_v_heads, bias=False)
        self.mem_read_gate = nn.Linear(config.hidden_size, self.num_v_heads, bias=False)

    def forward(self, hidden_states, cache_params=None, **kwargs):
        # ... the stock forward (produces q, k, v, the delta rule update of S) ...

        # ADDED: write to the global M1, M2 (additive, gated)
        mem_write = torch.sigmoid(self.mem_write_gate(hidden_states))
        self.global_M1 = self.global_M1 + mem_write * k_norm  # additive write
        self.global_M2 = self.global_M2 + mem_write * v       # additive write

        # ADDED: read from the global M1, M2 (attention)
        mem_read = torch.sigmoid(self.mem_read_gate(hidden_states))
        mem_attn = F.softmax(q @ self.global_M1.T / sqrt(head_k_dim), dim=-1)
        o_mem = mem_attn @ self.global_M2
        output = output + mem_read * o_mem  # combine delta output + memory read

        return output
```

| Object | Shape | Size (fp16, mem_size=128) |
|---|---|---|
| M1 (global key-memory) | `(1, 32, mem_size, 128)` | 1.0 MiB |
| M2 (global value-memory) | `(1, 32, mem_size, 128)` | 1.0 MiB |
| **Total M1+M2** | 2 tensors | **2.0 MiB** |

### 1.4 The full-attn layers — NOT snapshotted

The 8 full-attn layers (`Qwen3_5Attention`) are NOT modified, NOT snapshotted. They process the query tokens fresh. The user said: "It keeps the same state after user query is prefilled... and then converges by the next forward passes. No snapshots here."

The full-attn layers' contribution to the model is in the hidden states they produce — which feed into the next linear-attn layer's input. The linear-attn layer's S captures the accumulated information (including what the full-attn contributed via the hidden state). We don't need to snapshot the full-attn separately.

---

## 2. The retrieval vector

```
cache_vector = concat([
    S_0.flatten(),      # layer 0's recurrent state — (32, 128, 128) → 524,288
    S_1.flatten(),      # layer 1's
    S_2.flatten(),      # layer 2's
    S_4.flatten(),      # layer 4's (layer 3 is full-attn, no S)
    S_5.flatten(),
    S_6.flatten(),
    S_8.flatten(),
    ...all 24 linear layers...
    S_30.flatten(),
    M1.flatten(),       # the global key-memory — (32, mem_size, 128) → 524,288 (at mem_size=128)
    M2.flatten(),       # the global value-memory
])
# at mem_size=128:
# 24 × 524,288 + 2 × 524,288 = 26 × 524,288 = 13,631,488 dims
# size: 26.0 MiB (fp16)
```

**24 per-layer S + 2 global M1/M2 = 26 tensors, flattened into one ~26 MiB vector.**

---

## 3. Ingestion (one-time, per chunk)

```python
# poc/ingest.py
from transformers import DynamicCache
import torch, numpy as np

# the 9 hook points (after these layers, capture the linear layers since the last hook)
HOOKS = [
    (0, [0]),           # after layer 0: capture S_0
    (3, [1, 2]),        # after layer 3 (full-attn): capture S_1, S_2
    (7, [4, 5, 6]),     # after layer 7: capture S_4, S_5, S_6
    (11, [8, 9, 10]),
    (15, [12, 13, 14]),
    (19, [16, 17, 18]),
    (23, [20, 21, 22]),
    (27, [24, 25, 26]),
    (31, [28, 29, 30]),
]

def ingest_chunk(model, chunk_token_ids, device):
    """Prefill a chunk. Capture S at 9 hooks + M1/M2 (global).
    NO full-attn snapshot."""
    # create a fresh cache (conv-reset = all 24 conv_states to zero)
    cache = DynamicCache(config=model.config)

    # register forward hooks on the 9 boundary layers
    captured_S = {}  # layer_idx → S tensor
    captured_conv = {}

    def make_hook(layers_to_capture):
        def hook(module, input, output):
            for layer_idx in layers_to_capture:
                if cache.has_previous_state(layer_idx, state_idx=0):
                    S = cache.layers[layer_idx].recurrent_states[0]
                    conv = cache.layers[layer_idx].conv_states[0]
                    captured_S[layer_idx] = S.detach().clone()
                    captured_conv[layer_idx] = conv.detach().clone()
        return hook

    handles = []
    for hook_after_layer, layers_to_capture in HOOKS:
        h = model.model.layers[hook_after_layer].register_forward_hook(
            make_hook(layers_to_capture))
        handles.append(h)

    # prefill
    with torch.no_grad():
        model(input_ids=chunk_token_ids, past_key_values=cache, use_cache=True)

    # remove hooks
    for h in handles:
        h.remove()

    # snapshot M1, M2 (the two global caches)
    M1 = model.global_M1.detach().clone()
    M2 = model.global_M2.detach().clone()

    # the cache vector = flattened S (24) + M1 + M2
    s_list = [captured_S[i] for i in sorted(captured_S.keys())]  # 24 tensors
    cache_vector = torch.cat(
        [s.flatten() for s in s_list] + [M1.flatten(), M2.flatten()]
    ).numpy().astype(np.float16)

    return {
        's_per_layer': {i: captured_S[i] for i in sorted(captured_S.keys())},  # 24 S tensors
        'conv_per_layer': {i: captured_conv[i] for i in sorted(captured_conv.keys())},  # 24 conv
        'M1': M1,           # the global key-memory delta
        'M2': M2,           # the global value-memory delta
        'cache_vector': cache_vector,  # (13631488,) — the retrieval vector
    }
```

### What's saved per chunk

| Component | Count | Size each | Total (fp16) |
|---|---|---|---|
| S (per-layer recurrent state) | 24 | 1.0 MiB | 24 MiB |
| conv_state (per-layer) | 24 | 64 KiB | 1.5 MiB |
| M1 (global) | 1 | 1.0 MiB | 1.0 MiB |
| M2 (global) | 1 | 1.0 MiB | 1.0 MiB |
| cache_vector (redundant) | 1 | 26 MiB | 26 MiB |
| **Total per chunk** | | | **~27.5 MiB** (without redundant: ~27.5 MiB) |

**For 50,000 chunks:** ~1.375 TiB on disk.

---

## 4. The query flow

```
[1. Tokenize the query]
[2. Prefill the query → capture S (24 per-layer, at 9 hooks) + M1 + M2]
[3. IVFADC preselect on the cache vector → top-100]
[4. Cos sim rerank → top-3 chunk indices]
[5. Load the top-3 chunks' snapshots from disk]
[6. Install: sum the 24 S deltas + sum the M1/M2 deltas (2 global)]
   — monkey-patch: directly assign to cache.layers[L].recurrent_states[0]
[7. Answer from installed caches — full-attn runs fresh, converges]
[8. Decode the answer]
```

### The installation (monkey-patched)

```python
# poc/augment.py
def install_and_answer(model, query_token_ids, retrieved_snapshots, system_snapshot, device):
    """Install S (24 per-layer) + M1/M2 (2 global) via monkey-patch.
    Full-attn runs fresh."""

    cache = DynamicCache(config=model.config)

    # 1. install the 24 per-layer S states (sum the deltas)
    for layer_idx in range(32):
        if model.config.layer_types[layer_idx] == "linear_attention":
            # start from the system prompt's S
            restored_S = system_snapshot['s_per_layer'][layer_idx].clone()
            restored_conv = system_snapshot['conv_per_layer'][layer_idx].clone()
            # sum the retrieved chunks' deltas
            for snap in retrieved_snapshots:
                restored_S = restored_S + snap['s_per_layer'][layer_idx]
                restored_conv = snap['conv_per_layer'][layer_idx]  # last chunk's conv
            # monkey-patch: direct assignment to the cache
            cache.layers[layer_idx].recurrent_states[0] = restored_S.to(device)
            cache.layers[layer_idx].conv_states[0] = restored_conv.to(device)

    # 2. install the 2 global M1/M2 (sum the deltas)
    restored_M1 = system_snapshot['M1'].clone()
    restored_M2 = system_snapshot['M2'].clone()
    for snap in retrieved_snapshots:
        restored_M1 = restored_M1 + snap['M1']
        restored_M2 = restored_M2 + snap['M2']
    model.global_M1 = restored_M1.to(device)
    model.global_M2 = restored_M2.to(device)

    # 3. the full-attn layers run fresh — NO installation
    # 4. answer from the installed caches
    with torch.no_grad():
        logits = model(input_ids=query_token_ids, past_key_values=cache, use_cache=True)
    return logits
```

---

## 5. Pretraining

Same as before: next-token prediction on the OfficeQA corpus via the W10 LUT path (`scripts/qlora.py::attach_qlora` + `scripts/trainer.py`). The pretrain trains:
- The 24 per-layer linear-attn parameters (in_proj_qkv, A_log, dt_bias, etc.) → so S carries discriminative info
- The M1/M2 write/read gates (the NEW parameters) → so the global caches carry discriminative info
- The LUTs (via W10) → the quantized weights

---

## 6. Summary

| Question | Answer |
|---|---|
| How many global caches? | **TWO.** M1 and M2. For the whole model. Shared across all 24 linear layers. ADDED as new parameters. |
| How many S states? | **24.** One per linear-attn layer. Each is `(1, 32, 128, 128)` = 1.0 MiB. |
| Where do we capture S? | At **9 hooks**: after layer 0 (first), and after each full-attn layer (3, 7, 11, 15, 19, 23, 27, 31). All 24 S tensors are captured. |
| Full-attn KV? | **NOT snapshotted.** Runs fresh. Keeps state after query prefill, converges by next forward passes. |
| Retrieval vector? | Flattened S (24) + M1 + M2 = ~26 MiB (13.6M dims at mem_size=128). |
| Augmentation? | Sum the 24 S deltas + sum the 2 M1/M2 deltas. Monkey-patch into the cache. Answer from installed caches. |
| Per-chunk disk? | ~27.5 MiB. For 50k: ~1.375 TiB. |
| Cache API? | Monkey-patched — direct assignment to `cache.layers[L].recurrent_states[0]`, not the stock `update_recurrent_state`. |
| M1/M2 exist in the model? | **NO.** They must be ADDED (new parameters + forward logic + training). |
