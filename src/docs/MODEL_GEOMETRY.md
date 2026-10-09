# MODEL_GEOMETRY.md — Qwen3.5-9B web-verified geometry of record

**Status:** geometry of record (v1.0). **Source:** the live checkpoint
config fetched 2026-10-02 from
`https://huggingface.co/Qwen/Qwen3.5-9B/raw/main/config.json`
(byte-pinned at `docs/qwen3_5_9b_config.json`, sha256 below). This file
supersedes every other geometry statement in the repository: where any
doc, test, or script disagrees with this table, the doc is wrong and a
correction commit fixes it (the corrections applied 2026-10-02 are in
§5).

## 1. Config constants (text_config, verbatim)

| key | value |
|---|---|
| `hidden_size` | 4096 |
| `intermediate_size` | 12288 |
| `num_hidden_layers` | 32 |
| `layer_types` | 24 × `linear_attention`, 8 × `full_attention` (`full_attention_interval: 4`) |
| `vocab_size` | **248320** |
| `tie_word_embeddings` | **false** (top level) |
| `head_dim` | 256 |
| `num_attention_heads` | 16 |
| `num_key_value_heads` | 4 |
| `attn_output_gate` | true (q_proj packs Q+gate: out = heads·head_dim·2) |
| `linear_num_key_heads` / `linear_key_head_dim` | 16 / 128 |
| `linear_num_value_heads` / `linear_value_head_dim` | 32 / 128 |
| `linear_conv_kernel_dim` | 4 |
| `mtp_num_hidden_layers` | 1 |
| `mtp_use_dedicated_embeddings` | false |
| top-level architecture | `Qwen3_5ForConditionalGeneration` (text + 27-depth ViT vision tower + 1 MTP layer) |

Scope note: palettization targets the **text backbone only** (the 32
decoder layers + `lm_head` + `embed_tokens`). The vision tower and the
MTP layer are out of scope; `mtp_use_dedicated_embeddings: false` means
the MTP shares the text embedding — the head pass records this fact and
never writes a second artifact set for it.

## 2. Derived module inventory (per layer, from scripts/modeling.py shapes)

**Linear-attention layer (24 of them):**

| module | shape (N, K) | elements | idx4-eligible (N%128, K%64) |
|---|---|---|---|
| `in_proj_qkv` → Q | (2048, 4096) | 8,388,608 | yes (per-component, D6 split) |
| `in_proj_qkv` → K | (2048, 4096) | 8,388,608 | yes (per-component) |
| `in_proj_qkv` → V | (4096, 4096) | 16,777,216 | yes (per-component) |
| `in_proj_z` | (4096, 4096) | 16,777,216 | yes |
| `out_proj` | (4096, 4096) | 16,777,216 | yes |
| `gate_proj` | (12288, 4096) | 50,331,648 | yes |
| `up_proj` | (12288, 4096) | 50,331,648 | yes |
| `down_proj` | (4096, 12288) | 50,331,648 | yes |
| `in_proj_b` / `in_proj_a` | (32, 4096) | 131,072 each | **no** (N=32; dense remainder) |
| `conv1d` | conv (2048, 4096, k=4) | 33,554,432 | **no** (conv, not a GEMM) |

Layer total (eligible): 218,103,808 elements, 8 palettized modules.

**Full-attention layer (8 of them):**

| module | shape (N, K) | elements | idx4-eligible |
|---|---|---|---|
| `q_proj` (Q+gate packed) | (8192, 4096) | 33,554,432 | yes |
| `k_proj` | (1024, 4096) | 4,194,304 | yes |
| `v_proj` | (1024, 4096) | 4,194,304 | yes |
| `o_proj` | (4096, 4096) | 16,777,216 | yes |
| `gate_proj` / `up_proj` | (12288, 4096) | 50,331,648 each | yes |
| `down_proj` | (4096, 12288) | 50,331,648 | yes |

Layer total: 209,715,200 elements, 7 palettized modules.

**Whole text backbone:**

| quantity | value |
|---|---|
| palettized modules | 248 (= 24×8 + 8×7) |
| palettized elements | 6,912,212,992 (= 24×218,103,808 + 8×209,715,200) |
| `lm_head` / `embed_tokens` | **(248320, 4096)** each, 1,017,118,720 elements |
| dense remainder (embed + lm_head, fp16 today) | 2,034,237,440 elements ≈ 4.07 GB fp16 |

## 3. The head arithmetic (corrected, at the §2/§3 use sites)

| item | value |
|---|---|
| 248320 % 128 | 0 → **1940 x-tiles** (idx4-eligible) |
| 248320 % 64 | 0 (gs=64 → 3880 groups; gs=32 → 7760) |
| fp16 bytes (per head) | 2,034,237,440 B = **2.034 GB** |
| R2 hybrid bytes (per head) | 2 idx4 blobs 1,017,118,720 B + LUTs 2×3880×16×2 B (gs=64) + r32 residual 16,154,624 B = **1.034 GB** |
| bits/elt (R2) | 8 + 512·(1/248320 + 1/4096) = **8.127** |
| full W-read at 600 GB/s | fp16 3.39 ms → hybrid **1.72 ms** |

## 4. sha256 pin

```
d0883072e01861ed0b2d47be3c16c36a8e81c224c7ffaa310c6558fb3f932b05  docs/qwen3_5_9b_config.json
```

The JSON is committed verbatim as fetched (no reformatting), so the pin
is reproducible. Re-verification (any session):
`curl -s <url> | sha256sum` must match.

## 5. Corrections applied by this document (2026-10-02)

The pre-correction docs carried the **Qwen3/Qwen2.5 vocab 151936** — a
stale constant that never matched this checkpoint. Corrected sites:

| site | old | new |
|---|---|---|
| PROPOSAL.md §0-evidence, §2, §3, §9, App. A | vocab 151936; lm_head [151936, 4096]; 1187 tiles; 1.246/0.633 GB; 2.08/1.06 ms | 248320; [248320, 4096]; 1940 tiles; 2.034/1.034 GB; 3.39/1.72 ms |
| README.md "Model facts" | "text vocab_size 151936" | "text vocab_size 248320" |
| tests/test_lut_gradients.py `test_box_geometry_arithmetic` | `kl_graph` from vocab 151936 → 2.32 GiB | vocab 248320 → 3.79 GiB (assertion updated; the historical R14 note keeps its own number in its own comment) |
| the deleted "toy world proxy" claim | w6_toy teacher config as vocab source | this file + the pinned JSON |

Unchanged (verified correct against the real config): hidden 4096,
intermediate 12288, 32 layers (24 linear + 8 full, interval 4), full-attn
Q/K/V = 4096/1024/1024 (16×256, 4×256 GQA), linear-attn Q/K/V =
2048/2048/4096 (16×128, 32×128), q_proj Q+gate packing (8192),
`tie_word_embeddings: false`, 248 modules, 6,912,212,992 palettized
elements.

**Binding rule going forward:** no doc, test, or script may state a
model dimension that is not either (a) a row of §1/§2/§3 of this file,
or (b) read dynamically from `config.vocab_size`-style attributes at
runtime. The geometry-audit gate (TASKS.md) enforces this mechanically.
