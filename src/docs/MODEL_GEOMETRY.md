# Model geometry — Qwen3.5-9B

The dimensions of the deployment target, pinned against the live
checkpoint config. The config is committed verbatim at
[docs/qwen3_5_9b_config.json](qwen3_5_9b_config.json):

```
d0883072e01861ed0b2d47be3c16c36a8e81c224c7ffaa310c6558fb3f932b05  docs/qwen3_5_9b_config.json
```

Re-verify any time: `curl -s https://huggingface.co/Qwen/Qwen3.5-9B/raw/main/config.json | sha256sum`.

Palettization targets the **text backbone**: 32 decoder layers + `lm_head`
+ `embed_tokens`. The vision tower and the MTP layer are out of scope
(`mtp_use_dedicated_embeddings: false` — the MTP layer shares the text
embedding; no second artifact set is written for it).

## 1. Config constants (text_config)

| key | value |
|---|---|
| `hidden_size` | 4096 |
| `intermediate_size` | 12288 |
| `num_hidden_layers` | 32 |
| `layer_types` | 24 × `linear_attention`, 8 × `full_attention` (interval 4) |
| `vocab_size` | 248320 |
| `tie_word_embeddings` | false |
| `head_dim` | 256 |
| `num_attention_heads` | 16 |
| `num_key_value_heads` | 4 |
| `attn_output_gate` | true (`q_proj` packs Q+gate → out = 16·256·2 = 8192) |
| `linear_num_key_heads` × `linear_key_head_dim` | 16 × 128 (= 2048) |
| `linear_num_value_heads` × `linear_value_head_dim` | 32 × 128 (= 4096) |
| `linear_conv_kernel_dim` | 4 |
| top-level architecture | `Qwen3_5ForConditionalGeneration` (text + vision tower + 1 MTP layer) |

## 2. Module inventory (shapes are (N, K): output rows × input columns)

**Linear-attention layer — 24 of them:**

| module | shape (N, K) | elements | palettized |
|---|---|---|---|
| `in_proj_qkv` → Q | (2048, 4096) | 8,388,608 | yes (per-component) |
| `in_proj_qkv` → K | (2048, 4096) | 8,388,608 | yes (per-component) |
| `in_proj_qkv` → V | (4096, 4096) | 16,777,216 | yes (per-component) |
| `in_proj_z` | (4096, 4096) | 16,777,216 | yes |
| `out_proj` | (4096, 4096) | 16,777,216 | yes |
| `gate_proj` | (12288, 4096) | 50,331,648 | yes |
| `up_proj` | (12288, 4096) | 50,331,648 | yes |
| `down_proj` | (4096, 12288) | 50,331,648 | yes |
| `in_proj_b` / `in_proj_a` | (32, 4096) | 131,072 each | no (dense remainder) |
| `conv1d` | conv (2048, 4096, k=4) | 33,554,432 | no (convolution, not a GEMM) |

8 palettized modules, 218,103,808 elements per layer.

**Full-attention layer — 8 of them:**

| module | shape (N, K) | elements | palettized |
|---|---|---|---|
| `q_proj` (Q+gate packed) | (8192, 4096) | 33,554,432 | yes |
| `k_proj` | (1024, 4096) | 4,194,304 | yes |
| `v_proj` | (1024, 4096) | 4,194,304 | yes |
| `o_proj` | (4096, 4096) | 16,777,216 | yes |
| `gate_proj` / `up_proj` | (12288, 4096) | 50,331,648 each | yes |
| `down_proj` | (4096, 12288) | 50,331,648 | yes |

7 palettized modules, 209,715,200 elements per layer.

**Text backbone totals:**

| quantity | value |
|---|---|
| palettized modules | 248 (= 24×8 + 8×7) |
| palettized elements | 6,912,212,992 |
| `lm_head` / `embed_tokens` | (248320, 4096) each, 1,017,118,720 elements |
| dense remainder (both heads, FP16) | 2,034,237,440 elements ≈ 4.07 GiB |

## 3. The head arithmetic

`lm_head` and `embed_tokens` are the widest tensors in the model and get
the dedicated head pass (`--palettize-heads`):

| item | value |
|---|---|
| 248320 % 128 | 0 → 1940 x-tiles (idx4-eligible) |
| 248320 % 64 | 0 (gs=64 → 3880 LUT rows; gs=32 → 7760) |
| FP16 bytes per head | 2,034,237,440 B ≈ 1.89 GiB |
| two idx4 streams + LUTs + rank-32 residual | ≈ 0.96 GiB (8.127 bits/elt) |
| full W-read at 600 GB/s | FP16 3.39 ms → hybrid 1.72 ms |

The head pass runs decoupled from the body (`--palettize-heads` /
`--only-heads` modes) because its 1940-tile geometry needs a different
memory plan than the 32-layer body sweep; a head-only run reproduces the
full run's head artifacts byte-identically, so the two passes compose
without a combined rerun.

## 4. Notes for code that consumes this table

- The probe table's per-module `MB` column (code-stream bytes) follows
  directly from these shapes: e.g. `gate_proj` at idx4 = 12288·4096·4/8 B
  ≈ 24 MiB of indices + LUT.
- Model-scale arithmetic in tests (`tests/test_lut_gradients.py` and
  friends) derives from this file, never from hardcoded copies.
- Q+gate packing means the full-attention `q_proj` output is 8192 rows:
  16 heads × 256 dim for Q, plus the same again for the gate values.
  Splitting it for kernel dispatch respects the 2048-row Q / K / V
  component boundaries of the linear-attention variant, not this 8192
  split.
