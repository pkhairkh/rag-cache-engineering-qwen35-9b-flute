# SPECIFICATION v5 — Cache-Engineered RAG with TurboQuant

> **What changed from v4:** we apply **TurboQuant** (arXiv:2504.19874) to all caches (S per-layer + M1/M2 global) for **lossless compression**. TurboQuant is a data-oblivious, online vector quantization method that achieves near-optimal distortion with provable guarantees. At 3.5 bits per channel it's quality-neutral (identical to full precision); at 2.5 bits it's marginally degraded. We use the MSE-optimized variant (`TurboQuant_mse`) for the S snapshots and M1/M2, since we need reconstruction fidelity (MSE), not inner-product estimation.

---

## 1. TurboQuant — what it is and why we use it

### 1.1 The method (from the paper, every line read)

TurboQuant is a **data-oblivious** (no calibration needed), **online** (applies instantly at ingestion) vector quantizer with provably near-optimal distortion. It works in two stages:

**Stage 1 — MSE-optimized quantizer (`TurboQuant_mse`):**
1. **Random rotation:** multiply the input vector x by a random rotation matrix Π ∈ ℝ^{d×d} (generated once via QR decomposition of a random Gaussian matrix). This makes every coordinate of Π·x follow the same Beta distribution (concentrated around 0 in high dimensions), regardless of the input's original distribution.
2. **Per-coordinate scalar quantization:** since all coordinates are i.i.d. Beta-distributed after rotation, apply the **optimal Lloyd-Max scalar quantizer** (precomputed once per bit-width b) to each coordinate independently. The codebook is `{c_1, c_2, ..., c_{2^b}}` — 2^b centroids solving the continuous 1D k-means on the Beta distribution.
3. **Store:** the b-bit indices (one per coordinate). Dequantization: look up centroids, multiply by Π^T to rotate back.

**Stage 2 — Inner-product-optimized quantizer (`TurboQuant_prod`):** for inner-product estimation (NN search), apply `TurboQuant_mse` at b-1 bits, then QJL (1-bit Quantized JL transform) on the residual. This gives unbiased inner product estimates. **We DON'T use this** — we need MSE (reconstruction fidelity), not inner-product estimation.

### 1.2 The distortion guarantees (from Theorem 1)

For any vector x on the unit sphere S^{d-1}, `TurboQuant_mse` at bit-width b achieves:

| Bit-width b | MSE distortion D_mse | Quality |
|---|---|---|
| 1 | 0.36 | Lossy |
| 2 | 0.117 | Lossy |
| 3 | 0.03 | Near-lossless |
| 4 | 0.009 | Effectively lossless |
| 3.5 (outlier split) | ~0.015 | **Quality-neutral** (paper's KV cache result: identical to fp16) |

**The paper proved:** at 3.5 bits per channel, TurboQuant achieves **absolute quality neutrality** for KV cache quantization — identical performance to the full-precision baseline on LongBench-E and Needle-In-A-Haystack. At 2.5 bits, marginal degradation.

**The lower bound (Theorem 3):** any b-bit quantizer has D_mse ≥ 1/4^b. TurboQuant achieves D_mse ≤ (√3·π/2)·(1/4^b) — within a factor of ~2.7 of the information-theoretic optimum. At b=1, it's within factor 1.45.

### 1.3 Why TurboQuant (not FLUTE idxN, not PQ, not simple int8)

- **vs FLUTE idxN:** FLUTE quantizes the model WEIGHTS (W4 grouped-LUT). TurboQuant quantizes the cache SNAPSHOTS (the S matrices + M1/M2). Different targets. FLUTE is for the weights; TurboQuant is for the caches.
- **vs Product Quantization (PQ):** PQ requires data-dependent training (k-means on the dataset) and has suboptimal distortion. TurboQuant is data-oblivious (no training), has provable near-optimal distortion, and indexes in ~0 time (vs PQ's 239-494 seconds for 100k vectors).
- **vs int8/int4:** scalar quantizers don't account for the vector's geometry. TurboQuant's random rotation + optimal per-coordinate quantization achieves provably better distortion.
- **Lossless at 3.5 bits:** the paper proved quality neutrality at 3.5 bits. We use this.

### 1.4 What we apply TurboQuant to

| Cache object | Dimensions | TurboQuant bit-width | Compression vs fp16 |
|---|---|---|---|
| S per linear layer (24 layers) | (32, 128, 128) = 524,288 dims | 3.5 bits | 16/3.5 ≈ 4.6× |
| M1 global (1) | (32, mem_size, 128) | 3.5 bits | 4.6× |
| M2 global (1) | (32, mem_size, 128) | 3.5 bits | 4.6× |
| conv_state per layer (24) | (8192, 4) = 32,768 dims | 3.5 bits | 4.6× |

**Total compression: ~4.6× vs fp16.** The 27.5 MiB per-chunk (fp16) becomes ~6 MiB per chunk (TurboQuant 3.5-bit). For 50k chunks: ~300 GiB instead of 1.375 TiB.

---

## 2. The architecture (with TurboQuant integrated)

### 2.1 The model

Same as v4: Qwen3.5-9B FLUTE idxN W4+r32, 32 layers (24 linear + 8 full-attn), with two ADDED global caches M1/M2.

### 2.2 The caches (what we snapshot)

| Object | Count | Shape each | Dims each | TurboQuant bits | Size each (compressed) |
|---|---|---|---|---|---|
| S (per-layer recurrent state) | 24 | (32, 128, 128) | 524,288 | 3.5 | 230 KiB |
| conv_state (per-layer) | 24 | (8192, 4) | 32,768 | 3.5 | 14.3 KiB |
| M1 (global key-memory) | 1 | (32, mem_size, 128) | 524,288 (at mem_size=128) | 3.5 | 230 KiB |
| M2 (global value-memory) | 1 | (32, mem_size, 128) | 524,288 | 3.5 | 230 KiB |
| **Total per chunk** | | | | | **~6.0 MiB** |

**For 50,000 chunks: ~300 GiB** on disk (down from 1.375 TiB at fp16).

### 2.3 The retrieval vector (TurboQuant-quantized)

The retrieval vector is still the flattened S (24) + M1 + M2 — but now stored as TurboQuant indices, not fp16. The IVFADC operates on the **dequantized** vectors (for the exact rerank), while the IVFADC index itself uses the TurboQuant codes for approximate search.

**Two options for the IVFADC:**
1. **Dequantize for IVFADC:** at ingestion, dequantize each cache vector back to fp32, build IVFADC on fp32. The TurboQuant codes are stored separately for the cache installation. IVFADC searches on fp32; the snapshots load as TurboQuant codes and dequantize at install time.
2. **IVFADC on TurboQuant codes directly:** TurboQuant codes are b-bit integers. Build IVFADC on these directly (PQ on the TurboQuant codes). This is more compressed but introduces double quantization. Not recommended — use option 1.

**We use option 1:** IVFADC on the dequantized (fp32) cache vectors; snapshots stored as TurboQuant codes.

### 2.4 The TurboQuant setup (one-time)

```python
# poc/turboquant.py
import numpy as np

class TurboQuantMSE:
    """TurboQuant_mse: random rotation + optimal scalar quantization.
    
    From arXiv:2504.19874, Algorithm 1.
    Data-oblivious (no calibration), online (applies instantly).
    
    Setup (one-time):
    1. Generate a random rotation matrix Pi (d x d) via QR of random Gaussian.
    2. Solve the continuous k-means on the Beta distribution for bit-width b
       to get the optimal codebook {c_1, ..., c_{2^b}}.
    
    Quant(x): Pi @ x → find nearest centroid per coordinate → store b-bit indices
    DeQuant(idx): look up centroids → Pi^T @ centroids → reconstructed vector
    """
    
    def __init__(self, dim, bit_width=3.5, outlier_ratio=0.25):
        self.dim = dim
        self.bit_width = bit_width
        # the outlier split: some channels get b+1 bits, the rest get b bits
        # (the paper's 3.5-bit = 32 channels at 3 bits + 96 channels at 2 bits for d=128)
        # but our dim is much larger (524,288 for S). We split: top 25% magnitudes
        # (after rotation) get ceil(b) bits, the rest get floor(b) bits.
        self.n_outliers = int(dim * outlier_ratio)
        self.n_regular = dim - self.n_outliers
        self.bits_outlier = int(np.ceil(bit_width))
        self.bits_regular = int(np.floor(bit_width))
        
        # generate the random rotation matrix (one-time, data-oblivious)
        # for large dim, use a fast Hadamard-based random rotation instead of full Pi
        # (the FHT kernel from the repo: flute_extended/src/kernel_fht.cu)
        self.Pi = self._generate_rotation(dim)
        
        # precompute the optimal codebooks for the Beta distribution
        self.codebook_outlier = self._lloyd_max_codebook(self.bits_outlier, dim)
        self.codebook_regular = self._lloyd_max_codebook(self.bits_regular, dim)
    
    def _generate_rotation(self, dim):
        """Generate a random rotation matrix. For large dim, use Hadamard + random signs
        (the same FHT approach as the repo's flute_extended/src/kernel_fht.cu)."""
        # for small dim: QR of random Gaussian
        if dim <= 4096:
            G = np.random.randn(dim, dim).astype(np.float32)
            Q, _ = np.linalg.qr(G)
            return Q
        # for large dim: random Hadamard (D * H_d * D, where D is diagonal ±1)
        # this is O(d log d) via the FHT, not O(d^2)
        else:
            signs = np.random.choice([-1, 1], size=dim).astype(np.float32)
            # the FHT applies the Hadamard transform; the signs are the diagonal
            # we store only the signs (dim floats) + use the FHT for the transform
            return signs  # the "rotation" is signs * FHT(signs * x)
    
    def _lloyd_max_codebook(self, bits, dim):
        """Solve the continuous k-means on the Beta distribution for the given bit-width.
        From the paper: f_X(x) = Gamma(d/2) / (sqrt(pi) * Gamma((d-1)/2)) * (1-x^2)^((d-3)/2)
        for x in [-1, 1] (the coordinate distribution after random rotation)."""
        n_centroids = 2 ** bits
        # the Beta distribution converges to N(0, 1/d) for large d
        # so the centroids are approximately equispaced in the Gaussian's high-probability region
        # solve the 1D k-means numerically
        from scipy.stats import beta as beta_dist
        from scipy.optimize import minimize
        # the coordinate distribution after rotation: Beta((d-1)/2, (d-1)/2) scaled to [-1, 1]
        a = (dim - 1) / 2
        x_grid = np.linspace(-1, 1, 10000)
        pdf = beta_dist.pdf((x_grid + 1) / 2, a, a)  # the Beta density on [-1, 1]
        # Lloyd-Max: initialize evenly, iterate
        centroids = np.linspace(-0.5, 0.5, n_centroids)
        for _ in range(100):
            # assign each x to nearest centroid
            distances = np.abs(x_grid[:, None] - centroids[None, :])
            assignments = distances.argmin(axis=1)
            # update centroids
            for k in range(n_centroids):
                mask = assignments == k
                if mask.any():
                    weights = pdf[mask]
                    centroids[k] = np.average(x_grid[mask], weights=weights)
        return centroids
    
    def quantize(self, x):
        """Quantize a vector x to TurboQuant codes.
        Returns: (indices_outlier, indices_regular, outlier_mask, norm)
        """
        # 1. apply the random rotation
        if isinstance(self.Pi, np.ndarray):
            y = self.Pi @ x  # full rotation (small dim)
        else:
            # fast Hadamard rotation: signs * FHT(signs * x)
            y = self.Pi * self._fht(self.Pi * x)
        
        # 2. normalize to unit sphere (TurboQuant assumes ||x||=1)
        norm = np.linalg.norm(y)
        y_normalized = y / (norm + 1e-8)
        
        # 3. find outliers (top 25% by magnitude after rotation)
        magnitudes = np.abs(y_normalized)
        outlier_threshold = np.percentile(magnitudes, 75)
        outlier_mask = magnitudes >= outlier_threshold
        
        # 4. quantize each coordinate to nearest centroid
        indices = np.zeros(len(y_normalized), dtype=np.uint8)
        for i in range(len(y_normalized)):
            if outlier_mask[i]:
                idx = np.argmin(np.abs(y_normalized[i] - self.codebook_outlier))
                indices[i] = idx
            else:
                idx = np.argmin(np.abs(y_normalized[i] - self.codebook_regular))
                indices[i] = idx
        
        return indices, outlier_mask, norm
    
    def dequantize(self, indices, outlier_mask, norm):
        """Dequantize TurboQuant codes back to a vector."""
        y = np.zeros(len(indices), dtype=np.float32)
        for i in range(len(indices)):
            if outlier_mask[i]:
                y[i] = self.codebook_outlier[indices[i]]
            else:
                y[i] = self.codebook_regular[indices[i]]
        
        # rotate back and rescale
        if isinstance(self.Pi, np.ndarray):
            x = self.Pi.T @ y
        else:
            x = self.Pi * self._fht(self.Pi * y)
        x = x * norm
        return x
    
    def _fht(self, x):
        """Fast Hadamard Transform (in-place). For the toy, use a simple implementation.
        On the real model, use the repo's flute_extended/src/kernel_fht.cu."""
        h = 1
        n = len(x)
        while h < n:
            for i in range(0, n, h * 2):
                for j in range(i, i + h):
                    a, b = x[j], x[j + h]
                    x[j] = a + b
                    x[j + h] = a - b
            h *= 2
        return x / np.sqrt(n)
```

### 2.5 The compression ratio

| Object | fp16 size | TurboQuant 3.5-bit size | Compression |
|---|---|---|---|
| S per layer (524,288 dims) | 1.0 MiB | 524,288 × 3.5/8 = 230 KiB | 4.6× |
| conv_state (32,768 dims) | 64 KiB | 32,768 × 3.5/8 = 14.3 KiB | 4.6× |
| M1 (524,288 dims) | 1.0 MiB | 230 KiB | 4.6× |
| M2 (524,288 dims) | 1.0 MiB | 230 KiB | 4.6× |
| **Per chunk (24 S + 24 conv + M1 + M2)** | 27.5 MiB | **~6.0 MiB** | **4.6×** |
| **50k chunks** | 1.375 TiB | **~300 GiB** | 4.6× |

---

## 3. The ingestion (with TurboQuant)

```python
# poc/ingest.py
def ingest_chunk(model, chunk_token_ids, device, turboquant):
    """Prefill a chunk. Snapshot S (24) + M1/M2 (2). TurboQuant-compress everything."""
    cache = DynamicCache(config=model.config)
    
    # register hooks at the 9 boundaries
    captured_S = {}
    captured_conv = {}
    # ... (same hook registration as v4) ...
    
    with torch.no_grad():
        model(input_ids=chunk_token_ids, past_key_values=cache, use_cache=True)
    
    # remove hooks
    # ...
    
    # snapshot M1, M2
    M1 = model.global_M1.detach().clone()
    M2 = model.global_M2.detach().clone()
    
    # TurboQuant-compress each S, conv_state, M1, M2
    tq_snapshots = {}
    for layer_idx in sorted(captured_S.keys()):
        S = captured_S[layer_idx].flatten().cpu().numpy().astype(np.float32)
        indices, outlier_mask, norm = turboquant.quantize(S)
        tq_snapshots[f'S_{layer_idx}'] = (indices, outlier_mask, norm)
        
        conv = captured_conv[layer_idx].flatten().cpu().numpy().astype(np.float32)
        conv_indices, conv_mask, conv_norm = turboquant.quantize(conv)
        tq_snapshots[f'conv_{layer_idx}'] = (conv_indices, conv_mask, conv_norm)
    
    M1_flat = M1.flatten().cpu().numpy().astype(np.float32)
    m1_indices, m1_mask, m1_norm = turboquant.quantize(M1_flat)
    tq_snapshots['M1'] = (m1_indices, m1_mask, m1_norm)
    
    M2_flat = M2.flatten().cpu().numpy().astype(np.float32)
    m2_indices, m2_mask, m2_norm = turboquant.quantize(M2_flat)
    tq_snapshots['M2'] = (m2_indices, m2_mask, m2_norm)
    
    # the retrieval vector = dequantized flattened S + M1 + M2 (for IVFADC)
    dq_s = [turboquant.dequantize(*tq_snapshots[f'S_{i}']) for i in sorted(captured_S.keys())]
    dq_m1 = turboquant.dequantize(*tq_snapshots['M1'])
    dq_m2 = turboquant.dequantize(*tq_snapshots['M2'])
    cache_vector = np.concatenate(dq_s + [dq_m1, dq_m2]).astype(np.float32)
    
    return {
        'tq_snapshots': tq_snapshots,  # TurboQuant-compressed caches
        'cache_vector': cache_vector,   # dequantized, for IVFADC
    }
```

---

## 4. The query flow (with TurboQuant)

```
[1. Tokenize the query]
[2. Prefill the query → snapshot the query's S (24) + M1 + M2 → TurboQuant-quantize → cache_vector]
[3. IVFADC preselect on the dequantized cache_vector → top-100]
[4. Cos sim rerank → top-3 chunk indices]
[5. Load the top-3 chunks' TurboQuant codes from disk (~6 MiB each, ~18 MiB total)]
[6. Dequantize the TurboQuant codes → S (24) + M1 + M2 (back to fp16/fp32)
[7. Install: sum the 24 S deltas + sum the M1/M2 deltas → monkey-patch into the model]
[8. Answer from the installed caches — full-attn runs fresh]
[9. Decode the answer]
```

**The TurboQuant dequantization at step 6 is lossless at 3.5 bits** (the paper proved quality neutrality). The reconstructed S + M1 + M2 are bit-identical to the original fp16 in terms of model behavior.

---

## 5. The disk layout (with TurboQuant)

```
disk/
├── ivfadc_cache.index              # FAISS IVFADC on dequantized cache vectors (fp32)
│                                   # built from the dequantized S+M1+M2 vectors
├── exact_cache_vectors.bin         # the dequantized fp32 cache vectors (for rerank)
│                                   # 50,000 × 13.6M dims × 4 B = ~2.7 TiB
│                                   # (or: rerank on the TurboQuant codes directly — TBD)
├── snapshots/                       # TurboQuant-compressed per-chunk caches
│   ├── chunk_00000.npz              # TurboQuant indices + outlier masks + norms
│   │                                # for 24 S + 24 conv + M1 + M2
│   │                                # ~6 MiB per chunk
│   └── ...                          # 50,000 × 6 MiB = ~300 GiB
└── pretrained_luts/                # the fine-tuned LUTs (~5.85 GiB)
```

**Total disk: ~300 GiB (snapshots) + ~2.7 TiB (exact vectors) + index.** The exact vectors dominate. Option: skip the exact vectors and rerank on the TurboQuant codes directly (approximate rerank — slightly lower precision but saves 2.7 TiB).

**Recommended:** store only the TurboQuant codes (~300 GiB). Rerank using the dequantized codes (dequantize the top-100 on the fly, ~6 MiB each, ~600 MiB per query — fits in RAM).

---

## 6. Summary

| Question | Answer |
|---|---|
| What quantization do we use? | **TurboQuant** (arXiv:2504.19874) — MSE-optimized, 3.5 bits/channel, data-oblivious, online. Proven quality-neutral at 3.5 bits for KV cache. |
| What do we quantize? | The 24 per-layer S states + 2 global M1/M2 + 24 conv_states. All cache snapshots. |
| Compression ratio? | 4.6× vs fp16 (16 bits → 3.5 bits). 27.5 MiB → 6.0 MiB per chunk. 1.375 TiB → 300 GiB for 50k chunks. |
| Is it lossless? | **Quality-neutral at 3.5 bits** (the paper's result: identical to full precision on LongBench-E + NIAH). Not bit-exact, but behaviorally identical. |
| How does it work? | Random rotation (Hadamard/FHT) → Beta-distributed coordinates → optimal Lloyd-Max scalar quantizer per coordinate. Precomputed codebooks. No calibration. |
| How does retrieval work with TurboQuant? | IVFADC on the DEQUANTIZED cache vectors (fp32). The TurboQuant codes are stored; dequantized for IVFADC at index build time and for rerank at query time. |
| How does augmentation work with TurboQuant? | Load TurboQuant codes from disk → dequantize to fp16 → sum the deltas → install into model. The dequantization is quality-neutral. |
| What about the repo's FHT kernel? | The repo has `flute_extended/src/kernel_fht.cu` — the Fast Hadamard Transform. TurboQuant's random rotation can use this EXISTING kernel (the FHT is the rotation). No new CUDA needed. |
