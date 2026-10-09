#!/usr/bin/env python3
"""
Example: Using FLUTE-Extended for quantized GEMM with the production
         Tensor Core streaming-dequant backend (idx4 layout).

Run:
    python example.py
"""

import sys
from pathlib import Path

import numpy as np
import torch

# Make the local package importable regardless of checkout location.
sys.path.insert(0, str(Path(__file__).resolve().parent))

import flute_extended  # noqa: F401
from flute_extended import qgemm_per_group_lut
from flute_extended.idxN import pack_idxn as pack_idx4

# Example dimensions: small batch, Qwen3.5-9B gate_proj shape
M, N, K = 1024, 4096, 4096
num_bits = 4
group_size = 32  # MLP layers use 32; attention layers use 64

# Activations: [M, K] FP16
input_tensor = torch.randn(M, K, dtype=torch.float16, device="cuda")

# Palettized weights (idx4 layout):
# - logical: [N, K] uint8, one 4-bit LUT index (0-15) per weight
# - indices: flat uint8 blob of N*K/2 bytes produced by pack_idx4 — the
#  canonical on-disk artifact format consumed by the kernel
# - lut:     [N // group_size, 16] FP16
logical = torch.randint(0, 16, (N, K), dtype=torch.uint8, device="cuda")
indices = torch.from_numpy(pack_idx4(logical.cpu().numpy())).to("cuda")
lut = torch.randn(N // group_size, 16, dtype=torch.float16, device="cuda")

# Run quantized GEMM with the production Tensor Core kernel.
output = qgemm_per_group_lut(input_tensor,
    indices,
    lut,
    bitwidth=num_bits,
    group_size=group_size,
    backend="cutlass_streaming",   # production; "auto" picks this too
    indices_layout="idx4",
)

print(f"Input shape:  {input_tensor.shape}")
print(f"Indices blob: {indices.numel()} bytes (idx4 layout, N*K/2)")
print(f"LUT shape:    {lut.shape}")
print(f"Output shape: {output.shape}")

# Correctness sanity vs a vectorized reference:
#  W[n, k] = LUT[n // group_size, logical[n, k]]
with torch.no_grad():
    n_groups = lut.shape[0]
    lut_rows = lut.view(n_groups, 1, 16).expand(n_groups, group_size, 16)
    lut_rows = lut_rows.reshape(-1, 16)[:N]                    # [N, 16]
    W = torch.gather(lut_rows, 1, logical.long()).to(torch.float16)
    ref = (input_tensor.float() @ W.float().T).half()

cos = torch.nn.functional.cosine_similarity(output.flatten().float(), ref.flatten().float(), dim=0
).item()
print(f"\nCosine similarity vs reference: {cos:.6f}  ({'PASS' if cos > 0.99 else 'FAIL'})")
