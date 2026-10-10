#!/usr/bin/env python3
"""
test_qwen_weights.py — validate the kernel against REAL Qwen3.5-9B
palettized weights on disk (not synthetic tensors).

For a sample of layers (default 0, 15, 31) this checks:
  1. file/shape consistency against metadata.json
  2. the production streaming kernel (idx4 layout) vs a pure-torch FP32
     reference dequantization (cosine > 0.999 AND rel-max < 1e-3 — same
     gates as the kernel test suite), where the reference indices are
     recovered from the on-disk blob with flute_extended.idx4.unpack_idx4
  3. the idx4 path is BIT-EXACT vs the kernel-internal legacy byte order
     on the same real weights (unpacked blob, q_layout=0 via _C)

Exit code 0 only if every check passes.

Usage:
    python test_qwen_weights.py                       # layers 0, 15, 31
    python test_qwen_weights.py --model-dir DIR --layers 0 3 7
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import flute_extended  # noqa: E402
from flute_extended import qgemm_per_group_lut  # noqa: E402
from flute_extended.idxN import unpack_idxn as unpack_idx4  # noqa: E402


def load_palettized_weight(weight_name: str, model_dir: Path):
    """Load one palettized weight pair from disk (file naming per the
    palettized export: underscores for dots)."""
    idx_files = sorted(model_dir.glob(f"{weight_name}*.idx4"))
    lut_files = sorted(model_dir.glob(f"{weight_name}*.lut_scalar"))
    if not idx_files or not lut_files:
        raise FileNotFoundError(f"Could not find weight files for {weight_name} in {model_dir} "
            f"(got {len(idx_files)} .idx4, {len(lut_files)} .lut_scalar)")
    indices = torch.from_file(str(idx_files[0]), dtype=torch.uint8)
    lut = torch.from_file(str(lut_files[0]), dtype=torch.float16)
    return indices, lut, idx_files[0], lut_files[0]


def dequantize_reference(blob: torch.Tensor, lut: torch.Tensor,
                         dense_shape, group_size: int) -> torch.Tensor:
    """Reference dequantization (src/docs/QUANTIZATION.md
    sections 1-4, 8).

    The on-disk blob is unpacked back to the logical packed matrix with
    flute_extended.idx4.unpack_idx4 first; groups run along N:
    W[n, k] = LUT[n // group_size, nibble(n, k)], nibbles LSB-first
    (even k = low nibble). LUT is [ceil(N/gs), 16].
    """
    N, K = dense_shape
    assert blob.numel() == N * (K // 2), \
        f"index blob size {blob.numel()} != N*K/2 = {N * (K // 2)}"
    q = torch.from_numpy(unpack_idx4(blob.cpu().numpy(), N, K))          # [N, K/2] logical
    n_groups = (N + group_size - 1) // group_size
    lut2 = lut[:n_groups * 16].view(n_groups, 16)

    lo = (q & 0x0F).long()                     # even k indices
    hi = ((q >> 4) & 0x0F).long()               # odd k indices
    idx = torch.stack([lo, hi], dim=-1).reshape(N, K)
    groups = torch.arange(N) // group_size
    return lut2[groups.unsqueeze(1).expand(N, K), idx]   # W [N, K] fp16


def test_layer(weight_name: str, meta_key: str, info: dict,
               model_dir: Path, device: str) -> bool:
    print(f"\n[{weight_name}]")
    N, K = info["dense_shape"]
    gs = info["group_size"]
    print(f"  Shape: [{N}, {K}], group_size: {gs}")

    indices, lut, idx_file, lut_file = load_palettized_weight(weight_name, model_dir)

    # ---- 1. shape consistency ------------------------------------------
    ok_shapes = (indices.numel() == N * (K // 2)
                 and lut.numel() == ((N + gs - 1) // gs) * 16)
    print(f"  [{'PASS' if ok_shapes else 'FAIL'}] blob sizes match metadata "
          f"(idx={indices.numel()} B, lut={lut.numel} halves)")
    if not ok_shapes:
        return False

    indices_dev = indices.to(device)
    lut_dev = lut.view(-1, 16).to(device)
    M = 128
    X = torch.randn(M, K, dtype=torch.float16, device=device)

    # ---- 2. streaming kernel (idx4) vs FP32 reference --------------------
    W = dequantize_reference(indices, lut, (N, K), gs)
    ref = (X.float() @ W.float().T).half()

    Y = qgemm_per_group_lut(X, indices_dev, lut_dev,
                            bitwidth=4, group_size=gs,
                            backend="cutlass_streaming",
                            indices_layout="idx4")
    cos = torch.nn.functional.cosine_similarity(
        Y.float().flatten(), ref.float().flatten(), dim=0).item()
    relmax = ((Y.float() - ref.float()).abs().max()
              / ref.float().abs().max()).item()
    ok_kernel = cos > 0.999 and relmax < 1e-3
    print(f"  [{'PASS' if ok_kernel else 'FAIL'}] streaming(idx4) vs fp32 ref: "
          f"cos={cos:.6f}, rel-max={relmax:.3e}")
    if not ok_kernel:
        return False

    # ---- 3. idx4 bit-exact vs kernel-internal legacy byte order ----------
    q_logical = torch.from_numpy(unpack_idx4(indices.cpu().numpy(), N, K)).to(device)
    W_empty = torch.empty(0, dtype=torch.float16, device=device)
    Y_leg = flute_extended._C.qgemm_per_group_lut(
        X, q_logical, lut_dev, W_empty, 4, gs, "cutlass_streaming", 0)
    exact = torch.equal(Y, Y_leg)
    print(f"  [{'PASS' if exact else 'FAIL'}] idx4 bit-exact vs legacy")
    return exact


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", type=Path,
                    default=Path("/home/ubuntu/qwen3_5_9b_palettized"))
    ap.add_argument("--layers", type=int, nargs="+", default=[0, 15, 31])
    args = ap.parse_args()

    metadata_file = args.model_dir / "metadata.json"
    if not metadata_file.exists():
        print(f"ERROR: Metadata file not found: {metadata_file}")
        sys.exit(1)
    with open(metadata_file) as f:
        metadata = json.load(f)

    if not torch.cuda.is_available():
        print("CUDA is not available — this test requires a GPU.")
        sys.exit(2)

    print("=" * 70)
    print("Testing FLUTE kernel against Qwen3.5-9B palettized weights")
    print("=" * 70)

    all_ok = True
    found = 0
    for layer_idx in args.layers:
        weight_name = f"model_layers_{layer_idx}_mlp_gate_proj_weight"
        meta_key = f"model.layers.{layer_idx}.mlp.gate_proj.weight"
        if meta_key not in metadata:
            print(f"\nLayer {layer_idx} ({meta_key}) not found in metadata")
            continue
        found += 1
        all_ok &= test_layer(weight_name, meta_key, metadata[meta_key],
                             args.model_dir, "cuda")

    if found == 0:
        print("ERROR: no matching layers found in metadata.json")
        sys.exit(1)

    print("\n" + "=" * 70)
    print("ALL LAYER TESTS " + ("PASSED" if all_ok else "FAILED"))
    print("=" * 70)
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
