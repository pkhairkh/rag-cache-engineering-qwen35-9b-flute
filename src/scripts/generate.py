#!/usr/bin/env python3
"""Text generation comparison: Dense FP16 vs FLUTE-palettized + QLoRA adapters.

Loads and generates with each model sequentially (not simultaneously) to avoid
memory pressure. Useful for comparing output quality side-by-side.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import torch
from transformers import AutoTokenizer

from eval_common import load_dense_fp16, load_quant_model, release_model_memory


def generate_dense(args, tokenizer):
    """Generate with dense FP16 model."""
    print(f"\n{'='*60}")
    print("[DENSE FP16]")
    print(f"{'='*60}")

    model = load_dense_fp16(args.model, args.device)
    
    inputs = tokenizer(args.prompt, return_tensors="pt").to(args.device)
    input_len = inputs.input_ids.shape[1]
    print(f"[input] {input_len} tokens")
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,  # Greedy
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    generated_ids = outputs[0][input_len:]
    generated = tokenizer.decode(generated_ids, skip_special_tokens=True)
    print("-" * 60)
    print(args.prompt, end="")
    print(generated)
    print("-" * 60)
    print(f"[output] {generated_ids.numel()} tokens")
    
    # Free memory
    del model
    release_model_memory()

    return generated_ids


def generate_palettized(args, tokenizer):
    """Generate with FLUTE-palettized model + optional QLoRA adapters."""
    print(f"\n{'='*60}")
    label = "PALETTIZED" if not args.adapters_dir else f"PALETTIZED + QLoRA ({args.adapters_dir})"
    print(f"[{label}]")
    print(f"{'='*60}")
    
    if args.adapters_dir:
        print(f"[loading] palettized + QLoRA adapters")
    else:
        print(f"[loading] palettized model (no adapters)")
    model, _ = load_quant_model(
        args.artifacts_dir, args.model, args.device,
        qlora_adapters=args.adapters_dir, dtype=torch.float16)

    model.eval()
    
    inputs = tokenizer(args.prompt, return_tensors="pt").to(args.device)
    input_len = inputs.input_ids.shape[1]
    print(f"[input] {input_len} tokens")
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,  # Greedy
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    generated_ids = outputs[0][input_len:]
    generated = tokenizer.decode(generated_ids, skip_special_tokens=True)
    print("-" * 60)
    print(args.prompt, end="")
    print(generated)
    print("-" * 60)
    print(f"[output] {generated_ids.numel()} tokens")
    
    # Free memory
    del model
    release_model_memory()

    return generated_ids


def main():
    parser = argparse.ArgumentParser(description="Generate text: Dense vs Palettized comparison")
    parser.add_argument("--artifacts-dir", required=True, help="Path to palettized model artifacts")
    parser.add_argument("--adapters-dir", default=None, help="Path to trained QLoRA adapters")
    parser.add_argument("--model", default="Qwen/Qwen3.5-9B", help="Base model name")
    parser.add_argument("--prompt", required=True, help="Input prompt")
    parser.add_argument("--max-new-tokens", type=int, default=256, help="Max tokens to generate")
    parser.add_argument("--device", default="cuda:0", help="Device")
    parser.add_argument("--skip-dense", action="store_true", help="Skip dense model generation")
    parser.add_argument("--skip-palettized", action="store_true", help="Skip palettized model generation")
    args = parser.parse_args()

    # Load tokenizer once
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    
    print(f"[prompt] {args.prompt!r}")
    print(f"[max_new_tokens] {args.max_new_tokens}")

    dense_ids = None
    palettized_ids = None

    # Generate with dense model
    if not args.skip_dense:
        dense_ids = generate_dense(args, tokenizer)

    # Generate with palettized model
    if not args.skip_palettized:
        palettized_ids = generate_palettized(args, tokenizer)

    # Summary comparison: token ids, not decoded strings (decode is
    # lossy around special tokens and rendering-dependent)
    if dense_ids is not None and palettized_ids is not None:
        print(f"\n{'='*60}")
        print("[COMPARISON]")
        print(f"{'='*60}")
        if torch.equal(dense_ids, palettized_ids):
            print("Outputs are IDENTICAL (token ids)")
        else:
            n = min(dense_ids.numel(), palettized_ids.numel())
            first_diff = next(
                (k for k in range(n)
                 if int(dense_ids[k]) != int(palettized_ids[k])), n)
            print("Outputs DIFFER")
            print(f"\nDense tokens:      {dense_ids.numel()}")
            print(f"Palettized tokens: {palettized_ids.numel()}")
            print(f"First divergence at generated token {first_diff}")


if __name__ == "__main__":
    main()
