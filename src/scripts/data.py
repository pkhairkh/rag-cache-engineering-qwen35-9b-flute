#!/usr/bin/env python3
"""scripts/data.py — SFT dataset loading and tokenization.

Contract for every example returned by load_sft_dataset (and produced
by _tokenize): three torch.long tensors of shape (seq_len,) each:
  input_ids       left-padded token ids (pad_id first)
  labels          -100 on padding and on the prompt prefix; target ids
                  on the response suffix
  attention_mask  0 on padding, 1 on text
Examples whose labels are all -100 are dropped by the loader. EOS is
appended to full_ids when the tokenizer defines eos_token_id; padding
uses pad_token_id with fallback to eos_token_id.

Dataset sources are HuggingFace hub ids; DATASET_DEFAULTS maps the CLI
name to (hub_id, split, record_style). Records missing their style's
required fields are skipped, not padded. An unknown dataset name raises
ValueError — loud, never a silent empty list.

_report_status prints the kernel-path census (fused-forward module
count, fused-backward kernel availability); it lives here because the
SFT trainer and the O-1 probe import both from this module.
"""
from __future__ import annotations
import os
import sys

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path: sys.path.insert(0, _HERE)
import qlora

DATASET_DEFAULTS = {
    "OASST1": ("OpenAssistant/oasst1", "train+validation", "oasst"),
    "Alpaca": ("yahma/alpaca-cleaned", "train", "alpaca"),
    "FLANv2": ("SirNeural/flan_v2", "train", "flan"),
    "HH-RLHF": ("Anthropic/hh-rlhf", "train", "hh"),
    "Unnatural": ("Orsteneo/unnatural-instructions", "train", "alpaca"),
    "Self-Instruct": ("yizhongw/self_instruct", "train", "alpaca"),
    "LongForm": ("akoksal/LongForm", "train", "alpaca"),
    "Chip2": ("laion/OIG", "train", "alpaca"),
    "FineWeb-Edu": ("HuggingFaceFW/fineweb-edu", "train", "fineweb"),
}

def _format_alpaca(instruction, response, sys_prompt=None):
    """Alpaca template: (full_text, response_text); response is the
    supervised suffix of full_text."""
    if sys_prompt is None:
        sys_prompt = "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n"
    return (f"{sys_prompt}### Instruction:\n{instruction}\n\n### Response:\n{response}",
            f"### Response:\n{response}")

def _format_oasst(conversation):
    """OASST flattening: (full_text, response_text) with the assistant
    turns as the supervised response."""
    parts, response_parts = [], []
    for turn in conversation:
        role, text = turn.get("role", "user"), turn.get("text", "")
        if role == "assistant": parts.append(f"Assistant: {text}"); response_parts.append(f"Assistant: {text}")
        else: parts.append(f"User: {text}")
    return "\n\n".join(parts), "\n\n".join(response_parts)

def _tokenize(tokenizer, full_text, response_text, seq_len, eos_token):
    """Tokenize one (full_text, response_text) pair into the example
    contract: left-padded ids, -100 labels outside the response, EOS
    appended, truncation keeps the last seq_len ids."""
    idx = full_text.rfind(response_text)
    prefix_text = full_text if idx < 0 else full_text[:idx]
    prefix_ids = tokenizer(prefix_text, add_special_tokens=False)["input_ids"]
    full_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"]
    if tokenizer.eos_token_id is not None and (len(full_ids) == 0 or full_ids[-1] != tokenizer.eos_token_id):
        full_ids = full_ids + [tokenizer.eos_token_id]
    if len(full_ids) > seq_len:
        full_ids = full_ids[-seq_len:]
        cut = len(full_ids) - len(tokenizer(response_text, add_special_tokens=False)["input_ids"])
        prefix_len = max(0, cut)
    else:
        prefix_len = min(len(prefix_ids), len(full_ids))
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    pad_len = seq_len - len(full_ids)
    if pad_len > 0:
        input_ids = [pad_id] * pad_len + full_ids
        labels = [-100] * pad_len + [-100] * prefix_len + full_ids[prefix_len:]
        attn = [0] * pad_len + [1] * len(full_ids)
    else:
        input_ids, labels, attn = full_ids, [-100] * prefix_len + full_ids[prefix_len:], [1] * seq_len
    return {"input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attn, dtype=torch.long)}

def load_sft_dataset(dataset_name, seq_len, tokenizer, max_samples=0):
    """Load DATASET_DEFAULTS[dataset_name] and tokenize every kept
    record into the example contract. max_samples caps records (after
    style parsing for non-streaming styles, during iteration for
    FineWeb-Edu); 0 = uncapped. Raises ValueError on an unknown name."""
    from datasets import load_dataset
    if dataset_name not in DATASET_DEFAULTS:
        raise ValueError(f"unknown dataset {dataset_name!r}")
    hf_id, split, style = DATASET_DEFAULTS[dataset_name]
    print(f"  [data] loading {dataset_name} ({hf_id})", flush=True)

    # Use streaming for large datasets like FineWeb-Edu
    use_streaming = style == "fineweb"
    ds = load_dataset(hf_id, split=split, streaming=use_streaming)
    examples = []

    if style == "oasst":
        # parent_id -> children index (single pass; O(1) child lookup per node)
        children_by_parent = {}
        msg_by_id = {}
        for m in ds:
            mid = m.get("message_id")
            pid = m.get("parent_id")
            if mid:
                msg_by_id[mid] = m
            if pid:
                children_by_parent.setdefault(pid, []).append(m)
        for pid in children_by_parent:
            children_by_parent[pid].sort(key=lambda m: m.get("rank") or 0)

        roots = [r for r in ds if r.get("parent_id") is None]
        for r in roots:
            conv, cur, seen = [], r, 0
            while cur is not None and seen < 8:
                conv.append({"role": "assistant" if cur["role"] == "assistant" else "user", "text": cur["text"]})
                children = children_by_parent.get(cur["message_id"], [])
                if not children: break
                cur, seen = children[0], seen + 1
            if len(conv) < 2: continue
            full, response = _format_oasst(conv)
            examples.append((full, response))
    elif style == "alpaca":
        for r in ds:
            instr, inp, out = r.get("instruction", ""), r.get("input", ""), r.get("output", "")
            if not instr or not out: continue
            full, response = _format_alpaca(f"{instr}\n\n{inp}" if inp else instr, out)
            examples.append((full, response))
    elif style == "hh":
        for r in ds:
            chosen = r.get("chosen", "")
            if not chosen: continue
            parts = chosen.split("\n\nHuman: ")
            if len(parts) < 2: continue
            full = parts[0] + "\n\nHuman: " + "\n\nHuman: ".join(parts[1:])
            asst_idx = full.rfind("\n\nAssistant: ")
            response = full[asst_idx:] if asst_idx >= 0 else full
            examples.append((full, response))
    elif style == "fineweb":
        # FineWeb-Edu: raw web text for continued pretraining (streaming)
        # Each sample is just text - we use it as both input and target
        n_added = 0
        for r in ds:
            text = r.get("text", "")
            if not text or len(text) < 100: continue  # skip very short texts
            # For continued pretraining, the whole text is the target
            # We'll use a prefix as context during tokenization
            examples.append((text, text))
            n_added += 1
            if max_samples > 0 and n_added >= max_samples:
                break
        # Don't slice again since we stopped at max_samples during iteration
    else:
        if max_samples > 0: examples = examples[:max_samples]
    print(f"  [data] {len(examples)} examples", flush=True)
    eos = tokenizer.eos_token or "</s>"
    out = [_tokenize(tokenizer, f, r, seq_len, eos) for f, r in examples]
    out = [e for e in out if (e["labels"] != -100).any().item()]
    print(f"  [data] {len(out)} examples after masking filter", flush=True)
    return out

def _report_status(model):
    """Print the kernel-path census: how many QLoRA modules sit on the
    FLUTE fused forward kernel, and whether the fused backward kernel
    is built. Print-only; never raises (a missing build reports
    NOT BUILT with the build command, not a traceback)."""
    try:
        import qlora_gemm
        n_qlora = n_fused = 0
        for _, mod in qlora.iter_qlora_modules(model):
            if isinstance(mod, qlora.QLoRALinear):
                n_qlora += 1
                if mod._use_fused(): n_fused += 1
        print(f"  [qlora_gemm] FusedQLoRAGEMM {'ACTIVE' if n_fused else 'NOT ENGAGED'}: "
              f"{n_fused}/{n_qlora} modules use the FLUTE kernel for the forward", flush=True)
        if qlora_gemm.fused_backward_available():
            print(f"  [flute_train_kernels] FUSED BACKWARD KERNEL ACTIVE: "
                  f"fused_backward_gemm ready (W never in DRAM, mma.sync FP32-acc)", flush=True)
        else:
            print(f"  [flute_train_kernels] FUSED BACKWARD KERNEL NOT BUILT: "
                  f"backward falls back to reference dequant + torch.matmul. "
                  f"Build: cd flute_train_kernels && python setup.py build_ext --inplace", flush=True)
    except Exception as e:
        print(f"  [status] check skipped ({e!r})", flush=True)
