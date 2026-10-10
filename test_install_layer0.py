#!/usr/bin/env python3
"""Test generation after install - install just layer 0 S codes."""

import sys
import os
_RAG = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _RAG)
sys.path.insert(0, os.path.join(_RAG, 'src', 'rag'))
sys.path.insert(0, os.path.join(_RAG, 'src', 'scripts'))
sys.path.insert(0, os.path.join(_RAG, 'src', 'flute_extended'))

import torch
from loader import load_quant_model
from transformers import AutoTokenizer
from tq_cache import TQCache
from ingest import load_system_state, reseed_cache
from snapshot import load_chunk
from install import sum_turboquant_codes

print('Loading model...')
model, meta = load_quant_model(
    artifacts_dir='/home/ubuntu/qwen3_5_9B_palettized',
    model_name='Qwen/Qwen3.5-9B',
    device='cuda',
    heads_dir='/home/ubuntu/qwen3_5_9B_palettized_heads'
)
model = model.eval()
tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen3.5-9B')

disk_dir = '/home/ubuntu/RAGGA/disk/ingested_50k'
system = load_system_state(disk_dir)
snap0 = load_chunk(os.path.join(disk_dir, 'snapshots', 'chunk_00000.npz'))

query = "What is 2+2?"
query_ids = tokenizer.encode(query, return_tensors='pt').cuda()

# Test: Install ONLY layer 0 S codes
print('\n[Test] Install ONLY layer 0 S codes...')
cache = TQCache(config=model.config, bits=3.5, online=True)
reseed_cache(cache, system)

# Only install layer 0 S codes
delta0 = snap0.s_codes.get(0)
if delta0:
    summed = sum_turboquant_codes(system.s_codes[0], [delta0], kind='S', bits=3.5)
    cache.set_s_codes(0, summed)
    print(f'  Installed layer 0 S codes')

with torch.no_grad():
    out = model(input_ids=query_ids, past_key_values=cache, use_cache=True)
    for i in range(5):
        next_id = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        print(f'  {tokenizer.decode(next_id[0])}', end='', flush=True)
        query_ids = torch.cat([query_ids, next_id], dim=1)
        if int(next_id) == tokenizer.eos_token_id:
            break
        out = model(input_ids=next_id, past_key_values=cache, use_cache=True)
print()

# Test baseline: Just system (no install at all)
print('\n[Test] Baseline - no install...')
cache = TQCache(config=model.config, bits=3.5, online=True)
reseed_cache(cache, system)

query_ids = tokenizer.encode(query, return_tensors='pt').cuda()
with torch.no_grad():
    out = model(input_ids=query_ids, past_key_values=cache, use_cache=True)
    for i in range(5):
        next_id = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        print(f'  {tokenizer.decode(next_id[0])}', end='', flush=True)
        query_ids = torch.cat([query_ids, next_id], dim=1)
        if int(next_id) == tokenizer.eos_token_id:
            break
        out = model(input_ids=next_id, past_key_values=cache, use_cache=True)
print()
