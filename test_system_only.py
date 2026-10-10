#!/usr/bin/env python3
"""Test answer_query flow WITHOUT chunk installation."""

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
from query import query_cache_vector

print('Loading model...')
model, meta = load_quant_model(
    artifacts_dir='/home/ubuntu/qwen3_5_9B_palettized',
    model_name='Qwen/Qwen3.5-9B',
    device='cuda',
    heads_dir='/home/ubuntu/qwen3_5_9B_palettized_heads'
)
model = model.eval()
tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen3.5-9B')

# Load system state
disk_dir = '/home/ubuntu/RAGGA/disk/ingested_50k'
system = load_system_state(disk_dir)

print(f'\nSystem: {system.reference()}')

# Simple query test
query = "What is 2+2?"
query_ids = tokenizer.encode(query, return_tensors='pt').cuda()
print(f'\nQuery: {query}')

# Test: Generate with system state but NO chunk installation
print('\nTest 1: Prefill on system state, then generate WITHOUT chunk install...')
cache = TQCache(config=model.config, bits=3.5, online=True)
reseed_cache(cache, system)

# Prefill query
with torch.no_grad():
    out = model(input_ids=query_ids, past_key_values=cache, use_cache=True)
    logits = out.logits
    
    # Generate 10 tokens
    for i in range(10):
        next_id = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        print(f'  {tokenizer.decode(next_id[0])}', end='', flush=True)
        query_ids = torch.cat([query_ids, next_id], dim=1)
        if int(next_id) == tokenizer.eos_token_id:
            break
        out = model(input_ids=next_id, past_key_values=cache, use_cache=True)
        logits = out.logits

print(f'\nFull: {tokenizer.decode(query_ids[0])}')

# Test 2: Just the cache after system prefill - what's in it?
print('\nTest 2: System state analysis...')
print(f'  system.s_codes keys: {sorted(system.s_codes.keys())[:5]}...')
print(f'  cache has {len(cache._s_layers)} S layers')
