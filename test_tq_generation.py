#!/usr/bin/env python3
"""Test generation with TQCache."""

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

print('Loading model...')
model, meta = load_quant_model(
    artifacts_dir='/home/ubuntu/qwen3_5_9B_palettized',
    model_name='Qwen/Qwen3.5-9B',
    device='cuda',
    heads_dir='/home/ubuntu/qwen3_5_9B_palettized_heads'
)
model = model.eval()
tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen3.5-9B')

# Create TQCache
cache = TQCache(config=model.config, bits=3.5, online=True)

prompt = "The capital of France is"
print(f'\nPrompt: {prompt}')
ids = tokenizer.encode(prompt, return_tensors='pt').cuda()

print('\nGenerating with TQCache...')
with torch.no_grad():
    # First forward with cache
    out = model(input_ids=ids, past_key_values=cache, use_cache=True)
    
    # Generate 10 more tokens
    for i in range(10):
        logits = out.logits
        next_id = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        ids = torch.cat([ids, next_id], dim=1)
        out = model(input_ids=next_id, past_key_values=cache, use_cache=True)
        print(f'  {tokenizer.decode(next_id[0])}', end='', flush=True)

print(f'\n\nFull output: {tokenizer.decode(ids[0])}')
