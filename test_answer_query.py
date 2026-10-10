#!/usr/bin/env python3
"""Test answer_query flow directly."""

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
from ingest import load_system_state
from index import ChunkVectorLoader
from query import answer_query

print('Loading model...')
model, meta = load_quant_model(
    artifacts_dir='/home/ubuntu/qwen3_5_9B_palettized',
    model_name='Qwen/Qwen3.5-9B',
    device='cuda',
    heads_dir='/home/ubuntu/qwen3_5_9B_palettized_heads'
)
model = model.eval()
tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen3.5-9B')

# Load system state and loader
disk_dir = '/home/ubuntu/RAGGA/disk/ingested_50k'
loader = ChunkVectorLoader(disk_dir, bits=3.5)
system = load_system_state(disk_dir)

print(f'\nSystem: {system.reference()}')
print(f'Chunks: {list(loader.chunk_ids())[:5]}...')

# Simple query test
query = "What is 2+2?"
query_ids = tokenizer.encode(query, return_tensors='pt').cuda()
print(f'\nQuery: {query}')

def cache_factory():
    return TQCache(config=model.config, bits=3.5, online=True)

# Use answer_query with a single chunk (chunk 0)
print('\nRunning answer_query with chunk 0...')
result = answer_query(
    model=model,
    query_token_ids=query_ids,
    system=system,
    loader=loader,
    cache_factory=cache_factory,
    retrieved_ids=[0],  # Just chunk 0
    max_new_tokens=20
)

print(f'Retrieved: {result.retrieved_ids}')
print(f'New tokens: {result.new_token_ids}')
print(f'Answer: {tokenizer.decode(result.new_token_ids, skip_special_tokens=True)}')
print(f'Timings: {result.timings}')
