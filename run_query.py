#!/usr/bin/env python3
"""Run RAG queries using the proper framework."""

import sys
import os
_RAG = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _RAG)
sys.path.insert(0, os.path.join(_RAG, 'src', 'rag'))
sys.path.insert(0, os.path.join(_RAG, 'src', 'scripts'))
sys.path.insert(0, os.path.join(_RAG, 'src', 'flute_extended'))

import json
import numpy as np
import torch
from loader import load_quant_model
from transformers import AutoTokenizer
from index import ChunkVectorLoader, rerank
from query import query_cache_vector
from ingest import load_system_state, reseed_cache
from tq_cache import TQCache

print('=' * 60)
print('RAGGA QUERY TEST (Using Framework)')
print('=' * 60)

# Load model
print('\n[1] Loading model...')
model, meta = load_quant_model(
    artifacts_dir='/home/ubuntu/qwen3_5_9B_palettized',
    model_name='Qwen/Qwen3.5-9B',
    device='cuda',
    heads_dir='/home/ubuntu/qwen3_5_9B_palettized_heads'
)
model = model.eval()
tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen3.5-9B')
print('    Model loaded')

# Load system state and loader
print('\n[2] Loading ingested data...')
disk_dir = '/home/ubuntu/RAGGA/disk/ingested_50k'
loader = ChunkVectorLoader(disk_dir, bits=3.5)
system = load_system_state(disk_dir)
chunks = list(loader.chunk_ids())
print(f'    System: {system.reference()}')
print(f'    {len(chunks)} chunks')

# Load questions
print('\n[3] Loading questions...')
questions_path = '/home/ubuntu/enterprise_rag_bench/questions/questions.jsonl'
questions = []
with open(questions_path) as f:
    for i, line in enumerate(f):
        if i >= 5:
            break
        q = json.loads(line)
        questions.append(q.get('text', q.get('question', '')))
print(f'    Loaded {len(questions)} questions')

# Run queries using direct rerank (no FAISS index for 12.5M dims)
print('\n[4] Running queries (direct rerank over all chunks)...')

def cache_factory():
    return TQCache(config=model.config, bits=3.5, online=True)

for i, question in enumerate(questions):
    print(f'\n    Query {i+1}: {question[:80]}...')
    
    # Tokenize query
    query_ids = tokenizer.encode(question, return_tensors='pt').cuda()
    
    # [2] Prefill query through cache to get query vector
    cache = cache_factory()
    reseed_cache(cache, system)
    with torch.no_grad():
        model(input_ids=query_ids, past_key_values=cache, use_cache=True)
    qvec = query_cache_vector(cache, system)
    
    # [3-4] Direct rerank against all chunks (no FAISS preselect)
    ids, scores = rerank(loader, qvec, np.array(chunks), k=3)
    
    print(f'    Top 3 chunks: {[int(x) for x in ids]}')
    print(f'    Scores: {[float(f"{s:.4f}") for s in scores]}')
    
    # [5-8] Install top chunk and generate answer
    from query import answer_query
    cache = cache_factory()
    result = answer_query(
        model=model,  # Full CausalLM for generation
        query_token_ids=query_ids,
        system=system,
        loader=loader,
        cache_factory=cache_factory,
        retrieved_ids=[int(x) for x in ids],  # Use retrieved chunks
        max_new_tokens=128
    )
    
    # Decode answer
    answer_text = tokenizer.decode(result.new_token_ids, skip_special_tokens=True)
    print(f'    Answer: {answer_text[:200]}...')

print('\n' + '=' * 60)
print('QUERY TEST COMPLETE!')
print('=' * 60)
