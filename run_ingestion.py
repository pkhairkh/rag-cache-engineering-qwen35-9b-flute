#!/usr/bin/env python3
"""Run RAGGA ingestion on 50K documents from EnterpriseRAG-Bench."""

import sys
import os

# Set up paths
_RAG = os.path.dirname(os.path.abspath(__file__))
if _RAG not in sys.path:
    sys.path.insert(0, _RAG)
sys.path.insert(0, os.path.join(_RAG, 'src', 'rag'))
sys.path.insert(0, os.path.join(_RAG, 'src', 'scripts'))
sys.path.insert(0, os.path.join(_RAG, 'src', 'flute_extended'))

import torch
import json
from loader import load_quant_model
from ingest import IngestDriver
from transformers import AutoTokenizer

print('=' * 60)
print('RAGGA INGESTION - 50K DOCUMENTS')
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
print(f'    Model loaded, vocab_size={model.config.vocab_size}')

# Load tokenizer
print('\n[2] Loading tokenizer...')
tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen3.5-9B')
print('    Tokenizer loaded')

# Load 50K documents
print('\n[3] Loading 50K documents...')
docs = []
with open('/home/ubuntu/enterprise_rag_bench/documents/documents.jsonl') as f:
    for i, line in enumerate(f):
        if i >= 100:
            break
        doc = json.loads(line)
        docs.append(doc.get('text', doc.get('content', '')))
print(f'    Loaded {len(docs)} documents')

# System prompt
print('\n[4] Preparing system prompt...')
system_prompt = 'You are a helpful AI assistant.'
system_ids = tokenizer.encode(system_prompt, add_special_tokens=False)
system_ids = torch.tensor([system_ids], dtype=torch.long)
print(f'    System prompt: {len(system_ids[0])} tokens')

# Create chunks (each doc is a chunk, NO truncation)
print('\n[5] Tokenizing documents (full text, NO chunking)...')
chunks = []
for i, doc in enumerate(docs):
    if i % 10000 == 0:
        print(f'    Tokenizing doc {i}/{len(docs)}...')
    ids = tokenizer.encode(doc, add_special_tokens=False)
    chunks.append(torch.tensor([ids], dtype=torch.long))
print(f'    Created {len(chunks)} document chunks')

# Run ingestion
out_dir = '/home/ubuntu/RAGGA/disk/ingested_50k'
print(f'\n[6] Running ingestion to {out_dir}...')

drv = IngestDriver(
    model=model.model,  # Use inner TextModel (not CausalLM wrapper)
    system_token_ids=system_ids,
    chunks=chunks,
    out_dir=out_dir,
    bits=3.5
)

drv.run()
print('\n' + '=' * 60)
print('INGESTION COMPLETE!')
print('=' * 60)
