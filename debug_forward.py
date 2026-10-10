#!/usr/bin/env python3
"""Debug script to trace conv state shapes."""

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

print('Loading model on CUDA...')
model, meta = load_quant_model(
    artifacts_dir='/home/ubuntu/qwen3_5_9B_palettized',
    model_name='Qwen/Qwen3.5-9B',
    device='cuda',
    heads_dir='/home/ubuntu/qwen3_5_9B_palettized_heads'
)
model = model.eval()

print('Loading tokenizer...')
tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen3.5-9B')

print('Tokenizing test input...')
text = "Hello world"
ids = tokenizer.encode(text, return_tensors='pt').cuda()
print(f'Input shape: {ids.shape}')

print('Running forward pass...')
with torch.no_grad():
    out = model.model(ids, use_cache=True)
print('Forward pass succeeded!')
print(f'Output shape: {out.last_hidden_state.shape}')
