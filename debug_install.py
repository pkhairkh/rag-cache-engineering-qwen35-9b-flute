#!/usr/bin/env python3
"""Debug install_snapshot - check if installed codes match expected."""

import sys
import os
_RAG = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _RAG)
sys.path.insert(0, os.path.join(_RAG, 'src', 'rag'))
sys.path.insert(0, os.path.join(_RAG, 'src', 'scripts'))
sys.path.insert(0, os.path.join(_RAG, 'src', 'flute_extended'))

import torch
import numpy as np
from loader import load_quant_model
from transformers import AutoTokenizer
from tq_cache import TQCache, resolve_quantizer
from ingest import load_system_state
from snapshot import load_chunk
from install import install_snapshot

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

# Load chunk 0 snapshot
snap0 = load_chunk(os.path.join(disk_dir, 'snapshots', 'chunk_00000.npz'))
print(f'\nChunk 0:')
print(f'  s_codes: {list(snap0.s_codes.keys())[:5]}...')

# Create cache and install
cache = TQCache(config=model.config, bits=3.5, online=True)
from ingest import reseed_cache
reseed_cache(cache, system)

# Check cache before install
q0 = resolve_quantizer('S', system.s_codes[0].d, 3.5)
dequant_sys = q0.dequant(system.s_codes[0])
print(f'\nDequant system layer 0: mean={dequant_sys.mean():.4f}, std={dequant_sys.std():.4f}')

# Get current cache codes for layer 0
codes_before = cache.snapshot_codes()
s0_before = codes_before['s'].get(0)
if s0_before:
    dequant_before = q0.dequant(s0_before)
    print(f'Dequant cache layer 0 BEFORE install: mean={dequant_before.mean():.4f}, std={dequant_before.std():.4f}')
else:
    print('Cache layer 0 BEFORE install: None')

# Now install
print(f'\nInstalling chunk 0...')
report = install_snapshot(cache, system, [snap0])
print(f'  Report: layer 0 = {report.get(0)}')

# Get cache codes after install
codes_after = cache.snapshot_codes()
s0_after = codes_after['s'].get(0)
if s0_after:
    dequant_after = q0.dequant(s0_after)
    print(f'Dequant cache layer 0 AFTER install: mean={dequant_after.mean():.4f}, std={dequant_after.std():.4f}')
else:
    print('Cache layer 0 AFTER install: None')

# Expected: dequant(sys) + dequant(chunk0_delta)
dequant_chunk0 = q0.dequant(snap0.s_codes[0])
expected = dequant_sys + dequant_chunk0
print(f'\nExpected (sys + chunk0): mean={expected.mean():.4f}, std={expected.std():.4f}')
