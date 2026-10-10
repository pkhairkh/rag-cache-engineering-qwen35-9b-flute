#!/usr/bin/env python3
"""Build IVFADC index from ingested chunks."""

import sys
import os
_RAG = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _RAG)
sys.path.insert(0, os.path.join(_RAG, 'src', 'rag'))
sys.path.insert(0, os.path.join(_RAG, 'src', 'scripts'))
sys.path.insert(0, os.path.join(_RAG, 'src', 'flute_extended'))

from index import ChunkVectorLoader, build_index, IndexConfig

print('=' * 60)
print('RAGGA INDEX BUILD')
print('=' * 60)

disk_dir = '/home/ubuntu/RAGGA/disk/ingested_50k'
index_path = '/home/ubuntu/RAGGA/disk/index.faiss'

print(f'\n[1] Loading chunk vectors from {disk_dir}...')
loader = ChunkVectorLoader(disk_dir, bits=3.5)
chunks = list(loader.chunk_ids())
print(f'    Found {len(chunks)} chunks')
print(f'    Manifest: {loader.manifest.get("protocol", "unknown")}')

# Get vector dims from manifest
vector_dims = loader.manifest.get("vector_dims")
if vector_dims is None:
    # Load first chunk to get dims
    first_vec = loader.vector(chunks[0])
    vector_dims = first_vec.shape[0]
print(f'    Vector dims: {vector_dims}')

print('\n[2] Building simple flat index (FAISS IVFPQ needs >= 256 train points)...')
# For small test corpus, use IndexFlatIP directly
import faiss
import numpy as np

# Collect all vectors
print('    Loading vectors...')
vectors = []
for chunk_id in chunks:
    v = loader.vector(chunk_id)
    # L2 normalize
    v = v / np.linalg.norm(v)
    vectors.append(v)
    
vectors_np = np.vstack(vectors).astype('float32')
print(f'    Shape: {vectors_np.shape}')

# Build flat index
index = faiss.IndexFlatIP(vector_dims)
index.add(vectors_np)

# Save
print(f'    Writing index to {index_path}...')
faiss.write_index(index, index_path)

print('\n' + '=' * 60)
print('INDEX BUILD COMPLETE!')
print(f'Index written to: {index_path}')
print('=' * 60)
