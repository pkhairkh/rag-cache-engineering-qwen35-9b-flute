#!/usr/bin/env python3
"""toy_sweep_matrix.py — sweep the dimensions that actually decide whether
two global caches beat one, on a tiny CPU model.

Sweeps:
  - retrieval_overlap ∈ {0.0, 0.25, 0.5, 0.75, 1.0}
  - system_prompt_len ∈ {32, 128, 512, 2048}
  - cache_size_kb ∈ {1, 4, 16, 64, 256}  (the eviction pressure axis)
  - n_unique_chunks ∈ {4, 16, 64}  (corpus diversity)

For each (overlap, sys_len, cache_kb, n_chunks), runs 50 sessions with
3 cache configs: kv_only, state_only, both. Reports hit_rate, bytes,
evictions for each. Writes a single JSON matrix.
"""
import argparse, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from toy_global_cache_sweep import (TinyHybridModel, gen_corpus,
                                     sweep_one_config, GlobalKVCache, GlobalStateCache)
import torch

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n-sessions', type=int, default=50)
    ap.add_argument('--out', default='/home/z/my-project/scripts/poc_toy/sweep_matrix.json')
    args = ap.parse_args()

    overlaps = [0.0, 0.5, 1.0]
    sys_lens = [32, 256, 1024]
    cache_kbs = [4, 64]
    n_unique_chunks_list = [4, 16]

    torch.manual_seed(0)
    device = 'cpu'
    vocab = 256
    model = TinyHybridModel(hidden=32, vocab=vocab).to(device)
    model.eval()

    results = {
        'n_sessions': args.n_sessions,
        'sweeps': [],
        'summary': {},
    }

    # Summary accumulators
    n_configs = 0
    state_wins = 0
    both_wins = 0
    state_bytes_saved = 0
    state_more_evictions = 0

    for ov in overlaps:
        for sl in sys_lens:
            for ck in cache_kbs:
                for nuc in n_unique_chunks_list:
                    sessions = gen_corpus(args.n_sessions, sl, 32, 8, 8, vocab, 0, ov, nuc)
                    cache_bytes = ck * 1024
                    configs = [
                        ('kv_only', True, False),
                        ('state_only', False, True),
                        ('both', True, True),
                    ]
                    row = {'overlap': ov, 'sys_len': sl, 'cache_kb': ck,
                           'n_unique_chunks': nuc, 'configs': {}}
                    for name, uk, us in configs:
                        m = sweep_one_config(model, device, sessions, sl, 32,
                                             cache_bytes, uk, us)
                        row['configs'][name] = {
                            'hit_rate': round(m.hit_rate, 4),
                            'cross_session': round(m.cross_session_hit_rate, 4),
                            'bytes': m.bytes_cached,
                            'evictions': m.evictions,
                            'prefill_saved': m.prefill_tokens_saved,
                        }
                    results['sweeps'].append(row)
                    n_configs += 1
                    # track wins
                    kv = row['configs']['kv_only']
                    st = row['configs']['state_only']
                    bt = row['configs']['both']
                    if st['hit_rate'] >= kv['hit_rate'] and st['bytes'] < kv['bytes']:
                        state_wins += 1
                    if bt['hit_rate'] > max(kv['hit_rate'], st['hit_rate']):
                        both_wins += 1
                    if kv['bytes'] > 0:
                        state_bytes_saved += (kv['bytes'] - st['bytes'])
                    if st['evictions'] > kv['evictions']:
                        state_more_evictions += 1

    results['summary'] = {
        'n_configs': n_configs,
        'state_wins_on_hitrate_and_bytes': state_wins,
        'state_wins_pct': round(state_wins / n_configs * 100, 1),
        'both_beats_either': both_wins,
        'both_beats_pct': round(both_wins / n_configs * 100, 1),
        'state_total_bytes_saved_vs_kv': state_bytes_saved,
        'configs_where_state_evicts_more': state_more_evictions,
    }

    with open(args.out, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nSwept {n_configs} configurations.")
    print(f"State cache wins on (hit_rate ≥ kv) AND (bytes < kv): {state_wins}/{n_configs} ({state_wins/n_configs*100:.1f}%)")
    print(f"Both caches beats either: {both_wins}/{n_configs} ({both_wins/n_configs*100:.1f}%)")
    print(f"State cache total bytes saved vs KV: {state_bytes_saved}")
    print(f"Configs where state evicts MORE than KV: {state_more_evictions}/{n_configs}")
    print(f"\nResults: {args.out}")

    # Print a few interesting rows: high overlap, low cache, big sys prompt
    print("\n=== High-pressure rows (overlap=0.5, sys_len=512, cache_kb=4) ===")
    print(f"{'n_chunks':<10} {'config':<10} {'hit%':<8} {'bytes':<8} {'evict':<6}")
    for r in results['sweeps']:
        if r['overlap'] == 0.5 and r['sys_len'] == 512 and r['cache_kb'] == 4:
            for name, c in r['configs'].items():
                print(f"{r['n_unique_chunks']:<10} {name:<10} {c['hit_rate']*100:<8.1f} {c['bytes']:<8} {c['evictions']:<6}")

    print("\n=== Low-pressure rows (overlap=1.0, sys_len=32, cache_kb=256) ===")
    print(f"{'n_chunks':<10} {'config':<10} {'hit%':<8} {'bytes':<8} {'evict':<6}")
    for r in results['sweeps']:
        if r['overlap'] == 1.0 and r['sys_len'] == 32 and r['cache_kb'] == 256:
            for name, c in r['configs'].items():
                print(f"{r['n_unique_chunks']:<10} {name:<10} {c['hit_rate']*100:<8.1f} {c['bytes']:<8} {c['evictions']:<6}")

if __name__ == '__main__':
    main()
