#!/usr/bin/env python3
"""toy_finetune_test.py — test the CacheBlend-FT hypothesis (arxiv 2609.09768):
does fine-tuning the model with cache restoration in the loop produce states
that survive re-installation better?

We measure "restoration error" = ||S_restored - S_recomputed|| before and
after a few hundred fine-tune steps where the model sees its own restored
state during forward passes.

This is the kernel of the user's "finetune on operating on global cache"
request: the model is trained to be cache-friendly, not just trained on
the downstream task.
"""
import argparse, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from toy_global_cache_sweep import TinyHybridModel, gen_corpus
import torch
import torch.nn.functional as F


def measure_restoration_error(model, sessions, device, n_samples=20, sys_len=256):
    """Measure the divergence between (a) a fresh prefill that produces state
    S, then continuing with one more token t, and (b) snapshotting S to fp16,
    restoring it, then continuing with the same token t.

    The cache hit semantics: we prefill the system prompt, get state S.
    A later request wants to continue from S + t. Path A: S is in fp32 (no
    cache). Path B: S was stored in fp16 (the cache), restored to fp32, then
    we continue. The difference in the OUTPUT token's logits is the
    restoration error — what the user actually sees."""
    errors = []
    model.eval()
    with torch.no_grad():
        for s in sessions[:n_samples]:
            tokens = torch.tensor([s], device=device, dtype=torch.long)
            sys_tokens = tokens[:, :sys_len]
            next_token = tokens[:, sys_len:sys_len+1]  # one more token
            # path A: fresh prefill, then continue with next_token
            _, state_a, conv_a, kv_a = model(sys_tokens)
            logits_a, _, _, _ = model(next_token, linear_state=state_a,
                                       conv_state=conv_a, kv_cache=kv_a)
            # path B: snapshot state to fp16 (simulate cache), restore, continue
            state_b_init = state_a.half().float()
            conv_b_init = conv_a.half().float()
            kv_b_init = (kv_a[0].half().float(), kv_a[1].half().float())
            logits_b, _, _, _ = model(next_token, linear_state=state_b_init,
                                       conv_state=conv_b_init, kv_cache=kv_b_init)
            # the error is the divergence in the next-token logits
            err = (logits_a - logits_b).norm().item() / max(1e-8, logits_a.norm().item())
            errors.append(err)
    return sum(errors) / len(errors)


def finetune_on_cache_operation(model, sessions, device, n_steps=200, lr=1e-3,
                                  use_restoration_loss=True, restoration_weight=0.5):
    """Fine-tune the model with cache-operation-aware training. Two variants:
    (a) LM-only (use_restoration_loss=False): train on the answer-token LM loss
        with the cache round-trip in the forward pass. (The CacheBlend-FT idea.)
    (b) LM + restoration loss (use_restoration_loss=True): add an auxiliary loss
        term that penalizes the divergence between the fp32-fresh logits and
        the fp16-restored logits on the next token. (The explicit cache-friendly
        objective.)
    """
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    losses = []
    for step in range(n_steps):
        s = sessions[step % len(sessions)]
        tokens = torch.tensor([s], device=device, dtype=torch.long)
        sys_tokens = tokens[:, :256]
        rest_tokens = tokens[:, 256:]
        # forward through system prompt, snapshot state to fp16 (simulating cache)
        with torch.no_grad():
            _, state_snap, conv_snap, kv_snap = model(sys_tokens)
        # simulate the cache round-trip: fp16 store + restore
        state_restored = state_snap.half().float()
        conv_restored = conv_snap.half().float()
        kv_restored = (kv_snap[0].half().float(), kv_snap[1].half().float())

        # the LM loss path: forward through the rest with the restored state
        logits, _, _, _ = model(rest_tokens, linear_state=state_restored,
                                conv_state=conv_restored, kv_cache=kv_restored)
        targets = rest_tokens[:, -8:]
        logits_ans = logits[:, -8:, :]
        lm_loss = F.cross_entropy(logits_ans.reshape(-1, logits_ans.size(-1)),
                                  targets.reshape(-1))

        if use_restoration_loss:
            # the restoration loss: forward ONE more token with both the fresh
            # fp32 state and the restored fp16 state, penalize the divergence
            next_token = rest_tokens[:, :1]
            with torch.no_grad():
                _, state_fresh, conv_fresh, kv_fresh = model(sys_tokens)
            logits_fresh, _, _, _ = model(next_token, linear_state=state_fresh,
                                           conv_state=conv_fresh, kv_cache=kv_fresh)
            logits_restored, _, _, _ = model(next_token, linear_state=state_restored,
                                              conv_state=conv_restored, kv_cache=kv_restored)
            rest_loss = F.mse_loss(logits_restored, logits_fresh.detach())
            loss = lm_loss + restoration_weight * rest_loss
        else:
            loss = lm_loss

        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
        if step % 50 == 0:
            print(f'  step {step}: loss {loss.item():.4f} (lm={lm_loss.item():.4f})')
    return losses


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n-steps', type=int, default=200)
    ap.add_argument('--out', default='/home/z/my-project/scripts/poc_toy/finetune_test.json')
    args = ap.parse_args()

    torch.manual_seed(0)
    device = 'cpu'
    vocab = 256
    model = TinyHybridModel(hidden=32, vocab=vocab).to(device)
    # Make the state values large enough that fp16 quantization bites.
    # In the real Qwen3.5-9B, the recurrent state accumulates over
    # thousands of tokens across 24 layers, with values reaching into
    # the hundreds. Here we artificially amplify the value projections
    # so the state magnitude is in fp16's lossy range (>~2 where the
    # mantissa has <10 bits of precision).
    with torch.no_grad():
        model.linear_attn.in_proj_qkv.weight.mul_(8.0)
        model.linear_attn.out_proj.weight.mul_(0.125)  # keep the output magnitude sane
        model.linear_attn.A_log.mul_(0.1)  # very small decay = state grows over the prefill

    sessions = gen_corpus(50, 256, 32, 8, 8, vocab, 0, 0.5, 8)

    # before fine-tune
    err_before = measure_restoration_error(model, sessions, device)
    print(f'Before fine-tune: mean restoration error = {err_before:.6e}')

    # Variant A: LM-only fine-tune (the CacheBlend-FT idea)
    print(f'\n--- Variant A: LM-only fine-tune (CacheBlend-FT) for {args.n_steps} steps ---')
    losses_a = finetune_on_cache_operation(model, sessions, device, n_steps=args.n_steps,
                                            use_restoration_loss=False)
    err_after_a = measure_restoration_error(model, sessions, device)
    print(f'After LM-only fine-tune: restoration error = {err_after_a:.6e}')
    if err_before > 0:
        print(f'Reduction: {(1 - err_after_a/err_before)*100:.1f}%')

    # reset the model to fresh
    torch.manual_seed(0)
    model = TinyHybridModel(hidden=32, vocab=vocab).to(device)
    with torch.no_grad():
        model.linear_attn.in_proj_qkv.weight.mul_(8.0)
        model.linear_attn.out_proj.weight.mul_(0.125)
        model.linear_attn.A_log.mul_(0.1)

    # Variant B: LM + restoration loss (explicit cache-friendly objective)
    print(f'\n--- Variant B: LM + restoration-loss fine-tune for {args.n_steps} steps ---')
    losses_b = finetune_on_cache_operation(model, sessions, device, n_steps=args.n_steps,
                                            use_restoration_loss=True, restoration_weight=0.5)
    err_after_b = measure_restoration_error(model, sessions, device)
    print(f'After LM+restoration fine-tune: restoration error = {err_after_b:.6e}')
    if err_before > 0:
        print(f'Reduction: {(1 - err_after_b/err_before)*100:.1f}%')

    result = {
        'n_steps': args.n_steps,
        'restoration_error_before': err_before,
        'restoration_error_after_lm_only': err_after_a,
        'restoration_error_after_lm_plus_restoration': err_after_b,
        'reduction_lm_only_pct': (1 - err_after_a/err_before)*100 if err_before > 0 else None,
        'reduction_lm_plus_restoration_pct': (1 - err_after_b/err_before)*100 if err_before > 0 else None,
        'loss_lm_only_first': losses_a[0],
        'loss_lm_only_last': losses_a[-1],
        'loss_lm_plus_restoration_first': losses_b[0],
        'loss_lm_plus_restoration_last': losses_b[-1],
    }
    with open(args.out, 'w') as f:
        json.dump(result, f, indent=2)
    print(f'\n{args.out}')

if __name__ == '__main__':
    main()
