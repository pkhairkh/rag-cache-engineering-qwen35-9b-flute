#!/usr/bin/env python3
"""
calibrate_real_text.py — real-text calibration capture.

The default calibration source of palettize_qwen3_5_9b.py, accumulated over
real text (WikiText-2 train / fineweb-edu / a local file) as:

  * running per-input-channel Hessian diagonal  h_k = sum_t x_tk^2   (always,
    GPU, a few MB total — the quantity the weighted-Lloyd quantizer needs);
  * a retained activation sample (first `retain_rows` rows per tensor, CPU)
    for the per-tensor cosine gate;
  * full Gram matrices X^T X — for every hooked tensor when `full_gram=True`
    (the GPTQ assignment and the whitened-SVD residual both consume them),
    or for a selected `gram_keys` subset;
  * the head Gram (W5): top-level modules (lm_head) passed via
    extra_head_modules, hooked input-side under the ("head", name)
    pseudo-block key, full-gram-always and accumulated across capture
    forwards at zero extra forward cost;
  * optional running input mean.

Split discipline (binding): calibration uses the WikiText-2 **train**
split; the perplexity evaluation (scripts/eval_ppl.py) uses the **test**
split — CALIB_SPLIT / EVAL_SPLIT below are the single source of truth.

Memory model: nothing here stores full activations except the retained
sample, and hooks can be restricted to a layer range, so n_seqs x seq_len
scales to the community-standard 128x2048 (~262k tokens) in layer blocks
(the palettizer drives `layer_range` per block).

W11 streaming note: the palettizer no longer captures block-by-block
with CPU Grams (the per-hook .cpu() sync storm was the A10G 24 GiB
bottleneck). It drives this class with gram_device='cuda' over
dynamically-sized layer windows: each window's Grams accumulate on GPU
across ALL sequences (zero per-hook CPU sync) and the caller bulk-
flushes them to CPU masters at the window boundary — see
run_streaming_calibration in scripts/palettize_qwen3_5_9b.py.

Usage (standalone smoke test):
    python calibrate_real_text.py --source wikitext2 --n-seqs 2 --seq-len 512
"""

import argparse
import hashlib
import os
from typing import Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn

# Single source of truth for split disjointness:
# calibration on train, perplexity evaluation on test. Never the same split.
CALIB_SPLIT = {"wikitext2": "train"}
EVAL_SPLIT = {"wikitext2": "test"}


def stream_sha256(token_sequences: torch.Tensor) -> str:
    """Deterministic sha256 of a token-id stream (provenance anchor).

    The hash covers dtype shape and the raw little-endian token ids, so any
    change of source, seed, sequence count or length changes the digest
    (recorded in metadata.json by the palettizer).
    """
    ids = torch.as_tensor(token_sequences, dtype=torch.long).cpu().contiguous()
    h = hashlib.sha256()
    h.update(f"{tuple(ids.shape)}|int64|".encode())
    h.update(ids.numpy().tobytes())
    return h.hexdigest()


# --------------------------------------------------------------------------- #
# Calibration data sources
# --------------------------------------------------------------------------- #

def load_calibration_sequences(tokenizer, source: str = "wikitext2",
                               n_seqs: int = 8, seq_len: int = 2048,
                               seed: int = 0
                               ) -> torch.Tensor:
    """Return a LongTensor [n_seqs, seq_len] of real-text token ids.

    Sources:
      wikitext2 : Salesforce/wikitext wikitext-2-raw-v1 train split
                  (the OmniQuant-standard calibration set: 128x2048)
      fineweb   : HuggingFaceFW/fineweb-edu sample-10BT, streaming
      file:PATH : a local plain-text file (one document per line, or free
                  text — anything readable)
    """
    texts: List[str] = []
    if source.startswith("file:"):
        with open(source[5:]) as f:
            texts = [ln.strip() for ln in f if len(ln.strip()) > 40]
        if not texts:
            raise ValueError(f"no usable lines in {source}")
    elif source == "wikitext2":
        from datasets import load_dataset
        # CALIB_SPLIT is the single source of truth (train; eval uses test)
        ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1",
                          split=CALIB_SPLIT["wikitext2"])
        # concatenate the corpus, then re-chunk (standard practice)
        blob = "\n".join(t for t in ds["text"] if len(t.strip()) > 0)
        texts = [blob]
    elif source == "fineweb":
        from datasets import load_dataset
        ds = iter(load_dataset("HuggingFaceFW/fineweb-edu",
                               name="sample-10BT", split="train",
                               streaming=True))
        while len(texts) < n_seqs:
            try:
                texts.append(next(ds)["text"])
            except StopIteration:
                break
    else:
        raise ValueError(f"unknown calibration source {source!r}")

    if source in ("wikitext2",) or source.startswith("file:"):
        # tokenize the blob(s) and cut into seq_len chunks
        ids: List[int] = []
        for t in texts:
            ids.extend(tokenizer(t, add_special_tokens=False)["input_ids"])
        n_avail = len(ids) // seq_len
        if n_avail == 0:
            raise ValueError(
                f"calibration source {source!r} yielded <{seq_len} tokens")
        take = min(n_seqs, n_avail)
        start = (len(ids) - take * seq_len) // 2  # middle of the corpus
        return torch.tensor(
            [ids[start + i * seq_len: start + (i + 1) * seq_len]
             for i in range(take)], dtype=torch.long)
    else:
        # per-document sequences (fineweb): tokenize each, truncate/pad
        seqs = []
        for t in texts[:n_seqs]:
            tok = tokenizer(t, add_special_tokens=False,
                            truncation=True, max_length=seq_len)["input_ids"]
            if len(tok) < seq_len:
                tok = tok + [tokenizer.pad_token_id or 0] * (seq_len - len(tok))
            seqs.append(tok[:seq_len])
        return torch.tensor(seqs, dtype=torch.long)


# --------------------------------------------------------------------------- #
# Running-sum capture
# --------------------------------------------------------------------------- #

class CalibrationCapture:
    """Forward passes over real text, accumulating per-tensor statistics.

    Args:
        model: the dense model (eval mode, on device).
        should_palettize_fn: callable(name, weight) -> bool — pass the
            palettize script's `should_palettize` so the hooked set matches
            exactly what will be quantized.
        retain_rows: rows of input activations kept per tensor for the
            cosine gate (stored on CPU).
        gram_keys: optional set of (layer_idx, module_name) keys (or name
            suffixes matched by `in`) for which the full Gram X^T X is
            accumulated. Ignored when full_gram is True.
        gram_device: 'cpu' (default, safe) or 'cuda'.
        track_mean: also accumulate the running input mean.
        full_gram: accumulate the full Gram X^T X for EVERY hooked tensor
            (the GPTQ assignment and the whitened-SVD residual both
            consume per-tensor Grams).
        layer_range: optional (lo, hi) half-open range of layer indices to
            hook. The palettizer drives capture layer-block by layer-block
            so only one block's Grams are resident at a time (bounded GPU
            memory at the 128x2048 calibration scale).
        extra_head_modules: optional [(name, module), ...] of TOP-LEVEL
            modules (e.g. [("lm_head", model.lm_head)]) hooked under the
            pseudo-block key ("head", name) — see the head contract below.

    Head contract (extra_head_modules): head hooks are input-side (lm_head
    sees the final-norm output), ride the same _make_hook path as layer
    tensors, and are full-gram-always: h_sum, x_sum (when track_mean), the
    retained sample AND the full Gram accumulate under ("head", name).
    Zero extra forward passes — the head module runs on every calibration
    forward anyway, so its hook just rides along; the hook closures
    persist, so head statistics accumulate across multiple run() calls
    within one capture instance (per-block layer_range scans included).
    """

    def __init__(self, model, should_palettize_fn: Callable,
                 retain_rows: int = 512, gram_keys=(), gram_device: str = "cpu",
                 track_mean: bool = True, full_gram: bool = False,
                 layer_range: Optional[Tuple[int, int]] = None,
                 extra_head_modules: Optional[List[Tuple[str, nn.Module]]] = None):
        self.model = model
        self.should_palettize = should_palettize_fn
        self.retain_rows = retain_rows
        self.gram_keys = set(gram_keys)
        self.gram_device = gram_device
        self.track_mean = track_mean
        self.full_gram = full_gram
        self.layer_range = layer_range

        self.h_sum: Dict[Tuple[Union[int, str], str], torch.Tensor] = {}  # (K,)
        self.x_sum: Dict[Tuple[Union[int, str], str], torch.Tensor] = {}  # (K,)
        self.grams: Dict[Tuple[Union[int, str], str], torch.Tensor] = {}  # (K,K)
        self.x_sample: Dict[Tuple[Union[int, str], str], torch.Tensor] = {}  # (R,K) CPU
        self.n_tokens: int = 0
        self.n_forwards: int = 0   # one per batch in run() (W5 zero-extra-forwards proof)
        self._hooks: List = []

        layers = self._find_layers(model)
        for layer_idx, layer in enumerate(layers):
            if self.layer_range is not None:
                lo, hi = self.layer_range
                if not (lo <= layer_idx < hi):
                    continue
            for name, module in layer.named_modules():
                if isinstance(module, nn.Linear):
                    full_name = f"model.layers.{layer_idx}.{name}"
                    if self.should_palettize(full_name, module.weight):
                        self._hooks.append(module.register_forward_hook(
                            self._make_hook(layer_idx, name)))

        # Head contract (W5): top-level modules (lm_head) hooked through
        # the same _make_hook, under the ("head", name) pseudo-block key,
        # force_full_gram=True — the head pass needs the full Gram (plus
        # h_sum / x_sum / the retained sample) regardless of full_gram /
        # gram_keys. The layer loop above is untouched (with the default
        # extra_head_modules=None no new hooks appear: layer path inert).
        for name, module in (extra_head_modules or ()):
            self._hooks.append(module.register_forward_hook(
                self._make_hook("head", name, force_full_gram=True)))

    @staticmethod
    def _find_layers(model):
        if hasattr(model, "model") and hasattr(model.model, "language_model"):
            return model.model.language_model.layers
        if hasattr(model, "model") and hasattr(model.model, "layers"):
            return model.model.layers
        raise AttributeError("Cannot find layers in model structure")

    def _make_hook(self, layer_idx: Union[int, str], name: str,
                   force_full_gram: bool = False):
        def hook(module, inp, out):
            x = inp[0] if isinstance(inp, tuple) else inp
            x = x.detach().float().reshape(-1, x.shape[-1])   # (T, K)
            key = (layer_idx, name)
            sq = (x * x).sum(dim=0)
            if key in self.h_sum:
                self.h_sum[key] += sq
                if self.track_mean:
                    self.x_sum[key] += x.sum(dim=0)
            else:
                self.h_sum[key] = sq
                if self.track_mean:
                    self.x_sum[key] = x.sum(dim=0)
                # retain a sample for the cosine gate (v1-compatible access)
                self.x_sample[key] = x[: self.retain_rows].cpu()
            # force_full_gram: the head pseudo-block ("head", name) is
            # full-gram-always; it short-circuits before the suffix scan
            # (which must not run for head keys).
            want_gram = (self.full_gram or force_full_gram
                         or (key in self.gram_keys
                             or any(suffix in f"layers.{layer_idx}.{name}"
                                    for suffix in self.gram_keys)))
            if want_gram:
                g = x.t() @ x
                if key in self.grams:
                    self.grams[key] += (g.to(self.gram_device)
                                        if self.gram_device == "cuda" else g.cpu())
                else:
                    self.grams[key] = (g if self.gram_device == "cuda"
                                       else g.cpu())
        return hook

    @torch.no_grad()
    def run(self, token_sequences: torch.Tensor, device: str = "cuda:0",
            batch_size: int = 1, log_every: int = 1):
        """Feed `token_sequences` [N, L] through the model in batches.

        Each batch is exactly ONE model forward pass, counted in
        self.n_forwards (the head hooks ride along — zero extra passes).
        Progress lines use the W11 homogeneous pipeline format
        ([CALIB] tag)."""
        self.model.eval()
        n = token_sequences.shape[0]
        for i in range(0, n, batch_size):
            batch = token_sequences[i: i + batch_size].to(device)
            _ = self.model(batch)
            self.n_forwards += 1
            self.n_tokens += int(batch.numel())
            if log_every and (i // batch_size) % log_every == 0:
                print(f"[CALIB] {i + batch.shape[0]}/{n} sequences "
                      f"({self.n_tokens} tokens, "
                      f"{len(self.h_sum)} tensors tracked)", flush=True)

    def feed_head(self, name, x):
        """W15 head-only bypass: accumulate the ("head", name) statistics
        from a provided input tensor WITHOUT any module forward.

        The palettizer's --only-heads path never runs the transformer
        layers; it feeds the lm_head's input statistics directly from
        the embed_tokens output. This method invokes the SAME closure
        the extra_head_modules hook installs (bit-identical math:
        h_sum, x_sum, the retained first-batch sample, the full Gram
        under ("head", name)), so the bypass needs neither the hook nor
        the lm_head forward — the (rows, vocab) logits GEMM is never
        materialized. Use EITHER this OR the extra_head_modules hook
        for a given head name, never both (statistics would
        double-accumulate); n_forwards is NOT incremented (a feed is
        not a forward — the caller records its own provenance).
        """
        self._make_hook("head", name, force_full_gram=True)(None, (x,), None)

    def remove_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks = []

    # -- accessors ----------------------------------------------------------- #
    def hess_diag(self, key: Tuple[Union[int, str], str]) -> Optional[torch.Tensor]:
        return self.h_sum.get(key)

    def input_mean(self, key: Tuple[Union[int, str], str]) -> Optional[torch.Tensor]:
        if key in self.x_sum and self.n_tokens:
            return self.x_sum[key] / self.n_tokens
        return None

    def sample(self, key: Tuple[Union[int, str], str]) -> Optional[torch.Tensor]:
        """v1-compatible activation sample: real-text rows (n, K) on CPU."""
        return self.x_sample.get(key)

    def summary(self) -> Dict:
        return {"tensors": len(self.h_sum), "tokens": self.n_tokens,
                "gram_tensors": len(self.grams),
                "retained_rows": self.retain_rows,
                "full_gram": self.full_gram,
                "layer_range": (list(self.layer_range)
                                if self.layer_range else None)}


# --------------------------------------------------------------------------- #
# Standalone smoke test
# --------------------------------------------------------------------------- #

def _smoke_test(args):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    seqs = load_calibration_sequences(tok, args.source, args.n_seqs,
                                       args.seq_len)
    print(f"token sequences: {tuple(seqs.shape)}")
    if not args.forward:
        return
    model = AutoModelForCausalLM.from_pretrained(
        args.model, trust_remote_code=True, torch_dtype=torch.float16,
        low_cpu_mem_usage=True).to(args.device)
    model.eval()

    # one implementation: the palettizer's own predicate, imported (a
    # local mirror would drift from the idx4 N%128/K%64 contract)
    import palettize_qwen3_5_9b as _palettizer
    cap = CalibrationCapture(model, _palettizer.should_palettize,
                             retain_rows=args.retain_rows)
    cap.run(seqs, device=args.device, batch_size=args.batch_size)
    cap.remove_hooks()
    print("summary:", cap.summary())
    key0 = next(iter(cap.h_sum))
    h = cap.hess_diag(key0)
    print(f"example key {key0}: h_k shape {tuple(h.shape)}, "
          f"sum={h.sum().item():.3e}; sample "
          f"{tuple(cap.sample(key0).shape)}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3.5-9B")
    p.add_argument("--source", default="wikitext2")
    p.add_argument("--n-seqs", type=int, default=2)
    p.add_argument("--seq-len", type=int, default=512)
    p.add_argument("--retain-rows", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--forward", action="store_true",
                   help="run a forward pass (requires the model)")
    _smoke_test(p.parse_args())
