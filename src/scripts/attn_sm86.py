"""scripts/attn_sm86.py — SM86 Triton flash attention: forward, backward dQ/dK/dV, and the autograd Function.

Full-attention student layers get a Triton flash-attention kernel at the
model geometry, where torch SDPA dispatches to the math backend and
materializes (B, H, S, S) fp32 scores. The teacher stays on torch SDPA.
Design of record: reports/design_attn_sm86.md.

Public surface:
  - kernel_available() -> (bool, reason): the guard callers must check
    before the kernel path. triton is imported lazily, never at module
    top level, so this module imports on any box.
  - sm86_attention_forward(q, k, v, sm_scale, ...) -> (out, lse): the
    kernel wrapper. Loud asserts with shapes in every message; never a
    silent no-op — an unavailable kernel raises.
  - sm86_attention_backward_dq(...) -> dq; sm86_attention_backward_dkv(
    ...) -> (dk, dv): the backward wrappers (FA2-style recompute from
    LSE; delta = rowsum(dO o O) precomputed on the host; the dK/dV
    q-head loop accumulates in-program, no atomics).
  - Sm86AttentionFn: the autograd Function gluing the three kernels
    (FLUTE_ATTN_DEBUG_NAN=1 turns a NaN grad into a RuntimeError).
  - flute_sm86_attention(q, k, v, sm_scale): the public differentiable
    entry — kernel path on CUDA or under TRITON_INTERPRET=1, the
    reference autograd path on CPU, S == 1 short-circuits to the
    reference math; FLUTE_ATTN_KERNEL in ("", "0") is the loud escape
    hatch forcing the reference path.
  - reference_attention_forward(q, k, v, sm_scale): eager dense
    reference — the parity oracle and the CPU fallback.

Contract: q (B, H, S, 256), k/v (B, H_kv, S, 256), fp16 or bf16,
contiguous (the wrappers coerce); GQA kv_head = q_head // (H // H_kv)
with no repeat_kv materialization; causal (S == 1 is full attention);
dropout p=0 asserted upstream. LSE (B, H, S) fp32 natural log is always
produced — the backward's recompute key.

Tiling (sm_86, no TMA/wgmma): forward BLOCK 64/64, backward 32/32,
D_CHUNK=128, num_warps=4, num_stages=1; grid (ceil(S/BLOCK_M), H, B).
Online softmax (FA1-style) with fp32 running m/l; tl.dot on fp16
operands with fp32 accumulation. Interpreter testing: TRITON_INTERPRET=1
set before import relaxes the CUDA asserts to CPU tensors (a test-only
seam; the interpreter runs tl.dot via numpy, so parity is semantic).
"""
import os

import torch

# Test-only seam: TRITON_INTERPRET=1 must be set in the environment before
# this module is imported (and before the first kernel launch, so
# @triton.jit decorates in interpreter mode). It relaxes the CUDA asserts
# below to CPU tensors for parity tests on this CPU box.
_INTERPRET_MODE = os.environ.get("TRITON_INTERPRET") == "1"

# Lazily decorated @triton.jit kernels (triton must never load at module
# import — see kernel_available()).
_FWD_KERNEL = None
_DQ_KERNEL = None
_DKV_KERNEL = None


def kernel_available():
    """(bool, reason) guard for the SM86 Triton kernel path.

    Imports triton lazily (never at module import). Callers gate the
    kernel path on this and fall back to reference_attention_forward /
    SDPA when False. Note: TRITON_INTERPRET=1 interpreter mode is a
    test seam and is deliberately NOT reported here — this function
    reports the real CUDA path.
    """
    try:
        import triton  # noqa: F401 — importability is the check
    except ImportError as e:
        return (False, f"triton import failed: {e}")
    if not torch.cuda.is_available():
        return (False, "triton importable but no CUDA device")
    return (True, "triton importable and CUDA device available")



def _check_qkv(ctx, q, k, v, out=None, grad_out=None, lse=None):
    """Shared shape/dtype contract for the attention wrappers.

    Validates q/k/v — and out/grad_out/lse when given — against the
    module contract; returns (B, H, S, D, H_kv). Every message names
    the calling function (`ctx`) and the offending shapes. out and
    grad_out are validated together (both or neither).
    """
    assert q.dim() == 4, (
        f"{ctx}: q must be 4-D (B, H, S, D), got "
        f"{q.dim()} dims, shape {tuple(q.shape)}")
    assert k.dim() == 4, (
        f"{ctx}: k must be 4-D (B, H_kv, S, D), got "
        f"{k.dim()} dims, shape {tuple(k.shape)}")
    assert v.dim() == 4, (
        f"{ctx}: v must be 4-D (B, H_kv, S, D), got "
        f"{v.dim()} dims, shape {tuple(v.shape)}")
    if out is not None:
        assert out.dim() == 4, (
            f"{ctx}: out must be 4-D (B, H, S, D), "
            f"got {out.dim()} dims, shape {tuple(out.shape)}")
    if grad_out is not None:
        assert grad_out.dim() == 4, (
            f"{ctx}: grad_out must be 4-D (B, H, S, D), "
            f"got {grad_out.dim()} dims, shape {tuple(grad_out.shape)}")
    if lse is not None:
        assert torch.is_tensor(lse), (
            f"{ctx}: lse must be a (B, H, S) fp32 tensor "
            f"(the forward's recompute key; the need_lse=False shortcut "
            f"output cannot backpropagate), got {type(lse).__name__}")
        assert lse.dim() == 3, (
            f"{ctx}: lse must be 3-D (B, H, S), got "
            f"{lse.dim()} dims, shape {tuple(lse.shape)}")

    B, H, S, D = q.shape
    B_k, H_kv, S_k, D_k = k.shape
    B_v, H_kv_v, S_v, D_v = v.shape

    assert (B_k == B and B_v == B and S_k == S and S_v == S
            and D_k == D and D_v == D), (
        f"{ctx}: q/k/v must share B, S, D; got "
        f"q {tuple(q.shape)}, k {tuple(k.shape)}, v {tuple(v.shape)}")
    D_o = D_g = None
    if out is not None:
        B_o, H_o, S_o, D_o = out.shape
        assert (B_o == B and H_o == H and S_o == S and D_o == D), (
            f"{ctx}: out must match q's (B, H, S, D); "
            f"got out {tuple(out.shape)} vs q {tuple(q.shape)}")
    if grad_out is not None:
        B_g, H_g, S_g, D_g = grad_out.shape
        assert (B_g == B and H_g == H and S_g == S and D_g == D), (
            f"{ctx}: grad_out must match q's "
            f"(B, H, S, D); got grad_out {tuple(grad_out.shape)} vs "
            f"q {tuple(q.shape)}")
    if lse is not None:
        B_l, H_l, S_l = lse.shape
        assert (B_l == B and H_l == H and S_l == S), (
            f"{ctx}: lse must be (B, H, S) matching q; "
            f"got lse {tuple(lse.shape)} vs q {tuple(q.shape)}")

    d_parts = [f"q D={D}", f"k D={D_k}", f"v D={D_v}"]
    if out is not None:
        d_parts.append(f"out D={D_o}")
    if grad_out is not None:
        d_parts.append(f"grad_out D={D_g}")
    assert D == 256, (
        f"{ctx}: design geometry is D=256 "
        f"(reports/design_attn_sm86.md); got " + ", ".join(d_parts))
    assert H % H_kv == 0 and H_kv_v == H_kv, (
        f"{ctx}: GQA requires H % H_kv == 0 and k/v "
        f"to share H_kv; got H={H}, H_kv={H_kv} (k), H_kv={H_kv_v} (v); "
        f"shapes q {tuple(q.shape)}, k {tuple(k.shape)}, "
        f"v {tuple(v.shape)}")
    assert S >= 1, (
        f"{ctx}: S must be >= 1, got S={S} "
        f"(q {tuple(q.shape)}, k {tuple(k.shape)}, v {tuple(v.shape)})")
    assert q.dtype in (torch.float16, torch.bfloat16), (
        f"{ctx}: q must be fp16 or bf16, got "
        f"{q.dtype} (q {tuple(q.shape)}, k {tuple(k.shape)}, "
        f"v {tuple(v.shape)})")
    assert k.dtype == q.dtype and v.dtype == q.dtype, (
        f"{ctx}: q/k/v must share dtype, got "
        f"q={q.dtype}, k={k.dtype}, v={v.dtype} "
        f"(q {tuple(q.shape)}, k {tuple(k.shape)}, v {tuple(v.shape)})")
    if out is not None and grad_out is not None:
        assert out.dtype == q.dtype and grad_out.dtype == q.dtype, (
            f"{ctx}: out/grad_out must share q's dtype, "
            f"got q={q.dtype}, out={out.dtype}, grad_out={grad_out.dtype} "
            f"(q {tuple(q.shape)}, out {tuple(out.shape)}, "
            f"grad_out {tuple(grad_out.shape)})")
    if lse is not None:
        assert lse.dtype == torch.float32, (
            f"{ctx}: lse must be fp32 (the forward "
            f"produces (B, H, S) fp32 natural log), got {lse.dtype} "
            f"(lse {tuple(lse.shape)}, q {tuple(q.shape)})")
    return B, H, S, D, H_kv


def _check_tiling(ctx, D, block_m, block_n, d_chunk):
    """Kernel tiling contract: exactly two D-chunks; power-of-two
    block sizes (tl.arange constraint)."""
    assert 2 * d_chunk == D, (
        f"{ctx}: the kernel processes D in exactly "
        f"two D-chunks, so 2*d_chunk must equal D={D}; got "
        f"d_chunk={d_chunk}")
    for name, val in (("block_m", block_m), ("block_n", block_n),
                      ("d_chunk", d_chunk)):
        assert val > 0 and (val & (val - 1)) == 0, (
            f"{ctx}: {name} must be a power of two "
            f"(tl.arange constraint), got {name}={val}")


def _get_fwd_kernel():
    """Lazily import triton and define/decorate _attn_fwd_kernel.

    The module must import without triton, so the kernel is defined here
    (first use) and cached. TRITON_INTERPRET must be set before the first
    call so @triton.jit picks interpreter mode on CPU test boxes.
    """
    global _FWD_KERNEL
    if _FWD_KERNEL is not None:
        return _FWD_KERNEL
    try:
        import triton
        import triton.language as tl
    except ImportError as e:
        raise ImportError(
            f"attn_sm86: _attn_fwd_kernel needs triton but the import "
            f"failed ({e}); kernel_available() would have returned False — "
            f"guard the kernel path with it and fall back to "
            f"reference_attention_forward") from e

    @triton.jit
    def _attn_fwd_kernel(
        q_ptr, k_ptr, v_ptr, out_ptr, lse_ptr, sm_scale,
        stride_qb, stride_qh, stride_qm,
        stride_kh, stride_kn,
        stride_vh, stride_vn,
        stride_ob, stride_oh, stride_om,
        stride_lb, stride_lh,
        B, H, H_KV, S,
        D: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
        D_CHUNK: tl.constexpr,
    ):
        # One program = one (M-tile, q-head, batch): grid
        # (ceil(S/BLOCK_M), H, B). GQA: q head pid_h maps to kv head
        # pid_h // (H // H_KV). K/V batch is folded into the head stride
        # (flat kv head index = pid_b * H_KV + kv_h), so k_ptr/v_ptr
        # address (B*H_KV, S, D) views — no repeat_kv materialization.
        pid_m = tl.program_id(0)
        pid_h = tl.program_id(1)
        pid_b = tl.program_id(2)
        kv_h = pid_h // (H // H_KV)

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_d1 = tl.arange(0, D_CHUNK)
        offs_d2 = D_CHUNK + tl.arange(0, D_CHUNK)  # D == 2*D_CHUNK (asserted)
        m_mask = offs_m < S

        # Q tile, loaded ONCE per program in two 128-wide D-chunks
        # (BLOCK_M x D_CHUNK, operand dtype — fp16/bf16 — held for tl.dot).
        q_base = (q_ptr + pid_b * stride_qb + pid_h * stride_qh
                  + offs_m[:, None] * stride_qm)
        q1 = tl.load(q_base + offs_d1[None, :],
                     mask=m_mask[:, None], other=0.0)
        q2 = tl.load(q_base + offs_d2[None, :],
                     mask=m_mask[:, None], other=0.0)

        # Online softmax (FA1-style): fp32 running max m and sum l.
        m_i = tl.full([BLOCK_M], float("-inf"), tl.float32)
        l_i = tl.zeros([BLOCK_M], tl.float32)
        acc1 = tl.zeros([BLOCK_M, D_CHUNK], tl.float32)
        acc2 = tl.zeros([BLOCK_M, D_CHUNK], tl.float32)

        # Causal N loop: Q row i attends K rows 0..i, so this M-tile only
        # schedules N blocks up to its own diagonal — hi in KV row units
        # (min with S handles non-multiple S; fully-masked blocks are
        # never scheduled).
        hi = tl.minimum((pid_m + 1) * BLOCK_M, S)
        for start_n in range(0, hi, BLOCK_N):
            offs_n = start_n + tl.arange(0, BLOCK_N)
            n_mask = offs_n < S
            kv_flat = pid_b * H_KV + kv_h
            k_base = k_ptr + kv_flat * stride_kh + offs_n[:, None] * stride_kn
            k1 = tl.load(k_base + offs_d1[None, :],
                         mask=n_mask[:, None], other=0.0)
            k2 = tl.load(k_base + offs_d2[None, :],
                         mask=n_mask[:, None], other=0.0)

            # scores = q @ k^T over both D-chunks; tl.dot accumulates fp32.
            scores = tl.dot(q1, tl.trans(k1)) + tl.dot(q2, tl.trans(k2))
            scores = scores * sm_scale
            # Causal + S-boundary mask (exp, not exp2 — interpreter-safe).
            scores = tl.where(
                (offs_m[:, None] >= offs_n[None, :]) & (offs_n[None, :] < S),
                scores, float("-inf"))

            m_new = tl.maximum(m_i, tl.max(scores, 1))
            alpha = tl.exp(m_i - m_new)
            p = tl.exp(scores - m_new[:, None])
            l_i = l_i * alpha + tl.sum(p, 1)

            v_base = v_ptr + kv_flat * stride_vh + offs_n[:, None] * stride_vn
            v1 = tl.load(v_base + offs_d1[None, :],
                         mask=n_mask[:, None], other=0.0)
            v2 = tl.load(v_base + offs_d2[None, :],
                         mask=n_mask[:, None], other=0.0)
            p_h = p.to(v1.dtype)  # fp32 probabilities -> operand dtype
            acc1 = acc1 * alpha[:, None] + tl.dot(p_h, v1)
            acc2 = acc2 * alpha[:, None] + tl.dot(p_h, v2)
            m_i = m_new

        # Epilogue: out = acc / l in two D-chunks (operand dtype, masked
        # store); LSE = m + log(l) fp32 (natural log, masked store) —
        # the backward's recompute key.
        o_base = (out_ptr + pid_b * stride_ob + pid_h * stride_oh
                  + offs_m[:, None] * stride_om)
        out1 = acc1 / l_i[:, None]
        out2 = acc2 / l_i[:, None]
        tl.store(o_base + offs_d1[None, :], out1.to(q1.dtype),
                 mask=m_mask[:, None])
        tl.store(o_base + offs_d2[None, :], out2.to(q2.dtype),
                 mask=m_mask[:, None])
        lse = m_i + tl.log(l_i)
        tl.store(lse_ptr + pid_b * stride_lb + pid_h * stride_lh + offs_m,
                 lse, mask=m_mask)

    _FWD_KERNEL = _attn_fwd_kernel
    return _FWD_KERNEL


def sm86_attention_forward(q, k, v, sm_scale, need_lse=True,
                           block_m=64, block_n=64, d_chunk=128):
    """Forward flash attention (Triton kernel, SM86 design geometry).

    q (B, H, S, 256), k/v (B, H_kv, S, 256), fp16/bf16; causal; GQA with
    kv_head = q_head // (H // H_kv). Returns (out, lse): out (B, H, S, D)
    in q.dtype and lse (B, H, S) fp32 natural log (None when
    need_lse=False — the kernel still computes it, it is the backward's
    recompute key). Tiling defaults: BLOCK_M=64, BLOCK_N=64, D_CHUNK=128,
    num_warps=4, num_stages=1 (launch values; overridable for interpreter
    tests). Loud asserts, never a silent
    no-op: the CUDA path raises if the kernel is unavailable.
    """
    B, H, S, D, H_kv = _check_qkv("sm86_attention_forward", q, k, v)

    if _INTERPRET_MODE:
        # Test-only seam (see module docstring): interpreter mode runs the
        # kernel on CPU — relax the CUDA assert to CPU tensors.
        assert not q.is_cuda and not k.is_cuda and not v.is_cuda, (
            f"sm86_attention_forward: TRITON_INTERPRET=1 test seam expects "
            f"CPU tensors, got devices q={q.device}, k={k.device}, "
            f"v={v.device} (shapes q {tuple(q.shape)}, "
            f"k {tuple(k.shape)}, v {tuple(v.shape)})")
    else:
        assert q.is_cuda and k.is_cuda and v.is_cuda, (
            f"sm86_attention_forward: the Triton kernel is the CUDA path; "
            f"got devices q={q.device}, k={k.device}, v={v.device} "
            f"(shapes q {tuple(q.shape)}, k {tuple(k.shape)}, "
            f"v {tuple(v.shape)}). CPU callers must guard with "
            f"kernel_available() and use reference_attention_forward.")
        ok, reason = kernel_available()
        if not ok:
            raise RuntimeError(
                f"sm86_attention_forward: SM86 Triton kernel unavailable "
                f"({reason}); shapes q {tuple(q.shape)}, "
                f"k {tuple(k.shape)}, v {tuple(v.shape)}. Guard the kernel "
                f"path with kernel_available() and fall back to "
                f"reference_attention_forward — never a silent no-op.")

    # Tiling contract with the kernel: exactly two D-chunks, powers
    # of two for tl.arange.
    _check_tiling("sm86_attention_forward", D, block_m, block_n, d_chunk)

    # Contiguous coercion (kernel strides assume unit last-dim stride).
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    # K/V batch folded into the head dim: (B*H_kv, S, D) — the kernel's
    # flat kv head index pid_b * H_KV + kv_h addresses these.
    k_flat = k.view(B * H_kv, S, D)
    v_flat = v.view(B * H_kv, S, D)

    out = torch.empty((B, H, S, D), dtype=q.dtype, device=q.device)
    lse = torch.empty((B, H, S), dtype=torch.float32, device=q.device)

    kernel = _get_fwd_kernel()
    grid = ((S + block_m - 1) // block_m, H, B)  # (ceil(S/BLOCK_M), H, B)
    # Strides in ELEMENTS (torch strides are element strides already).
    kernel[grid](
        q, k_flat, v_flat, out, lse,
        sm_scale,
        q.stride(0), q.stride(1), q.stride(2),
        k_flat.stride(0), k_flat.stride(1),
        v_flat.stride(0), v_flat.stride(1),
        out.stride(0), out.stride(1), out.stride(2),
        lse.stride(0), lse.stride(1),
        B, H, H_kv, S,
        D=D, BLOCK_M=block_m, BLOCK_N=block_n, D_CHUNK=d_chunk,
        num_warps=4, num_stages=1,  # Reduced from 2 for A10G shared memory
    )
    return out, (lse if need_lse else None)


def _get_dq_kernel():
    """Lazily import triton and define/decorate _attn_dq_kernel.

    Mirrors _get_fwd_kernel(): the module must import without triton, so
    the kernel is defined here (first use) and cached. TRITON_INTERPRET
    must be set before the first call so @triton.jit picks interpreter
    mode on CPU test boxes.
    """
    global _DQ_KERNEL
    if _DQ_KERNEL is not None:
        return _DQ_KERNEL
    try:
        import triton
        import triton.language as tl
    except ImportError as e:
        raise ImportError(
            f"attn_sm86: _attn_dq_kernel needs triton but the import "
            f"failed ({e}); kernel_available() would have returned False — "
            f"guard the kernel path with it and differentiate "
            f"reference_attention_forward instead") from e

    @triton.jit
    def _attn_dq_kernel(
        q_ptr, k_ptr, v_ptr, do_ptr, dq_ptr, lse_ptr, delta_ptr, sm_scale,
        stride_qb, stride_qh, stride_qm,
        stride_kh, stride_kn,
        stride_vh, stride_vn,
        stride_dob, stride_doh, stride_dom,
        stride_dqb, stride_dqh, stride_dqm,
        stride_lb, stride_lh,
        stride_db, stride_dh,
        B, H, H_KV, S,
        D: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
        D_CHUNK: tl.constexpr,
    ):
        # One program = one (M-tile, q-head, batch): grid
        # (ceil(S/BLOCK_M), H, B) — the SAME tiling as the forward. GQA:
        # q head pid_h maps to kv head pid_h // (H // H_KV); K/V batch is
        # folded into the head stride (flat kv head index =
        # pid_b * H_KV + kv_h), so k_ptr/v_ptr address (B*H_KV, S, D)
        # views — no repeat_kv materialization.
        pid_m = tl.program_id(0)
        pid_h = tl.program_id(1)
        pid_b = tl.program_id(2)
        kv_h = pid_h // (H // H_KV)

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_d1 = tl.arange(0, D_CHUNK)
        offs_d2 = D_CHUNK + tl.arange(0, D_CHUNK)  # D == 2*D_CHUNK (asserted)
        m_mask = offs_m < S

        # Q and dO tiles, loaded ONCE per program in two 128-wide D-chunks
        # (BLOCK_M x D_CHUNK, operand dtype — fp16/bf16 — held for tl.dot).
        # NOTE: the forward's out is deliberately NOT a kernel argument:
        # the only quantity it contributes, delta = rowsum(dO ∘ O), is
        # precomputed on the host by the wrapper (simpler and exact) and
        # passed in via delta_ptr.
        q_base = (q_ptr + pid_b * stride_qb + pid_h * stride_qh
                  + offs_m[:, None] * stride_qm)
        q1 = tl.load(q_base + offs_d1[None, :],
                     mask=m_mask[:, None], other=0.0)
        q2 = tl.load(q_base + offs_d2[None, :],
                     mask=m_mask[:, None], other=0.0)
        do_base = (do_ptr + pid_b * stride_dob + pid_h * stride_doh
                   + offs_m[:, None] * stride_dom)
        do1 = tl.load(do_base + offs_d1[None, :],
                      mask=m_mask[:, None], other=0.0)
        do2 = tl.load(do_base + offs_d2[None, :],
                      mask=m_mask[:, None], other=0.0)

        # LSE tile (BLOCK_M fp32 — the recompute key) and the host-side
        # delta = rowsum(dO ∘ O) tile (BLOCK_M fp32).
        lse = tl.load(lse_ptr + pid_b * stride_lb + pid_h * stride_lh
                      + offs_m, mask=m_mask, other=0.0)
        delta = tl.load(delta_ptr + pid_b * stride_db + pid_h * stride_dh
                        + offs_m, mask=m_mask, other=0.0)

        # Running dQ accumulator: two D-chunks of (BLOCK_M, D_CHUNK) fp32.
        # fp32 accumulation, NO atomics — each (q_head, M-block) program
        # owns its dQ tile exclusively.
        dq1 = tl.zeros([BLOCK_M, D_CHUNK], tl.float32)
        dq2 = tl.zeros([BLOCK_M, D_CHUNK], tl.float32)

        # Causal N loop: Q row i attends K rows 0..i, so this M-tile only
        # schedules N blocks up to its own diagonal — the SAME bound as
        # the forward (hi in KV row units; min with S handles non-multiple
        # S; fully-masked blocks are never scheduled).
        hi = tl.minimum((pid_m + 1) * BLOCK_M, S)
        for start_n in range(0, hi, BLOCK_N):
            offs_n = start_n + tl.arange(0, BLOCK_N)
            n_mask = offs_n < S
            kv_flat = pid_b * H_KV + kv_h
            k_base = k_ptr + kv_flat * stride_kh + offs_n[:, None] * stride_kn
            k1 = tl.load(k_base + offs_d1[None, :],
                         mask=n_mask[:, None], other=0.0)
            k2 = tl.load(k_base + offs_d2[None, :],
                         mask=n_mask[:, None], other=0.0)
            v_base = v_ptr + kv_flat * stride_vh + offs_n[:, None] * stride_vn
            v1 = tl.load(v_base + offs_d1[None, :],
                         mask=n_mask[:, None], other=0.0)
            v2 = tl.load(v_base + offs_d2[None, :],
                         mask=n_mask[:, None], other=0.0)

            # FA2-style recompute (no saved probabilities): scores =
            # q @ k^T over both D-chunks (fp32 accumulate) * sm_scale,
            # causal + S-boundary mask — exactly the forward's block.
            scores = tl.dot(q1, tl.trans(k1)) + tl.dot(q2, tl.trans(k2))
            scores = scores * sm_scale
            scores = tl.where(
                (offs_m[:, None] >= offs_n[None, :]) & (offs_n[None, :] < S),
                scores, float("-inf"))

            # P = exp(scores − lse) (fp32, natural exp — interpreter-
            # safe): exact softmax probabilities, denominator in LSE.
            p = tl.exp(scores - lse[:, None])
            # dP = dO @ V^T over both D-chunks (fp32 accumulate)...
            dp = tl.dot(do1, tl.trans(v1)) + tl.dot(do2, tl.trans(v2))
            # ...dS = P ∘ (dP − delta) (fp32)...
            ds = p * (dp - delta[:, None])
            # ...and dQ += (sm_scale·dS) @ K in two D-chunks. sm_scale
            # rides on dS (chain rule: scores = sm_scale·q·k^T); dS is
            # cast to K's operand dtype for tl.dot — the design's
            # fp16-operands / fp32-accumulate contract.
            ds_h = (ds * sm_scale).to(k1.dtype)
            dq1 = dq1 + tl.dot(ds_h, k1)
            dq2 = dq2 + tl.dot(ds_h, k2)

        # Epilogue: store dQ in two D-chunks (cast to q's dtype, masked).
        dq_base = (dq_ptr + pid_b * stride_dqb + pid_h * stride_dqh
                   + offs_m[:, None] * stride_dqm)
        tl.store(dq_base + offs_d1[None, :], dq1.to(q1.dtype),
                 mask=m_mask[:, None])
        tl.store(dq_base + offs_d2[None, :], dq2.to(q2.dtype),
                 mask=m_mask[:, None])

    _DQ_KERNEL = _attn_dq_kernel
    return _DQ_KERNEL


def sm86_attention_backward_dq(q, k, v, out, lse, grad_out, sm_scale,
                                block_m=32, block_n=32, d_chunk=128):
    """dQ backward (Triton kernel, FA2-style LSE recompute).

    Computes dq = dL/dq from the forward's out and lse: each (q_head,
    M-block) program recomputes P = exp(q·k^T·sm_scale − lse) over the
    causal N range, forms dS = P ∘ (dP − delta) with dP = dO @ V^T and
    delta = rowsum(dO ∘ O) — precomputed HERE on the host as (B, H, S)
    fp32 (design note: simpler and exact, which is also why out is not a
    kernel argument) — and accumulates dQ = (sm_scale · dS) @ K in two
    D-chunks (fp32 accumulation, no atomics; program-private tiles).

    q (B, H, S, 256), k/v (B, H_kv, S, 256), out and grad_out (B, H, S,
    256), all fp16/bf16, contiguous (coerced); lse (B, H, S) fp32
    natural log (sm86_attention_forward / reference_attention_forward
    output). Returns dq (B, H, S, D) in q.dtype. Tiling defaults:
    BLOCK_M=32, BLOCK_N=32, D_CHUNK=128, num_warps=4, num_stages=1
    (launch values; overridable for interpreter tests). Loud
    asserts, never a silent no-op: the CUDA path raises if the kernel
    is unavailable.
    """
    B, H, S, D, H_kv = _check_qkv("sm86_attention_backward_dq", q, k, v,
                                  out=out, grad_out=grad_out, lse=lse)

    if _INTERPRET_MODE:
        # Test-only seam (see module docstring): interpreter mode runs the
        # kernel on CPU — relax the CUDA assert to CPU tensors.
        assert (not q.is_cuda and not k.is_cuda and not v.is_cuda
                and not out.is_cuda and not lse.is_cuda
                and not grad_out.is_cuda), (
            f"sm86_attention_backward_dq: TRITON_INTERPRET=1 test seam "
            f"expects CPU tensors, got devices q={q.device}, "
            f"k={k.device}, v={v.device}, out={out.device}, "
            f"lse={lse.device}, grad_out={grad_out.device}")
    else:
        assert (q.is_cuda and k.is_cuda and v.is_cuda and out.is_cuda
                and lse.is_cuda and grad_out.is_cuda), (
            f"sm86_attention_backward_dq: the Triton kernel is the CUDA "
            f"path; got devices q={q.device}, k={k.device}, "
            f"v={v.device}, out={out.device}, lse={lse.device}, "
            f"grad_out={grad_out.device}. CPU callers must guard with "
            f"kernel_available() and differentiate "
            f"reference_attention_forward instead.")
        ok, reason = kernel_available()
        if not ok:
            raise RuntimeError(
                f"sm86_attention_backward_dq: SM86 Triton kernel "
                f"unavailable ({reason}); shapes q {tuple(q.shape)}, "
                f"k {tuple(k.shape)}, v {tuple(v.shape)}. Guard the "
                f"kernel path with kernel_available() — never a silent "
                f"no-op.")

    _check_tiling("sm86_attention_backward_dq", D, block_m, block_n, d_chunk)

    # Contiguous coercion (kernel strides assume unit last-dim stride).
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    out = out.contiguous()
    lse = lse.contiguous()
    grad_out = grad_out.contiguous()
    # K/V batch folded into the head dim: (B*H_kv, S, D) — the kernel's
    # flat kv head index pid_b * H_KV + kv_h addresses these.
    k_flat = k.view(B * H_kv, S, D)
    v_flat = v.view(B * H_kv, S, D)

    # delta = rowsum(dO ∘ O), (B, H, S) fp32, precomputed on the host
    # (design note: simpler and exact) — the dS = P ∘ (dP − delta)
    # correction term.
    delta = (grad_out.float() * out.float()).sum(-1).contiguous()

    dq = torch.empty((B, H, S, D), dtype=q.dtype, device=q.device)

    kernel = _get_dq_kernel()
    grid = ((S + block_m - 1) // block_m, H, B)  # (ceil(S/BLOCK_M), H, B)
    # Strides in ELEMENTS (torch strides are element strides already).
    kernel[grid](
        q, k_flat, v_flat, grad_out, dq, lse, delta,
        sm_scale,
        q.stride(0), q.stride(1), q.stride(2),
        k_flat.stride(0), k_flat.stride(1),
        v_flat.stride(0), v_flat.stride(1),
        grad_out.stride(0), grad_out.stride(1), grad_out.stride(2),
        dq.stride(0), dq.stride(1), dq.stride(2),
        lse.stride(0), lse.stride(1),
        delta.stride(0), delta.stride(1),
        B, H, H_kv, S,
        D=D, BLOCK_M=block_m, BLOCK_N=block_n, D_CHUNK=d_chunk,
        num_warps=4, num_stages=1,  # Reduced from 2 for A10G shared memory
    )
    return dq


def _get_dkv_kernel():
    """Lazily import triton and define/decorate _attn_dkv_kernel.

    Mirrors _get_fwd_kernel()/_get_dq_kernel(): the module must import
    without triton, so the kernel is defined here (first use) and cached.
    TRITON_INTERPRET must be set before the first call so @triton.jit
    picks interpreter mode on CPU test boxes.
    """
    global _DKV_KERNEL
    if _DKV_KERNEL is not None:
        return _DKV_KERNEL
    try:
        import triton
        import triton.language as tl
    except ImportError as e:
        raise ImportError(
            f"attn_sm86: _attn_dkv_kernel needs triton but the import "
            f"failed ({e}); kernel_available() would have returned False — "
            f"guard the kernel path with it and differentiate "
            f"reference_attention_forward instead") from e

    @triton.jit
    def _attn_dkv_kernel(
        q_ptr, k_ptr, v_ptr, do_ptr, dk_ptr, dv_ptr, lse_ptr, delta_ptr,
        sm_scale,
        stride_qb, stride_qh, stride_qm,
        stride_kh, stride_kn,
        stride_vh, stride_vn,
        stride_dob, stride_doh, stride_dom,
        stride_dkh, stride_dkn,
        stride_dvh, stride_dvn,
        stride_lb, stride_lh,
        stride_db, stride_dh,
        B, H, H_KV, S,
        D: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
        D_CHUNK: tl.constexpr, GROUP: tl.constexpr,
    ):
        # One program = one (N-tile, kv-head, batch): grid
        # (ceil(S/BLOCK_N), H_KV, B) — the N-transposed tiling of the
        # forward/dQ kernels. This program OWNS one N-block of K/V rows
        # for one kv head exclusively, so dK/dV accumulate in registers
        # across the q-head × M-block loops — GQA-aware, NO atomics
        # (the design note's in-program accumulation). K/V batch is
        # folded into the head stride (flat kv head index =
        # pid_b * H_KV + pid_h), so k_ptr/v_ptr/dk_ptr/dv_ptr address
        # (B*H_KV, S, D) views — no repeat_kv materialization.
        pid_n = tl.program_id(0)
        pid_h = tl.program_id(1)  # kv head (NOT a q head — see GROUP loop)
        pid_b = tl.program_id(2)
        kv_h = pid_h
        kv_flat = pid_b * H_KV + kv_h

        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_d1 = tl.arange(0, D_CHUNK)
        offs_d2 = D_CHUNK + tl.arange(0, D_CHUNK)  # D == 2*D_CHUNK (asserted)
        n_mask = offs_n < S

        # K/V tiles, loaded ONCE per program in two 128-wide D-chunks
        # (BLOCK_N x D_CHUNK, operand dtype — fp16/bf16 — held for
        # tl.dot): they are FIXED for this program; every (q head,
        # M-block) pair below reuses them from registers.
        k_base = k_ptr + kv_flat * stride_kh + offs_n[:, None] * stride_kn
        k1 = tl.load(k_base + offs_d1[None, :],
                     mask=n_mask[:, None], other=0.0)
        k2 = tl.load(k_base + offs_d2[None, :],
                     mask=n_mask[:, None], other=0.0)
        v_base = v_ptr + kv_flat * stride_vh + offs_n[:, None] * stride_vn
        v1 = tl.load(v_base + offs_d1[None, :],
                     mask=n_mask[:, None], other=0.0)
        v2 = tl.load(v_base + offs_d2[None, :],
                     mask=n_mask[:, None], other=0.0)

        # dK/dV accumulators: two D-chunks of (BLOCK_N, D_CHUNK) fp32
        # each (fp32 accumulation, no atomics — see the program comment).
        dk1 = tl.zeros([BLOCK_N, D_CHUNK], tl.float32)
        dk2 = tl.zeros([BLOCK_N, D_CHUNK], tl.float32)
        dv1 = tl.zeros([BLOCK_N, D_CHUNK], tl.float32)
        dv2 = tl.zeros([BLOCK_N, D_CHUNK], tl.float32)

        # GQA: loop the GROUP = H // H_KV q heads sharing this kv head
        # (q head kv_h*GROUP + g reads kv head kv_h — the SAME tile-side
        # mapping as kv_h = pid_h // (H // H_KV) in the forward/dQ
        # kernels). GROUP is a constexpr, so static_range unrolls the
        # loop at compile time.
        for g in tl.static_range(GROUP):
            q_h = kv_h * GROUP + g
            # Causal M loop: Q row i attends K rows j <= i, so K row j is
            # seen by Q rows i >= j — the N-block [start_n, start_n +
            # BLOCK_N) is touched by M blocks from (start_n // BLOCK_M) *
            # BLOCK_M (the first M-block whose causal range reaches N row
            # start_n) up to S (the transposed bound of the forward's
            # hi = min((pid_m+1)*BLOCK_M, S); min with S handles
            # non-multiple S; fully-masked blocks are never scheduled).
            lo = (pid_n * BLOCK_N // BLOCK_M) * BLOCK_M
            for start_m in range(lo, S, BLOCK_M):
                offs_m = start_m + tl.arange(0, BLOCK_M)
                m_mask = offs_m < S

                # Q/dO tiles (BLOCK_M x D_CHUNK, q_h addressing) plus the
                # LSE tile (BLOCK_M fp32 — the recompute key) and the
                # host-side delta = rowsum(dO ∘ O) tile (BLOCK_M fp32).
                q_base = (q_ptr + pid_b * stride_qb + q_h * stride_qh
                          + offs_m[:, None] * stride_qm)
                q1 = tl.load(q_base + offs_d1[None, :],
                             mask=m_mask[:, None], other=0.0)
                q2 = tl.load(q_base + offs_d2[None, :],
                             mask=m_mask[:, None], other=0.0)
                do_base = (do_ptr + pid_b * stride_dob + q_h * stride_doh
                           + offs_m[:, None] * stride_dom)
                do1 = tl.load(do_base + offs_d1[None, :],
                              mask=m_mask[:, None], other=0.0)
                do2 = tl.load(do_base + offs_d2[None, :],
                              mask=m_mask[:, None], other=0.0)
                lse = tl.load(lse_ptr + pid_b * stride_lb
                              + q_h * stride_lh + offs_m,
                              mask=m_mask, other=0.0)
                delta = tl.load(delta_ptr + pid_b * stride_db
                                + q_h * stride_dh + offs_m,
                                mask=m_mask, other=0.0)

                # FA2-style recompute (no saved probabilities), N-major:
                # scores^T = (k @ q^T) over both D-chunks (fp32
                # accumulate) * sm_scale — shape (BLOCK_N, BLOCK_M).
                scores_t = tl.dot(k1, tl.trans(q1)) + tl.dot(k2, tl.trans(q2))
                scores_t = scores_t * sm_scale
                # Causal + S-boundary mask, TRANSPOSED: K row offs_n is
                # seen by Q rows offs_m >= offs_n (both bounded by S) —
                # -inf where violated, applied BEFORE exp (exp, not exp2
                # — interpreter-safe).
                scores_t = tl.where(
                    (offs_n[:, None] <= offs_m[None, :])
                    & (offs_m[None, :] < S) & (offs_n[:, None] < S),
                    scores_t, float("-inf"))

                # P^T = exp(scores^T − lse[None, :]) (fp32, natural exp):
                # exact softmax probabilities, denominator in LSE.
                p_t = tl.exp(scores_t - lse[None, :])
                # dP^T = v @ do^T over both D-chunks (BLOCK_N, BLOCK_M),
                # dS^T = P^T ∘ (dP^T − delta[None, :]) (fp32)...
                dp_t = tl.dot(v1, tl.trans(do1)) + tl.dot(v2, tl.trans(do2))
                ds_t = p_t * (dp_t - delta[None, :])
                # ...dK += scale · dS^T @ Q (sm_scale rides on dS^T: chain
                # rule, scores = sm_scale·q·k^T) and dV += P^T @ dO, both
                # in two D-chunks. dS^T/P^T are cast to the operand dtype
                # for tl.dot — the design's fp16-operands / fp32-accumulate
                # contract (mirrors the dQ kernel's ds_h cast).
                ds_h = (ds_t * sm_scale).to(q1.dtype)
                dk1 = dk1 + tl.dot(ds_h, q1)
                dk2 = dk2 + tl.dot(ds_h, q2)
                p_h = p_t.to(do1.dtype)
                dv1 = dv1 + tl.dot(p_h, do1)
                dv2 = dv2 + tl.dot(p_h, do2)

        # Epilogue: store dK/dV in two D-chunks each (cast to K/V's dtype,
        # masked by the N-boundary) through the flat (B*H_kv, S, D) views.
        dk_base = dk_ptr + kv_flat * stride_dkh + offs_n[:, None] * stride_dkn
        tl.store(dk_base + offs_d1[None, :], dk1.to(k1.dtype),
                 mask=n_mask[:, None])
        tl.store(dk_base + offs_d2[None, :], dk2.to(k2.dtype),
                 mask=n_mask[:, None])
        dv_base = dv_ptr + kv_flat * stride_dvh + offs_n[:, None] * stride_dvn
        tl.store(dv_base + offs_d1[None, :], dv1.to(v1.dtype),
                 mask=n_mask[:, None])
        tl.store(dv_base + offs_d2[None, :], dv2.to(v2.dtype),
                 mask=n_mask[:, None])

    _DKV_KERNEL = _attn_dkv_kernel
    return _DKV_KERNEL


def sm86_attention_backward_dkv(q, k, v, out, lse, grad_out, sm_scale,
                                 block_m=32, block_n=32, d_chunk=128):
    """dK/dV backward (Triton kernel, FA2-style LSE recompute).

    Computes (dk, dv) = (dL/dk, dL/dv) from the forward's out and lse:
    one program per (kv_head, N-block) — grid (ceil(S/BLOCK_N), H_KV, B)
    — loads its K/V tiles ONCE, then loops the GROUP = H // H_kv q heads
    sharing the kv head and, within each q head, the M blocks whose causal
    range reaches the N block (start_m from (start_n // BLOCK_M) *
    BLOCK_M to S). Per (q head, M block) it recomputes P^T =
    exp(q·k^T·sm_scale − lse), forms dS^T = P^T ∘ (dP^T − delta) with
    dP^T = V @ dO^T and delta = rowsum(dO ∘ O) — precomputed HERE on the
    host as (B, H, S) fp32 (per-Q-row, indexed by q head in the kernel) —
    and accumulates dK += (sm_scale · dS^T) @ Q and dV += P^T @ dO in two
    D-chunks each (fp32 accumulation, NO atomics: the q-head loop is
    in-program accumulation; each program owns its dK/dV tile
    exclusively).

    q (B, H, S, 256), k/v (B, H_kv, S, 256), out and grad_out (B, H, S,
    256), all fp16/bf16, contiguous (coerced); lse (B, H, S) fp32
    natural log (sm86_attention_forward / reference_attention_forward
    output). Returns (dk, dv), each (B, H_kv, S, D) in k.dtype/v.dtype
    (== q.dtype, asserted). Tiling defaults: BLOCK_M=32, BLOCK_N=32,
    D_CHUNK=128, num_warps=4, num_stages=1 (launch values;
    overridable for interpreter tests). Loud asserts, never a silent
    no-op: the CUDA path raises if the kernel is unavailable.
    """
    B, H, S, D, H_kv = _check_qkv("sm86_attention_backward_dkv", q, k, v,
                                  out=out, grad_out=grad_out, lse=lse)

    if _INTERPRET_MODE:
        # Test-only seam (see module docstring): interpreter mode runs the
        # kernel on CPU — relax the CUDA assert to CPU tensors.
        assert (not q.is_cuda and not k.is_cuda and not v.is_cuda
                and not out.is_cuda and not lse.is_cuda
                and not grad_out.is_cuda), (
            f"sm86_attention_backward_dkv: TRITON_INTERPRET=1 test seam "
            f"expects CPU tensors, got devices q={q.device}, "
            f"k={k.device}, v={v.device}, out={out.device}, "
            f"lse={lse.device}, grad_out={grad_out.device}")
    else:
        assert (q.is_cuda and k.is_cuda and v.is_cuda and out.is_cuda
                and lse.is_cuda and grad_out.is_cuda), (
            f"sm86_attention_backward_dkv: the Triton kernel is the CUDA "
            f"path; got devices q={q.device}, k={k.device}, "
            f"v={v.device}, out={out.device}, lse={lse.device}, "
            f"grad_out={grad_out.device}. CPU callers must guard with "
            f"kernel_available() and differentiate "
            f"reference_attention_forward instead.")
        ok, reason = kernel_available()
        if not ok:
            raise RuntimeError(
                f"sm86_attention_backward_dkv: SM86 Triton kernel "
                f"unavailable ({reason}); shapes q {tuple(q.shape)}, "
                f"k {tuple(k.shape)}, v {tuple(v.shape)}. Guard the "
                f"kernel path with kernel_available() — never a silent "
                f"no-op.")

    _check_tiling("sm86_attention_backward_dkv", D, block_m, block_n, d_chunk)

    # Contiguous coercion (kernel strides assume unit last-dim stride).
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    out = out.contiguous()
    lse = lse.contiguous()
    grad_out = grad_out.contiguous()
    # K/V batch folded into the head dim: (B*H_kv, S, D) — the kernel's
    # flat kv head index pid_b * H_KV + kv_h addresses these.
    k_flat = k.view(B * H_kv, S, D)
    v_flat = v.view(B * H_kv, S, D)

    # delta = rowsum(dO ∘ O), (B, H, S) fp32, precomputed on the host
    # (design note: simpler and exact) — per-Q-row (indexed by q head in
    # the kernel, NOT by kv head), the dS = P ∘ (dP − delta) correction.
    delta = (grad_out.float() * out.float()).sum(-1).contiguous()

    dk = torch.empty((B, H_kv, S, D), dtype=k.dtype, device=k.device)
    dv = torch.empty((B, H_kv, S, D), dtype=v.dtype, device=v.device)
    dk_flat = dk.view(B * H_kv, S, D)
    dv_flat = dv.view(B * H_kv, S, D)

    kernel = _get_dkv_kernel()
    grid = ((S + block_n - 1) // block_n, H_kv, B)  # (ceil(S/BLOCK_N), H_KV, B)
    # Strides in ELEMENTS (torch strides are element strides already).
    kernel[grid](
        q, k_flat, v_flat, grad_out, dk_flat, dv_flat, lse, delta,
        sm_scale,
        q.stride(0), q.stride(1), q.stride(2),
        k_flat.stride(0), k_flat.stride(1),
        v_flat.stride(0), v_flat.stride(1),
        grad_out.stride(0), grad_out.stride(1), grad_out.stride(2),
        dk_flat.stride(0), dk_flat.stride(1),
        dv_flat.stride(0), dv_flat.stride(1),
        lse.stride(0), lse.stride(1),
        delta.stride(0), delta.stride(1),
        B, H, H_kv, S,
        D=D, BLOCK_M=block_m, BLOCK_N=block_n, D_CHUNK=d_chunk,
        GROUP=H // H_kv,
        num_warps=4, num_stages=1,  # Reduced from 2 for A10G shared memory
    )
    return dk, dv


def reference_attention_forward(q, k, v, sm_scale):
    """Eager dense reference: repeat_kv + fp32 softmax (SDPA-math
    numerics) — the parity oracle and the CPU fallback.

    Causal when S > 1; S == 1 is full attention by definition (single
    query row, no mask). Returns (out, lse): out in q.dtype, lse (B, H, S)
    fp32 natural log. repeat_kv is expand+reshape (tile semantics: q head
    h -> kv head h // n_rep), matching the kernel's GQA mapping —
    repeat_interleave would be WRONG here.
    """
    assert q.dim() == 4, (
        f"reference_attention_forward: q must be 4-D (B, H, S, D), got "
        f"{q.dim()} dims, shape {tuple(q.shape)}")
    assert k.dim() == 4, (
        f"reference_attention_forward: k must be 4-D (B, H_kv, S, D), got "
        f"{k.dim()} dims, shape {tuple(k.shape)}")
    assert v.dim() == 4, (
        f"reference_attention_forward: v must be 4-D (B, H_kv, S, D), got "
        f"{v.dim()} dims, shape {tuple(v.shape)}")

    B, H, S, D = q.shape
    B_k, H_kv, S_k, D_k = k.shape
    B_v, H_kv_v, S_v, D_v = v.shape

    assert (B_k == B and B_v == B and S_k == S and S_v == S
            and D_k == D and D_v == D and H_kv_v == H_kv), (
        f"reference_attention_forward: q/k/v must share B, S, D and k/v "
        f"must share H_kv; got q {tuple(q.shape)}, k {tuple(k.shape)}, "
        f"v {tuple(v.shape)}")
    assert H % H_kv == 0, (
        f"reference_attention_forward: GQA requires H % H_kv == 0; got "
        f"H={H}, H_kv={H_kv} (q {tuple(q.shape)}, k {tuple(k.shape)}, "
        f"v {tuple(v.shape)})")

    n_rep = H // H_kv
    # Tile-style repeat_kv (NOT repeat_interleave): q head h reads kv head
    # h // n_rep, exactly the kernel's kv_h = pid_h // (H // H_KV).
    k_r = k[:, :, None, :, :].expand(B, H_kv, n_rep, S, D).reshape(B, H, S, D)
    v_r = v[:, :, None, :, :].expand(B, H_kv, n_rep, S, D).reshape(B, H, S, D)

    q_f = q.float()
    scores = torch.matmul(q_f, k_r.float().transpose(-1, -2)) * sm_scale
    if S > 1:
        row_idx = torch.arange(S, device=q.device).unsqueeze(1)
        col_idx = torch.arange(S, device=q.device).unsqueeze(0)
        scores = scores.masked_fill(row_idx < col_idx, float("-inf"))
    # fp32 softmax with rowmax subtraction for stability.
    row_max = scores.amax(dim=-1, keepdim=True)
    exp_scores = torch.exp(scores - row_max)
    probs = exp_scores / exp_scores.sum(dim=-1, keepdim=True)
    out = torch.matmul(probs, v_r.float()).to(q.dtype)
    lse = torch.logsumexp(scores, dim=-1)  # (B, H, S) fp32, natural log
    return out, lse


# ---------------------------------------------------------------------------
# The autograd Function gluing the kernels. Mirrors the repo's
# autograd-Function style (scripts/qlora_gemm.py): frozen tensors as plain
# ctx attributes (NOT save_for_backward), an escape-hatch env flag, and a
# NaN debug env flag.
# ---------------------------------------------------------------------------


def _debug_nan_enabled() -> bool:
    """FLUTE_ATTN_DEBUG_NAN=1 → backward grad NaNs raise (qlora_gemm's
    FLUTE_DEBUG_NAN pattern: off by default, zero overhead when off)."""
    return os.environ.get("FLUTE_ATTN_DEBUG_NAN", "") == "1"


_attn_escape_banner = False


def _attn_kernel_enabled() -> bool:
    """Escape hatch for flute_sm86_attention: FLUTE_ATTN_KERNEL in
    ("", "0") disables the Triton kernel path (loud one-time banner,
    qlora_gemm's FLUTE_FUSED_BWD pattern) — forward and backward fall
    back to the reference autograd path. Default ("1"): kernel on."""
    global _attn_escape_banner
    if os.environ.get("FLUTE_ATTN_KERNEL", "1") in ("", "0"):
        if not _attn_escape_banner:
            _attn_escape_banner = True
            print("[attn_sm86] FLUTE_ATTN_KERNEL=0: Triton attention "
                  "kernel disabled; flute_sm86_attention uses the "
                  "reference autograd path (reference_attention_forward)",
                  flush=True)
        return False
    return True


class Sm86AttentionFn(torch.autograd.Function):
    """Autograd surface over the three SM86 kernels.

    Forward: (out, lse) = sm86_attention_forward(q, k, v, sm_scale,
    need_lse=True); returns out ONLY — lse is internal (the backward's
    recompute key, never part of the public output). q/k/v/out/lse/
    sm_scale are saved as PLAIN ctx attributes (qlora_gemm's frozen-tensor
    style: they survive checkpoint recompute and skip save_for_backward's
    version bookkeeping; the tradeoff — an in-place mutation of q/k/v
    between forward and backward is not detected — is the documented
    qlora_gemm contract). Operand dtype is kept AS-IS: fp16 and bf16 are
    both first-class kernel operand dtypes (bf16 is NOT cast to fp16).

    Backward: dq from sm86_attention_backward_dq, (dk, dv) from
    sm86_attention_backward_dkv; returns (dq, dk, dv, None) — sm_scale
    is a python float, no grad. FLUTE_ATTN_DEBUG_NAN=1 turns a NaN grad
    into a RuntimeError naming the tensor (see _debug_nan_enabled).
    """

    @staticmethod
    def forward(ctx, q, k, v, sm_scale):
        assert (q.dtype in (torch.float16, torch.bfloat16)
                and k.dtype == q.dtype and v.dtype == q.dtype), (
            f"Sm86AttentionFn.forward: q/k/v must be fp16 or bf16 and "
            f"share dtype, kept AS-IS (the kernels take both operand "
            f"dtypes; bf16 is NOT cast to fp16); got q={q.dtype}, "
            f"k={k.dtype}, v={v.dtype} (shapes q {tuple(q.shape)}, "
            f"k {tuple(k.shape)}, v {tuple(v.shape)})")
        out, lse = sm86_attention_forward(q, k, v, sm_scale,
                                          need_lse=True)
        # Frozen tensors as plain ctx attributes (qlora_gemm style — see
        # the class docstring); lse detached (it is fp32, graph-free
        # anyway — detach is the belt-and-braces contract).
        ctx.q = q
        ctx.k = k
        ctx.v = v
        ctx.out = out.detach()
        ctx.lse = lse.detach()
        ctx.sm_scale = sm_scale
        return out

    @staticmethod
    def backward(ctx, grad_out):
        q, k, v = ctx.q, ctx.k, ctx.v
        out, lse, sm_scale = ctx.out, ctx.lse, ctx.sm_scale

        assert torch.is_tensor(grad_out), (
            f"Sm86AttentionFn.backward: grad_out must be a tensor "
            f"matching out's (B, H, S, D)={tuple(out.shape)}, got "
            f"{type(grad_out).__name__}")
        assert tuple(grad_out.shape) == tuple(out.shape), (
            f"Sm86AttentionFn.backward: grad_out must match out's shape "
            f"{tuple(out.shape)}, got {tuple(grad_out.shape)}")
        # autograd hands us whatever layout the downstream op produced
        # (possibly non-contiguous / a view); the kernels' stride
        # contract needs a unit last-dim stride — coerce, never assume.
        grad_out = grad_out.contiguous()

        dq = sm86_attention_backward_dq(q, k, v, out, lse, grad_out,
                                        sm_scale)
        dk, dv = sm86_attention_backward_dkv(q, k, v, out, lse, grad_out,
                                             sm_scale)

        if _debug_nan_enabled():
            for name, grad in (("dq", dq), ("dk", dk), ("dv", dv)):
                if torch.isnan(grad).any():
                    raise RuntimeError(
                        f"Sm86AttentionFn.backward produced NaN in {name} "
                        f"(shape {tuple(grad.shape)}; q {tuple(q.shape)}, "
                        f"k {tuple(k.shape)}, v {tuple(v.shape)}, dtype "
                        f"{q.dtype}, sm_scale={sm_scale}); see "
                        f"reports/design_attn_sm86.md — FLUTE_ATTN_DEBUG_NAN")

        return dq, dk, dv, None


def flute_sm86_attention(q, k, v, sm_scale):
    """Public differentiable SM86 attention entry point.

    Routing (never a silent downgrade on CUDA):
      - S == 1 → the reference math (design non-goal: single-query decode
        is full attention by definition — the engine's short-circuit,
        not a kernel branch; the kernels are never launched for it).
      - FLUTE_ATTN_KERNEL in ("", "0") → the reference autograd path
        (loud banner — the documented escape hatch).
      - CUDA q/k/v, or TRITON_INTERPRET=1 (test seam) →
        Sm86AttentionFn.apply (the Triton kernels). If triton is somehow
        unavailable on a CUDA box the WRAPPER raises loudly — this
        function never falls back behind the caller's back.
      - otherwise (CPU box, no interpreter) → the reference autograd
        path: reference_attention_forward(q, k, v, sm_scale)[0], pure
        torch ops, so autograd flows natively (the documented CPU
        behavior parity tests rely on).

    q (B, H, S, 256), k/v (B, H_kv, S, 256), same dtype (fp16/bf16 on
    the kernel path — see Sm86AttentionFn), GQA with H % H_kv == 0.
    Returns out (B, H, S, D) in q.dtype. Loud asserts, shapes in every
    message.
    """
    assert q.dim() == 4, (
        f"flute_sm86_attention: q must be 4-D (B, H, S, D), got "
        f"{q.dim()} dims, shape {tuple(q.shape)}")
    assert k.dim() == 4, (
        f"flute_sm86_attention: k must be 4-D (B, H_kv, S, D), got "
        f"{k.dim()} dims, shape {tuple(k.shape)}")
    assert v.dim() == 4, (
        f"flute_sm86_attention: v must be 4-D (B, H_kv, S, D), got "
        f"{v.dim()} dims, shape {tuple(v.shape)}")

    B, H, S, D = q.shape
    B_k, H_kv, S_k, D_k = k.shape
    B_v, H_kv_v, S_v, D_v = v.shape

    assert (B_k == B and B_v == B and S_k == S and S_v == S
            and D_k == D and D_v == D and H_kv_v == H_kv), (
        f"flute_sm86_attention: q/k/v must share B, S, D and k/v must "
        f"share H_kv; got q {tuple(q.shape)}, k {tuple(k.shape)}, "
        f"v {tuple(v.shape)}")
    assert D == 256, (
        f"flute_sm86_attention: design geometry is D=256 "
        f"(reports/design_attn_sm86.md); got q D={D}, k D={D_k}, "
        f"v D={D_v}")
    assert H % H_kv == 0, (
        f"flute_sm86_attention: GQA requires H % H_kv == 0; got H={H}, "
        f"H_kv={H_kv} (q {tuple(q.shape)}, k {tuple(k.shape)}, "
        f"v {tuple(v.shape)})")
    assert S >= 1, (
        f"flute_sm86_attention: S must be >= 1, got S={S} "
        f"(q {tuple(q.shape)}, k {tuple(k.shape)}, v {tuple(v.shape)})")
    assert q.dtype == k.dtype and k.dtype == v.dtype, (
        f"flute_sm86_attention: q/k/v must share dtype, got q={q.dtype}, "
        f"k={k.dtype}, v={v.dtype} (q {tuple(q.shape)}, "
        f"k {tuple(k.shape)}, v {tuple(v.shape)})")
    assert q.device == k.device and k.device == v.device, (
        f"flute_sm86_attention: q/k/v must share device, got "
        f"q={q.device}, k={k.device}, v={v.device} (shapes q "
        f"{tuple(q.shape)}, k {tuple(k.shape)}, v {tuple(v.shape)})")

    # S == 1 decode short-circuit (design non-goal — see docstring): the
    # reference math IS the answer, on every device.
    if S == 1:
        return reference_attention_forward(q, k, v, sm_scale)[0]

    # Escape hatch first (it must win on CUDA boxes too — that is its
    # point), then the device/interpreter routing.
    if not _attn_kernel_enabled():
        return reference_attention_forward(q, k, v, sm_scale)[0]

    if _INTERPRET_MODE or (q.is_cuda and k.is_cuda and v.is_cuda):
        return Sm86AttentionFn.apply(q, k, v, sm_scale)

    # CPU box without the interpreter: the documented CPU behavior —
    # reference numerics with native autograd (kernel_available() is
    # (False, ...) here; the kernels cannot run).
    return reference_attention_forward(q, k, v, sm_scale)[0]
