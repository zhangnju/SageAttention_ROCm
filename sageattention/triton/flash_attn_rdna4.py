"""
Flash Attention Triton kernel optimized for RDNA4 (gfx1201).

Optimization history:
  v1: Basic Flash Attention with online softmax, exp2 trick
      → 1.15-1.37x faster than SDPA
  v2: FP16 P@V dot product (out_dtype=fp32) for D=128
      → Additional 10-17% for D=128 non-causal
      → Final: 1.3-2.2x faster than SDPA, up to 3.7x for D=64 causal

Key design:
  - Online softmax (Flash Attention style) with exp2 trick
  - FP16 P@V GEMM with fp32 accumulation (out_dtype=tl.float32)
    for D=128 to improve WMMA utilization
  - Triton autotune: searches BLOCK_M, BLOCK_N, num_warps, num_stages

Performance (gfx1201, AMD Radeon AI PRO R9700):
  D=128 non-causal: 44-59 TFLOPS  vs SDPA 33-44 TFLOPS  (1.2-1.4x faster)
  D=128 causal:     22-35 TFLOPS  vs SDPA 12-14 TFLOPS  (1.7-2.2x faster)
  D=64  non-causal: 48-60 TFLOPS  vs SDPA 17-34 TFLOPS  (1.5-1.8x faster)
  D=64  causal:     40-54 TFLOPS  vs SDPA 14-15 TFLOPS  (2.8-3.7x faster)
"""

import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        # D=128: FP16 P@V benefits from BN=32, nw=8
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32, 'num_warps': 8, 'num_stages': 1}),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32, 'num_warps': 8, 'num_stages': 2}),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32, 'num_warps': 4, 'num_stages': 1}),
        triton.Config({'BLOCK_M':  64, 'BLOCK_N': 32, 'num_warps': 4, 'num_stages': 2}),
        triton.Config({'BLOCK_M':  64, 'BLOCK_N': 64, 'num_warps': 4, 'num_stages': 2}),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'num_warps': 8, 'num_stages': 1}),
        # D=64: larger BN, fewer warps
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'num_warps': 4, 'num_stages': 2}),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'num_warps': 8, 'num_stages': 2}),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N':128, 'num_warps': 8, 'num_stages': 1}),
        triton.Config({'BLOCK_M':  64, 'BLOCK_N':128, 'num_warps': 4, 'num_stages': 1}),
    ],
    key=['N_CTX', 'BLOCK_DMODEL', 'IS_CAUSAL'],
)
@triton.jit
def flash_attn_fwd_kernel(
    Q, K, V, Out,
    sm_scale,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vn, stride_vk,
    stride_oz, stride_oh, stride_om, stride_ok,
    Z, H, N_CTX,
    BLOCK_M: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
):
    """
    Flash Attention forward kernel for RDNA4 gfx1201.
    Grid: (ceil(N_CTX/BLOCK_M), H*Z)

    Uses FP16 P@V dot product with FP32 out_dtype for improved WMMA utilization
    on RDNA4's v_wmma_f32_16x16x16_f16 instruction.
    """
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_z = off_hz // H
    off_h = off_hz % H

    # Pointer offsets for this (batch, head)
    q_offset = off_z * stride_qz + off_h * stride_qh
    k_offset = off_z * stride_kz + off_h * stride_kh
    v_offset = off_z * stride_vz + off_h * stride_vh
    o_offset = off_z * stride_oz + off_h * stride_oh

    # Block pointers
    Q_block_ptr = tl.make_block_ptr(
        base=Q + q_offset,
        shape=(N_CTX, BLOCK_DMODEL),
        strides=(stride_qm, stride_qk),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_DMODEL),
        order=(1, 0),
    )
    K_block_ptr = tl.make_block_ptr(
        base=K + k_offset,
        shape=(BLOCK_DMODEL, N_CTX),
        strides=(stride_kk, stride_kn),
        offsets=(0, 0),
        block_shape=(BLOCK_DMODEL, BLOCK_N),
        order=(0, 1),
    )
    V_block_ptr = tl.make_block_ptr(
        base=V + v_offset,
        shape=(N_CTX, BLOCK_DMODEL),
        strides=(stride_vn, stride_vk),
        offsets=(0, 0),
        block_shape=(BLOCK_N, BLOCK_DMODEL),
        order=(1, 0),
    )
    O_block_ptr = tl.make_block_ptr(
        base=Out + o_offset,
        shape=(N_CTX, BLOCK_DMODEL),
        strides=(stride_om, stride_ok),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_DMODEL),
        order=(1, 0),
    )

    # Initialize online softmax state
    m_i = tl.full((BLOCK_M,), float("-inf"), dtype=tl.float32)  # running max
    l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)                # running sum
    acc = tl.zeros((BLOCK_M, BLOCK_DMODEL), dtype=tl.float32)   # output accumulator

    # Load Q tile (stays in registers for entire KV loop — Flash Attention key insight)
    q = tl.load(Q_block_ptr, boundary_check=(0, 1), padding_option="zero")

    # Scale factor: use log2 trick so softmax uses exp2 instead of exp
    # exp(x * sm_scale) = exp2(x * sm_scale * log2(e))
    qk_scale = sm_scale * 1.44269504088896340736  # = sm_scale * log2(e)

    # Causal: only attend to keys at positions <= current query position
    hi = tl.minimum((start_m + 1) * BLOCK_M, N_CTX) if IS_CAUSAL else N_CTX

    # Main KV tile loop — streams KV through LDS
    for start_n in range(0, hi, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)

        # Load K tile (transposed: [BLOCK_DMODEL, BLOCK_N])
        k = tl.load(K_block_ptr, boundary_check=(0, 1), padding_option="zero")

        # QK^T scaled attention scores: [BLOCK_M, BLOCK_N]
        qk = tl.dot(q, k) * qk_scale

        # Apply causal mask + out-of-bounds padding mask
        if IS_CAUSAL:
            q_idx = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
            k_idx = start_n + tl.arange(0, BLOCK_N)
            mask = (q_idx[:, None] >= k_idx[None, :]) & (k_idx[None, :] < N_CTX)
            qk = tl.where(mask, qk, float("-inf"))
        else:
            k_idx = start_n + tl.arange(0, BLOCK_N)
            qk = tl.where(k_idx[None, :] < N_CTX, qk, float("-inf"))

        # Online softmax update (log2 space for exp2 efficiency)
        m_ij  = tl.max(qk, axis=1)          # row max of this tile
        m_new = tl.maximum(m_i, m_ij)       # updated running max

        # p = exp2(qk - m_new):  softmax weights for this tile
        p = tl.math.exp2(qk - m_new[:, None])   # [BLOCK_M, BLOCK_N]
        # alpha = exp2(m_i - m_new): correction factor for previous output
        alpha = tl.math.exp2(m_i - m_new)        # [BLOCK_M]

        # Load V tile [BLOCK_N, BLOCK_DMODEL]
        v = tl.load(V_block_ptr, boundary_check=(0, 1), padding_option="zero")

        # Accumulate: O = O * alpha + P @ V
        # Using FP16 P (cast from FP32 softmax) with FP32 accumulation via out_dtype.
        # This allows Triton to emit v_wmma_f32_16x16x16_f16_w32_gfx12 instructions
        # which have higher throughput than the implicit FP32 path.
        acc = acc * alpha[:, None]
        acc = tl.dot(p.to(tl.float16), v, acc=acc, out_dtype=tl.float32)

        # Update running normalizer
        l_i = l_i * alpha + tl.sum(p, axis=1)
        m_i = m_new

        # Advance K and V block pointers
        K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_N))
        V_block_ptr = tl.advance(V_block_ptr, (BLOCK_N, 0))

    # Normalize output
    acc = acc / l_i[:, None]

    # Write FP16 output
    tl.store(O_block_ptr, acc.to(tl.float16), boundary_check=(0, 1))


def flash_attn_triton(q, k, v, sm_scale=None, is_causal=False):
    """
    Flash Attention for AMD RDNA4 (gfx1201) via Triton.

    Outperforms PyTorch SDPA (which uses CK Flash Attention backend):
      - D=128 non-causal: 1.2–1.4x faster
      - D=128 causal:     1.7–2.2x faster
      - D=64  non-causal: 1.5–1.8x faster
      - D=64  causal:     2.8–3.7x faster

    Args:
        q, k, v: [batch, heads, seq, head_dim] FP16 (HND layout, contiguous)
        sm_scale: attention scale. Default: 1/sqrt(head_dim)
        is_causal: causal masking

    Returns:
        out: [batch, heads, seq, head_dim] FP16
    """
    assert q.is_cuda and k.is_cuda and v.is_cuda
    assert q.dtype == torch.float16 and k.dtype == torch.float16 and v.dtype == torch.float16
    assert q.dim() == 4, "Expected [B, H, S, D] layout"

    B, H, S, D = q.shape
    assert D in (64, 128), f"head_dim must be 64 or 128, got {D}"

    if sm_scale is None:
        sm_scale = D ** -0.5

    o = torch.empty_like(q)

    # Grid: (num_Q_tiles, H*B) — autotune picks BLOCK_M
    grid = lambda meta: (triton.cdiv(S, meta['BLOCK_M']), H * B)

    flash_attn_fwd_kernel[grid](
        q, k, v, o,
        sm_scale,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        o.stride(0), o.stride(1), o.stride(2), o.stride(3),
        B, H, S,
        BLOCK_DMODEL=D,
        IS_CAUSAL=is_causal,
    )
    return o
