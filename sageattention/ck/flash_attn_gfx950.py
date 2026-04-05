"""
CK tile Flash Attention wrapper for AMD CDNA4 (gfx950, MI355X).

Uses the Composable Kernel (CK) FMHA implementation from flash_attn_2_cuda,
which achieves ~770-893 TFLOPS on MI355X.

Optimizations vs naive permute-based wrapper:
  1. Strided (non-contiguous) inputs accepted by fa_cuda.fwd — avoids 3x memcpy
  2. NHD layout path: zero permutes when caller already has NHD tensors
  3. return_lse: CK computes lse for free, just pass results[1] through
  4. bfloat16: CK natively supports bf16, no extra casting needed

Layout:
  HND mode: Input  [B, H, S, D] -> transpose to strided NHD view -> CK
             Output [B, S, H, D] strided view -> transpose back (no copy)
  NHD mode: Input  [B, S, H, D] -> CK directly
             Output [B, S, H, D] as returned by CK
"""

import torch
from typing import Optional, Tuple, Union


def flash_attn_ck(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    tensor_layout: str = "HND",
    sm_scale: Optional[float] = None,
    is_causal: bool = False,
    return_lse: bool = False,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """
    CK tile Flash Attention for gfx950 (MI355X).

    Parameters
    ----------
    q, k, v : torch.Tensor
        HND layout [B, H, S, D] or NHD layout [B, S, H, D] per tensor_layout.
    tensor_layout : str
        "HND" (default) or "NHD".
    sm_scale : float, optional
        Softmax scale. Defaults to 1/sqrt(headdim).
    is_causal : bool
        Whether to apply causal mask.
    return_lse : bool
        If True, also return log-sum-exp [B, H, S] in natural-log space.

    Returns
    -------
    out : torch.Tensor  (same layout as input)
    lse : torch.Tensor  (only if return_lse=True)  shape [B, H, S], float32
    """
    import flash_attn_2_cuda as fa_cuda

    if sm_scale is None:
        sm_scale = q.size(-1) ** -0.5

    if tensor_layout == "NHD":
        # Already NHD [B, S, H, D] — pass directly, zero layout conversion cost
        q_nhd, k_nhd, v_nhd = q, k, v
    else:
        # HND [B, H, S, D] -> strided NHD view via transpose (no data copy)
        q_nhd = q.transpose(1, 2)   # [B, S, H, D], strided, no copy
        k_nhd = k.transpose(1, 2)
        v_nhd = v.transpose(1, 2)

    # fa_cuda.fwd accepts strided tensors.
    # Returns [out_nhd, lse, S_dmask, rng_state]
    #   out_nhd : [B, S, H, D] float16/bf16
    #   lse     : [B, H, S]   float32, in log2 space (exp2 trick internally)
    results = fa_cuda.fwd(
        q_nhd, k_nhd, v_nhd,
        None,           # alibi_slopes
        None,           # pre-allocated out (None = allocate)
        0.0,            # dropout_p
        sm_scale,
        is_causal,
        -1,             # window_size_left  (-1 = full context)
        -1,             # window_size_right
        0.0,            # softcap
        False,          # return_softmax
        None,           # generator
    )
    out_nhd = results[0]   # [B, S, H, D]
    lse_log2 = results[1]  # [B, H, S], log2 space

    if tensor_layout == "NHD":
        out = out_nhd
    else:
        # Strided transpose back: [B, S, H, D] -> [B, H, S, D], no data copy
        out = out_nhd.transpose(1, 2)

    if return_lse:
        # Convert from log2 space to natural-log space (sageattn convention)
        lse = lse_log2 / 1.44269504088896340736  # log2(e) = 1.4427
        return out, lse
    return out
