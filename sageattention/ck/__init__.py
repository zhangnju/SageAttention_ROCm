"""
CK tile (Composable Kernel) backend for SageAttention on AMD CDNA4 (gfx950, MI355X).

Uses flash_attn's CK FMHA kernel which is natively optimized for gfx950.
The rocwmma-based INT8/FP8 kernel is not efficient on gfx950 because:
  - rocwmma sgattn uses CUDA Core for QK and rocwmma for PV
  - TBLOCK_X=256 (gfx9Params) leads to low occupancy on MI355X (256 CU)
  - Result: ~67 TFLOPS vs ~893 TFLOPS for CK FA2

This module wraps the CK FMHA forward kernel from flash_attn_2_cuda.
"""

from .flash_attn_gfx950 import flash_attn_ck
