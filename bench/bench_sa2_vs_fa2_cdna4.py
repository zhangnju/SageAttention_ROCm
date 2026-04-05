"""
SageAttention2 vs Flash Attention 2 (CK) Benchmark
AMD CDNA4 (gfx950) — AMD Instinct MI355X

三方对比：
  1. SageAttention2   : 本项目 INT8/FP8 rocwmma 内核 (HND layout)
                        在 gfx950 上使用 sageattn_qk_int8_pv_fp8_cuda
                        需要先编译: pip install -e . (in SageAttention_ROCm)
  2. Flash Attention 2: CK 实现
                        优先使用 torch._C._ROCmFABackend.Ck (若 PyTorch 构建时包含)
                        备选: flash_attn 包 (需用 --use_ck 编译安装)
                        当前 fallback: SDPA FLASH_ATTENTION backend (aotriton)
  3. PyTorch SDPA     : torch.nn.functional.scaled_dot_product_attention
                        ROCm 上默认使用 aotriton Flash Attention

布局说明:
  flash_attn_func:    [B, S, H, D]  (NHD)
  sageattn / SDPA:    [B, H, S, D]  (HND)

与 RDNA4 脚本的主要差异:
  - FA2 后端: CK (Composable Kernel) 而非 Triton
  - SageAttention2: INT8/FP8 rocwmma (CDNA4 优化), 而非 FP16 Triton FA
  - 目标 GPU: CDNA4 (gfx950) 而非 RDNA4 (gfx1201)

运行方式:
  # 首先构建 sageattention 扩展 (如尚未构建):
  # cd /home/work/SageAttention_ROCm && pip install -e .
  #
  # 若已安装 flash_attn with CK:
  # python bench/bench_sa2_vs_fa2_cdna4.py
  #
  # 若需要安装 flash_attn with CK (from source):
  # pip install flash-attn --use-ck (if available for gfx950)
"""

import os, time, argparse, warnings
import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.attention import sdpa_kernel, SDPBackend


# ── Imports ───────────────────────────────────────────────────────────────────

try:
    from sageattention import sageattn
    HAS_SAGE = True
except ImportError:
    HAS_SAGE = False
    print("WARNING: sageattention not found or not compiled.")
    print("  Build with: cd /home/work/SageAttention_ROCm && pip install -e .")

# ── FA2 backend detection (CK priority) ───────────────────────────────────────
# Strategy: try CK backends in priority order
FA2_BACKEND = "none"
FA2_BACKEND_DESC = "NOT available"
flash_attn_func = None

# 1. Try flash_attn package (supports both CK and aotriton via FLASH_ATTN_ROCM_FA_BACKEND env)
try:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from flash_attn import flash_attn_func as _flash_attn_func
    flash_attn_func = _flash_attn_func
    import flash_attn
    # Check which backend flash_attn uses on ROCm (CK vs aotriton)
    fa_backend_env = os.environ.get("FLASH_ATTN_ROCM_FA_BACKEND", "CK").upper()
    FA2_BACKEND = "flash_attn_ck" if "CK" in fa_backend_env else "flash_attn_aotriton"
    FA2_BACKEND_DESC = f"flash_attn {flash_attn.__version__} ({fa_backend_env} backend)"
    HAS_FA2 = True
except ImportError:
    HAS_FA2 = False

# 2. If no flash_attn, try torch built-in CK backend for ROCm
if not HAS_FA2:
    try:
        _ck_backend = torch._C._ROCmFABackend.Ck
        torch._C._set_rocm_fa_preferred_backend(_ck_backend)
        # Verify it works
        _q = torch.randn(1, 1, 64, 64, dtype=torch.float16, device="cuda")
        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            F.scaled_dot_product_attention(_q, _q, _q)
        FA2_BACKEND = "sdpa_ck"
        FA2_BACKEND_DESC = "PyTorch SDPA (ROCm CK backend)"
        HAS_FA2 = True
        del _q
    except (RuntimeError, AttributeError) as _e:
        # CK not built into this PyTorch, fall back to aotriton
        FA2_BACKEND = "sdpa_aotriton"
        FA2_BACKEND_DESC = "PyTorch SDPA (aotriton, CK not available in this build)"
        HAS_FA2 = True  # aotriton-based SDPA is still a valid FA2 comparison
        print(f"NOTE: ROCm CK flash attention not available ({_e})")
        print(f"      Using aotriton-based SDPA as FA2 reference.")
        print(f"      To use CK, install flash-attn with CK support for gfx950.")

if not HAS_FA2:
    print("WARNING: No Flash Attention 2 backend available.")


def fa2_forward_nhd(q_nhd, k_nhd, v_nhd, sm_scale, is_causal):
    """Run FA2 with NHD input (flash_attn_func layout). Returns NHD output."""
    assert FA2_BACKEND in ("flash_attn_ck", "flash_attn_aotriton")
    return flash_attn_func(q_nhd, k_nhd, v_nhd,
                           softmax_scale=sm_scale, causal=is_causal)


def fa2_forward_hnd(q_hnd, k_hnd, v_hnd, sm_scale, is_causal):
    """Run FA2 with HND input (SDPA layout). Returns HND output."""
    with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
        return F.scaled_dot_product_attention(
            q_hnd, k_hnd, v_hnd, scale=sm_scale, is_causal=is_causal)


# ── Benchmark helpers ─────────────────────────────────────────────────────────

def bench(fn, warmup=50, rep=300):
    """Returns (median_ms, std_ms)."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(rep):
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1000)
    return float(np.median(ts)), float(np.std(ts))


def flops(B, H, S, D, is_causal):
    f = 2 * 2 * B * H * S * S * D
    return f // 2 if is_causal else f


def acc(out, ref):
    d = (out.float() - ref.float()).abs()
    return d.max().item(), d.mean().item()


# ── Single config run ─────────────────────────────────────────────────────────

def run_config(B, H, S, D, is_causal, warmup, rep):
    sm = D ** -0.5
    torch.manual_seed(42)

    # HND layout for SageAttention / SDPA
    q_hnd = torch.randn(B, H, S, D, dtype=torch.float16, device="cuda") * 0.5
    k_hnd = torch.randn(B, H, S, D, dtype=torch.float16, device="cuda") * 0.5
    v_hnd = torch.randn(B, H, S, D, dtype=torch.float16, device="cuda") * 0.5

    # NHD layout for flash_attn_func
    q_nhd = q_hnd.permute(0, 2, 1, 3).contiguous()  # [B,S,H,D]
    k_nhd = k_hnd.permute(0, 2, 1, 3).contiguous()
    v_nhd = v_hnd.permute(0, 2, 1, 3).contiguous()

    nf = flops(B, H, S, D, is_causal)

    # ── FP32 reference ──────────────────────────────────────────────────────
    ref = F.scaled_dot_product_attention(
        q_hnd.float(), k_hnd.float(), v_hnd.float(),
        scale=sm, is_causal=is_causal
    )  # [B,H,S,D]

    result = {"B": B, "H": H, "S": S, "D": D, "causal": is_causal}

    # ── SageAttention2 (INT8/FP8 rocwmma, CDNA4) ────────────────────────────
    if HAS_SAGE:
        try:
            t_sage, std_sage = bench(
                lambda: sageattn(q_hnd, k_hnd, v_hnd, sm_scale=sm, is_causal=is_causal),
                warmup=warmup, rep=rep)
            out_sage = sageattn(q_hnd, k_hnd, v_hnd, sm_scale=sm, is_causal=is_causal)
            maxd_sage, meand_sage = acc(out_sage, ref)
            result.update({
                "sage_ms": t_sage, "sage_tflops": nf / t_sage / 1e9,
                "sage_maxd": maxd_sage, "sage_meand": meand_sage,
            })
        except Exception as e:
            result.update({"sage_ms": None, "sage_tflops": None,
                           "sage_maxd": None, "sage_meand": None,
                           "sage_err": str(e)})
    else:
        result.update({"sage_ms": None, "sage_tflops": None,
                       "sage_maxd": None, "sage_meand": None})

    # ── Flash Attention 2 (CK / aotriton) ───────────────────────────────────
    if HAS_FA2:
        try:
            if FA2_BACKEND in ("flash_attn_ck", "flash_attn_aotriton"):
                # flash_attn uses NHD layout
                _fa2_fn = lambda: fa2_forward_nhd(q_nhd, k_nhd, v_nhd, sm, is_causal)
                t_fa2, std_fa2 = bench(_fa2_fn, warmup=warmup, rep=rep)
                out_fa2 = fa2_forward_nhd(q_nhd, k_nhd, v_nhd, sm, is_causal).permute(0, 2, 1, 3)
            else:
                # SDPA-based backends (sdpa_ck, sdpa_aotriton) use HND layout directly
                _fa2_fn = lambda: fa2_forward_hnd(q_hnd, k_hnd, v_hnd, sm, is_causal)
                t_fa2, std_fa2 = bench(_fa2_fn, warmup=warmup, rep=rep)
                out_fa2 = fa2_forward_hnd(q_hnd, k_hnd, v_hnd, sm, is_causal)
            maxd_fa2, meand_fa2 = acc(out_fa2, ref)
            result.update({
                "fa2_ms": t_fa2, "fa2_tflops": nf / t_fa2 / 1e9,
                "fa2_maxd": maxd_fa2, "fa2_meand": meand_fa2,
            })
        except Exception as e:
            result.update({"fa2_ms": None, "fa2_tflops": None,
                           "fa2_maxd": None, "fa2_meand": None,
                           "fa2_err": str(e)})
    else:
        result.update({"fa2_ms": None, "fa2_tflops": None,
                       "fa2_maxd": None, "fa2_meand": None})

    # ── PyTorch SDPA (aotriton default) ─────────────────────────────────────
    t_sdpa, std_sdpa = bench(
        lambda: F.scaled_dot_product_attention(q_hnd, k_hnd, v_hnd,
                                               scale=sm, is_causal=is_causal),
        warmup=warmup, rep=rep)
    out_sdpa = F.scaled_dot_product_attention(q_hnd, k_hnd, v_hnd,
                                              scale=sm, is_causal=is_causal)
    maxd_sdpa, meand_sdpa = acc(out_sdpa, ref)
    result.update({
        "sdpa_ms": t_sdpa, "sdpa_tflops": nf / t_sdpa / 1e9,
        "sdpa_maxd": maxd_sdpa, "sdpa_meand": meand_sdpa,
    })

    # Speedup ratios
    if result.get("sage_ms") and result.get("fa2_ms"):
        result["sa2_vs_fa2"]  = result["fa2_ms"]  / result["sage_ms"]
        result["sa2_vs_sdpa"] = result["sdpa_ms"] / result["sage_ms"]
        result["fa2_vs_sdpa"] = result["sdpa_ms"] / result["fa2_ms"]
    elif result.get("sage_ms"):
        result["sa2_vs_sdpa"] = result["sdpa_ms"] / result["sage_ms"]
    elif result.get("fa2_ms"):
        result["fa2_vs_sdpa"] = result["sdpa_ms"] / result["fa2_ms"]

    return result


# ── Table printing ────────────────────────────────────────────────────────────

def print_table(results, fa2_label):
    W = 182
    print("=" * W)
    print(f"{'Configuration':42s} │ {'SageAttention2 (INT8/FP8 rocwmma)':34s} │ {fa2_label:38s} │ {'PyTorch SDPA (aotriton)':26s} │ SA2/FA2")
    print(f"{'':42s} │ {'TFLOPS':>8s} {'ms':>7s} {'maxΔ':>8s} {'meanΔ':>8s} │ {'TFLOPS':>8s} {'ms':>7s} {'maxΔ':>8s} {'meanΔ':>8s} │ {'TFLOPS':>8s} {'ms':>7s} │")
    print("=" * W)

    nc = [r for r in results if not r["causal"]]
    ca = [r for r in results if r["causal"]]

    def fmt_sage(r):
        if r.get("sage_ms"):
            return f"{r['sage_tflops']:8.1f} {r['sage_ms']:7.3f} {r['sage_maxd']:8.5f} {r['sage_meand']:8.6f}"
        elif r.get("sage_err"):
            return f"{'ERR':>34s}"
        return f"{'N/A':>34s}"

    def fmt_fa2(r):
        if r.get("fa2_ms"):
            return f"{r['fa2_tflops']:8.1f} {r['fa2_ms']:7.3f} {r['fa2_maxd']:8.5f} {r['fa2_meand']:8.6f}"
        elif r.get("fa2_err"):
            return f"{'ERR':>38s}"
        return f"{'N/A':>38s}"

    def print_group(rows, label):
        if not rows:
            return
        print(f"  ── {label} ──")
        for r in rows:
            c = "causal" if r["causal"] else "      "
            cfg = f"B={r['B']} H={r['H']:2d} S={r['S']:5d} D={r['D']:3d} {c}"
            sage_s = fmt_sage(r)
            fa2_s  = fmt_fa2(r)
            sdpa_s = f"{r['sdpa_tflops']:8.1f} {r['sdpa_ms']:7.3f}"
            vs_s   = f"{r.get('sa2_vs_fa2', 0):5.2f}x" if r.get("sa2_vs_fa2") else "  N/A"
            print(f"  {cfg:42s} │ {sage_s} │ {fa2_s} │ {sdpa_s} │ {vs_s}")

    print_group(nc, "Non-Causal")
    print()
    print_group(ca, "Causal")
    print("=" * W)


def print_summary(results, fa2_label):
    rows_with_both = [r for r in results if r.get("sage_ms") and r.get("fa2_ms")]
    rows_with_sage = [r for r in results if r.get("sage_ms")]
    rows_with_fa2  = [r for r in results if r.get("fa2_ms")]

    print(f"\nSpeedup Summary  (SA2 = SageAttention2, FA2 = {fa2_label}):")
    print(f"  {'Scenario':25s} {'SA2/FA2':>10s} {'SA2/SDPA':>10s} {'FA2/SDPA':>10s}")
    print(f"  {'─'*57}")

    nc_both = [r for r in rows_with_both if not r["causal"]]
    ca_both = [r for r in rows_with_both if r["causal"]]

    for label, rows in [("Non-Causal", nc_both), ("Causal", ca_both), ("Overall", rows_with_both)]:
        if not rows:
            continue
        sa2_fa2  = np.mean([r["sa2_vs_fa2"]  for r in rows])
        sa2_sdpa = np.mean([r["sa2_vs_sdpa"] for r in rows])
        fa2_sdpa = np.mean([r["fa2_vs_sdpa"] for r in rows])
        print(f"  {label:25s} {sa2_fa2:10.2f}x {sa2_sdpa:10.2f}x {fa2_sdpa:10.2f}x")

    if not rows_with_both:
        # Partial summaries
        if rows_with_sage:
            sdpa_ratios = [r["sa2_vs_sdpa"] for r in rows_with_sage if r.get("sa2_vs_sdpa")]
            if sdpa_ratios:
                print(f"  {'SA2 vs SDPA (overall)':25s} {'N/A':>10s} {np.mean(sdpa_ratios):10.2f}x {'N/A':>10s}")
        return

    print()
    best = max(rows_with_both, key=lambda r: r["sa2_vs_fa2"])
    print(f"  Peak SA2 speedup vs FA2:  {best['sa2_vs_fa2']:.2f}x "
          f"(S={best['S']} D={best['D']} causal={best['causal']})")
    print()

    print("Accuracy vs FP32 reference:")
    if rows_with_sage:
        sage_maxds = [r["sage_maxd"] for r in rows_with_sage]
        print(f"  SageAttention2: avg maxΔ = {np.mean(sage_maxds):.6f},  max maxΔ = {max(sage_maxds):.6f}")
    if rows_with_fa2:
        fa2_maxds = [r["fa2_maxd"] for r in rows_with_fa2]
        print(f"  FA2 ({fa2_label[:12]:12s}): avg maxΔ = {np.mean(fa2_maxds):.6f},  max maxΔ = {max(fa2_maxds):.6f}")
    sdpa_maxds = [r["sdpa_maxd"] for r in results]
    print(f"  PyTorch SDPA:   avg maxΔ = {np.mean(sdpa_maxds):.6f},  max maxΔ = {max(sdpa_maxds):.6f}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Benchmark SageAttention2 vs FA2 CK vs SDPA on AMD CDNA4 (MI355X)")
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--rep",    type=int, default=300)
    args = parser.parse_args()

    # Build FA2 label for display
    if FA2_BACKEND.startswith("flash_attn"):
        fa2_label = f"FA2 ({FA2_BACKEND_DESC[:30]})"
    elif FA2_BACKEND == "sdpa_ck":
        fa2_label = "Flash Attention 2 (CK via SDPA)"
    else:
        fa2_label = "FA2 (aotriton, CK fallback)"

    # Environment info
    print(f"{'='*65}")
    print(f" SageAttention2 vs Flash Attention 2 — CDNA4 (MI355X) Benchmark")
    print(f"{'='*65}")
    print(f" GPU:       {torch.cuda.get_device_name(0)}")
    print(f" GCN Arch:  {torch.cuda.get_device_properties(0).gcnArchName}")
    print(f" ROCm:      {torch.version.hip}")
    print(f" PyTorch:   {torch.__version__}")
    print(f" Triton:    {__import__('triton').__version__}")
    print(f" SA2:       {'available' if HAS_SAGE else 'NOT installed (need: pip install -e .)'}")
    print(f" FA2:       {FA2_BACKEND_DESC}")
    print(f" FA2 note:  {FA2_BACKEND}")
    print(f" Warmup:    {args.warmup}  Rep: {args.rep}")
    print(f"{'='*65}")

    if FA2_BACKEND == "sdpa_aotriton":
        print()
        print("NOTE: CK Flash Attention is not available in this PyTorch build.")
        print("      FA2 column shows aotriton backend for reference.")
        print("      To benchmark with CK FA2, install flash-attn with CK support:")
        print("        pip install flash-attn  (for gfx950, CK support needed)")
        print()

    CONFIGS = [
        # (B,  H,    S,   D,  causal)
        # ─── head_dim=128, non-causal (image/video diffusion) ────────────────
        (1,  16,   512, 128, False),
        (1,  16,  1024, 128, False),
        (1,  16,  2048, 128, False),
        (1,  16,  4096, 128, False),
        (1,  16,  8192, 128, False),
        # ─── head_dim=128, causal (LLM prefill/decode) ───────────────────────
        (1,  16,   512, 128, True),
        (1,  16,  1024, 128, True),
        (1,  16,  2048, 128, True),
        (1,  16,  4096, 128, True),
        # ─── head_dim=64, non-causal ──────────────────────────────────────────
        (1,  16,  1024,  64, False),
        (1,  16,  2048,  64, False),
        (1,  16,  4096,  64, False),
        # ─── head_dim=64, causal ─────────────────────────────────────────────
        (1,  16,  1024,  64, True),
        (1,  16,  2048,  64, True),
        (1,  16,  4096,  64, True),
        # ─── multi-batch / multi-head (typical LLM shapes) ───────────────────
        (2,  16,  2048, 128, False),
        (1,  32,  2048, 128, False),
        (4,  16,  1024, 128, False),
        # ─── GQA-style (fewer KV heads) ──────────────────────────────────────
        (1,  32,  2048, 128, True),
        (2,   8,  4096, 128, True),
    ]

    print(f"\nRunning {len(CONFIGS)} configurations...\n")

    results = []
    for i, (B, H, S, D, causal) in enumerate(CONFIGS):
        tag = f"B={B} H={H:2d} S={S:5d} D={D:3d} {'causal' if causal else '      '}"
        try:
            r = run_config(B, H, S, D, causal, args.warmup, args.rep)
            results.append(r)

            sage_str = f"SA2={r['sage_tflops']:.1f}T" if r.get("sage_ms") else "SA2=N/A"
            fa2_str  = f"FA2={r['fa2_tflops']:.1f}T"  if r.get("fa2_ms")  else "FA2=N/A"
            vs_str   = f"SA2/FA2={r['sa2_vs_fa2']:.2f}x" if r.get("sa2_vs_fa2") else ""
            print(f"  [{i+1:2d}/{len(CONFIGS)}] {tag}: "
                  f"{sage_str}  {fa2_str}  "
                  f"SDPA={r['sdpa_tflops']:.1f}T  {vs_str}")
        except Exception as e:
            print(f"  [{i+1:2d}/{len(CONFIGS)}] {tag}: ERROR: {e}")
            import traceback
            traceback.print_exc()

    print()
    print_table(results, fa2_label)
    print_summary(results, fa2_label)


if __name__ == "__main__":
    main()
