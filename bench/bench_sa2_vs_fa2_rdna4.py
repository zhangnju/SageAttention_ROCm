"""
SageAttention2 vs Flash Attention 2 (AMD Triton) Benchmark
AMD RDNA4 (gfx1201) — AMD Radeon AI PRO R9700

三方对比：
  1. SageAttention2   : 本项目 Triton Flash Attention (FP16, HND layout)
  2. Flash Attention 2: dao-ailab/flash-attention, FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE
                        使用 aiter Triton 内核 (AMD 官方 FA2 实现, NHD layout)
  3. PyTorch SDPA     : torch.nn.functional.scaled_dot_product_attention
                        使用 ROCm CK (Composable Kernel) Flash Attention

布局说明:
  flash_attn_func:    [B, S, H, D]  (NHD)
  sageattn / SDPA:    [B, H, S, D]  (HND)

运行方式:
  PYTHONPATH=/path/to/SageAttention_ROCm \\
  FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE \\
  python bench/bench_sa2_vs_fa2_rdna4.py
"""

import os, time, argparse, warnings
import numpy as np
import torch
import torch.nn.functional as F

# Suppress aiter build messages
os.environ.setdefault("FLASH_ATTENTION_TRITON_AMD_ENABLE", "TRUE")

# ── Imports ───────────────────────────────────────────────────────────────────
try:
    from sageattention import sageattn
    HAS_SAGE = True
except ImportError:
    HAS_SAGE = False
    print("WARNING: sageattention not found")

try:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from flash_attn import flash_attn_func
    HAS_FA2 = True
except ImportError:
    HAS_FA2 = False
    print("WARNING: flash_attn not found. Install with:")
    print("  cd /home/work/flash-attention")
    print("  FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE pip install --no-build-isolation .")


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

    # NHD layout for flash_attn_func (transpose H and S dims)
    q_nhd = q_hnd.permute(0, 2, 1, 3).contiguous()  # [B,S,H,D]
    k_nhd = k_hnd.permute(0, 2, 1, 3).contiguous()
    v_nhd = v_hnd.permute(0, 2, 1, 3).contiguous()

    nf = flops(B, H, S, D, is_causal)

    # ── FP32 reference ──────────────────────────────────────────────────────
    ref_nhd = F.scaled_dot_product_attention(
        q_hnd.float(), k_hnd.float(), v_hnd.float(),
        scale=sm, is_causal=is_causal
    )  # [B,H,S,D]

    result = {"B": B, "H": H, "S": S, "D": D, "causal": is_causal}

    # ── SageAttention2 ──────────────────────────────────────────────────────
    if HAS_SAGE:
        t_sage, std_sage = bench(
            lambda: sageattn(q_hnd, k_hnd, v_hnd, sm_scale=sm, is_causal=is_causal),
            warmup=warmup, rep=rep)
        out_sage = sageattn(q_hnd, k_hnd, v_hnd, sm_scale=sm, is_causal=is_causal)
        maxd_sage, meand_sage = acc(out_sage, ref_nhd)
        result.update({
            "sage_ms": t_sage, "sage_tflops": nf / t_sage / 1e9,
            "sage_maxd": maxd_sage, "sage_meand": meand_sage,
        })
    else:
        result.update({"sage_ms": None, "sage_tflops": None,
                       "sage_maxd": None, "sage_meand": None})

    # ── Flash Attention 2 (AMD Triton) ──────────────────────────────────────
    if HAS_FA2:
        t_fa2, std_fa2 = bench(
            lambda: flash_attn_func(q_nhd, k_nhd, v_nhd,
                                    softmax_scale=sm, causal=is_causal),
            warmup=warmup, rep=rep)
        out_fa2_nhd = flash_attn_func(q_nhd, k_nhd, v_nhd,
                                      softmax_scale=sm, causal=is_causal)
        # Convert back to HND for accuracy comparison
        out_fa2 = out_fa2_nhd.permute(0, 2, 1, 3)  # [B,H,S,D]
        maxd_fa2, meand_fa2 = acc(out_fa2, ref_nhd)
        result.update({
            "fa2_ms": t_fa2, "fa2_tflops": nf / t_fa2 / 1e9,
            "fa2_maxd": maxd_fa2, "fa2_meand": meand_fa2,
        })
    else:
        result.update({"fa2_ms": None, "fa2_tflops": None,
                       "fa2_maxd": None, "fa2_meand": None})

    # ── PyTorch SDPA ────────────────────────────────────────────────────────
    t_sdpa, std_sdpa = bench(
        lambda: F.scaled_dot_product_attention(q_hnd, k_hnd, v_hnd,
                                               scale=sm, is_causal=is_causal),
        warmup=warmup, rep=rep)
    out_sdpa = F.scaled_dot_product_attention(q_hnd, k_hnd, v_hnd,
                                               scale=sm, is_causal=is_causal)
    maxd_sdpa, meand_sdpa = acc(out_sdpa, ref_nhd)
    result.update({
        "sdpa_ms": t_sdpa, "sdpa_tflops": nf / t_sdpa / 1e9,
        "sdpa_maxd": maxd_sdpa, "sdpa_meand": meand_sdpa,
    })

    # Speedup ratios
    if result["sage_ms"] and result["fa2_ms"]:
        result["sa2_vs_fa2"]  = result["fa2_ms"]  / result["sage_ms"]
        result["sa2_vs_sdpa"] = result["sdpa_ms"] / result["sage_ms"]
        result["fa2_vs_sdpa"] = result["sdpa_ms"] / result["fa2_ms"]
    elif result["sage_ms"]:
        result["sa2_vs_sdpa"] = result["sdpa_ms"] / result["sage_ms"]

    return result


# ── Table printing ────────────────────────────────────────────────────────────

def print_table(results):
    W = 175
    print("=" * W)
    print(f"{'Configuration':42s} │ {'SageAttention2 (ours)':28s} │ {'Flash Attention 2 (AMD Triton)':34s} │ {'PyTorch SDPA (CK FA)':26s} │ SA2/FA2")
    print(f"{'':42s} │ {'TFLOPS':>8s} {'ms':>7s} {'maxΔ':>8s} │ {'TFLOPS':>8s} {'ms':>7s} {'maxΔ':>8s} {'meanΔ':>8s} │ {'TFLOPS':>8s} {'ms':>7s} │")
    print("=" * W)

    nc = [r for r in results if not r["causal"]]
    ca = [r for r in results if r["causal"]]

    def print_group(rows, label):
        if not rows: return
        print(f"  ── {label} ──")
        for r in rows:
            c = "causal" if r["causal"] else "      "
            cfg = f"B={r['B']} H={r['H']:2d} S={r['S']:5d} D={r['D']:3d} {c}"

            sage_s = f"{r['sage_tflops']:8.1f} {r['sage_ms']:7.3f} {r['sage_maxd']:8.5f}" if r['sage_ms'] else f"{'N/A':>26s}"
            fa2_s  = f"{r['fa2_tflops']:8.1f} {r['fa2_ms']:7.3f} {r['fa2_maxd']:8.5f} {r['fa2_meand']:8.6f}" if r.get('fa2_ms') else f"{'N/A':>34s}"
            sdpa_s = f"{r['sdpa_tflops']:8.1f} {r['sdpa_ms']:7.3f}"
            vs_s   = f"{r.get('sa2_vs_fa2', 0):5.2f}x" if r.get('sa2_vs_fa2') else "  N/A"

            print(f"  {cfg:42s} │ {sage_s} │ {fa2_s} │ {sdpa_s} │ {vs_s}")

    print_group(nc, "Non-Causal")
    print()
    print_group(ca, "Causal")
    print("=" * W)


def print_summary(results):
    rows_with_all = [r for r in results if r.get("sage_ms") and r.get("fa2_ms")]
    if not rows_with_all:
        print("No complete results for summary.")
        return

    nc = [r for r in rows_with_all if not r["causal"]]
    ca = [r for r in rows_with_all if r["causal"]]

    print("\nSpeedup Summary  (SA2 = SageAttention2, FA2 = Flash Attention 2 AMD):")
    print(f"  {'Scenario':25s} {'SA2/FA2':>10s} {'SA2/SDPA':>10s} {'FA2/SDPA':>10s}")
    print(f"  {'─'*57}")
    for label, rows in [("Non-Causal", nc), ("Causal", ca), ("Overall", rows_with_all)]:
        if not rows: continue
        sa2_fa2  = np.mean([r["sa2_vs_fa2"]  for r in rows])
        sa2_sdpa = np.mean([r["sa2_vs_sdpa"] for r in rows])
        fa2_sdpa = np.mean([r["fa2_vs_sdpa"] for r in rows])
        print(f"  {label:25s} {sa2_fa2:10.2f}x {sa2_sdpa:10.2f}x {fa2_sdpa:10.2f}x")

    print()
    print(f"  Peak SA2 speedup vs FA2:  "
          f"{max(r['sa2_vs_fa2'] for r in rows_with_all):.2f}x "
          f"({next(r for r in rows_with_all if r['sa2_vs_fa2']==max(r['sa2_vs_fa2'] for r in rows_with_all))['S']}S "
          f"D={next(r for r in rows_with_all if r['sa2_vs_fa2']==max(r['sa2_vs_fa2'] for r in rows_with_all))['D']} "
          f"causal={next(r for r in rows_with_all if r['sa2_vs_fa2']==max(r['sa2_vs_fa2'] for r in rows_with_all))['causal']})")
    print()

    print("Accuracy vs FP32 reference:")
    sage_maxds = [r["sage_maxd"] for r in rows_with_all]
    fa2_maxds  = [r["fa2_maxd"]  for r in rows_with_all]
    sdpa_maxds = [r["sdpa_maxd"] for r in results]
    print(f"  SageAttention2: avg maxΔ = {np.mean(sage_maxds):.6f}, max maxΔ = {max(sage_maxds):.6f}")
    print(f"  Flash Attn 2:   avg maxΔ = {np.mean(fa2_maxds):.6f},  max maxΔ = {max(fa2_maxds):.6f}")
    print(f"  PyTorch SDPA:   avg maxΔ = {np.mean(sdpa_maxds):.6f}, max maxΔ = {max(sdpa_maxds):.6f}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--rep",    type=int, default=300)
    args = parser.parse_args()

    # Environment info
    print(f"{'='*60}")
    print(f" SageAttention2 vs Flash Attention 2 — RDNA4 Benchmark")
    print(f"{'='*60}")
    print(f" GPU:     {torch.cuda.get_device_name(0)}")
    print(f" ROCm:    {torch.version.hip}")
    print(f" PyTorch: {torch.__version__}")
    print(f" Triton:  {__import__('triton').__version__}")
    print(f" FA2:     {'2.8.4 (AMD Triton backend)' if HAS_FA2 else 'NOT installed'}")
    print(f" Warmup:  {args.warmup}  Rep: {args.rep}")
    print(f"{'='*60}\n")

    CONFIGS = [
        # (B,  H,    S,   D,  causal)
        # ─── head_dim=128, non-causal (image/video diffusion) ────────────
        (1,  16,   512, 128, False),
        (1,  16,  1024, 128, False),
        (1,  16,  2048, 128, False),
        (1,  16,  4096, 128, False),
        (1,  16,  8192, 128, False),
        # ─── head_dim=128, causal (LLM decode) ───────────────────────────
        (1,  16,   512, 128, True),
        (1,  16,  1024, 128, True),
        (1,  16,  2048, 128, True),
        (1,  16,  4096, 128, True),
        # ─── head_dim=64, non-causal ──────────────────────────────────────
        (1,  16,  1024,  64, False),
        (1,  16,  2048,  64, False),
        (1,  16,  4096,  64, False),
        # ─── head_dim=64, causal ─────────────────────────────────────────
        (1,  16,  1024,  64, True),
        (1,  16,  2048,  64, True),
        (1,  16,  4096,  64, True),
        # ─── multi-batch / multi-head ─────────────────────────────────────
        (2,  16,  2048, 128, False),
        (1,  32,  2048, 128, False),
        (4,  16,  1024, 128, False),
    ]

    print(f"Running {len(CONFIGS)} configurations...\n")

    results = []
    for i, (B, H, S, D, causal) in enumerate(CONFIGS):
        tag = f"B={B} H={H:2d} S={S:5d} D={D:3d} {'causal' if causal else '      '}"
        try:
            r = run_config(B, H, S, D, causal, args.warmup, args.rep)
            results.append(r)

            sa2_fa2_str = f"SA2/FA2={r.get('sa2_vs_fa2', 0):.2f}x" if r.get('sa2_vs_fa2') else ""
            print(f"  [{i+1:2d}/{len(CONFIGS)}] {tag}: "
                  f"SA2={r['sage_tflops']:.1f}T  "
                  f"FA2={r.get('fa2_tflops', 0):.1f}T  "
                  f"SDPA={r['sdpa_tflops']:.1f}T  "
                  f"{sa2_fa2_str}")
        except Exception as e:
            print(f"  [{i+1:2d}/{len(CONFIGS)}] {tag}: ERROR: {e}")
            import traceback; traceback.print_exc()

    print()
    print_table(results)
    print_summary(results)


if __name__ == "__main__":
    main()
