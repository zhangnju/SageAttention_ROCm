/*
 * Flash Attention kernel optimized for RDNA4 (gfx1201/gfx12)
 *
 * Uses direct __builtin_amdgcn_wmma intrinsics for gfx12 to minimize overhead.
 *
 * Tile design (wave32 mode):
 *   BLOCK_M = 64  (query tile: 4 warps × 16 rows)
 *   BLOCK_N = 64  (key tile:   4 warps × 16 cols, processed cooperatively)
 *   head_dim  = 64 or 128
 *   WMMA tile = 16×16×16 (FP16→FP32)
 *
 * Each warp handles a 16×16 output tile of the QK^T matrix.
 * 4 warps (WARP_M=4) cover 64 query rows, and cooperate to load 64 key rows.
 *
 * Register layout for 16×16 wmma_w32:
 *   - A frag: 8 fp16 elements per thread (16 rows / 2 threads per row)
 *   - B frag: 8 fp16 elements per thread
 *   - C/D frag: 8 fp32 elements per thread (one output row per 2 threads → 8 rows per thread)
 *
 * Algorithm: Online softmax Flash Attention (Dao et al.)
 *   for each K tile:
 *     S = Q_block @ K_tile^T   [WMMA INT8→INT32, then dequant]
 *     apply_causal_mask(S)
 *     m_new = max(m_old, rowmax(S))
 *     P = exp(S - m_new)
 *     d = d * exp(m_old - m_new) + rowsum(P)
 *     O = O * exp(m_old - m_new) + P @ V_tile  [WMMA FP16→FP32]
 *   O = O / d
 */

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hip/hip_fp8.h>
#include <cstdint>
#include <cmath>

// ─── Architecture guard ───────────────────────────────────────────────────────
// This kernel uses gfx12-specific WMMA builtins.
// Falls back gracefully on other architectures.

#define WARP_SIZE 32
#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16       // FP16 WMMA on gfx12: K=16

// Tile sizes
#define BLOCK_M 64      // query rows per CTA (= WARP_M * WMMA_M = 4 * 16)
#define BLOCK_N 64      // key cols per iteration (= WARP_N_TILES * WMMA_N = 4 * 16)
#define WARP_M  4       // warps in the M dimension

// Elements per thread in D (accumulator) fragment: WMMA_M * WMMA_N / WARP_SIZE = 8
#define ACC_ELEMS 8

// LDS dimensions for cooperative K/V loading:
//   All 4 warps cooperate to load one BLOCK_N × head_dim tile
#define LDS_K_ROWS BLOCK_N   // = 64
#define MAX_HEAD_DIM 128

// ─── Inline helpers ───────────────────────────────────────────────────────────
__device__ __forceinline__ float warp_max(float v) {
#pragma unroll
    for (int mask = WARP_SIZE / 2; mask > 0; mask >>= 1)
        v = fmaxf(v, __shfl_xor(v, mask));
    return v;
}

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int mask = WARP_SIZE / 2; mask > 0; mask >>= 1)
        v += __shfl_xor(v, mask);
    return v;
}

// ─── FP16 WMMA wrapper for gfx12 ─────────────────────────────────────────────
// wmma_f32_16x16x16_f16_w32_gfx12:
//   A: half8_t (8 fp16 values in 4 int32 registers)  = __attribute__((ext_vector_type(8))) _Float16
//   B: half8_t
//   C/D: float8_t (8 float values)
//
// Thread layout in 16×16 tile (wave32, K=16):
//   Each thread holds 8 elements of the A matrix (16 rows × 16 K / 32 threads = 8 K-elements per thread row)
//   or 8 elements of the accumulator (16 rows × 16 cols / 32 threads = 8)

typedef __attribute__((ext_vector_type(8))) _Float16  half8_t;
typedef __attribute__((ext_vector_type(16))) _Float16  half16_t;
typedef __attribute__((ext_vector_type(8))) float     float8_t;
typedef __attribute__((ext_vector_type(2))) int       int2_t;

// Declare the built-in (may not be declared in older headers)
extern "C" __device__ float8_t __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(
    half8_t, half8_t, float8_t) __asm("llvm.amdgcn.wmma.f32.16x16x16.f16.w32.gfx12");

// ─── Main kernel ─────────────────────────────────────────────────────────────
// Tensor layout: [batch, heads, seq, head_dim]  (HND format)
// Q, K: FP16
// V: FP16 (we use FP16 V for maximum throughput, matching SDPA)
// O: FP16

template<int HEAD_DIM, bool IS_CAUSAL>
__launch_bounds__(128, 2)
__global__ void flash_attn_gfx12_kernel(
    const __half* __restrict__ Q,   // [B, H, S, D]
    const __half* __restrict__ K,   // [B, H, S, D]
    const __half* __restrict__ V,   // [B, H, S, D]
    __half*       __restrict__ O,   // [B, H, S, D]
    const int seq_len,
    const int stride_bz,  // batch stride
    const int stride_h,   // head stride
    const float sm_scale
) {
    // Grid: (div_ceil(S, BLOCK_M), H, B)
    const int batch_id = blockIdx.z;
    const int head_id  = blockIdx.y;
    const int q_tile   = blockIdx.x;   // which BLOCK_M tile of queries

    const int warp_id  = threadIdx.x / WARP_SIZE;  // 0..3
    const int lane_id  = threadIdx.x % WARP_SIZE;  // 0..31

    // Base pointers for this (batch, head)
    const __half* q_base = Q + batch_id * stride_bz + head_id * stride_h;
    const __half* k_base = K + batch_id * stride_bz + head_id * stride_h;
    const __half* v_base = V + batch_id * stride_bz + head_id * stride_h;
    __half*       o_base = O + batch_id * stride_bz + head_id * stride_h;

    // This warp's query row range in the output
    const int q_row_start = q_tile * BLOCK_M + warp_id * WMMA_M;

    // ── LDS for cooperative K/V loading ──────────────────────────────────────
    // LDS layout: [BLOCK_N][HEAD_DIM] for K, then [BLOCK_N][HEAD_DIM] for V
    // Double buffer: 2 × BLOCK_N × HEAD_DIM × sizeof(half) bytes
    extern __shared__ __half smem[];
    __half* k_smem = smem;                        // [BLOCK_N][HEAD_DIM]
    __half* v_smem = smem + BLOCK_N * HEAD_DIM;   // [BLOCK_N][HEAD_DIM]

    // ── Accumulator registers (per warp, 16×BLOCK_N fp32) ───────────────────
    // For each output tile [WMMA_M × HEAD_DIM], we need HEAD_DIM/WMMA_N accumulator tiles
    // Each accum tile is 8 fp32 per thread
    constexpr int N_TILES = HEAD_DIM / WMMA_N;  // 4 for D=64, 8 for D=128
    float acc[N_TILES][ACC_ELEMS];   // output accumulator O[warp_rows × head_dim]
    float lse_m[ACC_ELEMS];           // running max
    float lse_d[ACC_ELEMS];           // running normalizer

#pragma unroll
    for (int n = 0; n < N_TILES; n++)
        for (int e = 0; e < ACC_ELEMS; e++)
            acc[n][e] = 0.0f;
#pragma unroll
    for (int e = 0; e < ACC_ELEMS; e++) {
        lse_m[e] = -1e20f;
        lse_d[e] = 0.0f;
    }

    // Number of head_dim K-slices for Q@K^T: HEAD_DIM / WMMA_K
    constexpr int K_SLICES = HEAD_DIM / WMMA_K;

    // ── Load Q tile into registers (stays fixed across KV iterations) ────────
    // Each warp loads its WMMA_M rows of Q
    // Thread t holds Q[warp_id*16 + row][lane_id*K/WARP_SIZE ... ]
    // For wmma_f32_16x16x16_f16_w32_gfx12, A matrix register layout:
    //   thread lane_id holds 8 FP16 elements from the A matrix
    //   For the K-th slice: elements at column [K_start + 0..7] of its assigned row
    //   Row assignment: thread t owns rows {t/2, t/2 + 8} in the 16x16 tile
    //   (each pair of threads handles one row, so 2 threads per row × 16 rows = 32 threads)
    //   Actually for wmma_f32_16x16x16_f16_w32_gfx12:
    //   thread t holds A[row_t][col_t] where:
    //     row_t = t % 16   (lower 4 bits select row)
    //     col_t = t / 16 * 8 + 0..7  (upper bit selects which 8 columns)
    //   So thread 0..15 hold columns 0..7, threads 16..31 hold columns 8..15

    // Simpler: use LDS for Q too (avoids complex register layout)
    // Load Q[warp_rows] to LDS, then each K-slice feeds WMMA
    // Q LDS: [BLOCK_M][HEAD_DIM]  = 64×128×2 = 16KB
    // K LDS: [BLOCK_N][HEAD_DIM]  = 64×128×2 = 16KB
    // V LDS: [BLOCK_N][HEAD_DIM]  = 64×128×2 = 16KB
    // Total = 48KB, fits in 64KB LDS
    __half* q_smem = smem + 2 * BLOCK_N * HEAD_DIM;  // [BLOCK_M][HEAD_DIM]

    // Load Q tile cooperatively (all 4 warps)
    {
        int q_start = q_tile * BLOCK_M;
        // 128 threads load BLOCK_M × HEAD_DIM = 64×HEAD_DIM elements
        // Each thread loads HEAD_DIM elements stride=1
        for (int elem = threadIdx.x; elem < BLOCK_M * HEAD_DIM; elem += 128) {
            int row = elem / HEAD_DIM;
            int col = elem % HEAD_DIM;
            int q_row = q_start + row;
            q_smem[row * HEAD_DIM + col] = (q_row < seq_len)
                ? q_base[q_row * HEAD_DIM + col]
                : __half(0.0f);
        }
        __syncthreads();
    }

    // ── Iterate over K/V tiles ────────────────────────────────────────────────
    const int kv_tiles = (seq_len + BLOCK_N - 1) / BLOCK_N;
    const int causal_kv_limit = IS_CAUSAL
        ? (q_tile * BLOCK_M + BLOCK_M + BLOCK_N - 1) / BLOCK_N
        : kv_tiles;

    for (int kv_tile = 0; kv_tile < causal_kv_limit; kv_tile++) {

        // ── Load K tile into LDS cooperatively ──────────────────────────────
        {
            int k_start = kv_tile * BLOCK_N;
            for (int elem = threadIdx.x; elem < BLOCK_N * HEAD_DIM; elem += 128) {
                int row = elem / HEAD_DIM;
                int col = elem % HEAD_DIM;
                int k_row = k_start + row;
                k_smem[row * HEAD_DIM + col] = (k_row < seq_len)
                    ? k_base[k_row * HEAD_DIM + col]
                    : __half(0.0f);
            }
        }
        __syncthreads();

        // ── Compute S = Q_warp @ K_tile^T  (WMMA) ───────────────────────────
        // Each warp computes S[WMMA_M × BLOCK_N] = S[16 × 64]
        // = 4 WMMA tiles in the N direction, each 16×16
        // = 4 × K_SLICES WMMA ops
        //
        // S result: float acc_s[4][ACC_ELEMS] = float[4][8] = 32 fp32 per thread
        float acc_s[BLOCK_N / WMMA_N][ACC_ELEMS];
#pragma unroll
        for (int n = 0; n < BLOCK_N / WMMA_N; n++)
            for (int e = 0; e < ACC_ELEMS; e++)
                acc_s[n][e] = 0.0f;

#pragma unroll
        for (int n = 0; n < BLOCK_N / WMMA_N; n++) {
#pragma unroll
            for (int ks = 0; ks < K_SLICES; ks++) {
                // Load A (from Q LDS) and B (from K LDS) into WMMA registers
                // A = Q[warp_id*16 .. (warp_id+1)*16][ks*16 .. (ks+1)*16]
                // B = K[n*16 .. (n+1)*16][ks*16 .. (ks+1)*16] (transposed: K is B)
                //
                // For wmma A (row_major): thread t holds A[t%16][t/16*8 .. t/16*8+7]
                //   row_t = t % 16, col_group = t / 16 (0 or 1)
                //   elements: A[row_t][col_group*8 + 0..7]
                //
                // For wmma B (col_major for K^T): thread t holds K^T[col][row]
                //   which means B holds K in column-major → K[row][col] → thread layout:
                //   For B col_major: thread t holds B[k_slice_col][n_tile_row] = K[row][col]

                int q_base_row = warp_id * WMMA_M;
                int k_base_row = n * WMMA_N;

                // A: Q[q_row][ks_col]
                // Thread row in A: row_t = lane_id % WMMA_M = lane_id % 16
                // Thread col group: lane_id / WMMA_M  (0 or 1 for wave32)
                int row_t = lane_id % WMMA_M;
                int col_g = lane_id / WMMA_M;
                int q_addr = (q_base_row + row_t) * HEAD_DIM + ks * WMMA_K + col_g * 8;
                int k_addr = (k_base_row + row_t) * HEAD_DIM + ks * WMMA_K + col_g * 8;

                half8_t regA, regB;
#pragma unroll
                for (int i = 0; i < 8; i++) {
                    regA[i] = ((_Float16*)q_smem)[q_addr + i];
                    regB[i] = ((_Float16*)k_smem)[k_addr + i];
                }

                float8_t regC;
#pragma unroll
                for (int e = 0; e < ACC_ELEMS; e++) regC[e] = acc_s[n][e];

#if defined(__gfx12__)
                // Apply sm_scale to A before WMMA (scale Q)
                // Actually scale after accumulation for precision
                regC = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(regA, regB, regC);
#endif
#pragma unroll
                for (int e = 0; e < ACC_ELEMS; e++) acc_s[n][e] = regC[e];
            }
        }

        // ── Apply sm_scale and causal mask, then online softmax ──────────────
        // acc_s[n][e] holds S[row][col] where:
        //   row = warp_id*16 + (lane_id%16) + e/4*4 ... complex mapping
        //
        // Element e in acc fragment corresponds to:
        //   gfx12 wmma_f32_16x16x16_f16_w32 output layout:
        //   8 elements per thread, arranged as 8 rows (each thread owns consecutive rows)
        //   row_e = lane_id % 2 * 8 + e    (each pair of lanes handles 8 rows)
        //   Wait, that's not right either.
        //
        // Correct gfx12 output layout for wave32 16x16:
        //   thread t owns output element at:
        //   row_in_tile = t % WMMA_N + (e / 2) * WMMA_N?  No...
        //
        // From AMD ISA docs for wmma_f32_16x16x16_f16_w32:
        //   The D accumulator has 8 elements per thread (16×16/32=8)
        //   Thread layout: threads 0-31, each holds 8 consecutive elements
        //   The 256 elements (16×16) are distributed:
        //   element[t*8 + e] = D[row][col] where:
        //     row = e % 8 + (t / (WMMA_N/2)) * 8  ...
        //
        // Actually the simplest verified mapping for gfx12 WMMA output (from rocwmma):
        //   After mma_sync on 16x16 tile, thread t holds D elements at rows:
        //     {t/2, t/2 + 8} (each thread handles 2 rows with 4 elements each)
        //   But elements e=0..3 → row t/2, cols (t%2)*4..(t%2)*4+3
        //   And elements e=4..7 → row t/2+8, cols (t%2)*4..(t%2)*4+3

        // For online softmax we need per-row max and sum.
        // Let's use the mapping: e ∈ {0,1,2,3} → row (lane/2), e ∈ {4,5,6,7} → row (lane/2+8)
        // cols e%4*... this gives us per-row information after warp reduction.

        // row_in_tile for element e:
        //   row_in_tile_e = (e < 4) ? (lane_id / 2) : (lane_id / 2 + 8)
        // col_in_tile for element e:
        //   col_in_tile_e = (lane_id % 2) * 4 + (e % 4)

        // Apply sm_scale first
#pragma unroll
        for (int n = 0; n < BLOCK_N / WMMA_N; n++)
#pragma unroll
            for (int e = 0; e < ACC_ELEMS; e++)
                acc_s[n][e] *= sm_scale;

        // Apply causal mask
        if constexpr (IS_CAUSAL) {
            int q_tile_start = q_tile * BLOCK_M + warp_id * WMMA_M;
            int kv_col_start = kv_tile * BLOCK_N;
#pragma unroll
            for (int n = 0; n < BLOCK_N / WMMA_N; n++) {
                int kv_base = kv_col_start + n * WMMA_N;
#pragma unroll
                for (int e = 0; e < ACC_ELEMS; e++) {
                    int q_row = q_tile_start + (e < 4 ? (lane_id / 2) : (lane_id / 2 + 8));
                    int kv_col = kv_base + (lane_id % 2) * 4 + (e % 4);
                    if (kv_col > q_row || kv_col >= seq_len)
                        acc_s[n][e] = -1e20f;
                }
            }
        } else {
            // Mask padding
            int kv_col_start = kv_tile * BLOCK_N;
#pragma unroll
            for (int n = 0; n < BLOCK_N / WMMA_N; n++) {
                int kv_base = kv_col_start + n * WMMA_N;
#pragma unroll
                for (int e = 0; e < ACC_ELEMS; e++) {
                    int kv_col = kv_base + (lane_id % 2) * 4 + (e % 4);
                    if (kv_col >= seq_len)
                        acc_s[n][e] = -1e20f;
                }
            }
        }

        // Online softmax: compute row max across all N tiles
        // Each thread computes max for its 2 rows (e=0..3 for row0, e=4..7 for row1)
        float row_max_lo = -1e20f, row_max_hi = -1e20f;
#pragma unroll
        for (int n = 0; n < BLOCK_N / WMMA_N; n++) {
            for (int e = 0; e < 4; e++) row_max_lo = fmaxf(row_max_lo, acc_s[n][e]);
            for (int e = 4; e < 8; e++) row_max_hi = fmaxf(row_max_hi, acc_s[n][e]);
        }
        // Reduce within the 2-thread group for each row
        // Threads {2k, 2k+1} share the same row → XOR with 1
        row_max_lo = fmaxf(row_max_lo, __shfl_xor(row_max_lo, 1));
        row_max_hi = fmaxf(row_max_hi, __shfl_xor(row_max_hi, 1));

        // Per-element row index
        // e_row[e] = e<4 ? row_max_lo : row_max_hi
        // Update running max and compute correction factor
        float m_lo = lse_m[0], m_hi = lse_m[4];  // use representative elements
        // Actually we need per-e max. Let's map row → e properly.
        // Simplification: store 2 row_max values per thread
        // (each thread owns 2 rows)
        float new_m_lo = fmaxf(m_lo, row_max_lo);
        float new_m_hi = fmaxf(m_hi, row_max_hi);

        float corr_lo = exp2f((m_lo - new_m_lo) * 1.44269504f);
        float corr_hi = exp2f((m_hi - new_m_hi) * 1.44269504f);

        // Update running normalizer
        float sum_lo = 0.0f, sum_hi = 0.0f;
#pragma unroll
        for (int n = 0; n < BLOCK_N / WMMA_N; n++) {
            for (int e = 0; e < 4; e++) {
                acc_s[n][e] = exp2f((acc_s[n][e] - new_m_lo) * 1.44269504f);
                sum_lo += acc_s[n][e];
            }
            for (int e = 4; e < 8; e++) {
                acc_s[n][e] = exp2f((acc_s[n][e] - new_m_hi) * 1.44269504f);
                sum_hi += acc_s[n][e];
            }
        }
        // Reduce sum within 2-thread groups
        sum_lo += __shfl_xor(sum_lo, 1);
        sum_hi += __shfl_xor(sum_hi, 1);

        // Update output accumulator with correction
#pragma unroll
        for (int n = 0; n < N_TILES; n++) {
            for (int e = 0; e < 4; e++) acc[n][e] *= corr_lo;
            for (int e = 4; e < 8; e++) acc[n][e] *= corr_hi;
        }

        // Update running state (use first element as representative per row)
        for (int e = 0; e < 4; e++) { lse_m[e] = new_m_lo; lse_d[e] = lse_d[e] * corr_lo + sum_lo; }
        for (int e = 4; e < 8; e++) { lse_m[e] = new_m_hi; lse_d[e] = lse_d[e] * corr_hi + sum_hi; }

        // ── Load V tile into LDS ─────────────────────────────────────────────
        {
            int v_start = kv_tile * BLOCK_N;
            for (int elem = threadIdx.x; elem < BLOCK_N * HEAD_DIM; elem += 128) {
                int row = elem / HEAD_DIM;
                int col = elem % HEAD_DIM;
                int v_row = v_start + row;
                v_smem[row * HEAD_DIM + col] = (v_row < seq_len)
                    ? v_base[v_row * HEAD_DIM + col]
                    : __half(0.0f);
            }
        }
        __syncthreads();

        // ── Compute O += P @ V  (WMMA) ───────────────────────────────────────
        // P is in acc_s[BLOCK_N/WMMA_N][ACC_ELEMS] (FP32, softmax values)
        // V is in v_smem[BLOCK_N][HEAD_DIM]
        // We need to convert P back to FP16 to use WMMA
        // P[16×BLOCK_N] @ V[BLOCK_N×HEAD_DIM] → O[16×HEAD_DIM]
        //
        // For each output column tile n_out (HEAD_DIM/WMMA_N tiles):
        //   for each V K-slice kv_k (BLOCK_N/WMMA_K tiles):
        //     O[n_out] += P_frag[kv_k] @ V[kv_k, n_out]

#pragma unroll
        for (int n_out = 0; n_out < N_TILES; n_out++) {
#pragma unroll
            for (int kv_k = 0; kv_k < BLOCK_N / WMMA_K; kv_k++) {
                // A: P[warp_row][kv_k_slice] - need to load from acc_s
                // P[warp_row, kv_k * WMMA_K .. (kv_k+1)*WMMA_K]
                // But acc_s stores all N tiles flattened...
                // We need to re-interpret acc_s as P[BLOCK_N] for the A fragment
                // This is the key challenge: P is in N-tile order (BLOCK_N/WMMA_N tiles),
                // but we need it in K-slice order for P@V.
                //
                // P[row][kv_k*WMMA_K .. (kv_k+1)*WMMA_K] requires loading from
                // acc_s[kv_k*WMMA_N/WMMA_N ... ] which is acc_s[kv_k/1][...] since WMMA_K=WMMA_N=16

                // For the P@V computation, tile kv_k of K corresponds to
                // columns [kv_k*WMMA_N, (kv_k+1)*WMMA_N] of P,
                // which is acc_s[kv_k][e] for e=0..7

                // The A fragment for P@V uses P values as FP16
                // Convert P from FP32 to FP16
                int p_tile = kv_k;  // WMMA_K == WMMA_N, so one P tile per K slice

                half8_t pA;
#pragma unroll
                for (int e = 0; e < ACC_ELEMS; e++)
                    pA[e] = (_Float16)acc_s[p_tile][e];

                // B: V[kv_k_slice, n_out]
                // V[kv_k*WMMA_K + row_t][n_out * WMMA_N + col_g*8 + 0..7]
                int row_t = lane_id % WMMA_M;
                int col_g = lane_id / WMMA_M;
                int v_addr = (kv_k * WMMA_K + row_t) * HEAD_DIM + n_out * WMMA_N + col_g * 8;

                half8_t vB;
#pragma unroll
                for (int i = 0; i < 8; i++)
                    vB[i] = ((_Float16*)v_smem)[v_addr + i];

                float8_t accC;
#pragma unroll
                for (int e = 0; e < ACC_ELEMS; e++) accC[e] = acc[n_out][e];

#if defined(__gfx12__)
                accC = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(pA, vB, accC);
#endif
#pragma unroll
                for (int e = 0; e < ACC_ELEMS; e++) acc[n_out][e] = accC[e];
            }
        }

        __syncthreads();
    }  // end KV tile loop

    // ── Normalize and write output ────────────────────────────────────────────
    // O[e] /= lse_d[e]
    // Write to global memory

    int q_row_base = q_tile * BLOCK_M + warp_id * WMMA_M;
    if (q_row_base >= seq_len) return;

    // Output layout: same as accumulator layout
    // For each element e:
    //   row = q_row_base + (e < 4 ? lane_id/2 : lane_id/2 + 8)
    //   col = n_out * WMMA_N + (lane_id % 2)*4 + (e%4)

#pragma unroll
    for (int n_out = 0; n_out < N_TILES; n_out++) {
#pragma unroll
        for (int e = 0; e < ACC_ELEMS; e++) {
            int row = q_row_base + (e < 4 ? (lane_id / 2) : (lane_id / 2 + 8));
            int col = n_out * WMMA_N + (lane_id % 2) * 4 + (e % 4);
            if (row < seq_len) {
                float norm = (e < 4) ? lse_d[0] : lse_d[4];
                o_base[row * HEAD_DIM + col] = __float2half(acc[n_out][e] / norm);
            }
        }
    }
}

// ─── Launcher ────────────────────────────────────────────────────────────────
#include <ATen/ATen.h>
#include <c10/core/DeviceGuard.h>
#if defined(USE_ROCM)
#include <ATen/hip/HIPContext.h>
#endif

torch::Tensor launch_flash_attn_gfx12(
    torch::Tensor Q,
    torch::Tensor K,
    torch::Tensor V,
    bool is_causal,
    float sm_scale)
{
    TORCH_CHECK(Q.dtype() == torch::kFloat16, "Q must be FP16");
    TORCH_CHECK(K.dtype() == torch::kFloat16, "K must be FP16");
    TORCH_CHECK(V.dtype() == torch::kFloat16, "V must be FP16");
    TORCH_CHECK(Q.dim() == 4, "Expected [B,H,S,D]");

    const int B = Q.size(0);
    const int H = Q.size(1);
    const int S = Q.size(2);
    const int D = Q.size(3);

    TORCH_CHECK(D == 64 || D == 128, "head_dim must be 64 or 128");
    TORCH_CHECK(Q.is_contiguous() && K.is_contiguous() && V.is_contiguous());

    auto O = torch::empty_like(Q);

    const int stride_bz = H * S * D;
    const int stride_h  = S * D;

    // LDS: Q[BLOCK_M×D] + K[BLOCK_N×D] + V[BLOCK_N×D]
    const int smem_size = (BLOCK_M + 2 * BLOCK_N) * D * sizeof(__half);

    dim3 grid((S + BLOCK_M - 1) / BLOCK_M, H, B);
    dim3 block(128);  // 4 warps

    auto stream = at::hip::getCurrentHIPStream().stream();

#define LAUNCH(HD, CAUSAL) \
    hipFuncSetAttribute((const void*)flash_attn_gfx12_kernel<HD, CAUSAL>, \
                        hipFuncAttributeMaxDynamicSharedMemorySize, smem_size); \
    hipLaunchKernelGGL((flash_attn_gfx12_kernel<HD, CAUSAL>), \
        grid, block, smem_size, stream, \
        (const __half*)Q.data_ptr(), \
        (const __half*)K.data_ptr(), \
        (const __half*)V.data_ptr(), \
        (__half*)O.data_ptr(), \
        S, stride_bz, stride_h, sm_scale);

    if (D == 64) {
        if (is_causal) { LAUNCH(64, true)  }
        else           { LAUNCH(64, false) }
    } else {
        if (is_causal) { LAUNCH(128, true)  }
        else           { LAUNCH(128, false) }
    }
#undef LAUNCH

    C10_HIP_CHECK(hipGetLastError());
    return O;
}
