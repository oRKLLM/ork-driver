/* tools/gdn_opcount.c — the GDN twin of ssd_opcount.c: decide the Gated-DeltaNet-on-NPU question
 * with the MEASURED RK3588 floor cost model, BEFORE any board work ("shape of the win").
 *
 * Pure arithmetic (no NPU). Counts the matmuls in one chunked Gated-DeltaNet prefill (per value head,
 * per chunk) and applies the measured per-op costs (floor_decomp.c, 2026-07-12). The chunked GDN stage
 * list (fla chunk_gated_delta_rule / delta-net-base.cpp; d = head_dim = S_k = S_v, square d×d state):
 *   [1] kkt   A = tril(diag(b)·K·Kᵀ)      [CS,CS] K=d      chunk-parallel gram
 *   [2] kqt   scores = tril(Q·Kᵀ ⊙ mask)  [CS,CS] K=d      chunk-parallel gram
 *   [ ] solve T = (I+A)^{-1}              CS×CS            *** the ONLY non-matmul (triangular solve) ***
 *   [3] W     = T·(β·K)                   [CS,d]  K=CS      WY factor
 *   [4] U     = T·(β·V)                   [CS,d]  K=CS      WY factor
 *   [5] Vnew  = U − W·Sᵀ                  [CS,d]  K=d       carry (per chunk, seq)  BIG d×d state read
 *   [6] ointer= Q·Sᵀ                      [CS,d]  K=d       carry                   BIG d×d state read
 *   [7] ointra= (Q·Kᵀ⊙M)·Vnew            [CS,d]  K=CS      carry
 *   [8] Swr   S += Kᵀ·Vnew                [d,d]   K=CS      carry                   BIG d×d state write
 * (cumsum(g), decay_mask exp, β/gate muls are SDP/elementwise, not matmul submits.)
 *
 * KEY vs Mamba-2 SSD: GDN's state read/write [5,6,8] are d×d = 128×128 matmuls — BIGGER than SSD's ops,
 * so LESS floor-bound (better NPU efficiency), and the awkward non-GEMM piece (solve) is a small %.
 *
 *   cc -O2 -o gdn_opcount tools/gdn_opcount.c -lm && ./gdn_opcount
 */
#include <stdio.h>

/* measured RK3588 cost model (floor_decomp.c) — same as ssd_opcount.c */
#define NPU_SUBMIT_FIXED_US 48.0
#define NPU_TASK_US         16.6
#define NPU_STREAM_US       25.0
#define NPU_GMAC_INT8       120.0
#define NPU_FP16_PENALTY    3.3     /* GDN state math is fp16 (exp/decay dynamic range) */
#define CPU_GMAC_PER_S      30.0    /* CPU-favorable: fused, no per-op dispatch floor */

typedef struct { const char*name; int Hk,Hv,d,CS,layers,gdn_layers; } model;

static double npu_op_us(double macs,int fp16){
    double g = NPU_GMAC_INT8/(fp16?NPU_FP16_PENALTY:1.0);
    double compute = macs/(g*1e9)*1e6;
    return compute<NPU_TASK_US? NPU_TASK_US : compute;
}

static void analyze(model m, int L){
    int NC = (L + m.CS - 1)/m.CS;
    const int CS=m.CS, d=m.d;
    /* per (value-head, chunk) matmul MACs, by stage. gram stages [1,2] are per KEY head (Hk), shared
     * across the Hv/Hk value heads in the group; state stages are per VALUE head (Hv). */
    long macs_kkt = (long)CS*CS*d,  macs_kqt = (long)CS*CS*d;      /* per key head */
    long macs_W   = (long)CS*d*CS,  macs_U   = (long)CS*d*CS;      /* per value head */
    long macs_Vn  = (long)CS*d*d,   macs_oi  = (long)CS*d*d;       /* per value head, d×d read */
    long macs_ointra = (long)CS*d*CS;                             /* per value head */
    long macs_Swr = (long)d*d*CS;                                  /* per value head, d×d write */
    long macs_solve  = (long)d*CS*(CS-1)/2;                        /* forward-subst (NOT a matmul) */

    /* totals per (chunk) across heads */
    double gram_macs  = (double)(macs_kkt+macs_kqt)*m.Hk;
    double state_macs = (double)(macs_W+macs_U+macs_Vn+macs_oi+macs_ointra+macs_Swr)*m.Hv;
    double solve_macs = (double)macs_solve*m.Hv;
    double mm_macs    = gram_macs + state_macs;                   /* offloadable (matmul) */

    double scale = (double)NC*m.gdn_layers;
    double tot_mm    = mm_macs*scale;
    double tot_solve = solve_macs*scale;
    double tot_ew    = (double)(CS*CS + 4*CS*d)*m.Hv*scale;       /* decay_mask + muls, rough */
    double tot_all   = tot_mm + tot_solve + tot_ew;

    /* NPU (chain, 3-core): sum per-op compute (fp16) + one amortized submit per (head,chunk) chain */
    int nmm_per_hc = 8;  /* matmul submits per value-head-chunk (approx; grams shared but count 8) */
    double per_hc_npu =
        npu_op_us(macs_kkt,1)+npu_op_us(macs_kqt,1)+npu_op_us(macs_W,1)+npu_op_us(macs_U,1)+
        npu_op_us(macs_Vn,1)+npu_op_us(macs_oi,1)+npu_op_us(macs_ointra,1)+npu_op_us(macs_Swr,1);
    double npu_chain_us = (per_hc_npu + NPU_SUBMIT_FIXED_US) * m.Hv * NC * m.gdn_layers;
    double npu_chain_3c = npu_chain_us/3.0;
    /* solve stays on CPU (forward subst): CPU rate, added to NPU path as a serial cost */
    double solve_cpu_us = tot_solve/(CPU_GMAC_PER_S*1e9)*1e6;
    double npu_path_us  = npu_chain_3c + solve_cpu_us;

    double cpu_all_us   = tot_all/(CPU_GMAC_PER_S*1e9)*1e6;

    double tok_npu = L/(npu_path_us*1e-6), tok_cpu = L/(cpu_all_us*1e-6);

    printf("== %s (Hk=%d Hv=%d d=%d CS=%d, %d gdn/%d layers)  prefill L=%d (NC=%d) ==\n",
           m.name,m.Hk,m.Hv,m.d,m.CS,m.gdn_layers,m.layers,L,NC);
    printf("  MAC/op: kkt/kqt %ldK  W/U %ldK  Vnew/ointer %ldK (d×d)  ointra %ldK  Swr %ldK (d×d)  | solve %ldK\n",
           macs_kkt/1000,macs_W/1000,macs_Vn/1000,macs_ointra/1000,macs_Swr/1000,macs_solve/1000);
    printf("  offloadable-fraction: matmul %.1f%% | triangular-solve %.1f%% | elementwise %.1f%%\n",
           100.0*tot_mm/tot_all, 100.0*tot_solve/tot_all, 100.0*tot_ew/tot_all);
    printf("  breakdown: gram(kkt+kqt) %.1f%% of matmul | state(WY+carry, d×d) %.1f%% of matmul\n",
           100.0*(gram_macs*scale)/tot_mm, 100.0*(state_macs*scale)/tot_mm);
    printf("  total GDN-scan matmul MACs: %.2f G  (%d matmul submits/value-head-chunk)\n", tot_mm/1e9, nmm_per_hc);
    printf("  GDN-scan prefill: NPU(chain3c+cpusolve) %.1f ms (%.0f tok/s) | CPU %.1f ms (%.0f tok/s)\n",
           npu_path_us/1000, tok_npu, cpu_all_us/1000, tok_cpu);
    double r = cpu_all_us/npu_path_us;
    printf("  --> NPU is %.2fx %s CPU   [solve on CPU = %.0f%% of the NPU-path time]\n\n",
           r>=1?r:1.0/r, r>=1?"FASTER than":"SLOWER than", 100.0*solve_cpu_us/npu_path_us);
}

int main(void){
    printf("Gated-DeltaNet op-count & NPU-vs-CPU under the MEASURED floor model (48us/submit,16.6us/task)\n");
    printf("(GDN SCAN ONLY — q/k/v/gate/out projections are separate big dense GEMMs that belong on the NPU.)\n\n");
    /* Ornith-1.0-9B / Qwen3-Next GDN dims: Hk=16, Hv=32, head_dim=128, CS=64, 3:1 hybrid */
    model ornith = {"Ornith-1.0-9B (dense GDN)", 16,32,128,64,32,24};
    model qwen3n = {"Qwen3-Next-80B-A3B GDN",    16,32,128,64,48,36};
    analyze(ornith, 512);
    analyze(ornith, 128);
    analyze(qwen3n, 512);
    return 0;
}
