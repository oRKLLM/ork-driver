/* tools/ssd_opcount.c — decide the SSM-on-NPU question with the MEASURED floor cost model.
 *
 * Pure arithmetic (no NPU): count the matmuls in one Mamba-2/SSD prefill and apply the measured
 * RK3588 per-op costs (floor_decomp.c, 2026-07-12) to compare NPU vs CPU. Decides whether the
 * improved floor (16.6µs/task, 48µs/submit amortizable, run_stream ~25µs/op) makes the SSD scan
 * an NPU win — or whether the ops are simply too small.
 *
 * The chunked-SSD matmul stages, with GROUP-BATCHING applied where B/C are shared (HG = H/G heads
 * per group stacked along the head_dim axis):
 *   scores  G[g] = C[g]·B[g]^T      per (group,chunk)  [CS x CS], K=Nst   -> group-batched (G/chunk)
 *   cstate  Xt[g]·B[g]              per (group,chunk)  [HG*P x Nst], K=CS  -> group-batched (G/chunk)
 *   Y_off   SI[g]·C[g]^T            per (group,chunk)  [HG*P x CS], K=Nst  -> group-batched (G/chunk)
 *   Y_diag  M[h]·xbar[h]            per (HEAD,chunk)   [CS x P], K=CS      -> per-head (H/chunk), tiny
 * (cumsum/exp/segsum/mul are SDP/elementwise, not counted as matmul submits.)
 *
 *   cc -O2 -o ssd_opcount tools/ssd_opcount.c -lm && ./ssd_opcount
 */
#include <stdio.h>
#include <string.h>

/* measured RK3588 cost model (int8, warm, 1GHz), from floor_decomp.c */
#define NPU_SUBMIT_FIXED_US 48.0   /* per-ioctl NPU fixed (pipeline fill + CDMA->CBUF weight load) */
#define NPU_TASK_US         16.6   /* per-task NPU compute floor (irreducible) */
#define NPU_STREAM_US       25.0   /* run_stream 3-core measured per-op wall (independent tiny matmuls) */
/* NPU effective int8 GMAC/s from floor_decomp: M=512,K=512,N=64 -> 132.9us/16.8M = 126 GMAC/s;
 * M=64,K=2048,N=64 -> 88us/8.4M = 95 GMAC/s. Use 120. fp16 (needed for exp/decay dynamic range) ~3.3x
 * slower (int8-fp16 memory). SSD scan state/decay math is fp16 => big ops pay the fp16 penalty. */
#define NPU_GMAC_INT8       120.0
#define NPU_FP16_PENALTY    3.3
/* A76 multicore NEON compute-bound int8 GEMM on cache-resident tiles: ~8 int8 MAC/cyc/core * 2.4GHz
 * * 4 big cores ~= 77 GMAC/s peak; realistic fused ~30-40. Use a CPU-FAVORABLE-to-NPU 30 GMAC/s.
 * CPU FUSES the whole head-batch in one loop => NO per-op dispatch floor. */
#define CPU_GMAC_PER_S      30.0

typedef struct { const char*name; int H,P,Nst,G,CS,layers; } model;

/* NPU time for ONE matmul of `macs`, `tiny`=too-small-for-efficiency (floor-bound), fp16 if state-math */
static double npu_op_us(double macs,int fp16){
    double g = NPU_GMAC_INT8/(fp16?NPU_FP16_PENALTY:1.0);
    double compute = macs/(g*1e9)*1e6;
    return compute<NPU_TASK_US? NPU_TASK_US : compute;   /* floor-bound below ~16.6us */
}

static void analyze(model m, int L){
    int NC = (L + m.CS - 1)/m.CS;
    int HG = m.H/m.G; if(HG<1)HG=1;
    /* per (layer,chunk) matmul counts + MAC each */
    long n_scores = m.G,          macs_scores = (long)m.CS*m.CS*m.Nst;        /* [CS,CS] K=Nst */
    long n_cstate = m.G,          macs_cstate = (long)HG*m.P*m.Nst*m.CS;      /* [HG*P,Nst] K=CS */
    long n_yoff   = m.G,          macs_yoff   = (long)HG*m.P*m.CS*m.Nst;      /* [HG*P,CS] K=Nst */
    long n_ydiag  = m.H,          macs_ydiag  = (long)m.CS*m.P*m.CS;          /* [CS,P] K=CS (tiny, per head) */

    long per_lc_matmuls = n_scores+n_cstate+n_yoff+n_ydiag;
    long total_matmuls  = per_lc_matmuls * NC * m.layers;
    long tiny_matmuls   = n_ydiag * NC * m.layers;   /* the per-head Y_diag ones */
    long grp_matmuls    = (n_scores+n_cstate+n_yoff) * NC * m.layers;

    /* NPU time per (layer,chunk): sum the actual per-op compute (fp16 for decay-bearing ops), plus
     * ONE amortized 48us submit-fixed if the whole (layer,chunk) batch is chained into one ioctl. */
    double per_lc_npu =
        npu_op_us(macs_scores,1)*n_scores +   /* scores: G=C·B^T, plain matmul (int8 ok) but small */
        npu_op_us(macs_cstate,1)*n_cstate +   /* cstate: fp16 (decay-weighted) */
        npu_op_us(macs_yoff,  1)*n_yoff   +   /* Yoff: fp16 (decay) */
        npu_op_us(macs_ydiag, 1)*n_ydiag;     /* Ydiag: tiny, floor-bound */
    double npu_chain_us  = (per_lc_npu + NPU_SUBMIT_FIXED_US) * NC * m.layers;   /* /3 cores below */
    double npu_chain_3c  = npu_chain_us/3.0;   /* 3-core: independent (group,chunk,head) ops overlap */
    double npu_stream_us = NPU_STREAM_US * total_matmuls / 1.0;   /* stream already 3-core (25us/op wall) */

    /* CPU: total MACs / rate, FUSED (no per-op floor — the CPU's structural advantage on tiny batched ops) */
    double total_macs = (double)macs_scores*n_scores + (double)macs_cstate*n_cstate +
                        (double)macs_yoff*n_yoff + (double)macs_ydiag*n_ydiag;
    total_macs *= (double)NC*m.layers;
    double cpu_total_us = total_macs/(CPU_GMAC_PER_S*1e9)*1e6;

    double tok_npu_chain  = L / (npu_chain_3c*1e-6);
    double tok_npu_stream = L / (npu_stream_us*1e-6);
    double tok_cpu        = L / (cpu_total_us*1e-6);

    printf("== %s (H=%d P=%d N=%d G=%d CS=%d, %d layers)  prefill L=%d (NC=%d) ==\n",
           m.name,m.H,m.P,m.Nst,m.G,m.CS,m.layers,L,NC);
    printf("  matmuls/(layer,chunk): scores %ld + cstate %ld + Yoff %ld (group-batched, M=%d) + Ydiag %ld (per-head, tiny)\n",
           n_scores,n_cstate,n_yoff,HG*m.P,n_ydiag);
    printf("  total SSD-scan matmuls: %ld  (group-batched %ld / tiny per-head %ld = %.0f%% tiny)\n",
           total_matmuls, grp_matmuls, tiny_matmuls, 100.0*tiny_matmuls/total_matmuls);
    printf("  MAC/op: scores %ldK  cstate %ldK  Yoff %ldK  Ydiag %ldK\n",
           macs_scores/1000,macs_cstate/1000,macs_yoff/1000,macs_ydiag/1000);
    printf("  Ydiag on NPU per-op: %.1fus floor for %ldK MACs (fits floor) — the tiny per-head op\n",
           NPU_TASK_US, macs_ydiag/1000);
    printf("  total SSD-scan MACs: %.2f G  (NPU@%.0f/CPU@%.0f GMAC/s int8; big ops fp16 x%.1f)\n",
           total_macs/1e9, NPU_GMAC_INT8, CPU_GMAC_PER_S, NPU_FP16_PENALTY);
    printf("  SSD-scan prefill:  NPU(chain,3-core) %.1f ms (%.0f tok/s) | NPU(stream) %.1f ms (%.0f tok/s) | CPU %.1f ms (%.0f tok/s)\n",
           npu_chain_3c/1000, tok_npu_chain, npu_stream_us/1000, tok_npu_stream, cpu_total_us/1000, tok_cpu);
    double ratio = cpu_total_us/npu_chain_3c;
    printf("  --> best-NPU(chain,3-core) is %.2fx %s CPU\n\n", ratio>=1?ratio:1.0/ratio, ratio>=1?"FASTER than":"SLOWER than");
}

int main(void){
    printf("SSD-scan op-count & NPU-vs-CPU under the MEASURED floor model (48us/submit + 16.6us/task; stream 25us/op)\n");
    printf("(SSD SCAN ONLY — in/out projections are separate big dense GEMMs that DO belong on the NPU.)\n\n");
    /* representative dims (label as approximate; verdict is robust to exact values) */
    model m130 = {"mamba2-130m", 24,64,128,1,64,24};
    model m7b  = {"Codestral-7B(approx)",128,64,128,8,64,64};
    analyze(m130, 512);
    analyze(m7b,  512);
    analyze(m7b,  128);
    return 0;
}
