/* ws_model.c — pure-math DRAM-traffic + submit-overhead model (NO NPU / NO board).
 *
 * Decides, before any wedge-risky board work, whether a weight-STATIONARY int8 matmul
 * (CBUF-resident weight tiles reused across M via WEIGHT_REUSE, host/on-chip K-accumulate)
 * can beat the current output-stationary FOLD (M_tile=36, full weight re-streamed per M-tile).
 *
 * Grounding constants (measured / RE'd, see AGENTS.md + memory):
 *   - DRAM BW ~11 GB/s   (the 4.36 MB K=3584xN=1216 weight streams in the measured ~396 us)
 *   - CBUF   57344 int8 elems across 12 banks (~4779/bank)
 *   - fold M-tile = 36   (CBUF-capacity ceiling; the token-fold set max)
 *
 * Two dataflows can't both read weight-once AND write-output-once unless the whole op fits
 * on chip, so the candidate is weight-once + K-accumulate. We model the K-accumulate cost
 * two ways to BRACKET the opportunity:
 *   A) DRAM accumulate  (safe; partials written+reread per K-tile)   -> o = 2*nK*M*N*4
 *   B) on-chip accumulate (needs UNPROVEN cross-submit accumulator retention) -> o = M*N*4
 *
 *   cc -O2 -o ws_model ws_model.c -lm && ./ws_model
 */
#include <stdio.h>
#include <stdlib.h>

static const double BW    = 11e9;     /* DRAM bytes/s */
static const long   CBUF  = 57344;    /* int8 elems, 12 banks */
static const long   FOLD_MT = 36;

static long ceildiv(long a, long b){ return (a + b - 1) / b; }

typedef struct { double us, mb; long sub; int Kt, Nt, Mt; } res;

/* wall-clock combiner. overlap=0: serial (submit cost ADDS on top of DMA).
 * overlap=1: non-blocking/doorbell — submit issue HIDES under DMA, so the bottleneck is
 * whichever is larger, the DRAM stream or the host's submit-issue rate. */
static double wall(double bytes, long sub, double ov_us, int overlap){
    double dma = bytes/BW*1e6;
    double iss = (double)sub*ov_us;
    if(!overlap) { return dma + iss; }
    return dma > iss ? dma : iss;
}

/* output-stationary fold: full-K reduction in HW, weight re-streamed once per M-tile. */
static res fold_model(long M, long K, long N, double ov_us, int overlap){
    long nsub = ceildiv(M, FOLD_MT);
    double w = (double)nsub * K * N;      /* weight re-streamed per M-tile   */
    double a = (double)M * K;             /* A streamed once total           */
    double o = (double)M * N * 4;         /* single full-K int32 writeout    */
    double bytes = w + a + o;
    res r = { wall(bytes, nsub, ov_us, overlap), bytes/1e6, nsub, (int)K, (int)N, (int)FOLD_MT };
    return r;
}

/* weight-stationary: sweep the (Kt x Nt) resident weight tile, pick min time.
 * chip_accum=0 -> DRAM K-accumulate (safe);  chip_accum=1 -> on-chip (optimistic). */
static res ws_best(long M, long K, long N, double ov_us, int chip_accum, int overlap){
    long Wcap = 10 * CBUF / 12;   /* weight-heavy bank split: 10 of 12 banks resident weight */
    long Dcap =  2 * CBUF / 12;   /* 2 banks hold the Mt x Kt activation strip               */
    res best;
    best.us = 1e30; best.sub = 0; best.Kt = best.Nt = best.Mt = 0; best.mb = 0;
    for(long Kt = 64; Kt <= K; Kt += 64){
        if(K % Kt) { continue; }                    /* clean K tiling */
        long Nt = Wcap / Kt;
        if(Nt < 1) { continue; }
        if(Nt > N) { Nt = N; }
        long Mt = Dcap / Kt;
        if(Mt < 1) { continue; }
        if(Mt > FOLD_MT) { Mt = FOLD_MT; }
        long nK = ceildiv(K, Kt);
        long nN = ceildiv(N, Nt);
        long nM = ceildiv(M, Mt);
        double w = (double)K * N;                       /* each weight byte read ONCE (WR=1 reuse) */
        double a = (double)nN * M * K;                  /* A re-streamed once per N-tile column     */
        double o = chip_accum ? (double)M * N * 4       /* on-chip accumulate: single writeout      */
                              : (double)(2*nK) * M*N*4;  /* DRAM accumulate: write+reread per K-tile */
        double bytes = w + a + o;
        long sub = nK * nN * nM;
        double t = wall(bytes, sub, ov_us, overlap);
        if(t < best.us){
            best.us=t; best.mb=bytes/1e6; best.sub=sub;
            best.Kt=(int)Kt; best.Nt=(int)Nt; best.Mt=(int)Mt;
        }
    }
    return best;
}

static void row(long M, long K, long N, double ov, int overlap){
    res f  = fold_model(M, K, N, ov, overlap);
    res wd = ws_best(M, K, N, ov, 0, overlap);   /* DRAM accumulate (safe)      */
    res wc = ws_best(M, K, N, ov, 1, overlap);   /* on-chip accumulate (optim.) */
    printf("  M=%-5ld ov=%2.0fus | fold %7.0fus (%5.1fMB %ld sub) | "
           "WS-dram %8.0fus (%5.1fMB %ld sub) x%.2f | WS-chip %8.0fus (%5.1fMB %ld sub) x%.2f  [chip Kt=%d Nt=%d Mt=%d]\n",
           M, ov, f.us, f.mb, f.sub,
           wd.us, wd.mb, wd.sub, f.us/wd.us,
           wc.us, wc.mb, wc.sub, f.us/wc.us, wc.Kt, wc.Nt, wc.Mt);
}

int main(int argc, char**argv){
    long Ms[] = {128, 228, 512, 1024, 2048};
    double ovs[] = {0, 5, 10, 30, 60};   /* per-submit cost (us) — the decisive unknown */
    struct { long K, N; const char*name; } ops[] = {
        {3584, 1216, "fold-ref  (K=3584 N=1216, one N-slice)"},
        {3584, 3584, "ffn-down  (K=3584 N=3584, full op)"},
        {3584, 9472, "ffn-gate  (K=3584 N=9472)"},
    };
    for(int overlap=0; overlap<=1; overlap++){
        printf("\n########## %s submit model  |  ratio >1 = WS wins  |  BW=%.0fGB/s CBUF=%ld fold_Mt=%ld ##########\n",
               overlap ? "NON-BLOCKING/overlapped: wall=max(DMA, submits*cost)" : "SERIAL: wall=DMA + submits*cost",
               BW/1e9, CBUF, FOLD_MT);
        for(size_t o=0;o<sizeof ops/sizeof*ops;o++){
            printf("\n== %s ==\n", ops[o].name);
            for(size_t j=0;j<sizeof ovs/sizeof*ovs;j++){
                for(size_t i=0;i<sizeof Ms/sizeof*Ms;i++){ row(Ms[i], ops[o].K, ops[o].N, ovs[j], overlap); }
                printf("\n");
            }
        }
    }
    (void)argc;(void)argv;
    return 0;
}
