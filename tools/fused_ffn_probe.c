/* fused_ffn_probe — measure the CHAINED on-NPU FFN inner that keeps int8 intermediates on the NPU
 * (no fp32 round-trip / CPU op between the matmuls), vs the round-trip style.
 *
 *   CHAINED (on-NPU, int8 throughout):
 *     gate matmul + fused SiLU  (ork_mm_run_i8_silu)   -> silu_gate int8
 *     up   matmul + int8-out    (ork_mm_run_i8_out8)   -> up        int8
 *     silu_gate (x) up          (ork_npu_ewmul_i8)     -> glu       int8
 *     down matmul (int8-in)     (ork_mm_run_i8)        -> out       int32
 *
 * Times the chain warm + the per-op split, and (for the monitor harness) loops so NPU/CPU/RAM-BW can be
 * sampled. Dims default to a small FFN (K=2048, D_ff=2048) so down's K stays <=4096 (the full-K envelope).
 * Build with the CORE srcs; run: sudo ./fused_ffn_probe [M] [iters]
 */
#include "ork_npu.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <pthread.h>
#include <math.h>

static double now_us(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec*1e6 + t.tv_nsec/1e3; }

/* representative CPU per-token work that would run alongside the NPU FFN in a real layer (attention
 * scores / norm / activation-quant are all fp32 memory-bound sweeps). A tunable fp32 FMA sweep. */
static volatile float g_sink;
static void cpu_work(float *buf, int n, int reps){
    float acc=0;
    for(int r=0;r<reps;r++) for(int i=0;i<n;i++) acc = acc*1.0000001f + buf[i]*0.9999999f;
    g_sink = acc;
}

/* ---- pipeline threading: one thread runs the NPU FFN inner `iters` times ---- */
struct npu_arg { ork_npu*c; ork_w*wg,*wu,*wd; int M,Dff; const int8_t*x; int8_t*silu_gate,*up,*glu; int32_t*out; int iters; };
static void* npu_thread(void*a){
    struct npu_arg*p=a; double us=0;
    for(int it=0; it<p->iters; it++){
        ork_mm_run_i8_silu(p->c,p->wg,p->M,p->x,p->silu_gate,0x51aa,0x14,0xffffff9fu,0xffffc000u,0x56391100u,NULL,0);
        ork_mm_run_i8_out8(p->c,p->wu,p->M,p->x,p->up,0x4000,14);
        ork_npu_ewmul_i8(p->c,p->up,p->silu_gate,p->M,p->Dff,0x4000,14,p->glu,&us);
        ork_mm_run_i8(p->c,p->wd,p->M,p->glu,p->out);
    }
    return NULL;
}

int main(int argc,char**argv){
    int M = argc>1?atoi(argv[1]):64;
    int iters = argc>2?atoi(argv[2]):200;
    const int Kh=2048, Dff=2048;   /* hidden, ffn-inner (down K=Dff<=4096) */
    ork_npu *c = ork_npu_init();
    if(!c){ fprintf(stderr,"init failed\n"); return 2; }

    /* random int8 weights + activation */
    int8_t *Bg=malloc((size_t)Kh*Dff), *Bu=malloc((size_t)Kh*Dff), *Bd=malloc((size_t)Dff*Kh);
    int8_t *x=malloc((size_t)M*Kh);
    for(size_t i=0;i<(size_t)Kh*Dff;i++){ Bg[i]=(int8_t)((i*7)%13-6); Bu[i]=(int8_t)((i*5)%11-5); }
    for(size_t i=0;i<(size_t)Dff*Kh;i++) Bd[i]=(int8_t)((i*3)%9-4);
    for(size_t i=0;i<(size_t)M*Kh;i++) x[i]=(int8_t)((i*11)%17-8);

    ork_w *wg=ork_mm_pack_i8(c,Kh,Dff,Bg), *wu=ork_mm_pack_i8(c,Kh,Dff,Bu), *wd=ork_mm_pack_i8(c,Dff,Kh,Bd);
    if(!wg||!wu||!wd){ fprintf(stderr,"pack failed\n"); return 2; }

    int8_t  *silu_gate=malloc((size_t)M*Dff), *up=malloc((size_t)M*Dff), *glu=malloc((size_t)M*Dff);
    int32_t *out=malloc((size_t)M*Kh*4);
    double us=0;
    /* fused-silu g2 constants (op runs regardless of exact scale — this is a timing probe) */
    const int RM=0x51aa, RS=0x14; const unsigned OB=0xffffff9fu, IO=0xffffc000u, C4=0x56391100u;
    const int EM=0x4000, ES=14;   /* ewmul + out8 gain (identity-ish) */

    int rc=0;
    #define STEP(fn) do{ int r=(fn); if(r){fprintf(stderr,"step rc=%d\n",r); rc=1;} }while(0)
    /* warmup */
    for(int w=0;w<3 && !rc;w++){
        STEP(ork_mm_run_i8_silu(c,wg,M,x,silu_gate,RM,RS,OB,IO,C4,NULL,0));
        STEP(ork_mm_run_i8_out8(c,wu,M,x,up,EM,ES));
        STEP(ork_npu_ewmul_i8(c,up,silu_gate,M,Dff,EM,ES,glu,&us));
        STEP(ork_mm_run_i8(c,wd,M,glu,out));
    }
    if(rc){ fprintf(stderr,"warmup failed\n"); return 1; }

    /* timed loop + per-op split */
    double t_g=0,t_u=0,t_e=0,t_d=0,t0;
    double tot0=now_us();
    for(int it=0; it<iters && !rc; it++){
        t0=now_us(); STEP(ork_mm_run_i8_silu(c,wg,M,x,silu_gate,RM,RS,OB,IO,C4,NULL,0)); t_g+=now_us()-t0;
        t0=now_us(); STEP(ork_mm_run_i8_out8(c,wu,M,x,up,EM,ES));                          t_u+=now_us()-t0;
        t0=now_us(); STEP(ork_npu_ewmul_i8(c,up,silu_gate,M,Dff,EM,ES,glu,&us));           t_e+=now_us()-t0;
        t0=now_us(); STEP(ork_mm_run_i8(c,wd,M,glu,out));                                  t_d+=now_us()-t0;
    }
    double tot=now_us()-tot0;
    if(rc){ fprintf(stderr,"timed loop failed\n"); return 1; }

    printf("=== chained on-NPU FFN inner (M=%d Kh=%d Dff=%d, %d iters) ===\n", M, Kh, Dff, iters);
    printf("  gate+SiLU : %.1f us/ffn\n", t_g/iters);
    printf("  up (out8) : %.1f us/ffn\n", t_u/iters);
    printf("  ewmul     : %.1f us/ffn\n", t_e/iters);
    printf("  down      : %.1f us/ffn\n", t_d/iters);
    printf("  TOTAL     : %.1f us/ffn  (%.1f ffn/s)\n", tot/iters, iters*1e6/tot);

    if(argc>3 && !strcmp(argv[3],"pipe")){
        int n=512*1024; float*buf=malloc((size_t)n*4); for(int i=0;i<n;i++) buf[i]=(i%17)*0.01f;
        double per_npu=tot/iters;                                   /* us per FFN */
        double c0=now_us(); cpu_work(buf,n,1); double per_rep=now_us()-c0;
        int reps=(int)(per_npu/per_rep); if(reps<1)reps=1;          /* balance CPU work to ~one FFN */
        double cc0=now_us(); for(int it=0;it<iters;it++) cpu_work(buf,n,reps); double T_cpu=now_us()-cc0;
        double T_npu=tot;                                           /* NPU FFN over `iters` (from the timed loop) */
        struct npu_arg na={c,wg,wu,wd,M,Dff,x,silu_gate,up,glu,out,iters};
        pthread_t th; double o0=now_us();
        pthread_create(&th,NULL,npu_thread,&na);
        for(int it=0;it<iters;it++) cpu_work(buf,n,reps);           /* CPU runs concurrently with the NPU FFN */
        pthread_join(th,NULL);
        double T_ov=now_us()-o0;
        printf("=== PIPELINE (M=%d, %d iters, CPU work balanced to ~1 FFN) ===\n",M,iters);
        printf("  NPU FFN alone  : %.1f ms\n", T_npu/1000);
        printf("  CPU work alone : %.1f ms\n", T_cpu/1000);
        printf("  SERIAL (sum)   : %.1f ms\n", (T_npu+T_cpu)/1000);
        printf("  OVERLAPPED     : %.1f ms\n", T_ov/1000);
        printf("  speedup vs serial: %.2fx  (ideal max ~%.2fx)\n",
               (T_npu+T_cpu)/T_ov, (T_npu+T_cpu)/(T_npu>T_cpu?T_npu:T_cpu));
        free(buf);
    }

    ork_mm_free(c,wg); ork_mm_free(c,wu); ork_mm_free(c,wd); ork_npu_free(c);
    return 0;
}
