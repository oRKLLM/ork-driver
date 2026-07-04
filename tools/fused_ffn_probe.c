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

static double now_us(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec*1e6 + t.tv_nsec/1e3; }

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

    ork_mm_free(c,wg); ork_mm_free(c,wu); ork_mm_free(c,wd); ork_npu_free(c);
    return 0;
}
