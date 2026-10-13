/* attn_thrash_probe — does the fp16 attention bmm pay a MODE-TRANSITION tax in a real graph?
 *
 * attn_marshal_probe times the fp16 bmm in isolation: it stays in fp16 mode for the whole run, so
 * every ork_f16_mm_run after the first is warm. A real decoder layer does not look like that — the
 * graph is  int8/int4 dense matmul -> fp16 QK^T -> (softmax) -> fp16 A.V -> int8/int4 dense ...  so
 * EVERY attention node is a dtype transition, and ork_npu_enter may issue an RKNPU_ACT_RESET plus a
 * cold 2-pass re-warm on each one. That cost is invisible to an isolated benchmark and would be
 * paid 2x per layer per ubatch.
 *
 * Arms (same binary, same shapes, same order of work — only the interleaving differs):
 *   ISOLATED    : N fp16 bmm calls back to back.
 *   INTERLEAVED : int8 dense matmul, fp16 bmm, int8 dense matmul, fp16 bmm, ...
 * Reports per-call fp16 cost in each arm and the ACT_RESET count (run with ORK_DEBUG_RESET=1 to see
 * the sites). Board only. Diagnostic; not in `all`/`test`. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "ork_npu.h"

static double now_us(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec*1e6+t.tv_nsec*1e-3; }

int main(int argc,char**argv){
    int M   = argc>1?atoi(argv[1]):512;      /* ubatch */
    int HD  = argc>2?atoi(argv[2]):128;      /* head_dim (fp16 QK^T contraction) */
    int KV  = argc>3?atoi(argv[3]):1024;     /* n_kv  (fp16 QK^T output width) */
    int DK  = argc>4?atoi(argv[4]):2048;     /* dense int8 K (n_embd) */
    int DN  = argc>5?atoi(argv[5]):2048;     /* dense int8 N */
    int it  = argc>6?atoi(argv[6]):16;

    ork_npu *c=ork_npu_init(); if(!c){ fprintf(stderr,"init failed\n"); return 1; }
    printf("attn_thrash_probe  fp16 bmm [%d,%d]x[%d,%d]  vs  int8 dense [%d,%d]x[%d,%d]  iters=%d (ork %s)\n",
           M,HD,HD,KV,M,DK,DK,DN,it,ork_npu_version());

    /* fp16 attention operand (scratch weight, refreshed like the real handler does) */
    ork_f16 *A16=malloc((size_t)M*HD*2), *B16=malloc((size_t)HD*KV*2);
    float   *C32=malloc((size_t)M*KV*4);
    for(size_t i=0;i<(size_t)M*HD;i++) A16[i]=(ork_f16)(((i*2654435761u)%1000)/1000.0f-0.5f);
    for(size_t i=0;i<(size_t)HD*KV;i++) B16[i]=(ork_f16)(((i*40503u)%997)/997.0f-0.5f);
    ork_w *wf=ork_f16_mm_scratch(c,HD,KV);
    if(!wf){ fprintf(stderr,"f16 scratch failed\n"); return 1; }
    ork_f16_mm_repack(c,wf,HD,KV,B16);

    /* int8 resident dense weight, like a layer's attn_output / ffn projection */
    signed char *A8=malloc((size_t)M*DK), *B8=malloc((size_t)DK*DN);
    int *C8=malloc((size_t)M*DN*4);
    for(size_t i=0;i<(size_t)M*DK;i++) A8[i]=(signed char)((i*31u)%255-127);
    for(size_t i=0;i<(size_t)DK*DN;i++) B8[i]=(signed char)((i*17u)%255-127);
    ork_w *wi=ork_i8_mm_pack(c,DK,DN,B8);
    if(!wi){ fprintf(stderr,"i8 pack failed\n"); return 1; }

    /* warm both modes once */
    ork_f16_mm_run(c,wf,M,A16,C32);
    ork_i8_mm_run (c,wi,M,A8,C8);

    double t,f_iso,f_int,d_iso,d_int;
    t=now_us(); for(int i=0;i<it;i++) ork_f16_mm_run(c,wf,M,A16,C32);        f_iso=(now_us()-t)/it;
    t=now_us(); for(int i=0;i<it;i++) ork_i8_mm_run (c,wi,M,A8,C8);          d_iso=(now_us()-t)/it;
    ork_npu_floor_reset();
    t=now_us(); for(int i=0;i<it;i++){ ork_i8_mm_run(c,wi,M,A8,C8); ork_f16_mm_run(c,wf,M,A16,C32); }
    double pair=(now_us()-t)/it;
    f_int = pair - d_iso; d_int = pair - f_iso;
    double io,hw; long long raw; long ns; ork_npu_floor_timing(&io,&hw,&raw,&ns);

    double fmac=(double)M*HD*KV, dmac=(double)M*DK*DN;
    printf("\n  ISOLATED     fp16 bmm   %9.1f us  (%6.1f GMAC/s)\n", f_iso, fmac/f_iso/1e3);
    printf("  ISOLATED     int8 dense %9.1f us  (%6.1f GMAC/s)\n", d_iso, dmac/d_iso/1e3);
    printf("  INTERLEAVED  pair       %9.1f us   vs %9.1f us if additive  -> TRANSITION TAX %+9.1f us/pair (%.2fx)\n",
           pair, f_iso+d_iso, pair-(f_iso+d_iso), pair/(f_iso+d_iso));
    printf("               implied fp16-in-graph %9.1f us (%.2fx its isolated cost)\n", f_int, f_int/f_iso);
    printf("               submits/pair %.1f  ioctl %.1f us  hw %.1f us\n", (double)ns/it, io/it, hw/it);
    /* 2 attention matmuls per layer per ubatch */
    printf("\n  Per pp2048 (28 layers x 16 heads x 4 ubatches x 2 attn matmuls = 3584 fp16 nodes):\n");
    printf("     at isolated cost  %6.2f s     at in-graph cost  %6.2f s     tax %6.2f s\n",
           3584*f_iso/1e6, 3584*f_int/1e6, 3584*(f_int-f_iso)/1e6);

    /* fp16 PEAK reference: a big dense fp16 matmul at the same (M,K,N) as the int8 arm. The attention
     * bmm must be judged against THIS, not against the int8 dense rate — the brief's 222 GMAC/s figure
     * is int8, and the fp16 MAC array is half-rate. */
    { ork_f16 *Ad=malloc((size_t)M*DK*2), *Bd=malloc((size_t)DK*DN*2); float *Cd=malloc((size_t)M*DN*4);
      for(size_t i=0;i<(size_t)M*DK;i++) Ad[i]=(ork_f16)(((i*7u)%201)/201.0f-0.5f);
      for(size_t i=0;i<(size_t)DK*DN;i++) Bd[i]=(ork_f16)(((i*13u)%199)/199.0f-0.5f);
      ork_w *wd=ork_f16_mm_pack(c,DK,DN,Bd);
      if(wd){ ork_f16_mm_run(c,wd,M,Ad,Cd); ork_npu_floor_reset();
        double t2=now_us(); for(int i=0;i<it;i++) ork_f16_mm_run(c,wd,M,Ad,Cd); double w2=(now_us()-t2)/it;
        double io2,hw2; long long r2; long n2; ork_npu_floor_timing(&io2,&hw2,&r2,&n2);
        double dm=(double)M*DK*DN;
        printf("\n  fp16 DENSE [%d,%d]x[%d,%d]: wall %9.1f us (%6.1f GMAC/s)  submits %.1f  hw-sum %9.1f us\n",
               M,DK,DK,DN,w2,dm/w2/1e3,(double)n2/it,hw2/it);
        printf("               -> fp16 wall-rate is the ceiling the attention bmm is measured against\n");
        ork_mm_free(c,wd); }
      free(Ad);free(Bd);free(Cd); }

    ork_mm_free(c,wf); ork_mm_free(c,wi); ork_npu_free(c);
    free(A16);free(B16);free(C32);free(A8);free(B8);free(C8);
    return 0;
}
