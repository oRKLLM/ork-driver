/* tools/jit_inflate_check.c — validate int8 JIT-inflate to fp16 (emulated W8A16) is bit-exact.
 *
 * Claim: ork_mm_f16_scratch + ork_mm_inflate_i8_to_f16(i8,bscale) produces resident tiles that are
 * BYTE-IDENTICAL to ork_mm_pack of the row-major dequantized weight wf16[k,n]=(f16)(i8[k*N+n]*bscale[n]),
 * and therefore run bit-identically through the (already-validated) fp16 matmul kernel. So the scratch
 * costs only ONE fp16 buffer of IOVA (reused across layers) while giving the exact fp16-path result for
 * int8-precision weights + unquantized fp16 activations.
 *
 * Test: (1) DUMP compare — memcmp ork_w_dump(scratch) vs ork_w_dump(direct pack) == 0.
 *       (2) RUN compare — ork_mm_run on both, C bit-identical.
 *       (3) REUSE — re-inflate the same scratch with a second int8 weight, must match a fresh direct pack.
 *   make jit_inflate_check && sudo ./jit_inflate_check [K] [N] [M]   (board only)
 */
#define _GNU_SOURCE
#include "ork_npu.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

typedef ork_f16 f16;

/* deterministic int8 weight + positive per-channel scale (seed varies the weight for the reuse test) */
static void gen_weight(int8_t *i8, float *bscale, int K, int N, unsigned seed){
    for(int n=0;n<N;n++) bscale[n]=0.001f+0.0007f*(float)((n*13+seed*7)%37);
    for(int k=0;k<K;k++)for(int n=0;n<N;n++)
        i8[(size_t)k*N+n]=(int8_t)(((k*31+n*17+seed*101)%255)-127);
}
static void dequant_f16(f16 *out, const int8_t *i8, const float *bscale, int K, int N){
    for(int k=0;k<K;k++)for(int n=0;n<N;n++)
        out[(size_t)k*N+n]=(f16)((float)i8[(size_t)k*N+n]*bscale[n]);
}

/* dump a weight's resident tiles into a fresh malloc'd buffer; *len set to byte length */
static void *dump_w(const ork_w *w, size_t *len){
    size_t n=ork_w_dump(w,NULL,0); void *b=malloc(n?n:1); *len=ork_w_dump(w,b,n); return b;
}

int main(int argc,char**argv){
    int K=argc>1?atoi(argv[1]):1024, N=argc>2?atoi(argv[2]):512, M=argc>3?atoi(argv[3]):8;
    if(K%32||N%16){ printf("bad dims K%%32=%d N%%16=%d\n",K%32,N%16); return 2; }
    ork_npu*c=ork_npu_init(); if(!c){printf("no board\n");return 0;}
    printf("jit_inflate_check K=%d N=%d M=%d\n",K,N,M);

    int8_t *i8=malloc((size_t)K*N); float *bscale=malloc((size_t)N*sizeof(float));
    f16 *ref=malloc((size_t)K*N*sizeof(f16));
    f16 *A=malloc((size_t)M*K*sizeof(f16));
    for(int i=0;i<M*K;i++) A[i]=(f16)(0.5f*sinf(0.017f*(float)i));
    float *Ca=malloc((size_t)M*N*sizeof(float)),*Cb=malloc((size_t)M*N*sizeof(float));
    int fail=0;

    /* ---- weight #1 ---- */
    gen_weight(i8,bscale,K,N,1);
    dequant_f16(ref,i8,bscale,K,N);

    ork_w *sc=ork_mm_f16_scratch(c,K,N);        if(!sc){printf("scratch alloc fail\n");return 1;}
    if(ork_mm_inflate_i8_to_f16(c,sc,i8,bscale,K,N)){printf("inflate fail\n");return 1;}
    ork_w *dp=ork_mm_pack(c,K,N,ref);            if(!dp){printf("direct pack fail\n");return 1;}

    /* (1) DUMP compare — the decisive bit-exact check (no NPU run needed) */
    size_t la,lb; void*da=dump_w(sc,&la),*db=dump_w(dp,&lb);
    int dumpeq = (la==lb) && (memcmp(da,db,la)==0);
    printf("  [1] tile-byte dump: scratch=%zuB direct=%zuB  %s\n",la,lb,dumpeq?"IDENTICAL":"DIFFER");
    if(!dumpeq) fail=1;

    /* (2) RUN compare */
    int ra=ork_mm_run(c,sc,M,A,Ca), rb=ork_mm_run(c,dp,M,A,Cb);
    if(ra||rb){ printf("  [2] run rc: scratch=%d direct=%d\n",ra,rb); fail=1; }
    else {
        double maxd=0; int nbad=0;
        for(int i=0;i<M*N;i++){ double d=fabs((double)Ca[i]-(double)Cb[i]); if(d>maxd)maxd=d; if(d!=0.0)nbad++; }
        printf("  [2] run output: maxabsdiff=%.3e mismatched=%d/%d  %s\n",maxd,nbad,M*N,nbad?"DIFFER":"BIT-EXACT");
        if(nbad) fail=1;
    }
    free(da);free(db);

    /* (3) REUSE — re-inflate the SAME scratch with weight #2, compare to a fresh direct pack */
    gen_weight(i8,bscale,K,N,2);
    dequant_f16(ref,i8,bscale,K,N);
    if(ork_mm_inflate_i8_to_f16(c,sc,i8,bscale,K,N)){printf("re-inflate fail\n");return 1;}
    ork_w *dp2=ork_mm_pack(c,K,N,ref);
    size_t l2a,l2b; void*d2a=dump_w(sc,&l2a),*d2b=dump_w(dp2,&l2b);
    int reuseeq=(l2a==l2b)&&(memcmp(d2a,d2b,l2a)==0);
    printf("  [3] reuse re-inflate dump: %s\n",reuseeq?"IDENTICAL":"DIFFER");
    if(!reuseeq) fail=1;
    free(d2a);free(d2b);

    ork_mm_free(c,sc); ork_mm_free(c,dp); ork_mm_free(c,dp2); ork_npu_free(c);
    free(i8);free(bscale);free(ref);free(A);free(Ca);free(Cb);
    printf("VERDICT: %s\n", fail?"FAIL":"PASS (JIT-inflate == direct fp16 pack, bit-exact)");
    return fail?1:0;
}
