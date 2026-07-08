/* tools/i4_multim_probe.c — Tier 4b RE: can the int4 regcmd do MULTI-M in one submit, and what is the
 * multi-M output layout? Sets the M-count regs to M (ork_npu_probe_i4_mm -> synth_i4 mc=M), one submit,
 * and brute-forces candidate de-tile layouts against an EXACT CPU reference (int4*int4 is exact; with
 * small K the int16 sum never overflows, so a correct layout matches bit-for-bit).
 *
 *   make i4_multim_probe && sudo ./i4_multim_probe
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include "ork_npu.h"

static unsigned sd=99; static int8_t r4(void){sd=sd*1103515245+12345;return (int8_t)((int)((sd>>10)%15)-7);} /* [-7,7] */

/* candidate layouts: index of element (m,n) in the raw int16 buffer (M rows, N cols). */
static size_t L_rowmajor(int m,int n,int M,int N){(void)M;return (size_t)m*N+n;}
static size_t L_n8_m_8 (int m,int n,int M,int N){(void)N;return ((size_t)(n/8)*M+m)*8+(n%8);}
static size_t L_n4_m_4 (int m,int n,int M,int N){(void)N;return ((size_t)(n/4)*M+m)*4+(n%4);}
static size_t L_n64_m_64(int m,int n,int M,int N){(void)N;return ((size_t)(n/64)*M+m)*64+(n%64);}
static size_t L_n_m     (int m,int n,int M,int N){(void)N;return (size_t)n*M+m;}
/* surface-blocked (64-ch surfaces), stride-2 rows within each surface: physical row 2m */
static size_t L_surf64_s2(int m,int n,int M,int N){(void)N;return (size_t)(n/64)*(2*M*64)+(size_t)(2*m)*64+(n%64);}
/* surface-blocked (64-ch), contiguous rows within each surface (no stride-2) */
static size_t L_surf64_m (int m,int n,int M,int N){(void)N;return (size_t)(n/64)*(M*64)+(size_t)m*64+(n%64);}
struct lay{const char*name;size_t(*f)(int,int,int,int);};
static struct lay LAYS[]={
    {"row-major (m*N+n)",L_rowmajor},
    {"(N/8,M,8)",L_n8_m_8},
    {"(N/4,M,4)",L_n4_m_4},
    {"(N/64,M,64)",L_n64_m_64},
    {"(N,M) col-major",L_n_m},
    {"surf64 stride-2 [(n/64)(2M·64)+2m·64+n%64]",L_surf64_s2},
    {"surf64 contig  [(n/64)(M·64)+m·64+n%64]",L_surf64_m},
};

static void test(ork_npu*ctx,int M,int K,int N){
    int8_t*A=malloc((size_t)M*K),*B=malloc((size_t)K*N); int32_t*ref=malloc((size_t)M*N*4);
    for(size_t i=0;i<(size_t)M*K;i++)A[i]=r4();
    for(size_t i=0;i<(size_t)K*N;i++)B[i]=r4();
    for(int m=0;m<M;m++)for(int n=0;n<N;n++){int s=0;for(int k=0;k<K;k++)s+=A[(size_t)m*K+k]*B[(size_t)k*N+n];ref[(size_t)m*N+n]=s;}
    int16_t*raw=calloc((size_t)2*M*N,2);   /* 2x: stride-2 multi-M writes physical rows 0..2(M-1) */
    int rc=ork_npu_probe_i4_mm(ctx,M,K,N,A,B,raw);
    printf("M=%d K=%d N=%d  submit rc=%d\n",M,K,N,rc);
    if(rc==0){
        for(unsigned l=0;l<sizeof LAYS/sizeof*LAYS;l++){
            long bad=0,tot=(long)M*N; int firstm=-1,firstn=-1;
            for(int m=0;m<M;m++)for(int n=0;n<N;n++){ size_t idx=LAYS[l].f(m,n,M,N);
                int got=(idx<(size_t)2*M*N)?raw[idx]:0x7fff;
                if(got!=ref[(size_t)m*N+n]){ if(bad==0){firstm=m;firstn=n;} bad++; } }
            printf("   %-20s : %s (%ld/%ld mismatch%s)\n",LAYS[l].name, bad==0?"MATCH":"no",bad,tot,
                   bad&&firstm==0?"; row0 too":"");
        }
        /* explicit stride-2 detile (Exp-2026-06-19: logical row m -> physical row 2m) */
        { long b2=0,tot2=(long)M*N; for(int m=0;m<M;m++)for(int n=0;n<N;n++) if(raw[(size_t)(2*m)*N+n]!=ref[(size_t)m*N+n]) b2++;
          printf("   STRIDE-2 (raw[2m*N+n]) : %s (%ld/%ld mismatch)\n", b2==0?"MATCH":"no", b2, tot2); }
        /* also: is row 0 alone correct under row-major? (tells us M=1 part works, M>1 is the layout Q) */
        long b0=0; for(int n=0;n<N;n++) if(raw[n]!=ref[n]) b0++;
        printf("   row0 row-major: %s (%ld/%d)\n", b0==0?"ok":"no", b0, N);
        /* exhaustive: for each output row m, is its N-vector present ANYWHERE in raw (any start offset,
         * contiguous)? If rows 1..M-1 are nowhere, the hardware simply didn't compute them. */
        for(int m=1;m<M;m++){ long found=-1;
            for(size_t off=0; off+N<=(size_t)2*M*N; off++){ int ok=1;
                for(int n=0;n<N&&ok;n++) if(raw[off+n]!=ref[(size_t)m*N+n]) ok=0;
                if(ok){found=(long)off;break;} }
            if(found>=0) printf("   row%d contiguous: FOUND @ %ld (= physical row %ld, /N)\n", m, found, found/N);
            else         printf("   row%d contiguous: NOT computed (nowhere in output)\n", m); }
        printf("   raw[0..7]= "); for(int i=0;i<8&&i<2*M*N;i++)printf("%d ",raw[i]); printf(" | ref[0..7]= ");
        for(int i=0;i<8&&i<M*N;i++)printf("%d ",ref[i]); printf("\n");
        /* PER-64-BLOCK contiguous search (collision-proof): for each (row m, 64-wide block b), find the
         * physical raw offset where the whole 64-vector ref[m][b*64..] lands. Reveals the 2D surface layout. */
        printf("   block map (m,blk)->physrow (rawoff/64):\n");
        for(int m=0;m<M;m++){ printf("     m=%d:",m);
            for(int b=0;b<N/64;b++){ long found=-1;
                for(size_t off=0; off+64<=(size_t)2*M*N; off++){ int ok=1;
                    for(int c=0;c<64&&ok;c++) if(raw[off+c]!=ref[(size_t)m*N+b*64+c]) ok=0;
                    if(ok){found=(long)off;break;} }
                if(found>=0) printf("  blk%d@row%ld(off%ld)", b, found/64, found);
                else         printf("  blk%d=MISSING", b); }
            printf("\n"); }
    }
    free(A);free(B);free(ref);free(raw);
}
int main(void){
    ork_npu*ctx=ork_npu_init(); if(!ctx){printf("init failed (NPU?)\n");return 1;}
    test(ctx,8,32,64);     /* rows_computed*K=256 hypothesis: predicts 8 rows at K=32 */
    test(ctx,4,64,64);     /* the capture's exact config */
    test(ctx,2,64,64);
    test(ctx,4,128,128);
    test(ctx,8,256,256);
    /* single 64-wide N-block at REAL K (the shippable per-block-multi-M granularity) */
    test(ctx,8,2048,64);
    test(ctx,16,2048,64);
    test(ctx,16,512,64);
    /* multi-block (N>64) at production K WITH 0x107c=K/16 now default — does the 4-row batch span blocks? */
    test(ctx,4,512,128);
    test(ctx,4,2048,128);
    ork_npu_free(ctx);
    return 0;
}
