/* resident_handoff_probe — increment 1 of the heterogeneous resident spine: prove an NPU→CPU→NPU op chain
 * over ONE resident buffer is COHERENT (the bsync handoff is the historically-buggy part, e.g. ORK_ZC_OUT),
 * and measure whether CPU work OVERLAPS an NPU submit (the domain-free-CPU thesis: the CPU is outside the
 * IOMMU domain wall, so 1 NPU domain ∥ CPU is the one real parallelism axis).
 *
 * Chain (all activations in ork_dma_alloc resident buffers, NPU writes/reads them in place):
 *   NPU  mm1:  C1[M,N]  = A1[M,K]·W1[K,N]           (int8→int32, output in resident Bc)
 *   CPU  glue: A2[M,N]  = clamp(C1 >> SH)           (a stand-in norm/requant "glue" op, reads+writes resident)
 *   NPU  mm2:  C2[M,N2] = A2[M,N]·W2[N,N2]          (int8→int32; A2 read in place)
 * vs a pure-integer CPU reference of the SAME chain — must be BIT-EXACT (any diff = a coherency/bsync bug).
 *
 *   make resident_handoff_probe && sudo env ORK_ZC_OUT=1 ORK_MM_TIMEOUT=3000 ./resident_handoff_probe
 */
#define _GNU_SOURCE
#include "ork_npu.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <pthread.h>
#include <time.h>
static uint32_t g=0x1234u; static int s3(void){ g=g*1664525u+1013904223u; return (int)((g>>28)%3)-1; } /* [-1,1] */
static double now_us(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec*1e6+t.tv_nsec/1e3; }
enum { SH=2 };

/* worker: run one NPU matmul (blocks its thread on the NPU HW) so the main thread's CPU work overlaps it */
struct mmarg { ork_npu*c; ork_w*w; int M; const int8_t*A; int32_t*C; int rc; };
static void* mm_worker(void*p){ struct mmarg*a=p; a->rc=ork_mm_run_i8(a->c,a->w,a->M,a->A,a->C); return NULL; }

int main(void){
    int M=32, K=512, N=512, N2=512;
    setvbuf(stdout,0,_IONBF,0);
    ork_npu*c=ork_npu_init(); if(!c){ printf("init failed\n"); return 2; }
    printf("resident_handoff_probe: NPU->CPU->NPU over resident dma buffers (M=%d K=%d N=%d N2=%d)\n",M,K,N,N2);

    /* resident (dma) activation buffers — NPU-coherent + CPU-mapped */
    int8_t  *A1 = ork_dma_alloc(c,(size_t)M*K);
    int32_t *Bc = ork_dma_alloc(c,(size_t)M*N*4);
    int8_t  *A2 = ork_dma_alloc(c,(size_t)M*N);
    int32_t *C2 = ork_dma_alloc(c,(size_t)M*N2*4);
    if(!A1||!Bc||!A2||!C2){ printf("dma_alloc FAILED (A1=%p Bc=%p A2=%p C2=%p) — need the dma-heap for a resident handoff\n",(void*)A1,(void*)Bc,(void*)A2,(void*)C2); return 2; }
    int8_t *W1b=malloc((size_t)K*N), *W2b=malloc((size_t)N*N2);
    for(size_t i=0;i<(size_t)M*K;i++) A1[i]=(int8_t)s3();
    for(size_t i=0;i<(size_t)K*N;i++) W1b[i]=(int8_t)s3();
    for(size_t i=0;i<(size_t)N*N2;i++) W2b[i]=(int8_t)s3();
    ork_w *W1=ork_mm_pack_i8(c,K,N,W1b), *W2=ork_mm_pack_i8(c,N,N2,W2b);
    if(!W1||!W2){ printf("pack failed\n"); return 2; }

    /* ---- pure-integer CPU reference of the whole chain ---- */
    int32_t *C1r=malloc((size_t)M*N*4); int8_t *A2r=malloc((size_t)M*N); int32_t *C2r=malloc((size_t)M*N2*4);
    for(int m=0;m<M;m++) for(int n=0;n<N;n++){ long a=0; for(int k=0;k<K;k++) a+=A1[(size_t)m*K+k]*W1b[(size_t)k*N+n];
        C1r[(size_t)m*N+n]=(int32_t)a; long q=a>>SH; if(q>127)q=127; if(q<-127)q=-127; A2r[(size_t)m*N+n]=(int8_t)q; }
    for(int m=0;m<M;m++) for(int n=0;n<N2;n++){ long a=0; for(int nn=0;nn<N;nn++) a+=A2r[(size_t)m*N+nn]*W2b[(size_t)nn*N2+n]; C2r[(size_t)m*N2+n]=(int32_t)a; }

    /* ---- COHERENCY: run the chain on the NPU with a CPU glue op in the middle, all through resident buffers ---- */
    double t0=now_us(); int rc1=ork_mm_run_i8(c,W1,M,A1,Bc);              /* NPU writes Bc (resident) */
    double t1=now_us();
    if(rc1){ printf("mm1 rc=%d\n",rc1); return 1; }
    /* CPU reads Bc (NPU output) + writes A2 (matmul2 input) — the domain-free glue op on resident memory */
    for(size_t i=0;i<(size_t)M*N;i++){ long q=(long)Bc[i]>>SH; if(q>127)q=127; if(q<-127)q=-127; A2[i]=(int8_t)q; }
    double t2=now_us(); int rc2=ork_mm_run_i8(c,W2,M,A2,C2);              /* NPU reads A2 (resident) */
    double t3=now_us();
    if(rc2){ printf("mm2 rc=%d\n",rc2); return 1; }

    long badBc=0, badC2=0; int32_t mxBc=0, mxC2=0;
    for(size_t i=0;i<(size_t)M*N;i++){ int32_t d=Bc[i]-C1r[i]; if(d){badBc++; if(d<0)d=-d; if(d>mxBc)mxBc=d;} }
    for(size_t i=0;i<(size_t)M*N2;i++){ int32_t d=C2[i]-C2r[i]; if(d){badC2++; if(d<0)d=-d; if(d>mxC2)mxC2=d;} }
    printf("  mm1(NPU->resident Bc): %s (bad=%ld maxdiff=%d)  [%.0f us]\n", badBc?"MISMATCH":"bit-exact", badBc, mxBc, t1-t0);
    printf("  CPU glue on resident buffer: %.0f us\n", t2-t1);
    printf("  mm2(resident A2->NPU C2): %s (bad=%ld maxdiff=%d)  [%.0f us]\n", badC2?"MISMATCH":"bit-exact", badC2, mxC2, t3-t2);
    int coherent = (badBc==0 && badC2==0);
    printf("  HANDOFF: %s — NPU->CPU->NPU over one resident buffer %s\n", coherent?"COHERENT":"BROKEN",
           coherent?"is bit-exact (bsync handoff sound)":"DIVERGED (bsync/coherency bug)");

    /* ---- OVERLAP: does CPU work run concurrently with an NPU submit? (CPU is outside the IOMMU domain) ---- */
    int32_t *Cx=ork_dma_alloc(c,(size_t)M*N2*4); if(!Cx)Cx=C2;
    volatile long sink=0; size_t CPUN=(size_t)M*N*40;   /* a chunk of domain-free CPU glue work */
    int8_t *scratch=malloc(CPUN); for(size_t i=0;i<CPUN;i++) scratch[i]=(int8_t)(i*7);
    /* warm */ ork_mm_run_i8(c,W2,M,A2,Cx);
    double na=now_us(); for(int r=0;r<5;r++) ork_mm_run_i8(c,W2,M,A2,Cx); double npu1=(now_us()-na)/5;
    double ca=now_us(); for(int r=0;r<5;r++){ long acc=0; for(size_t i=0;i<CPUN;i++){ long q=(long)scratch[i]>>1; if(q>63)q=63; acc+=q; } sink+=acc; } double cpu1=(now_us()-ca)/5;
    double ba=now_us();
    for(int r=0;r<5;r++){ struct mmarg a={c,W2,M,A2,Cx,0}; pthread_t th; pthread_create(&th,0,mm_worker,&a);
        long acc=0; for(size_t i=0;i<CPUN;i++){ long q=(long)scratch[i]>>1; if(q>63)q=63; acc+=q; } sink+=acc;
        pthread_join(th,0); }
    double both=(now_us()-ba)/5;
    double eff = both>0 ? (npu1+cpu1)/both : 0;
    printf("  OVERLAP: NPU-alone %.0fus | CPU-alone %.0fus | concurrent %.0fus -> %.2fx (%.0f%% of CPU hidden under NPU)\n",
           npu1, cpu1, both, eff, both<npu1+cpu1 ? 100.0*(npu1+cpu1-both)/cpu1 : 0.0);
    (void)sink;

    printf("%s\n", coherent ? "PASS — resident NPU<->CPU handoff coherent; overlap measured" : "FAIL — coherency broken");
    ork_dma_free(c,A1); ork_dma_free(c,Bc); ork_dma_free(c,A2); ork_dma_free(c,C2); if(Cx!=C2) ork_dma_free(c,Cx);
    ork_mm_free(c,W1); ork_mm_free(c,W2); ork_npu_free(c);
    return coherent?0:1;
}
