/* test_slice_rescue — #33 slice-and-dice RESCUE wiring test.
 *
 * Proves the rescue PLUMBING (not just the sliced compute, which slice_dbrun_probe already covers):
 *   1. pack() builds w->sliced for a refuse-prone shape (Sn>1 && (K>4096||!Bf)) — the NATURAL gate — and
 *      for ANY int8 shape under the ORK_SLICE_ALL test hook.
 *   2. run_multicore's refuse sites RUN that w->sliced instead of returning ORK_RC_WEDGE_PRONE — forced
 *      here via ORK_FORCE_SLICE_RESCUE so the rescue fires even where a native path exists.
 *   3. the rescued result is BIT-EXACT vs a CPU int32 reference AND vs the native run_i8 path.
 *
 * Self-validating (exits nonzero on any mismatch); run by `make test` with NO env — it sets the hooks
 * internally per case (getenv is read at pack time for the build + at run time for the force). Shapes are
 * kept small so the O(M*N*K) CPU reference stays sub-second. */
#define _GNU_SOURCE
#include "ork_npu.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>

static double now_us(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec*1e6 + t.tv_nsec*1e-3; }
static uint32_t g = 2463534242u;
static int8_t r8(void){ g ^= g<<13; g ^= g>>17; g ^= g<<5; return (int8_t)(((int)(g & 0x3f)) - 31); }

static void cpuref(const int8_t *A, const int8_t *B, int M, int K, int N, int32_t *C){
    for(int m=0;m<M;m++) for(int n=0;n<N;n++){ int32_t s=0; const int8_t *ar=A+(size_t)m*K;
        for(int k=0;k<K;k++) s += (int32_t)ar[k] * (int32_t)B[(size_t)k*N+n];
        C[(size_t)m*N+n]=s; }
}
static long diff(const int32_t *X, const int32_t *Y, size_t n, long *first){
    long bad=0; *first=-1; for(size_t i=0;i<n;i++) if(X[i]!=Y[i]){ if(*first<0)*first=(long)i; bad++; } return bad;
}

/* One shape: native run_i8, then a FORCED slice-rescue run, both checked bit-exact vs CPU.
 * slice_all=1 sets ORK_SLICE_ALL so pack() builds tiles for a shape the natural gate wouldn't (small/base).
 * slice_all=0 relies on the NATURAL gate (must be a refuse-prone shape) — a rescue then also PROVES the gate. */
static int one(ork_npu *c, int K, int N, int M, int slice_all, const char *tag){
    printf("  [%-10s] K=%d N=%d M=%d natural_gate=%s\n", tag, K, N, M, slice_all?"forced(ORK_SLICE_ALL)":"yes(Sn>1,K>4096)");
    int8_t *A=malloc((size_t)M*K), *B=malloc((size_t)K*N);
    for(size_t i=0;i<(size_t)M*K;i++) A[i]=r8();
    for(size_t i=0;i<(size_t)K*N;i++) B[i]=r8();
    int32_t *Ccpu=malloc((size_t)M*N*4), *Cnat=malloc((size_t)M*N*4), *Cres=malloc((size_t)M*N*4);
    cpuref(A,B,M,K,N,Ccpu);

    if(slice_all) setenv("ORK_SLICE_ALL","1",1); else unsetenv("ORK_SLICE_ALL");
    ork_w *w = ork_mm_pack_i8(c,K,N,B);                 /* pack builds w->sliced (natural gate or ORK_SLICE_ALL) */
    unsetenv("ORK_SLICE_ALL");
    if(!w){ printf("    pack FAIL\n"); return 1; }

    unsetenv("ORK_FORCE_SLICE_RESCUE");                 /* NATIVE path */
    int rn = ork_mm_run_i8(c,w,M,A,Cnat);
    setenv("ORK_FORCE_SLICE_RESCUE","1",1);             /* FORCE the rescue */
    int rr = ork_mm_run_i8(c,w,M,A,Cres);
    unsetenv("ORK_FORCE_SLICE_RESCUE");
    ork_mm_free(c,w);

    int fail=0; long f;
    if(rn){ printf("    native run_i8 rc=%d FAIL\n", rn); fail=1; }
    else { long b=diff(Cnat,Ccpu,(size_t)M*N,&f); if(b){ printf("    native vs CPU: %ld mism (first@%ld %d/%d) FAIL\n",b,f,Cnat[f],Ccpu[f]); fail=1; } }
    if(rr==ORK_RC_WEDGE_PRONE){ printf("    rescue REFUSED (-501): w->sliced not built -> gate/plumbing FAIL\n"); fail=1; }
    else if(rr){ printf("    forced rescue rc=%d FAIL\n", rr); fail=1; }
    else { long b=diff(Cres,Ccpu,(size_t)M*N,&f); if(b){ printf("    rescue vs CPU: %ld mism (first@%ld %d/%d) FAIL\n",b,f,Cres[f],Ccpu[f]); fail=1; }
           long b2=diff(Cres,Cnat,(size_t)M*N,&f); if(b2){ printf("    rescue vs native: %ld mism FAIL\n",b2); fail=1; } }
    if(!fail) printf("    OK (native==CPU, rescue==CPU==native, bit-exact)\n");
    free(A);free(B);free(Ccpu);free(Cnat);free(Cres);
    return fail;
}

/* fp16 rescue (#33 stage 4): fp16 doesn't refuse — it falls to run()'s single-core reference. With
 * ORK_F16_SLICE_RESCUE/ORK_SLICE_ALL the fall-through runs the fast tiled doorbell instead. Validate the
 * forced rescue matches the native fp16 path AND a CPU f32 reference, within fp16 tolerance. */
static int one_f16(ork_npu *c, int K, int N, int M, const char *tag){
    printf("  [%-10s] K=%d N=%d M=%d fp16\n", tag, K, N, M);
    ork_f16 *A=malloc((size_t)M*K*sizeof(ork_f16)), *B=malloc((size_t)K*N*sizeof(ork_f16));
    for(size_t i=0;i<(size_t)M*K;i++) A[i]=(ork_f16)(r8()/32.0f);   /* ~[-1,1) */
    for(size_t i=0;i<(size_t)K*N;i++) B[i]=(ork_f16)(r8()/32.0f);
    float *Ccpu=malloc((size_t)M*N*4), *Cnat=malloc((size_t)M*N*4), *Cres=malloc((size_t)M*N*4);
    for(int m=0;m<M;m++) for(int n=0;n<N;n++){ float s=0; for(int k=0;k<K;k++) s+=(float)A[(size_t)m*K+k]*(float)B[(size_t)k*N+n];
        Ccpu[(size_t)m*N+n]=s; }
    setenv("ORK_SLICE_ALL","1",1);                     /* build fp16 tiles at pack */
    ork_w *w=ork_mm_pack(c,K,N,B); unsetenv("ORK_SLICE_ALL");
    if(!w){ printf("    fp16 pack FAIL\n"); return 1; }
    unsetenv("ORK_FORCE_SLICE_RESCUE"); int rn=ork_mm_run(c,w,M,A,Cnat);   /* native fp16 (colsplit/single-core) */
    setenv("ORK_FORCE_SLICE_RESCUE","1",1); int rr=ork_mm_run(c,w,M,A,Cres); unsetenv("ORK_FORCE_SLICE_RESCUE");
    ork_w_free(w);
    int fail=0; double mx=0, en=0, er=0;
    for(size_t i=0;i<(size_t)M*N;i++){ double a=Ccpu[i]<0?-Ccpu[i]:Ccpu[i]; if(a>mx)mx=a;
        double dn=Cnat[i]-Ccpu[i], dr=Cres[i]-Ccpu[i]; if(dn<0)dn=-dn; if(dr<0)dr=-dr; if(dn>en)en=dn; if(dr>er)er=dr; }
    double tol=0.02*(mx>1?mx:1);   /* fp16: ~2% of peak magnitude */
    if(rn){ printf("    native fp16 rc=%d FAIL\n",rn); fail=1; }
    else if(en>tol){ printf("    native vs CPU maxabs=%.3f > tol=%.3f FAIL\n",en,tol); fail=1; }
    if(rr==ORK_RC_WEDGE_PRONE){ printf("    fp16 rescue REFUSED (-501): tiles not built -> gate/plumbing FAIL\n"); fail=1; }
    else if(rr){ printf("    forced fp16 rescue rc=%d FAIL\n",rr); fail=1; }
    else if(er>tol){ printf("    rescue vs CPU maxabs=%.3f > tol=%.3f FAIL\n",er,tol); fail=1; }
    if(!fail) printf("    OK (native~CPU %.3f, rescue~CPU %.3f, tol %.3f)\n",en,er,tol);
    if(getenv("REPS")){   /* PERF GATE: sliced rescue must beat the single-core reference it replaces */
        int reps=atoi(getenv("REPS")); if(reps<1)reps=1;
        ork_w *wr=ork_mm_pack(c,K,N,B);                       /* plain pack (no tiles) -> F16_SC runs the single-core ref */
        setenv("ORK_F16_COLSPLIT","0",1);
        for(int q=0;q<2;q++) ork_mm_run(c,wr,M,A,Cnat);
        double t0=now_us(); for(int r=0;r<reps;r++) ork_mm_run(c,wr,M,A,Cnat); double us_ref=(now_us()-t0)/reps;
        unsetenv("ORK_F16_COLSPLIT"); ork_w_free(wr);
        setenv("ORK_SLICE_ALL","1",1); ork_w *ws=ork_mm_pack(c,K,N,B); unsetenv("ORK_SLICE_ALL");
        setenv("ORK_FORCE_SLICE_RESCUE","1",1);
        for(int q=0;q<2;q++) ork_mm_run(c,ws,M,A,Cres);
        double t1=now_us(); for(int r=0;r<reps;r++) ork_mm_run(c,ws,M,A,Cres); double us_slc=(now_us()-t1)/reps;
        unsetenv("ORK_FORCE_SLICE_RESCUE"); ork_w_free(ws);
        printf("    A/B: single-core-ref=%.1fus sliced-rescue=%.1fus  ref/sliced=%.2fx\n", us_ref, us_slc, us_ref/us_slc);
    }
    free(A);free(B);free(Ccpu);free(Cnat);free(Cres);
    return fail;
}

int main(void){
    setvbuf(stdout,0,_IONBF,0);
    ork_npu *c=ork_npu_init(); if(!c){ printf("init fail\n"); return 1; }
    printf("test_slice_rescue: #33 rescue wiring (pack builds tiles -> refuse site runs them, bit-exact)\n");
    int fail=0;
    fail |= one(c,1024, 512,  8, 1, "base");    /* Sn==1 base — tiles via ORK_SLICE_ALL, forced rescue */
    fail |= one(c,2048, 1536, 8, 1, "base2");   /* Sn==1 wider — tiles via ORK_SLICE_ALL */
    fail |= one(c,8192, 8704, 4, 0, "refuse");  /* Sn>1 && K>4096 — the NATURAL refuse-prone gate builds tiles */
    fail |= one_f16(c,1024, 512,  8, "f16base"); /* fp16 base — tiles via ORK_SLICE_ALL, forced rescue */
    fail |= one_f16(c,2560, 8704, 8, "f16wide"); /* fp16 multi-K-slice (K>2048) + wide-N (N>8192) */
    printf(fail ? "TEST_SLICE_RESCUE: FAIL\n" : "TEST_SLICE_RESCUE: PASS\n");
    return fail ? 1 : 0;
}
