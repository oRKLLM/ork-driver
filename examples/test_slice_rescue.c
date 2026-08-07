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

/* PADDING case: an unaligned K (K%kmul!=0) the sliced packer pads UP to kmul (zero-filled rows), so the
 * rescue can RUN it — bit-exact vs CPU on the real K. Checks ONLY the forced rescue vs CPU (the native path
 * isn't the subject here, and may not accept an unaligned K). slice_all=1 forces tiles for a non-refuse shape;
 * slice_all=0 uses the natural refuse-prone gate (Sn>1 && K>4096) on an also-unaligned K. */
static int one_pad(ork_npu *c, int K, int N, int M, int slice_all, const char *tag){
    printf("  [%-10s] K=%d N=%d M=%d (unaligned K%%512=%d -> pad)\n", tag, K, N, M, K%512);
    int8_t *A=malloc((size_t)M*K), *B=malloc((size_t)K*N);
    for(size_t i=0;i<(size_t)M*K;i++) A[i]=r8();
    for(size_t i=0;i<(size_t)K*N;i++) B[i]=r8();
    int32_t *Ccpu=malloc((size_t)M*N*4), *Cres=malloc((size_t)M*N*4);
    cpuref(A,B,M,K,N,Ccpu);
    if(slice_all) setenv("ORK_SLICE_ALL","1",1); else unsetenv("ORK_SLICE_ALL");
    ork_w *w=ork_mm_pack_i8(c,K,N,B); unsetenv("ORK_SLICE_ALL");
    if(!w){ printf("    pack FAIL\n"); free(A);free(B);free(Ccpu);free(Cres); return 1; }
    setenv("ORK_FORCE_SLICE_RESCUE","1",1); int rr=ork_mm_run_i8(c,w,M,A,Cres); unsetenv("ORK_FORCE_SLICE_RESCUE");
    ork_mm_free(c,w);
    int fail=0; long f;
    if(rr==ORK_RC_WEDGE_PRONE){ printf("    rescue REFUSED (-501): padded tiles not built -> FAIL\n"); fail=1; }
    else if(rr){ printf("    padded rescue rc=%d FAIL\n", rr); fail=1; }
    else { long b=diff(Cres,Ccpu,(size_t)M*N,&f); if(b){ printf("    padded rescue vs CPU: %ld mism (first@%ld %d/%d) FAIL\n",b,f,Cres[f],Ccpu[f]); fail=1; } }
    if(!fail) printf("    OK (padded rescue == CPU, bit-exact on the real K)\n");
    free(A);free(B);free(Ccpu);free(Cres);
    return fail;
}

int main(void){
    setvbuf(stdout,0,_IONBF,0);
    ork_npu *c=ork_npu_init(); if(!c){ printf("init fail\n"); return 1; }
    printf("test_slice_rescue: #33 rescue wiring (pack builds tiles -> refuse site runs them, bit-exact)\n");
    int fail=0;
    fail |= one(c,1024, 512,  8, 1, "base");     /* Sn==1 base — tiles via ORK_SLICE_ALL, forced rescue */
    fail |= one(c,2048, 1536, 8, 1, "base2");    /* Sn==1 wider — tiles via ORK_SLICE_ALL */
    fail |= one(c,8192, 8704, 4, 0, "refuse");   /* Sn>1 && K>4096 — the NATURAL refuse-prone gate builds tiles */
    fail |= one_pad(c,1408, 512,  8, 1, "pad-K");    /* K%512!=0 (1408->pad 1536) — PADDING fits the unaligned K, bit-exact */
    fail |= one_pad(c,4224, 8704, 4, 0, "pad-refuse"); /* NATURAL refuse (Sn>1,K>4096) AND unaligned K (4224->4608) — pad + rescue */
    printf(fail ? "TEST_SLICE_RESCUE: FAIL\n" : "TEST_SLICE_RESCUE: PASS\n");
    return fail ? 1 : 0;
}
