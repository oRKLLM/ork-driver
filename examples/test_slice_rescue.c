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
static int8_t r4(void){ g ^= g<<13; g ^= g>>17; g ^= g<<5; return (int8_t)(((int)(g & 0xf)) - 8); }   /* int4 [-8,7] in an int8 container */
/* Static golden (AGENTS convention): inputs are fixed-seed deterministic so each int4 rescue output is a
 * constant. Assert an O(M*N) fnv64 of the NPU output vs an embedded golden; the O(M*N*K) CPU int32 reference
 * runs ONLY under ORK_REGEN (print goldens) or ORK_FULL_REF (diagnose a mismatch). */
static uint64_t fnv64(const int32_t *x, size_t n){ uint64_t h=1469598103934665603ULL; const uint8_t *p=(const uint8_t*)x;
    for(size_t i=0;i<n*4;i++){ h^=p[i]; h*=1099511628211ULL; } return h; }
static const uint64_t GI4[] = {   /* [i4-base, i4-wideN, i4-wideK, i4-refuse] — regen: sudo env ORK_REGEN=1 ./test_slice_rescue */
    0x42a675c1a3080462ULL, 0x0166d951dfb84ba9ULL, 0xf1e54fbd3c9abce5ULL, 0x928fbeeca9293205ULL };
static int gi4 = 0;

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

/* int4 (W4A4) rescue: decompose a refused int4 shape into BCHAIN-legal sub-tiles. int4 matmul is exact
 * integer arithmetic (values [-8,7], int32 accumulate) so it's bit-exact vs a CPU int32 reference (the
 * QUANTIZATION is what's incoherent, not the matmul). A *naturally*-refusing shape needs large M (→ huge CPU
 * ref), so we FORCE the rescue on cheap shapes that still exercise the natural gate (Sn>1 and K>8192 both
 * build w->sliced) — plus a forced-small base. Validates slice_pack_i4 + slice_run_i4 (BCHAIN per tile) +
 * int32 K-accumulate + N-scatter. */
static int one_i4(ork_npu *c, int K, int N, int M, int slice_all, const char *tag){
    int idx = gi4++;
    printf("  [%-10s] K=%d N=%d M=%d int4 (Sn=%d K>8192=%d)\n", tag, K, N, M, (N+8191)/8192, K>8192);
    int8_t *A=malloc((size_t)M*K), *B=malloc((size_t)K*N);
    for(size_t i=0;i<(size_t)M*K;i++) A[i]=r4();
    for(size_t i=0;i<(size_t)K*N;i++) B[i]=r4();
    int32_t *Cres=malloc((size_t)M*N*4);
    if(slice_all) setenv("ORK_SLICE_ALL","1",1); else unsetenv("ORK_SLICE_ALL");
    ork_w *w=ork_mm_pack_i4(c,K,N,B); unsetenv("ORK_SLICE_ALL");
    if(!w){ printf("    pack_i4 FAIL\n"); free(A);free(B);free(Cres); return 1; }
    setenv("ORK_FORCE_SLICE_RESCUE","1",1); int rr=ork_mm_run_i4(c,w,M,A,Cres); unsetenv("ORK_FORCE_SLICE_RESCUE");
    int fail=0;
    if(rr==ORK_RC_WEDGE_PRONE){ printf("    rescue REFUSED (-501): int4 tiles not built -> FAIL\n"); fail=1; }
    else if(rr){ printf("    int4 rescue rc=%d FAIL\n", rr); fail=1; }
    else { uint64_t h=fnv64(Cres,(size_t)M*N);
        if(getenv("ORK_REGEN")) printf("    REGEN GI4[%d]=0x%016llxULL\n", idx, (unsigned long long)h);
        else if(h!=GI4[idx]){ printf("    fnv64=0x%016llx != golden 0x%016llx FAIL\n",(unsigned long long)h,(unsigned long long)GI4[idx]); fail=1; }
        if(getenv("ORK_FULL_REF")){ int32_t *Ccpu=malloc((size_t)M*N*4); cpuref(A,B,M,K,N,Ccpu); long f,b=diff(Cres,Ccpu,(size_t)M*N,&f);
            printf(b?"    vs CPU: %ld mism (first@%ld %d/%d)\n":"    vs CPU: bit-exact\n",b,f,b?Cres[f]:0,b?Ccpu[f]:0); free(Ccpu); }
        if(!fail && !getenv("ORK_REGEN")) printf("    OK (int4 rescue golden 0x%016llx)\n",(unsigned long long)GI4[idx]); }
    if(getenv("REPS")){ int reps=atoi(getenv("REPS")); if(reps<1)reps=1;
        setenv("ORK_FORCE_SLICE_RESCUE","1",1);
        for(int q=0;q<2;q++) ork_mm_run_i4(c,w,M,A,Cres);
        double t0=now_us(); for(int r=0;r<reps;r++) ork_mm_run_i4(c,w,M,A,Cres); double us=(now_us()-t0)/reps;
        unsetenv("ORK_FORCE_SLICE_RESCUE");
        printf("    PERF: int4 rescue = %.1f us/call (%d K-slice x %d N-tile)\n", us, (((K+31)/32*32)+8191)/8192, (N+8191)/8192); }
    ork_mm_free(c,w); free(A);free(B);free(Cres);
    return fail;
}

/* int4 NATURAL refuse -> rescue: a real reachable trigger (large M + Sn>1 makes run_i4_mc_db's per-core
 * program count exceed the cap -> -4). Proves the refuse SITE routes to the rescue (rc=0, not -501) and that
 * the natural path == the forced rescue. No CPU ref (compares the two NPU runs) -> cheap despite big M. */
static int one_i4_natural(ork_npu *c, int K, int N, int M, const char *tag){
    int idx = gi4++;
    printf("  [%-10s] K=%d N=%d M=%d int4 NATURAL refuse->rescue\n", tag, K, N, M);
    int8_t *A=malloc((size_t)M*K), *B=malloc((size_t)K*N);
    for(size_t i=0;i<(size_t)M*K;i++) A[i]=r4();
    for(size_t i=0;i<(size_t)K*N;i++) B[i]=r4();
    int32_t *Cnat=malloc((size_t)M*N*4), *Cfor=malloc((size_t)M*N*4);
    ork_w *w=ork_mm_pack_i4(c,K,N,B);   /* natural gate builds w->sliced (Sn>1) */
    if(!w){ printf("    pack_i4 FAIL\n"); free(A);free(B);free(Cnat);free(Cfor); return 1; }
    unsetenv("ORK_FORCE_SLICE_RESCUE"); int rn=ork_mm_run_i4(c,w,M,A,Cnat);   /* run_i4_mc_db -4 -> rescue */
    setenv("ORK_FORCE_SLICE_RESCUE","1",1); int rf=ork_mm_run_i4(c,w,M,A,Cfor); unsetenv("ORK_FORCE_SLICE_RESCUE");
    int fail=0; long f;
    if(rn==ORK_RC_WEDGE_PRONE){ printf("    NATURAL refuse did NOT route to rescue (-501) -> wiring FAIL\n"); fail=1; }
    else if(rn||rf){ printf("    rc natural=%d forced=%d FAIL\n",rn,rf); fail=1; }
    else { long b=diff(Cnat,Cfor,(size_t)M*N,&f); if(b){ printf("    natural != forced rescue: %ld mism FAIL\n",b); fail=1; }
        uint64_t h=fnv64(Cnat,(size_t)M*N);   /* golden: the natural-refuse output is correct (verify once via ORK_FULL_REF) */
        if(getenv("ORK_REGEN")) printf("    REGEN GI4[%d]=0x%016llxULL\n", idx, (unsigned long long)h);
        else if(h!=GI4[idx]){ printf("    fnv64=0x%016llx != golden 0x%016llx FAIL\n",(unsigned long long)h,(unsigned long long)GI4[idx]); fail=1; }
        if(getenv("ORK_FULL_REF")){ int32_t *Ccpu=malloc((size_t)M*N*4); cpuref(A,B,M,K,N,Ccpu); long b2=diff(Cnat,Ccpu,(size_t)M*N,&f);
            printf(b2?"    vs CPU: %ld mism\n":"    vs CPU: bit-exact\n",b2); free(Ccpu); } }
    if(!fail && !getenv("ORK_REGEN")) printf("    OK (natural refuse->rescue, rc=0, ==forced, golden 0x%016llx)\n",(unsigned long long)GI4[idx]);
    if(getenv("REPS")){ int reps=atoi(getenv("REPS")); if(reps<1)reps=1;
        for(int q=0;q<2;q++) ork_mm_run_i4(c,w,M,A,Cnat);   /* natural path (run_i4_mc_db -4 attempt -> rescue) */
        double t0=now_us(); for(int r=0;r<reps;r++) ork_mm_run_i4(c,w,M,A,Cnat); double usn=(now_us()-t0)/reps;
        setenv("ORK_FORCE_SLICE_RESCUE","1",1);             /* forced: rescue ONLY (skips the -4 attempt) */
        for(int q=0;q<2;q++) ork_mm_run_i4(c,w,M,A,Cfor);
        double t1=now_us(); for(int r=0;r<reps;r++) ork_mm_run_i4(c,w,M,A,Cfor); double usf=(now_us()-t1)/reps;
        unsetenv("ORK_FORCE_SLICE_RESCUE");
        printf("    PERF: natural(with -4 attempt)=%.1f us  forced(rescue only)=%.1f us  attempt-tax=%.1f us\n", usn, usf, usn-usf); }
    ork_mm_free(c,w); free(A);free(B);free(Cnat);free(Cfor);
    return fail;
}

/* GROUPED int4 rescue (float per-group W4A4): a shape whose per-core program count (M/nc)*Sn*Sk exceeds the
 * doorbell cap HARD-refuses (ORK_RC_WEDGE_PRONE) — unlike plain int4. The rescue M-chunks the rows. Validate
 * (gtest convention) the output matches the exact per-group dequant within float rounding (maxe<0.05); rc==0
 * proves the rescue fired (else the doorbell would have returned -501). */
static int one_i4g(ork_npu *c, int M, int K, int N, int G, const char *tag){
    int Sk=K/G; printf("  [%-10s] M=%d K=%d N=%d G=%d grouped (Sk=%d, Pcore~%d)\n", tag, M, K, N, G, Sk, (M+2)/3*Sk);
    float *Af=malloc((size_t)M*K*4), *aS=malloc((size_t)M*Sk*4), *bS=malloc((size_t)Sk*N*4), *C=malloc((size_t)M*N*4);
    signed char *Ai=malloc((size_t)M*K), *Bi=malloc((size_t)K*N); float *Bf=malloc((size_t)K*N*4);
    unsigned sd=7+M+K+N+G;
    for(size_t i=0;i<(size_t)M*K;i++){ sd=sd*1103515245+12345; Af[i]=((int)(sd>>9)%2001-1000)/1000.0f; }
    for(size_t i=0;i<(size_t)K*N;i++){ sd=sd*1103515245+12345; Bf[i]=((int)(sd>>9)%2001-1000)/1000.0f; }
    for(int m=0;m<M;m++)for(int g=0;g<Sk;g++){ float mx=1e-9f; for(int j=0;j<G;j++){ float a=Af[m*K+g*G+j]; if(a<0)a=-a; if(a>mx)mx=a; }
        aS[m*Sk+g]=mx/7; for(int j=0;j<G;j++){ int q=(int)(Af[m*K+g*G+j]/aS[m*Sk+g]+(Af[m*K+g*G+j]>=0?.5f:-.5f)); if(q>7)q=7; if(q<-8)q=-8; Ai[m*K+g*G+j]=(signed char)q; } }
    for(int g=0;g<Sk;g++)for(int n=0;n<N;n++){ float mx=1e-9f; for(int j=0;j<G;j++){ float b=Bf[(g*G+j)*N+n]; if(b<0)b=-b; if(b>mx)mx=b; }
        bS[g*N+n]=mx/7; for(int j=0;j<G;j++){ int q=(int)(Bf[(g*G+j)*N+n]/bS[g*N+n]+(Bf[(g*G+j)*N+n]>=0?.5f:-.5f)); if(q>7)q=7; if(q<-8)q=-8; Bi[(g*G+j)*N+n]=(signed char)q; } }
    ork_w *w=ork_mm_pack_i4_grouped(c,K,N,Bi,G);
    if(!w){ printf("    pack_i4_grouped FAIL\n"); free(Af);free(Bf);free(aS);free(bS);free(C);free(Ai);free(Bi); return 1; }
    int rc=ork_mm_run_i4_grouped(c,w,M,Ai,aS,bS,C); ork_w_free(w);
    int fail=0;
    if(rc==ORK_RC_WEDGE_PRONE){ printf("    grouped rescue REFUSED (-501) -> FAIL\n"); fail=1; }
    else if(rc){ printf("    grouped run rc=%d FAIL\n", rc); fail=1; }
    else { double maxe=0; for(int m=0;m<M;m++)for(int n=0;n<N;n++){ double ex=0;
            for(int g=0;g<Sk;g++){ long p=0; for(int j=0;j<G;j++) p+=(long)Ai[m*K+g*G+j]*Bi[(g*G+j)*N+n]; ex+=(double)aS[m*Sk+g]*bS[g*N+n]*p; }
            double e=C[m*N+n]-ex; if(e<0)e=-e; if(e>maxe)maxe=e; }
        if(maxe>=0.05){ printf("    grouped rescue maxerr=%.4f >= 0.05 FAIL\n", maxe); fail=1; }
        else printf("    OK (grouped rescue rc=0, dequant maxerr=%.4f EXACT)\n", maxe); }
    if(getenv("REPS")){ int reps=atoi(getenv("REPS")); if(reps<1)reps=1;
        ork_w *w2=ork_mm_pack_i4_grouped(c,K,N,Bi,G);
        for(int q=0;q<2;q++) ork_mm_run_i4_grouped(c,w2,M,Ai,aS,bS,C);
        double t0=now_us(); for(int r=0;r<reps;r++) ork_mm_run_i4_grouped(c,w2,M,Ai,aS,bS,C); double us=(now_us()-t0)/reps;
        ork_w_free(w2);
        int nc=3, per_row=Sk, rpc=64/per_row; if(rpc<1)rpc=1; int Msub=rpc*nc; int chunks=(M+Msub-1)/Msub;
        printf("    PERF: grouped rescue (M=%d Sk=%d) = %.1f us/call (~%d M-chunks of %d rows)\n", M, Sk, us, chunks, Msub); }
    free(Af);free(Bf);free(aS);free(bS);free(C);free(Ai);free(Bi);
    return fail;
}

/* GATED CONFIRMATION (ORK_TEST_BCH=1): a natural BCHAIN shape (Sn=1,Sk=1,N%64==0) at varying M.
 * Not in the default suite because a wedge-prone chain length would hang `make test`. Reports rc +
 * a "done" marker (absent => the submit wedged mid-chain). Set ORK_BCH_DEBUG=1 for the geometry line. */
static int one_i4_bch(ork_npu *c, int K, int N, int M, const char *tag){
    printf("  [%-12s] K=%d N=%d M=%d int4 NATURAL BCHAIN (Sn=%d Sk-wide=%d N%%64=%d)\n", tag, K, N, M, (N+8191)/8192, K>8192, N%64);
    int8_t *A=malloc((size_t)M*K), *B=malloc((size_t)K*N); int32_t *C=malloc((size_t)M*N*4);
    for(size_t i=0;i<(size_t)M*K;i++) A[i]=r4(); for(size_t i=0;i<(size_t)K*N;i++) B[i]=r4();
    ork_w *w=ork_mm_pack_i4(c,K,N,B); if(!w){ printf("    pack_i4 FAIL\n"); free(A);free(B);free(C); return 1; }
    printf("    submitting..."); fflush(stdout);
    int rc=ork_mm_run_i4(c,w,M,A,C);
    printf(" done rc=%d %s\n", rc, rc==0?"(completed, no wedge)":"(nonzero)");
    ork_mm_free(c,w); free(A);free(B);free(C);
    return rc==0?0:1;
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
    fail |= one_i4(c, 512,   256,   8, 1, "i4-base");  /* int4 forced small — slice_pack_i4/run_i4/accumulate */
    fail |= one_i4(c, 512,   16384, 8, 0, "i4-wideN"); /* int4 NATURAL Sn>1 gate — N-scatter (2 tiles) */
    fail |= one_i4(c, 10240, 128,   8, 0, "i4-wideK"); /* int4 NATURAL K>8192 gate — K-slice int32-accumulate (2 slices) */
    fail |= one_i4_natural(c, 2048, 16384, 128, "i4-refuse"); /* REAL trigger: M=128 + Sn=2 -> run_i4_mc_db -4 -> rescue */
    fail |= one_i4g(c, 64, 2048, 256, 128, "i4g-refuse");     /* GROUPED: M=64 Sk=16 -> Pcore~341 > cap -> hard refuse -> M-chunk rescue */
    if(getenv("ORK_TEST_BCH")){   /* P=128 wedge probe: the Qwen3-1.7B int4 down-proj (K=6144,N=2048) at M=64 (works) vs M=128 (wedges) */
        printf("  -- ORK_TEST_BCH: BCHAIN chain-length wedge probe (down-proj K=6144 N=2048) --\n");
        fail |= one_i4_bch(c, 6144, 2048, 64,  "bch-M64");
        fail |= one_i4_bch(c, 6144, 2048, 128, "bch-M128");
    }
    if(getenv("ORK_TEST_GRP")){   /* GROUPED W4A4 wedge sweep: ascending Sk=K/G at M=1 (isolates the grouped submit), then M=32 (rescue) */
        printf("  -- ORK_TEST_GRP: grouped W4A4 wedge sweep (N=2048, G=128; Sk=K/128) --\n");
        fail |= one_i4g(c, 1,  2048, 2048, 128, "grp-M1-Sk16");
        fail |= one_i4g(c, 1,  4096, 2048, 128, "grp-M1-Sk32");
        fail |= one_i4g(c, 1,  6144, 2048, 128, "grp-M1-Sk48");   /* down-proj group count */
        fail |= one_i4g(c, 1,  8192, 2048, 128, "grp-M1-Sk64");
        fail |= one_i4g(c, 32, 6144, 2048, 128, "grp-M32-Sk48");  /* the convert M_padded=32 rescue path */
    }
    printf(fail ? "TEST_SLICE_RESCUE: FAIL\n" : "TEST_SLICE_RESCUE: PASS\n");
    return fail ? 1 : 0;
}
