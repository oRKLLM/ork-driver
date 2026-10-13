/* tools/re/i4_e2_probe.c — E2: grade an int4 BCHAIN activation geometry against an ABSOLUTE CPU
 * reference, repeated and timed.
 *
 * i4_storm_probe grades iteration N against iteration 0, which cannot tell a consistently-wrong
 * geometry from a right one; E2 moves the activation line stride / declared row count / pack, each
 * of which changes the answer wholesale, so the reference has to be absolute. Both gates are
 * applied and reported separately:
 *     BIT-EXACT   every element equals the CPU int4xint4 reference
 *     NOT INFLATED  median wall time is not a self-heal (compare against the anchor's median)
 *
 *   make i4_e2_probe
 *   sudo tools/util/npu_guard.sh -- timeout -s TERM 200 env ORK_I4_NOLADDER=1 ORK_I4_H=4 \
 *        ./i4_e2_probe 512 64 8 6 1
 */
#define _GNU_SOURCE
#include "ork_npu.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double now_us(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec*1e6+t.tv_nsec/1e3; }
static unsigned long long fnv64(const void *p,size_t n){
    const unsigned char *b=p; unsigned long long h=1469598103934665603ULL;
    for(size_t i=0;i<n;i++){ h^=b[i]; h*=1099511628211ULL; } return h; }
static unsigned s_=12345u; static int rnd(int lo,int hi){ s_=s_*1103515245u+12345u; return lo+(int)((s_>>16)%(unsigned)(hi-lo+1)); }
static int cmpd(const void*a,const void*b){ double x=*(const double*)a,y=*(const double*)b; return x<y?-1:x>y; }

int main(int argc,char**argv){
    int K=argc>1?atoi(argv[1]):512, N=argc>2?atoi(argv[2]):64, M=argc>3?atoi(argv[3]):8;
    int iters=argc>4?atoi(argv[4]):6, dm=argc>5?atoi(argv[5]):1;
    if(getenv("ORK_STORM_SEED")) s_=(unsigned)atoi(getenv("ORK_STORM_SEED"));
    ork_npu *C_=ork_npu_init(); if(!C_){ printf("no board\n"); return 2; }
    /* ORK_I4_FUZZ="blk:reg=val,..." — arbitrary regcmd overrides, applied last in orki_i4_synth_full.
     * Lets one build sweep any register without a rebuild (the template cache turns itself off). */
    { const char *fz=getenv("ORK_I4_FUZZ");
      if(fz && *fz){ ork_i4_fuzz_clear(); char buf[512]; snprintf(buf,sizeof buf,"%s",fz);
        for(char *t=strtok(buf,","); t; t=strtok(NULL,",")){
            unsigned b=0,r=0,v=0; if(sscanf(t,"%x:%x=%x",&b,&r,&v)==3){ ork_i4_fuzz_add(b,r,v);
                printf("  fuzz %04x:%04x = 0x%08x\n",b,r,v); } } } }
    int8_t *B=malloc((size_t)K*N), *A=malloc((size_t)M*K);
    int32_t *Cb=malloc((size_t)M*N*4), *R=malloc((size_t)M*N*4);
    if(!A||!B||!Cb||!R){ printf("OOM\n"); return 2; }
    if(dm==3){ for(size_t i=0;i<(size_t)K*N;i++) B[i]=(int8_t)rnd(-1,1);
               for(size_t i=0;i<(size_t)M*K;i++) A[i]=(int8_t)rnd(-1,1); }
    else     { for(size_t i=0;i<(size_t)K*N;i++) B[i]=(int8_t)rnd(-8,7);
               for(size_t i=0;i<(size_t)M*K;i++) A[i]=(int8_t)rnd(-8,7); }
    /* CPU int4 x int4 reference (same convention as examples/i4.c: A[M][K] x B[K][N]) */
    double tr=now_us();
    long mx=0;
    for(int m=0;m<M;m++){ const int8_t *a=A+(size_t)m*K; int32_t *r=R+(size_t)m*N;
        memset(r,0,(size_t)N*4);
        for(int k=0;k<K;k++){ int av=a[k]; if(!av) continue; const int8_t *b=B+(size_t)k*N;
            for(int n=0;n<N;n++) r[n]+=av*b[n]; }
        for(int n=0;n<N;n++){ long v=r[n]<0?-r[n]:r[n]; if(v>mx)mx=v; } }
    printf("K=%d N=%d M=%d iters=%d dm=%d  refmaxabs=%ld%s  ref=%.0fms\n", K,N,M,iters,dm,mx,
           mx>32767?"  <<< SATURATES (reference invalid)":"", (now_us()-tr)/1000.0);
    printf("  knobs: DENSE=%s MCDECL=%s MCPHYS=%s LSTRIDE=%s OSTRIDE=%s H=%s WB=%s DBNK=%s\n",
           getenv("ORK_I4_DENSE")?:"-",getenv("ORK_I4_MCDECL")?:"-",getenv("ORK_I4_MCPHYS")?:"-",
           getenv("ORK_I4_LSTRIDE")?:"-",getenv("ORK_I4_OSTRIDE")?:"-",
           getenv("ORK_I4_H")?:"-",getenv("ORK_I4_WB")?:"-",getenv("ORK_I4_DBNK")?:"-");
    fflush(stdout);
    ork_w *W=ork_i4_mm_pack(C_,K,N,B); if(!W){ printf("pack fail\n"); return 2; }
    double *us=malloc(sizeof(double)*iters); int npass=0;
    for(int it=0; it<iters; it++){
        memset(Cb,0,(size_t)M*N*4);
        double t0=now_us(); int rc=ork_i4_mm_run(C_,W,M,A,Cb); double e=now_us()-t0; us[it]=e;
        long bad=0; int bm=-1,bn=-1;
        for(int m=0;m<M;m++)for(int n=0;n<N;n++) if(Cb[(size_t)m*N+n]!=R[(size_t)m*N+n]){
            if(!bad){ bm=m; bn=n; } bad++; }
        unsigned long long ck=fnv64(Cb,(size_t)M*N*4);
        if(!bad) npass++;
        printf("it=%-3d us=%-10.1f rc=%d bad=%-8ld/%-8d ck=%016llx %s",
               it,e,rc,bad,M*N,ck, bad?"MISMATCH":"EXACT");
        if(bad) printf("  first [%d,%d] got=%d want=%d",bm,bn,Cb[(size_t)bm*N+bn],R[(size_t)bm*N+bn]);
        printf("\n"); fflush(stdout);
    }
    double *so=malloc(sizeof(double)*iters); memcpy(so,us,sizeof(double)*iters);
    qsort(so,iters,sizeof(double),cmpd); double med=so[iters/2];
    long storms=0; for(int i=0;i<iters;i++) if(us[i]>3.0*med) storms++;
    printf("\nVERDICT pass=%d/%d  min=%.1f med=%.1f max=%.1f us  storms=%ld  %s\n",
           npass,iters,so[0],med,so[iters-1],storms,
           (npass==iters)?"BIT-EXACT":(npass?"FLAKY":"WRONG"));
    ork_mm_free(C_,W); ork_npu_free(C_); return npass==iters?0:1;
}
