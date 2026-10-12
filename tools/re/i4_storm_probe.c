/* tools/re/i4_storm_probe.c — grade an int4 BCHAIN geometry on THIS board: repeated, timed, and
 * checksum-compared, at a production shape.
 *
 * WHY IT IS NOT i4_hcap_probe. Neither `rc` nor timing alone is a validity test (the driver self-heals a
 * timed-out submit and returns the right answer ~1000x late), and neither is a single agreeing checksum
 * (a single standalone run passed K=2560 H=7, which is wrong). This loops ONE fixed geometry `iters`
 * times, times every call, prints a per-call checksum against the first, and counts how many outputs sit
 * at the int16 doorbell sentinel 0x7fff — so a self-heal (slow but correct), a silent miscompute (fast
 * but different) and a sentinel VALUE COLLISION (a saturating W4A4 accumulator reaching 0x7fff, which
 * looks exactly like "never written") are each distinguishable. Run it with ORK_I4_NOLADDER=1, or the
 * geometry ladder hides the failure and every row passes.
 *
 * Grade a candidate H against the H=2 anchor at the same shape: H only re-tiles M, so every valid H must
 * be BIT-IDENTICAL and must not be time-inflated. Use dm=1 — the patterned generator (dm=0) is the weaker
 * test, and the 2026-10-10 DRAM-clock discriminator was measured with it.
 *
 * data modes: 0 = i4_bank_sweep's deterministic pattern, 1 = seeded random, 2 = forced saturation
 *             (every output 0x7fff — reproduces the collision storm on purpose), 3 = small magnitude
 *             (saturation arithmetically impossible)
 *
 *   make i4_storm_probe
 *   sudo tools/util/npu_guard.sh -- timeout -s TERM 200 env ORK_I4_NOLADDER=1 ORK_I4_H=4 \
 *        ./i4_storm_probe 2048 2048 128 10 1
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
    int K=argc>1?atoi(argv[1]):2048, N=argc>2?atoi(argv[2]):2048, M=argc>3?atoi(argv[3]):128;
    int iters=argc>4?atoi(argv[4]):60, dm=argc>5?atoi(argv[5]):0;
    if(getenv("ORK_STORM_SEED")) s_=(unsigned)atoi(getenv("ORK_STORM_SEED"));
    ork_npu *C_=ork_npu_init(); if(!C_){ printf("no board\n"); return 0; }
    int8_t *B=malloc((size_t)K*N), *A=malloc((size_t)M*K); int32_t *Cb=malloc((size_t)M*N*4);
    if(!A||!B||!Cb){ printf("OOM\n"); return 2; }
    switch(dm){
      case 1: for(size_t i=0;i<(size_t)K*N;i++) B[i]=(int8_t)rnd(-8,7);
              for(size_t i=0;i<(size_t)M*K;i++) A[i]=(int8_t)rnd(-8,7); break;
      case 2: for(size_t i=0;i<(size_t)K*N;i++) B[i]=7; for(size_t i=0;i<(size_t)M*K;i++) A[i]=7; break;
      case 3: for(size_t i=0;i<(size_t)K*N;i++) B[i]=(int8_t)rnd(-1,1);
              for(size_t i=0;i<(size_t)M*K;i++) A[i]=(int8_t)(i%2?1:-1); break;
      default:for(size_t i=0;i<(size_t)K*N;i++) B[i]=(int8_t)((int)((i*7)%15)-7);
              for(size_t i=0;i<(size_t)M*K;i++) A[i]=(int8_t)((int)((i*13)%31)-15); break;
    }
    ork_w *W=ork_i4_mm_pack(C_,K,N,B); if(!W){ printf("pack fail\n"); return 2; }
    printf("K=%d N=%d M=%d iters=%d datamode=%d DBNK=%s WB=%s\n",K,N,M,iters,dm,
           getenv("ORK_I4_DBNK")?:"-", getenv("ORK_I4_WB")?:"-");
    double *us=malloc(sizeof(double)*iters); unsigned long long ref=0;
    long nstorm=0;
    for(int it=0; it<iters; it++){
        memset(Cb,0,(size_t)M*N*4);
        double t0=now_us(); int rc=ork_i4_mm_run(C_,W,M,A,Cb); double e=now_us()-t0; us[it]=e;
        long sp=0,sn=0; int32_t mx=0; for(size_t i=0;i<(size_t)M*N;i++){ int32_t v=Cb[i];
            if(v==32767)sp++; else if(v==-32768)sn++; int32_t a=v<0?-v:v; if(a>mx)mx=a; }
        unsigned long long ck=fnv64(Cb,(size_t)M*N*4); if(!it) ref=ck;
        printf("it=%-3d us=%-10.1f rc=%d sat+=%-6ld sat-=%-6ld maxabs=%-7d ck=%016llx%s\n",
               it,e,rc,sp,sn,mx,ck, ck!=ref?"  <<< CKSUM-DIFF":"");
        fflush(stdout);
    }
    double *sorted=malloc(sizeof(double)*iters); memcpy(sorted,us,sizeof(double)*iters);
    qsort(sorted,iters,sizeof(double),cmpd);
    double med=sorted[iters/2];
    for(int i=0;i<iters;i++) if(us[i]>3.0*med) nstorm++;
    printf("\nSUMMARY min=%.1f med=%.1f max=%.1f  storms(>3x med)=%ld/%d\n",sorted[0],med,sorted[iters-1],nstorm,iters);
    ork_mm_free(C_,W); ork_npu_free(C_); return 0;
}
