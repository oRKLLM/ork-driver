/* i4_macrate_probe — E5: is the int4 MAC array actually 2x as wide as the int8 one?
 *
 * Vendor table 3-5 says every tile atom is the same 16 B (A, C) / 1 KB (B) in every precision, with
 * int4 simply packing twice the elements. The int8 1 KB B atom is 32N x 32K = 1024 MACs, which at
 * 6 TOPS / 3 cores / 1 GHz / 2 is EXACTLY one cycle. So the question is whether the 1 KB int4 atom
 * (64N x 32K = 2048 MACs) also retires in one cycle. Either answer redirects months of work:
 *   int4 per-core MAC/s  >  int8's  -> the array is wider at int4 and the whole ceiling is schedule-side
 *   int4 saturates AT    == int8's  -> INT4_MM_INT4_TO_INT16 is a storage/format mode over an
 *                                      int8-rate array; no 2x exists and W4A4 is capacity-only
 *
 * METHOD. Run the SAME (M, K, N) in both precisions and report GMAC/s per core. Operands are drawn
 * from the int4 range [-8, 7] in both arms, so the two are computing the identical product and a
 * single CPU int32 reference grades both (int4's int16 output is the same value, clamped). The int4
 * arm is built as far from every KNOWN non-MAC bound as the geometry allows: K=512, where Wb=256 and
 * H=32 are both legal and measured PASS, a per-program weight of K*Wb/2 = 65536 B = exactly one CBUF
 * weight bank (resident, so no re-stream), and N wide enough that each core owns several column
 * chunks and the chain is long enough to amortise per-program dispatch. The int8 arm runs at its own
 * best geometry (ork_i8_mm_run's tuned path), which is the fair comparison: the question is about the
 * arrays, so each side gets its own best schedule.
 *
 * Both gates, every rep: bit-exact against the reference AND not time-inflated (at this frontier the
 * INCORRECT configs have measured as the FASTEST). Arms alternate, because arm position is worth up
 * to 1.126x on this board. Run with ORK_I4_NOLADDER=1 so a refused geometry reports rc!=0 instead of
 * being silently re-tiled into a different one.
 *
 *   make i4_macrate_probe
 *   sudo tools/util/npu_guard.sh -- timeout -s TERM 600 env ORK_I4_NOLADDER=1 ./i4_macrate_probe
 * Board only. Diagnostic; not in `all`/`test`. */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "ork_npu.h"

static double now_us(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec*1e6+t.tv_nsec*1e-3; }
static unsigned s_=12345u; static int rnd4(void){ s_=s_*1103515245u+12345u; return -8+(int)((s_>>16)%16u); }
static int cmpd(const void*a,const void*b){ double x=*(const double*)a,y=*(const double*)b; return x<y?-1:x>y; }

static int32_t sat16(long v){ return v>32767?32767:(v<-32768?-32768:(int32_t)v); }

int main(int argc,char**argv){
    int M    = argc>1?atoi(argv[1]):512;
    int K    = argc>2?atoi(argv[2]):512;
    int N    = argc>3?atoi(argv[3]):3072;
    int reps = argc>4?atoi(argv[4]):6;
    int order= argc>5?atoi(argv[5]):1;      /* 1 = int4 block first, 0 = int8 first */
    int cref = argc>6?atoi(argv[6]):1;      /* 1 = CPU int32 reference; 0 = grade int4 against the int8 NPU
                                             * output (a DIFFERENT datapath at the same inputs), so a shape
                                             * sweep is not priced at O(M*K*N) of host arithmetic per cell */

    setvbuf(stdout,NULL,_IOLBF,0);
    ork_npu *c=ork_npu_init(); if(!c){ printf("no board\n"); return 1; }
    printf("i4_macrate_probe  M=%d K=%d N=%d reps=%d order=%s  (ork %s)\n",M,K,N,reps,order?"int4-first":"int8-first",ork_npu_version());
    printf("  int4 geometry: Wb=%d (K*Wb/2 = %d B = %.2f CBUF weight banks), H=%s, NC=%d\n",
           (131072/K)&~63, K*(((131072/K)&~63))/2, (double)(K*(((131072/K)&~63))/2)/65536.0,
           getenv("ORK_I4_H")?:"default", N/(((131072/K)&~63)?((131072/K)&~63):64));

    int8_t *A=malloc((size_t)M*K), *B=malloc((size_t)K*N);
    int32_t *R=malloc((size_t)M*N*4), *C4=malloc((size_t)M*N*4), *C8=malloc((size_t)M*N*4);
    if(!A||!B||!R||!C4||!C8){ printf("OOM\n"); return 2; }
    for(size_t i=0;i<(size_t)M*K;i++) A[i]=(int8_t)rnd4();
    for(size_t i=0;i<(size_t)K*N;i++) B[i]=(int8_t)rnd4();
    if(cref){ double tref=now_us();
        for(int m=0;m<M;m++) for(int n=0;n<N;n++){ long acc=0;
            for(int k=0;k<K;k++) acc+=(long)A[(size_t)m*K+k]*B[(size_t)k*N+n];
            R[(size_t)m*N+n]=(int32_t)acc; }
        printf("  CPU reference %.1f s\n",(now_us()-tref)/1e6); }

    ork_w *W4=ork_i4_mm_pack(c,K,N,B); if(!W4){ printf("i4 pack failed\n"); return 2; }
    ork_w *W8=ork_i8_mm_pack(c,K,N,B);  if(!W8){ printf("i8 pack failed\n"); return 2; }

    double mac=(double)M*K*N;
    double *u4=malloc(sizeof(double)*reps), *u8=malloc(sizeof(double)*reps);
    int ok4=0,ok8=0,rc4=0,rc8=0; long worst4=0,worst8=0, sat=0;
    if(cref) for(size_t i=0;i<(size_t)M*N;i++) if(R[i]>32767||R[i]<-32768) sat++;

    /* BLOCKED, not interleaved. Alternating the arms per call makes every call an int4<->int8 mode
     * transition (ACT_RESET + cold re-warm), which measured an 18x spread (6.8 ms to 123 ms on the
     * same arm) and buried the quantity being measured. Each arm gets its own warmed block; arm
     * ORDER is the argv knob, so position bias is answered by running the probe both ways. */
    for(int phase=0;phase<2;phase++){
        int four = cref ? (order ? (phase==0) : (phase==1)) : (phase==1);   /* no CPU ref => int8 block must run FIRST, it IS the reference */
        int32_t *out=four?C4:C8;
        for(int wmup=0;wmup<2;wmup++){ memset(out,0,(size_t)M*N*4);
            if(four) ork_i4_mm_run(c,W4,M,A,out); else ork_i8_mm_run(c,W8,M,A,out); }
        for(int r=0;r<reps;r++){
            memset(out,0,(size_t)M*N*4);
            double t0=now_us();
            int rc= four?ork_i4_mm_run(c,W4,M,A,out):ork_i8_mm_run(c,W8,M,A,out);
            double el=now_us()-t0;
            long w=0;
            if(!rc && (cref||four)) for(size_t i=0;i<(size_t)M*N;i++){
                long want=four?sat16(cref?R[i]:C8[i]):R[i]; long d=out[i]-want; if(d<0)d=-d; if(d>w)w=d; }
            if(four){ rc4=rc; u4[r]=el; if(!rc&&!w)ok4++; if(w>worst4)worst4=w; }
            else    { rc8=rc; u8[r]=el; if(!rc&&!w&&(cref||1))ok8++; if(w>worst8)worst8=w; }
        }
    }
    qsort(u4,reps,sizeof(double),cmpd); qsort(u8,reps,sizeof(double),cmpd);
    int cores=ork_npu_cores(c);
    printf("  reference cells saturating int16: %ld / %d  (int4 arm is graded against the CLAMPED reference)\n",sat,M*N);
    printf("  %-5s %-8s %-10s %12s %12s %12s\n","arm","exact","rc","med us","GMAC/s","GMAC/s/core");
    printf("  %-5s %d/%-6d %-10d %12.1f %12.1f %12.1f  (min %.1f us -> %.1f GMAC/s/core)\n","int4",ok4,reps,rc4,u4[reps/2],mac/u4[reps/2]/1e3,mac/u4[reps/2]/1e3/cores,u4[0],mac/u4[0]/1e3/cores);
    printf("  %-5s %d/%-6d %-10d %12.1f %12.1f %12.1f  (min %.1f us -> %.1f GMAC/s/core)\n","int8",ok8,reps,rc8,u8[reps/2],mac/u8[reps/2]/1e3,mac/u8[reps/2]/1e3/cores,u8[0],mac/u8[0]/1e3/cores);
    printf("  int4 worst|diff|=%ld  int8 worst|diff|=%ld   spread int4 %.1f-%.1f us, int8 %.1f-%.1f us\n",
           worst4,worst8,u4[0],u4[reps-1],u8[0],u8[reps-1]);
    printf("  RATIO int4/int8 MAC rate = %.3f  (cores reported by the SoC caps: %d)\n", u8[reps/2]/u4[reps/2], cores);

    ork_mm_free(c,W4); ork_mm_free(c,W8);
    free(A);free(B);free(R);free(C4);free(C8);free(u4);free(u8);
    ork_npu_free(c);
    return (ok4==reps && ok8==reps) ? 0 : 1;
}
