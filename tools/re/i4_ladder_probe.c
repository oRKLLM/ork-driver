/* tools/re/i4_ladder_probe.c — grade the int4 BCHAIN GEOMETRY LADDER, and the individual rungs it
 * can land on, against an ABSOLUTE CPU int4xint4 reference.
 *
 * Two modes, one binary, because the ladder question needs both:
 *
 *   CELLS  (a "Wb:H,Wb:H,..." list): walks the named geometries IN THE GIVEN ORDER inside one
 *          process (ORK_I4_WB / ORK_I4_H are re-read per call), reps each. Use with
 *          ORK_I4_NOLADDER=1 so a refused geometry is reported, not silently recovered from.
 *   LADDER (no cell list): calls the shipped path with no geometry override, so the ladder in
 *          orki_i4_run_bchain_db picks the rung. Per-rep times are printed, which is the only way
 *          to separate the one-time LEARN from the steady state the memo lands on.
 *
 * Grading is ABSOLUTE on purpose. i4_storm_probe grades against its own iteration 0 and i4_wb_sweep
 * against a Wb=64 H=2 anchor; both anchors are themselves geometries, and a ladder A/B changes the
 * geometry by construction. Both gates are reported separately: BIT-EXACT (every element equals the
 * CPU reference) and NOT INFLATED (median wall time — a self-heal passes a checksum gate at 4.5 s).
 *
 *   make i4_ladder_probe
 *   sudo tools/util/npu_guard.sh -- timeout -s TERM 600 env ORK_I4_NOLADDER=1 \
 *        ./i4_ladder_probe 768 2048 128 6 0 320:21,128:21,64:21
 *   sudo tools/util/npu_guard.sh -- timeout -s TERM 600 ./i4_ladder_probe 768 2048 128 6
 */
#define _GNU_SOURCE
#include "ork_npu.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double now_us(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec*1e6+t.tv_nsec/1e3; }
static unsigned s_=777u; static int rnd(int lo,int hi){ s_=s_*1103515245u+12345u; return lo+(int)((s_>>16)%(unsigned)(hi-lo+1)); }
static int cmpd(const void*a,const void*b){ double x=*(const double*)a,y=*(const double*)b; return x<y?-1:x>y; }

int main(int argc,char**argv){
    int K=argc>1?atoi(argv[1]):768, N=argc>2?atoi(argv[2]):2048, M=argc>3?atoi(argv[3]):128;
    int reps=argc>4?atoi(argv[4]):6, desc=argc>5?atoi(argv[5]):0;
    const char *cells=argc>6?argv[6]:NULL;
    if(getenv("ORK_SWEEP_SEED")) s_=(unsigned)atoi(getenv("ORK_SWEEP_SEED"));

    ork_npu *c=ork_npu_init(); if(!c){ printf("no board\n"); return 2; }
    int8_t *B=malloc((size_t)K*N), *A=malloc((size_t)M*K);
    int32_t *C=malloc((size_t)M*N*4), *R=malloc((size_t)M*N*4);
    if(!B||!A||!C||!R){ printf("oom\n"); return 2; }
    for(size_t i=0;i<(size_t)K*N;i++) B[i]=(int8_t)rnd(-8,7);          /* FULL-RANDOM, int4 code range */
    for(size_t i=0;i<(size_t)M*K;i++) A[i]=(int8_t)rnd(-8,7);
    double tr=now_us();
    for(int m=0;m<M;m++){ const int8_t *ar=A+(size_t)m*K; int32_t *rr=R+(size_t)m*N;
        memset(rr,0,(size_t)N*4);
        for(int k=0;k<K;k++){ int av=ar[k]; if(!av) continue; const int8_t *br=B+(size_t)k*N;
            for(int n=0;n<N;n++) rr[n]+=av*br[n]; } }
    printf("# K=%d N=%d M=%d reps=%d order=%s cells=%s ladder=%s  cpu_ref %.0f ms\n",K,N,M,reps,
           desc?"desc":"asc", cells?cells:"(ladder)", getenv("ORK_I4_NOLADDER")?"off":"on",
           (now_us()-tr)/1000.0);

    ork_w *w=ork_i4_mm_pack(c,K,N,B); if(!w){ printf("pack fail\n"); return 2; }
    double *us=malloc(sizeof(double)*reps);

    /* one geometry (or the ladder's own choice): reps calls, absolute grade + the time distribution */
    int ncell=0; char *cl=cells?strdup(cells):NULL, *tok[64];
    if(cl) for(char *t=strtok(cl,",");t&&ncell<64;t=strtok(NULL,",")) tok[ncell++]=t;
    printf("cell\tpass\treps\tmed_us\tmin_us\tmax_us\trc_bad\tmism\tper_rep_us\n");
    for(int q=0;q<(ncell?ncell:1);q++){
        int ci = ncell ? (desc?ncell-1-q:q) : 0;
        if(ncell){ int Wb=0,H=0; char bb[32];
            if(sscanf(tok[ci],"%d:%d",&Wb,&H)!=2){ printf("bad cell %s\n",tok[ci]); continue; }
            snprintf(bb,sizeof bb,"%d",Wb); setenv("ORK_I4_WB",bb,1);
            snprintf(bb,sizeof bb,"%d",H);  setenv("ORK_I4_H",bb,1); }
        int pass=0,rcbad=0; long firstmm=-1; double raw[64];
        for(int r=0;r<reps && r<64;r++){
            memset(C,0,(size_t)M*N*4);
            double t0=now_us(); int rc=ork_i4_mm_run(c,w,M,A,C); us[r]=raw[r]=now_us()-t0;
            if(rc){ rcbad++; continue; }
            long mm=0; for(size_t i=0;i<(size_t)M*N;i++) if(C[i]!=R[i]) mm++;
            if(mm==0) pass++; else if(firstmm<0) firstmm=mm;
        }
        qsort(us,reps,sizeof(double),cmpd);
        printf("%s\t%d\t%d\t%.1f\t%.1f\t%.1f\t%d\t%ld\t",ncell?tok[ci]:"ladder",pass,reps,
               us[reps/2],us[0],us[reps-1],rcbad,firstmm);
        for(int r=0;r<reps && r<64;r++) printf("%.1f%s",raw[r],r+1<reps?" ":"");
        printf("\n"); fflush(stdout);
    }
    ork_mm_free(c,w); ork_npu_free(c); return 0;
}
