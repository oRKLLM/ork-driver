/* tools/re/i4_wb_sweep.c — map the int4 BCHAIN (Wb, H) dispatch surface on THIS board.
 *
 * The NBc>=2 "capacity law" was refuted; what is left is a RACE with a lower bound on H. This walks
 * H UPWARD (never bisects: the envelope is not monotone) for each candidate N-tile width Wb, grading
 * every cell by PASS-RATE against the Wb=64 H=2 anchor at the same (K,N,M) and by median wall time.
 * Both gates: bit-exact AND not time-inflated. One pack, one context; ORK_I4_H / ORK_I4_WB are re-read
 * per call, so a single process can sweep without re-paying the pack.
 *
 *   sudo tools/util/npu_guard.sh -- timeout -s TERM 3600 env ORK_I4_NOLADDER=1 \
 *        ./i4_wb_sweep <K> <N> <M> <Hmax> <reps> <desc?> <Wb,Wb,...>
 */
#define _GNU_SOURCE
#include "ork_npu.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double now_us(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec*1e6+t.tv_nsec/1e3; }
static unsigned long long fnv64(const void*p,size_t n){ const unsigned char*b=p; unsigned long long h=1469598103934665603ULL;
    for(size_t i=0;i<n;i++){ h^=b[i]; h*=1099511628211ULL; } return h; }
static unsigned s_=777u; static int rnd(int lo,int hi){ s_=s_*1103515245u+12345u; return lo+(int)((s_>>16)%(unsigned)(hi-lo+1)); }
static int cmpd(const void*a,const void*b){ double x=*(const double*)a,y=*(const double*)b; return x<y?-1:x>y; }

static ork_npu *C_; static ork_w *W_; static int8_t *A_; static int32_t *Cb_; static int M_,N_;

/* one cell: `reps` calls at (Wb,H); returns pass count, fills med/min and the first checksum seen */
static int cell(int Wb,int H,int reps,unsigned long long ref,double*med,double*mn,unsigned long long*ck0,int*rcbad){
    char b[32]; snprintf(b,sizeof b,"%d",Wb); setenv("ORK_I4_WB",b,1);
    snprintf(b,sizeof b,"%d",H); setenv("ORK_I4_H",b,1);
    double *us=malloc(sizeof(double)*reps); int pass=0; *rcbad=0; *ck0=0;
    for(int r=0;r<reps;r++){
        memset(Cb_,0,(size_t)M_*N_*4);
        double t0=now_us(); int rc=ork_i4_mm_run(C_,W_,M_,A_,Cb_); us[r]=now_us()-t0;
        unsigned long long ck=fnv64(Cb_,(size_t)M_*N_*4); if(!r)*ck0=ck;
        if(rc) (*rcbad)++;
        if(!rc && ck==ref) pass++;
    }
    qsort(us,reps,sizeof(double),cmpd); *med=us[reps/2]; *mn=us[0]; free(us);
    return pass;
}

int main(int argc,char**argv){
    int K=argc>1?atoi(argv[1]):1024, N=argc>2?atoi(argv[2]):1024, M=argc>3?atoi(argv[3]):64;
    int Hmax=argc>4?atoi(argv[4]):16, reps=argc>5?atoi(argv[5]):2, desc=argc>6?atoi(argv[6]):0;
    const char*wbl=argc>7?argv[7]:"128,192,256,320";
    if(getenv("ORK_SWEEP_SEED")) s_=(unsigned)atoi(getenv("ORK_SWEEP_SEED"));
    C_=ork_npu_init(); if(!C_){ printf("no board\n"); return 0; }
    int8_t *B=malloc((size_t)K*N); A_=malloc((size_t)M*K); Cb_=malloc((size_t)M*N*4); M_=M; N_=N;
    for(size_t i=0;i<(size_t)K*N;i++) B[i]=(int8_t)rnd(-8,7);
    for(size_t i=0;i<(size_t)M*K;i++) A_[i]=(int8_t)rnd(-8,7);
    W_=ork_i4_mm_pack(C_,K,N,B); if(!W_){ printf("pack fail\n"); return 2; }

    /* anchor: Wb=64 H=2, must agree across 3 reps or the whole sweep is ungraded */
    double am,an; unsigned long long a0; int arb;
    unsigned long long ref=0;
    { setenv("ORK_I4_WB","64",1); setenv("ORK_I4_H","2",1);
      memset(Cb_,0,(size_t)M*N*4); ork_i4_mm_run(C_,W_,M,A_,Cb_); ref=fnv64(Cb_,(size_t)M*N*4); }
    int ap=cell(64,2,3,ref,&am,&an,&a0,&arb);
    printf("# K=%d N=%d M=%d Hmax=%d reps=%d order=%s anchor(Wb=64,H=2) ck=%016llx agree=%d/3 med=%.1fus\n",
           K,N,M,Hmax,reps,desc?"desc":"asc",ref,ap,am);
    if(ap<3){ printf("# ANCHOR UNSTABLE -- sweep ungraded\n"); }
    printf("Wb\tH\tpass\treps\tmed_us\tmin_us\trc_bad\tck\n");
    fflush(stdout);

    /* Wb=64 reference column, so the H windows are comparable against a same-process baseline */
    if(!getenv("ORK_SWEEP_HLIST")) for(int H=2;H<=Hmax;H++){ double md,mi; unsigned long long ck; int rb;
        int p=cell(64,H,reps,ref,&md,&mi,&ck,&rb);
        printf("64\t%d\t%d\t%d\t%.1f\t%.1f\t%d\t%016llx\n",H,p,reps,md,mi,rb,ck); fflush(stdout); }

    /* ORK_SWEEP_HLIST="h1,h2,..." replaces the contiguous 2..Hmax walk with a named set
     * (phase B: 6 reps at the boundary and at the candidate shipping cells, both orders). */
    int hl[64], hn=0;
    { const char*e=getenv("ORK_SWEEP_HLIST");
      if(e){ char*c=strdup(e); for(char*t=strtok(c,",");t&&hn<64;t=strtok(NULL,",")) hl[hn++]=atoi(t); free(c); } }
    char *tmp=strdup(wbl);
    for(char*t=strtok(tmp,",");t;t=strtok(NULL,",")){ int Wb=atoi(t); if(Wb<=0) continue;
        int ncell = hn ? hn : Hmax-1;
        for(int i=0;i<ncell;i++){ int H = hn ? (desc?hl[ncell-1-i]:hl[i]) : (desc ? (Hmax-i) : (2+i));
            double md,mi; unsigned long long ck; int rb;
            int p=cell(Wb,H,reps,ref,&md,&mi,&ck,&rb);
            printf("%d\t%d\t%d\t%d\t%.1f\t%.1f\t%d\t%016llx\n",Wb,H,p,reps,md,mi,rb,ck); fflush(stdout); } }
    ork_mm_free(C_,W_); ork_npu_free(C_); return 0;
}
