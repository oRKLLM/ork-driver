/* tools/re/f16_k128_probe.c — root-cause the K=128 fp16 anomaly.
 *
 * f16_mcap_probe measured, at K=128 (sched=1): ceiling 256, while the 0x1040 K-reduction
 * schedule permits ~1499. Two things make it unlike every other sched=1 break:
 *   - other K report "first bad row == ceiling" (a TAIL truncation: rows below are correct)
 *   - K=128 reports "first bad row 0" (the WHOLE output is wrong)
 *
 * HYPOTHESIS. 0x1040 (RK_CNA_CBUF_CON0) is not only the K-schedule — ork_regs.h documents it as
 * DATA_BANK[3:0] / WEIGHT_BANK[7:4] bank split that ALSO carries the schedule. orki_f16_synth
 * writes v = base - slope*(mg-1) into the whole register, so every mg step REALLOCATES THE CBUF
 * BANKS. At K=128 base=184 slope=7, so v walks slowly through many splits:
 *     M=256 -> mg=4 -> v=163=0xA3 -> DATA_BANK=3,  WEIGHT_BANK=10
 *     M=257 -> mg=5 -> v=156=0x9C -> DATA_BANK=12, WEIGHT_BANK=9
 * If a split starves the WEIGHT bank the weights are corrupt for EVERY row -> first bad row 0.
 * At K=512/1024 slope is 30/60, so v saturates at 0x1b before reaching a bad split -> tail break.
 *
 * PREDICTION THAT DISCRIMINATES. If the limit is bank-split-driven rather than a row count, then
 * correctness is NON-MONOTONIC in M: some larger M (a different v) works again. A row-count or
 * area limit can only ever be monotonic. f16_mcap_probe stops at the first failure so it could
 * not see a recovery — this probe walks the WHOLE range regardless of failures.
 *
 * Also sweeps N to separate a row cap from an output-size (M*N) cap, and prints the failure
 * PATTERN (all-zero / scaled / garbage), which distinguishes "submit did nothing" from
 * "computed the wrong thing".
 *
 *   make f16_k128_probe
 *   sudo tools/util/npu_guard.sh -- env ORK_MM_TIMEOUT=2500 timeout 600 ./f16_k128_probe [K] [N]
 */
#define _GNU_SOURCE
#include "ork_npu.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* replicate orki_f16_synth's sched arithmetic so we can label each M with the v it programs */
static void sched_of(int K,int M,int *mg,int *v,int *sat){
    double scale=(double)K/256.0;
    int base=(int)(177.0-15.0*(scale-1.0)), slope=(int)(15.0*scale);
    int g=(M+63)/64; if(g<1)g=1;
    int vv=base-slope*(g-1);
    *sat = (vv<0x1b);
    if(vv<0x1b) vv=0x1b;
    *mg=g; *v=vv;
}
static int is_pow2(int K){ return K>0 && (K&(K-1))==0; }

/* classify HOW the output is wrong — the shape of the error names the mechanism */
static const char *pattern(const float *got,const float *ref,int M,int N,double *worst){
    int nz=0, n=M*N; double mx=0, ratio_sum=0; int ratio_n=0;
    for(int i=0;i<n;i++){
        if(got[i]!=0.0f) nz++;
        double d=fabs((double)got[i]-(double)ref[i]); if(d>mx) mx=d;
        if(fabs(ref[i])>1e-6){ ratio_sum += (double)got[i]/(double)ref[i]; ratio_n++; }
    }
    *worst=mx;
    if(nz==0) return "ALL-ZERO (submit produced nothing)";
    double r = ratio_n? ratio_sum/ratio_n : 0;
    if(fabs(r-1.0)<0.02) return "near-correct (rounding only)";
    if(fabs(r)>0.02 && fabs(r-floor(r+0.5))<0.02) return "INTEGER-SCALED (partial-K / repeated sum)";
    return "GARBAGE (unrelated values)";
}
static int first_bad_row(const float *got,const float *ref,int M,int N,double tol){
    for(int m=0;m<M;m++) for(int n=0;n<N;n++){
        double d=fabs((double)got[(size_t)m*N+n]-(double)ref[(size_t)m*N+n]), s=fabs((double)ref[(size_t)m*N+n]);
        if(d > tol*(s>1.0?s:1.0)) return m; }
    return -1;
}

int main(int argc,char**argv){
    int K=argc>1?atoi(argv[1]):128;
    int Nfix=argc>2?atoi(argv[2]):64;
    int do_pass2=argc>3;   /* PASS 2 bisects, which is INVALID on a non-monotonic predicate — opt-in only */
    const double TOL=1e-3;
    ork_npu *c=ork_npu_init(); if(!c){ printf("no board\n"); return 0; }

    printf("f16_k128_probe — SoC=%s  K=%d  sched=%d\n",ork_npu_soc(c),K,is_pow2(K)&&K>=128&&K<2048);

    /* ---------- PASS 1: walk M across many mg steps, NEVER stopping at a failure ---------- */
    printf("\n== PASS 1: M walk at N=%d (looking for NON-MONOTONIC recovery) ==\n",Nfix);
    printf("%-6s %-4s %-6s %-5s %-5s %-6s %-9s %s\n","M","mg","v(hex)","DBNK","WBNK","sat","firstbad","verdict / pattern");

    int Mlist[128], nM=0;
    for(int g=1;g<=24 && nM<120;g++){                 /* one point just below and at each mg step */
        int m1=g*64, m0=m1-1;
        if(m0>=1) Mlist[nM++]=m0;
        Mlist[nM++]=m1;
    }
    int Mmax=Mlist[nM-1];

    ork_f16 *A=malloc((size_t)Mmax*K*sizeof *A), *B=malloc((size_t)K*Nfix*sizeof *B);
    float *C=malloc((size_t)Mmax*Nfix*sizeof *C), *ref=malloc((size_t)Mmax*Nfix*sizeof *ref);
    if(!A||!B||!C||!ref){ printf("OOM\n"); return 2; }
    for(int m=0;m<Mmax;m++) for(int k=0;k<K;k++)
        A[(size_t)m*K+k]=(ork_f16)(0.5f+0.001f*(float)(((m*7+k*3)%17)-8));
    for(int k=0;k<K;k++) for(int n=0;n<Nfix;n++)
        B[(size_t)k*Nfix+n]=(ork_f16)(0.002f*(float)(((k+n*5)%11)-5));

    ork_w *w=ork_f16_mm_pack(c,K,Nfix,B);
    if(!w){ printf("pack fail\n"); return 2; }
    for(int m=0;m<Mmax;m++) if(ork_f16_mm_run(c,w,1,A+(size_t)m*K,ref+(size_t)m*Nfix)){ printf("ref fail row %d\n",m); return 2; }

    int nok=0,nbad=0,recovered=0,lastbad=0;
    for(int i=0;i<nM;i++){
        int M=Mlist[i], mg,v,sat; sched_of(K,M,&mg,&v,&sat);
        memset(C,0,(size_t)M*Nfix*sizeof *C);
        ork_mm_task_f16 t={w,M,A,C};
        int rc=ork_f16_mm_run_stream_chain(c,1,&t);
        if(rc){ printf("%-6d %-4d 0x%-4x %-5d %-5d %-6s %-9s rc=%d\n",M,mg,v,v&0xf,(v>>4)&0xf,sat?"SAT":"-","-",rc); continue; }
        int fb=first_bad_row(C,ref,M,Nfix,TOL);
        double worst; const char *pat = fb<0 ? "OK" : pattern(C,ref,M,Nfix,&worst);
        if(fb<0){ nok++; if(lastbad){ recovered=1; printf("   ^^^ RECOVERED after a failing M — envelope is NON-MONOTONIC\n"); } lastbad=0; }
        else { nbad++; lastbad=1; }
        printf("%-6d %-4d 0x%-4x %-5d %-5d %-6s %-9d %s\n",M,mg,v,v&0xf,(v>>4)&0xf,sat?"SAT":"-",fb,pat);
    }
    printf("\nPASS1: %d ok, %d bad, non-monotonic=%s\n",nok,nbad,recovered?"YES — bank-split hypothesis SUPPORTED":"no — monotonic ceiling (row/area limit)");
    ork_mm_free(c,w); free(A);free(B);free(C);free(ref);

    if(!do_pass2){ ork_npu_free(c); return 0; }
    /* ---------- PASS 2: N sweep — is the cap a ROW count or an OUTPUT-SIZE (M*N) limit? ---------- */
    printf("\n== PASS 2: N sweep at K=%d — row cap vs M*N cap ==\n",K);
    printf("%-6s %-9s %-9s %s\n","N","ceiling","ceiling*N","implies");
    int Ns[]={16,32,64,128,256};
    for(size_t i=0;i<sizeof Ns/sizeof*Ns;i++){
        int N=Ns[i];
        int MM=1024;
        ork_f16 *A2=malloc((size_t)MM*K*sizeof *A2), *B2=malloc((size_t)K*N*sizeof *B2);
        float *C2=malloc((size_t)MM*N*sizeof *C2), *r2=malloc((size_t)MM*N*sizeof *r2);
        if(!A2||!B2||!C2||!r2){ printf("%-6d OOM\n",N); free(A2);free(B2);free(C2);free(r2); continue; }
        for(int m=0;m<MM;m++) for(int k=0;k<K;k++) A2[(size_t)m*K+k]=(ork_f16)(0.5f+0.001f*(float)(((m*7+k*3)%17)-8));
        for(int k=0;k<K;k++) for(int n=0;n<N;n++)  B2[(size_t)k*N+n]=(ork_f16)(0.002f*(float)(((k+n*5)%11)-5));
        ork_w *w2=ork_f16_mm_pack(c,K,N,B2);
        if(!w2){ printf("%-6d pack fail\n",N); free(A2);free(B2);free(C2);free(r2); continue; }
        int ok=0;
        for(int m=0;m<MM;m++) if(ork_f16_mm_run(c,w2,1,A2+(size_t)m*K,r2+(size_t)m*N)){ ok=-1; break; }
        if(ok==0){
            int lo=0,hi=MM+1;                                     /* bisect the ceiling */
            while(hi-lo>1){ int mid=lo+(hi-lo)/2;
                memset(C2,0,(size_t)mid*N*sizeof *C2);
                ork_mm_task_f16 t2={w2,mid,A2,C2};
                int rc=ork_f16_mm_run_stream_chain(c,1,&t2);
                if(rc || first_bad_row(C2,r2,mid,N,TOL)>=0) hi=mid; else lo=mid; }
            ok=lo;
        }
        printf("%-6d %-9d %-9d %s\n",N,ok,ok*N, ok*N==16384?"M*N==16384 (output-size cap)":"");
        ork_mm_free(c,w2); free(A2);free(B2);free(C2);free(r2);
    }
    ork_npu_free(c);
    return 0;
}
