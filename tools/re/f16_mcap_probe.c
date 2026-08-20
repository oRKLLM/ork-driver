/* tools/re/f16_mcap_probe.c — map the TRUE fp16 M-tile envelope, per K, per entrypoint.
 *
 * WHY. Two bounds on fp16 M disagree by ~5x and nobody has measured which is real:
 *   (a) mg_max*64 — the 0x1040 (RK_CNA_CBUF_CON0) K-reduction schedule ceiling. orki_f16_synth
 *       programs v = base - slope*(ceil(M/64)-1) and SATURATES it at 0x1b; mg_max is the last
 *       mg before that clamp, so past it the HW reduces over the wrong number of K-groups.
 *       320 @K=512, 128 @K=1024. This is the int8 result transposed (fp16 scale=K/256 vs
 *       int8 K/512), where mg_max*64 was validated exact and its adoption was worth 2.1x.
 *   (b) M*K <= 32768 — the fp16 doorbell cap (src/npu/i8/dyn.c). = R = pow2_floor(cbuf/K),
 *       the CBUF row count feeding 0x1010, which ork_regs.h labels a *perf hint*. 64 @K=512.
 *       Archaeology (62e937c) shows it shipped as caution while fp16 was WIP-gated for a
 *       write-order flakiness that was LATER root-caused to a zero-copy-DMA-A test bug.
 *
 * WHAT. For each K, step M upward on each entrypoint and find the largest M that still matches
 * a reference, then print measured-vs-predicted for both bounds.
 *
 * REFERENCE = per-row M=1 submits of the SAME weight (in-envelope at every K, so it cannot
 * inherit a formula error the way a tiled reference would), cross-checked against a CPU double
 * accumulation. A has deterministic VARYING rows — constant rows would hide exactly the
 * cross-row error we are hunting.
 *
 * PATHS. chain = ork_f16_mm_run_stream_chain (single M-tile, NO M bound today — the hole, and
 * the only unbounded sweep vehicle); stream = ork_f16_mm_run_stream (doorbell, refuses above
 * min(64, 32768/K)); run = ork_f16_mm_run (M-tiles internally — the control, should never fail).
 *
 * SAFETY. Steps M upward through an explicit ladder (never jumps into unknown territory) and
 * stops a K once it has seen the failure confirmed. Out-of-envelope M is expected to
 * MISCOMPUTE, not wedge, but the far tail is unproven — run under npu_guard with a timeout.
 *
 *   make f16_mcap_probe
 *   sudo tools/util/npu_guard.sh -- env ORK_MM_TIMEOUT=2500 timeout 600 ./f16_mcap_probe [K...]
 */
#define _GNU_SOURCE
#include "ork_npu.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define NMAX_M 2048

/* mirror of orki_f16_mtile's ceiling (src/npu/f16/regcmd.c) — private, so replicated here on
 * purpose: measured-vs-this-prediction is the whole point of the probe. */
static int pred_mgmax64(int K){
    double scale=(double)K/256.0;
    int base=(int)(177.0-15.0*(scale-1.0)), slope=(int)(15.0*scale);
    if(slope<1) return -1;                       /* undefined (K<32 can't happen: K%32==0) */
    int mg_max = base>=0x1b ? (base-0x1b)/slope+1 : 0;
    return mg_max*64;                            /* 0 => even mg=1 saturates */
}
/* (c) the REAL-ARITHMETIC ceiling. orki_f16_mtile computes mg_max with INTEGER division, so it
 * truncates whenever slope does not divide (base-0x1b). But the HW clamp is on v itself: the
 * largest usable mg is where v == 0x1b exactly, i.e. mg = (base-0x1b)/slope + 1 in REALS.
 *   K=512: (162-27)/30 + 1 = 5.5 -> 5.5*64 = 352, vs the truncated 5*64 = 320.
 * Discovered from the K=512 sweep: the M=512 run's first bad row was exactly 352. */
static int pred_real(int K){
    double scale=(double)K/256.0;
    int base=(int)(177.0-15.0*(scale-1.0)), slope=(int)(15.0*scale);
    if(slope<1||base<0x1b) return 0;
    return (int)(((double)(base-0x1b)/(double)slope + 1.0) * 64.0);
}
static int pred_R(int K){ int cbuf=32768, R=cbuf/K; if(R<1)R=1; int p=1; while(p*2<=R)p*=2; return p; }
static int is_pow2(int K){ return K>0 && (K&(K-1))==0; }
/* the two entrypoints' sched predicates — they DIVERGE at K>=2048 */
static int sched_chain (int K){ return is_pow2(K) && K>=128 && K<2048; }
static int sched_stream(int K){ return is_pow2(K) && K>=128; }

static double relerr(const float *got,const float *ref,size_t n){
    double num=0,den=0;
    for(size_t i=0;i<n;i++){ double d=(double)got[i]-(double)ref[i]; num+=d*d; den+=(double)ref[i]*(double)ref[i]; }
    return den>0?sqrt(num/den):sqrt(num);
}
/* first row whose output departs from the reference — localises an M-tile break to a row index */
static int first_bad_row(const float *got,const float *ref,int M,int N,double tol){
    for(int m=0;m<M;m++) for(int n=0;n<N;n++){
        double a=got[(size_t)m*N+n], b=ref[(size_t)m*N+n];
        double d=fabs(a-b), s=fabs(b);
        if(d > tol*(s>1.0?s:1.0)) return m; }
    return -1;
}

enum { P_CHAIN=0, P_STREAM=1, P_RUN=2, NPATH=3 };
static const char *pname[NPATH]={"chain","stream","run"};

static int run_path(ork_npu *c,int p,ork_w *w,int M,const ork_f16 *A,float *C){
    ork_mm_task_f16 t={w,M,A,C};
    switch(p){
        case P_CHAIN:  return ork_f16_mm_run_stream_chain(c,1,&t);
        case P_STREAM: return ork_f16_mm_run_stream(c,1,&t);
        default:       return ork_f16_mm_run(c,w,M,A,C);
    }
}

int main(int argc,char**argv){
    static const int KDEF[]={256,512,1024,2048,3072,384,4096};
    const int *KS = KDEF; int nk=(int)(sizeof KDEF/sizeof*KDEF);
    int kbuf[32];
    if(argc>1){ nk=0; for(int i=1;i<argc && nk<32;i++) kbuf[nk++]=atoi(argv[i]); KS=kbuf; }
    const int N=64;
    const double TOL=1e-3;   /* a schedule break is a partial-K sum, orders of magnitude > fp16 rounding */

    ork_npu *c=ork_npu_init(); if(!c){ printf("no board\n"); return 0; }
    printf("f16_mcap_probe — SoC=%s  N=%d  tol=%.0e\n",ork_npu_soc(c),N,TOL);
    printf("%-6s %-7s %-9s %-9s %-9s %-7s | %-8s %-8s %-8s\n",
           "K","sched","mg_max*64","real-arith","R=cbuf/K","cpu-rel","chain","stream","run");
    printf("%-6s %-7s %-9s %-9s %-9s %-7s | %-8s %-8s %-8s\n",
           "","c/s","(pred a)","(pred c)","(pred b)","M=1","max ok M","max ok M","max ok M");

    int anyfail=0;
    for(int ki=0;ki<nk;ki++){
        int K=KS[ki];
        if(K<32||K%32){ printf("%-6d SKIP (K%%32)\n",K); continue; }
        int pa=pred_mgmax64(K), pb=pred_R(K), pc=pred_real(K);

        /* M ladder: dense around both predictions, then beyond. Upward only. */
        int ladder[64],nl=0;
        for(int m=1;m<=64;m*=2) ladder[nl++]=m;
        int cand[]={ pb, pb+1, pb*2, pa>0?pa/2:0, pa>0?pa-1:0, pa, pa>0?pa+1:0,
                     pc>0?pc-1:0, pc, pc>0?pc+1:0, pa>0?pa*2:0, 512, 1024 };
        for(size_t i=0;i<sizeof cand/sizeof*cand;i++){
            int m=cand[i]; if(m<1||m>NMAX_M) continue;
            int dup=0; for(int j=0;j<nl;j++) if(ladder[j]==m) dup=1;
            if(!dup && nl<64) ladder[nl++]=m; }
        for(int i=0;i<nl;i++) for(int j=i+1;j<nl;j++) if(ladder[j]<ladder[i]){int t=ladder[i];ladder[i]=ladder[j];ladder[j]=t;}
        int Mmax=ladder[nl-1];

        size_t szA=(size_t)Mmax*K, szC=(size_t)Mmax*N;
        ork_f16 *A=malloc(szA*sizeof *A), *B=malloc((size_t)K*N*sizeof *B);
        float *C=malloc(szC*sizeof *C), *ref=malloc(szC*sizeof *ref), *cpu=malloc(szC*sizeof *cpu);
        if(!A||!B||!C||!ref||!cpu){ printf("%-6d OOM\n",K); free(A);free(B);free(C);free(ref);free(cpu); continue; }

        /* deterministic, row-VARYING A; small-magnitude B so K-sums stay in fp16-friendly range */
        for(int m=0;m<Mmax;m++) for(int k=0;k<K;k++)
            A[(size_t)m*K+k]=(ork_f16)(0.5f+0.001f*(float)(((m*7+k*3)%17)-8));
        for(int k=0;k<K;k++) for(int n=0;n<N;n++)
            B[(size_t)k*N+n]=(ork_f16)(0.002f*(float)(((k+n*5)%11)-5));

        ork_w *w=ork_f16_mm_pack(c,K,N,B);
        if(!w){ printf("%-6d pack fail\n",K); free(A);free(B);free(C);free(ref);free(cpu); continue; }

        /* REFERENCE: per-row M=1 submits (in-envelope at every K) */
        int rref=0;
        for(int m=0;m<Mmax && !rref;m++) rref=ork_f16_mm_run(c,w,1,A+(size_t)m*K,ref+(size_t)m*N);
        if(rref){ printf("%-6d M=1 reference failed rc=%d\n",K,rref);
            ork_mm_free(c,w); free(A);free(B);free(C);free(ref);free(cpu); anyfail=1; continue; }

        /* CPU double cross-check of the M=1 reference itself (row 0 only — O(N*K)) */
        for(int n=0;n<N;n++){ double a=0; for(int k=0;k<K;k++) a+=(double)A[k]*(double)B[(size_t)k*N+n]; cpu[n]=(float)a; }
        double cpurel=relerr(ref,cpu,(size_t)N);

        int maxok[NPATH]={0,0,0}; int firstbad[NPATH]={-1,-1,-1}; int rcbad[NPATH]={0,0,0};
        int badrow[NPATH]={-1,-1,-1}; int hibad[NPATH]={0,0,0};
        for(int p=0;p<NPATH;p++){
            int done=0;
            for(int i=0;i<nl && !done;i++){
                int M=ladder[i];
                memset(C,0,(size_t)M*N*sizeof *C);
                int rc=run_path(c,p,w,M,A,C);
                if(rc){ if(!rcbad[p]) rcbad[p]=rc; if(firstbad[p]<0) firstbad[p]=-2; hibad[p]=M; done=1; break; }  /* refused */
                int fb=first_bad_row(C,ref,M,N,TOL);
                if(fb>=0){ if(firstbad[p]<0) firstbad[p]=fb; badrow[p]=fb; hibad[p]=M; done=1; break; }            /* MISCOMPUTE */
                maxok[p]=M;
            }
            /* BISECT the exact boundary in (maxok, hibad) — the ladder is coarse up there, and the
             * whole point is the exact ceiling, not the nearest ladder rung. Miscompute only; a
             * refusal boundary is a software constant, not worth the submits. */
            if(firstbad[p]>=0 && hibad[p]>maxok[p]+1){
                int lo=maxok[p], hi=hibad[p];
                while(hi-lo>1){
                    int mid=lo+(hi-lo)/2;
                    memset(C,0,(size_t)mid*N*sizeof *C);
                    int rc=run_path(c,p,w,mid,A,C);
                    if(rc){ hi=mid; continue; }
                    int fb=first_bad_row(C,ref,mid,N,TOL);
                    if(fb>=0){ hi=mid; badrow[p]=fb; } else lo=mid;
                }
                maxok[p]=lo;
            }
        }

        printf("%-6d %d/%d     %-9d %-9d %-9d %-7.1e | %-8d %-8d %-8d\n",
               K, sched_chain(K), sched_stream(K), pa, pc, pb, cpurel, maxok[0],maxok[1],maxok[2]);
        for(int p=0;p<NPATH;p++){
            if(firstbad[p]==-2) printf("        %-6s refused rc=%d above M=%d\n",pname[p],rcbad[p],maxok[p]);
            else if(firstbad[p]>=0) printf("        %-6s exact ceiling M=%d; M=%d miscomputes (first bad row %d)\n",
                                           pname[p],maxok[p],maxok[p]+1,badrow[p]);
        }
        /* the verdict this probe exists to produce */
        {
            const char *v = (maxok[0]==pc) ? "chain ceiling == REAL-arith ceiling (c) CONFIRMED — mg_max*64 under-reports"
                          : (maxok[0]==pa) ? "chain ceiling == mg_max*64 (a) CONFIRMED"
                          : (maxok[0]==pb) ? "chain ceiling == R (b) CONFIRMED"
                          : "chain ceiling matches NEITHER prediction";
            printf("        => %s   (measured %d: a=%d c=%d b=%d)\n",v,maxok[0],pa,pc,pb);
        }
        if(maxok[2]<Mmax && firstbad[2]>=0){ printf("        !! ork_f16_mm_run (tiled control) FAILED — its own M-tiling is broken\n"); anyfail=1; }

        ork_mm_free(c,w); free(A);free(B);free(C);free(ref);free(cpu);
    }
    ork_npu_free(c);
    return anyfail?1:0;
}
