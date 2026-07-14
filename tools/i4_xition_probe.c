/* i4_xition_probe — manufacture the {fp16,int8} -> int4 mode transition across a wide K/N/M sweep to
 * decide whether the XP_I4_* profiles' size-clear (buffer-realloc) gating is gratuitous drift or
 * mechanism-legit. The int4 paths are dormant/experimental in a real pipeline, so exercise them DIRECTLY.
 *
 * Profiles under test (per the mode-transition layer): run_i4 -> XP_I4_MC (clears mccsz on !nothrash),
 * run_i4_incr -> XP_I4_INCR (clears nothing, caller-local warm), run_chain_i4 -> XP_I4CHAIN (uncond
 * reset), run_stream_i4 -> XP_I4_STREAM (uncond reset). Question: entering int4 from fp16/int8, does
 * each stay COHERENT regardless of its clear policy? If all coherent, the size-clear differences are a
 * perf/realloc choice (candidate to converge); if one miscomputes, its clear was load-bearing.
 *
 * A=B=1 (int4 value 1) => C[m][n] = sum_k 1*1 = K (int32). Self-validating; exits nonzero on incoherence.
 * BOARD: sudo ./i4_xition_probe [K N M] (no args = sweep). K%32, N%64.  (ORK_XPROF=1 to see profiles.)
 */
#include "ork_npu.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

enum { S = 3 };
static double now_us(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec*1e6 + t.tv_nsec/1e3; }

/* contaminators: set c->last_dt to F16 / I8 so the int4 op sees a cross-precision predecessor */
static void contaminate_f16(ork_npu *c){
    static ork_f16 *A=NULL,*B=NULL; static float *C=NULL;
    if(!A){ A=malloc(64*64*2); B=malloc(64*16*2); C=malloc(64*16*4);
        for(int i=0;i<64*64;i++)A[i]=(ork_f16)0.01f; for(int i=0;i<64*16;i++)B[i]=(ork_f16)(1.0f/64); }
    ork_bmm_fp16(c,1,64,64,16,A,B,C);
}
static void contaminate_i8(ork_npu *c){
    static int8_t *A=NULL,*B=NULL; static int32_t *C=NULL;
    if(!A){ A=malloc(64*64); B=malloc(64*64); C=malloc(64*64*4); memset(A,1,64*64); memset(B,1,64*64); }
    ork_bmm_i8(c,1,64,64,64,A,B,C);
}

/* run int4 `mech` once; C must be all K. mech: 0=run_i4 1=run_i4_incr 2=chain_i4 3=stream_i4 */
static double run_mech(ork_npu *c,int mech,ork_w *w,int M,int N,int K,int8_t *A,int32_t *C,
                       ork_mm_task_i4 *tasks,int *bad){
    memset(C,0,(size_t)M*N*4);
    for(int i=0;i<S;i++) memset(tasks[i].C,0,(size_t)tasks[i].M*N*4);
    double t0=now_us(); int rc;
    switch(mech){
        case 0: rc=ork_mm_run_i4(c,w,M,A,C); break;
        case 1: rc=ork_mm_run_i4_incr(c,w,M,A,C); break;
        case 2: rc=ork_mm_run_chain_i4(c,S,tasks); break;   /* M=1 tasks */
        default:rc=ork_mm_run_stream_i4(c,S,tasks); break;
    }
    double us=now_us()-t0;
    *bad=0;
    if(rc){ *bad=-1; return us; }
    if(mech<2){ for(size_t e=0;e<(size_t)M*N;e++) if(C[e]!=K){ (*bad)++; break; } }
    else { for(int i=0;i<S;i++){ int32_t*Ci=tasks[i].C; for(size_t e=0;e<(size_t)tasks[i].M*N;e++) if(Ci[e]!=K){ (*bad)++; break; } } }
    return us;
}

static const char *MECH[4]={"run_i4    ","run_i4_incr","chain_i4  ","stream_i4 "};
static int probe_shape(ork_npu *c,int K,int N,int M){
    int8_t *B=malloc((size_t)K*N); memset(B,1,(size_t)K*N);      /* int4 value 1 */
    ork_w *w=ork_mm_pack_i4(c,K,N,B);
    if(!w){ printf("  K=%-5d N=%-5d M=%-4d  pack_i4 FAILED (unsupported shape) — skip\n",K,N,M); free(B); return 0; }
    int8_t *A=malloc((size_t)M*K); memset(A,1,(size_t)M*K);
    int32_t *C=malloc((size_t)M*N*4);
    /* chain/stream tasks: M=1 each (chain_i4 requires M==1), own A/C */
    ork_mm_task_i4 tasks[S]; int8_t *tA[S]; int32_t *tC[S];
    for(int i=0;i<S;i++){ tA[i]=malloc((size_t)1*K); memset(tA[i],1,(size_t)K); tC[i]=malloc((size_t)1*N*4);
        tasks[i]=(ork_mm_task_i4){w,1,tA[i],tC[i]}; }
    int fail=0;
    for(int mech=0;mech<4;mech++){
        int Mm = (mech>=2)?1:M;   /* chain/stream use M=1 tasks */
        int bc; double uc=run_mech(c,mech,w,Mm,N,K,A,C,tasks,&bc);   /* int4-live predecessor (prior mech left int4) */
        if(bc<0){ printf("  K=%-5d N=%-5d M=%-4d  %s  UNSUPPORTED (rc!=0)\n",K,N,Mm,MECH[mech]); continue; }
        contaminate_f16(c); int bf; double uf=run_mech(c,mech,w,Mm,N,K,A,C,tasks,&bf);   /* after fp16 */
        contaminate_i8(c);  int bi; double ui=run_mech(c,mech,w,Mm,N,K,A,C,tasks,&bi);   /* after int8 */
        if(bc||bf||bi) fail=1;
        printf("  K=%-5d N=%-5d M=%-4d  %s  int4-live=%s  afterF16=%s  afterI8=%s  | us: live=%.0f F16=%.0f I8=%.0f\n",
               K,N,Mm,MECH[mech], bc?"NO":"ok", bf?"NO":"ok", bi?"NO":"ok", uc,uf,ui);
    }
    ork_w_free(w); free(B); free(A); free(C);
    for(int i=0;i<S;i++){ free(tA[i]); free(tC[i]); }
    return fail;
}

int main(int argc,char**argv){
    ork_npu *c=ork_npu_init(); if(!c){ fprintf(stderr,"[i4_xition] no NPU — skip\n"); return 0; }
    fprintf(stderr,"[i4_xition] SoC=%s cores=%d\n",ork_npu_soc(c),ork_npu_cores(c));
    int fail=0;
    if(argc>=4){ fail=probe_shape(c,atoi(argv[1]),atoi(argv[2]),atoi(argv[3])); }
    else {
        int Ks[]={512,1024,2048,4096}, Ns[]={512,1024,2048}, Msw[]={1,128,256};
        printf("{fp16,int8} -> int4 transition sweep (A=B=1 => C==K; reset-cost = after-pred minus warm)\n");
        for(unsigned ki=0;ki<sizeof Ks/sizeof*Ks;ki++)
          for(unsigned ni=0;ni<sizeof Ns/sizeof*Ns;ni++)
            for(unsigned mi=0;mi<sizeof Msw/sizeof*Msw;mi++)
              fail |= probe_shape(c,Ks[ki],Ns[ni],Msw[mi]);
    }
    printf("\n%s\n", fail?"I4_XITION: FAIL (incoherent output)":"I4_XITION: PASS (all coherent)");
    ork_npu_free(c);
    return fail?1:0;
}
