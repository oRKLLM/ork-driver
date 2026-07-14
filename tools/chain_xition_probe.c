/* chain_xition_probe — manufacture the fp16 -> int8-chain transition across a wide K/M/N sweep and
 * measure (a) COHERENCE and (b) the reset cost the drifted profiles impose.
 *
 * The sw/hw/fused chain profiles are dormant in a real dense pipeline (the ggml graph never presents
 * chainable consecutive int8 mul_mats), so this exercises them DIRECTLY: for each shape it warms an
 * int8 chain mechanism, runs an fp16 matmul (last_dt := DT_F16), then re-runs the mechanism — the
 * "after-fp16" run pays whatever the profile's fp16->chain policy costs. XP_STREAM_I8 keeps warm
 * (ORK_SSM_KEEPWARM) so it should be ~free; XP_CHAIN_NT ignores f16warm (KWP_NTL) so it ACT_RESETs
 * (~105us). Coherence (C==K for all-ones operands) must hold in BOTH — proving the reset is a perf
 * drift, not a correctness requirement.
 *
 * A=B=1 => C[m][n] = sum_k 1*1 = K (int32). Self-validating: exits nonzero on any incoherent output.
 * BOARD: sudo ./chain_xition_probe [K N M]   (no args = sweep). Full-K only (K%512, K<=4096).
 */
#include "ork_npu.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

enum { S = 3 };                 /* independent tasks (multi-core for stream) */
static double now_us(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec*1e6 + t.tv_nsec/1e3; }

/* one fp16 matmul to force last_dt = DT_F16 (the contaminator) */
static void contaminate_f16(ork_npu *c){
    static ork_f16 *A=NULL,*B=NULL; static float *C=NULL;
    if(!A){ A=malloc(64*64*2); B=malloc(64*16*2); C=malloc(64*16*4);
        for(int i=0;i<64*64;i++)A[i]=(ork_f16)0.01f; for(int i=0;i<64*16;i++)B[i]=(ork_f16)(1.0f/64); }
    ork_bmm_fp16(c,1,64,64,16,A,B,C);
}

/* run `mech` (0=stream,1=chain) once; return us, set *bad to mismatch count (expect C==K) */
static double run_mech(ork_npu *c,int mech,ork_mm_task_i8 *tasks,int K,int N,int *bad){
    for(int i=0;i<S;i++) memset(tasks[i].C,0,(size_t)tasks[i].M*N*4);
    double t0=now_us();
    int rc = mech ? ork_mm_run_chain_i8(c,S,tasks) : ork_mm_run_stream_i8(c,S,tasks);
    double us=now_us()-t0;
    *bad=0;
    if(rc){ *bad=-1; return us; }
    for(int i=0;i<S;i++){ int32_t *Ci=tasks[i].C; size_t mn=(size_t)tasks[i].M*N;
        for(size_t e=0;e<mn;e++) if(Ci[e]!=K){ (*bad)++; break; } }
    return us;
}

static int probe_shape(ork_npu *c,int K,int N,int M){
    int Ms[S]; for(int i=0;i<S;i++) Ms[i]=M;
    int8_t *A[S]={0},*B=NULL; int32_t *C[S]={0}; ork_w *w=NULL; ork_mm_task_i8 tasks[S];
    B=malloc((size_t)K*N); memset(B,1,(size_t)K*N);
    w=ork_mm_pack_i8(c,K,N,B);
    if(!w){ printf("  K=%-5d N=%-5d M=%-4d  pack FAILED\n",K,N,M); free(B); return 0; }  /* shape unsupported: skip */
    for(int i=0;i<S;i++){ A[i]=malloc((size_t)M*K); memset(A[i],1,(size_t)M*K); C[i]=malloc((size_t)M*N*4);
        tasks[i]=(ork_mm_task_i8){w,M,A[i],C[i]}; }
    int fail=0;
    for(int mech=0;mech<2;mech++){
        const char *nm = mech?"chain(hw)":"stream(sw)";
        int b; run_mech(c,mech,tasks,K,N,&b);          /* warm (cold-start priming) */
        if(b<0){ printf("  K=%-5d N=%-5d M=%-4d  %-10s  UNSUPPORTED (rc!=0)\n",K,N,M,nm); continue; }
        contaminate_f16(c);                            /* last_dt := F16 */
        int ba; double ua=run_mech(c,mech,tasks,K,N,&ba);   /* after fp16 (pays the transition) */
        int bw; double uw=run_mech(c,mech,tasks,K,N,&bw);   /* warm again (chain->chain) */
        int coherent = (ba==0 && bw==0);
        if(!coherent) fail=1;
        printf("  K=%-5d N=%-5d M=%-4d  %-10s  coherent=%s  after-fp16=%7.1fus  warm=%7.1fus  reset-cost=%7.1fus\n",
               K,N,M,nm, coherent?"YES":"NO ", ua, uw, ua-uw);
    }
    ork_w_free(w); free(B); for(int i=0;i<S;i++){ free(A[i]); free(C[i]); }
    return fail;
}

int main(int argc,char**argv){
    ork_npu *c=ork_npu_init(); if(!c){ fprintf(stderr,"[chain_xition] no NPU — skip\n"); return 0; }
    fprintf(stderr,"[chain_xition] SoC=%s cores=%d\n",ork_npu_soc(c),ork_npu_cores(c));
    int fail=0;
    if(argc>=4){ fail=probe_shape(c,atoi(argv[1]),atoi(argv[2]),atoi(argv[3])); }
    else {
        int Ks[]={512,1024,2048,3072,4096}, Ns[]={512,1024,2048}, Msw[]={1,32,128,256};
        printf("fp16 -> int8-chain transition sweep (A=B=1 => C==K; reset-cost = after-fp16 minus warm)\n");
        for(unsigned ki=0;ki<sizeof Ks/sizeof*Ks;ki++)
          for(unsigned ni=0;ni<sizeof Ns/sizeof*Ns;ni++)
            for(unsigned mi=0;mi<sizeof Msw/sizeof*Msw;mi++)
              fail |= probe_shape(c,Ks[ki],Ns[ni],Msw[mi]);
    }
    printf("\n%s\n", fail?"CHAIN_XITION: FAIL (incoherent output)":"CHAIN_XITION: PASS (all coherent)");
    ork_npu_free(c);
    return fail?1:0;
}
