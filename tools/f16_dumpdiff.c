/* f16_dumpdiff — dump the regcmd/task/submit/buffer/bsync state of the WORKING fp16 chain
 * (ork_f16_mm_run_stream_chain) vs the FAILING fp16 doorbell (ork_dyn_begin_mc + ORK_DYN_F16)
 * for the SAME fp16 op (K=512,N=256,M=2), so the two [F16_DUMP] blocks can be diffed word-by-word.
 *   make f16_dumpdiff && sudo env ORK_DYN_F16=1 ORK_F16_DUMP=1 ORK_MM_TIMEOUT=3000 ./f16_dumpdiff [S]
 * Prints the doorbell dump then the chain dump on stderr; a pass/fail on each to stdout. */
#define _GNU_SOURCE
#include "ork_npu.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
static inline void civac(volatile void*p){ __asm__ volatile("dc civac,%0"::"r"(p):"memory"); }
int main(int argc,char**argv){
    int S=argc>1?atoi(argv[1]):4;
    int M=getenv("ORK_E_M")?atoi(getenv("ORK_E_M")):2, K=getenv("ORK_E_K")?atoi(getenv("ORK_E_K")):512, N=getenv("ORK_E_N")?atoi(getenv("ORK_E_N")):256;
    setvbuf(stdout,0,_IONBF,0);
    ork_npu*c=ork_npu_init(); if(!c){printf("init failed\n");return 1;}
    ork_f16 one=(ork_f16)1.0f;
    ork_f16*B=(ork_f16*)malloc((size_t)K*N*sizeof(ork_f16)); for(size_t i=0;i<(size_t)K*N;i++) B[i]=one;
    ork_w*w=ork_f16_mm_pack(c,K,N,B); if(!w){printf("pack fail\n");return 1;}
    ork_f16*Am=(ork_f16*)malloc((size_t)M*K*sizeof(ork_f16)); for(int i=0;i<M*K;i++) Am[i]=one;   /* references: malloc A (zero-copy DMA-A miscomputes) */
    ork_f16*A=(ork_f16*)ork_dma_alloc(c,(size_t)M*K*sizeof(ork_f16)); for(int i=0;i<M*K;i++) A[i]=one;   /* doorbell A (DMA, matches ork_dyn_test E) */
    float*Cd=(float*)ork_dma_alloc(c,(size_t)S*M*N*sizeof(float));   /* doorbell (direct) output */
    float*Cc=(float*)malloc((size_t)S*M*N*sizeof(float));            /* reference output */

    const char*only=getenv("ORK_ONLY");   /* "chain" or "door" to isolate; default runs chain THEN door */
    int iters=getenv("ORK_ITERS")?atoi(getenv("ORK_ITERS")):1;   /* repeat in-process to expose cold vs warm */
    ork_mm_task_f16*tf=malloc(sizeof(ork_mm_task_f16)*S);
    for(int i=0;i<S;i++){ tf[i].w=w; tf[i].M=M; tf[i].A=Am; tf[i].C=Cc+(size_t)i*M*N; }
    ork_mm_task_i8*ti=malloc(sizeof(ork_mm_task_i8)*S);
    ork_f16*doorA = (getenv("ORK_DOOR_A") && !strcmp(getenv("ORK_DOOR_A"),"mal")) ? Am : A;  /* door A src: DMA (default) or malloc */
    for(int i=0;i<S;i++){ ti[i].w=w; ti[i].M=M; ti[i].A=(const int8_t*)doorA; ti[i].C=(int32_t*)(Cd+(size_t)i*M*N); }

    for(int it=0; it<iters; it++){
    /* ---- REFERENCE A: ork_f16_mm_run (submit1/run_multicore) — the canonical bit-exact fp16 path ---- */
    if(only && !strcmp(only,"run")){
        for(int i=0;i<M*N;i++) Cc[i]=-1.0f;
        int rc=ork_f16_mm_run(c,w,M,Am,Cc);
        int okr = (Cc[N-1]>=(float)K-2 && Cc[N-1]<=(float)K+2 && Cc[M*N-1]>=(float)K-2 && Cc[M*N-1]<=(float)K+2);
        printf("[it%d] RUN     : rc=%d %s  C[0][last]=%.1f C[1][last]=%.1f C[0][0]=%.1f\n", it, rc, okr?"OK":"FAIL", Cc[N-1], Cc[M*N-1], Cc[0]);
    }
    /* ---- REFERENCE B: ork_f16_mm_run_stream (non-chain round-robin; the path the passing SSD test uses) ---- */
    if(only && !strcmp(only,"stream")){
        for(int i=0;i<S*M*N;i++) Cc[i]=-1.0f;
        int rc=ork_f16_mm_run_stream(c,S,tf);
        int okc=0;
        for(int i=0;i<S;i++){ int allk=1; for(int m=0;m<M;m++){ float v=Cc[(size_t)i*M*N+(size_t)m*N+(N-1)]; if(v<(float)K-2||v>(float)K+2){allk=0;break;} } if(allk)okc++; }
        printf("[it%d] STREAM  : rc=%d %d/%d ok  C[0..2][last]=%.1f %.1f %.1f  C[0][0]=%.1f\n",
               it, rc, okc, S, Cc[N-1], Cc[M*N+N-1], Cc[2*M*N+N-1], Cc[0]);
    }
    /* ---- fp16 chain (ork_f16_mm_run_stream_chain) ---- */
    if(!only || !strcmp(only,"chain")){
        for(int i=0;i<S*M*N;i++) Cc[i]=-1.0f;
        int rc=ork_f16_mm_run_stream_chain(c,S,tf);
        int okc=0;
        for(int i=0;i<S;i++){ int allk=1; for(int m=0;m<M;m++){ float v=Cc[(size_t)i*M*N+(size_t)m*N+(N-1)]; if(v<(float)K-2||v>(float)K+2){allk=0;break;} } if(allk)okc++; }
        printf("[it%d] CHAIN   : rc=%d %d/%d ok  C[0..2][last]=%.1f %.1f %.1f  C[0][0]=%.1f\n",
               it, rc, okc, S, Cc[N-1], Cc[M*N+N-1], Cc[2*M*N+N-1], Cc[0]);
    }
    /* ---- FAILING: fp16 doorbell (ork_dyn_begin_mc). ORK_DOOR_C=mal => malloc output (direct=0, scratch+copyback
     *      like the references); default DMA output (direct=1, zero-copy). ---- */
    if(!only || !strcmp(only,"door")){
        setenv("ORK_DYN_F16","1",1);
        if(getenv("ORK_DOOR_REFILL")) for(int i=0;i<M*K;i++) doorA[i]=one;   /* re-write A right before use (fresh, unevicted) */
        int indirect = getenv("ORK_DOOR_C") && !strcmp(getenv("ORK_DOOR_C"),"mal");
        float*Cout = indirect ? Cc : Cd;
        for(int i=0;i<S*M*N;i++) Cout[i]=-1.0f;
        for(int i=0;i<S;i++) ti[i].C=(int32_t*)(Cout+(size_t)i*M*N);
        int dnc=getenv("ORK_DOOR_NC")?atoi(getenv("ORK_DOOR_NC")):0;
        ork_dyn_chain*h=ork_dyn_begin_mc(c,S,ti,dnc);
        int okd=0;
        if(h){ ork_dyn_end(h);
            for(int i=0;i<S;i++){ int allk=1; for(int m=0;m<M;m++){ volatile float*d=(volatile float*)(Cout+(size_t)i*M*N+(size_t)m*N+(N-1)); if(!indirect)civac((void*)d); if(*d<(float)K-2||*d>(float)K+2){allk=0;break;} } if(allk)okd++; }
        }
        printf("[it%d] DOORBELL(%s): %d/%d ok  C[0..2][last]=%.1f %.1f %.1f  C[0][0]=%.1f\n", it, indirect?"mallocC/indirect":"dmaC/direct", okd, S,
               Cout[N-1], Cout[M*N+N-1], Cout[2*M*N+N-1], Cout[0]);
    }
    }
    ork_npu_free(c); return 0;
}
