/* ork_dyn_f16_interleave — validate the "int8-doorbell primes the fp16 doorbell" hypothesis and its
 * PERSISTENCE, mimicking a real mixed-precision async pipeline (single-stream NPU => all-or-nothing async;
 * int8 MoE + fp16 attention/SSM both flow through the non-blocking doorbell). Modes (argv[1]):
 *   cold        : fp16 mc ops only, no int8 ever (expect systemic K short-count — fp16 as first NPU op).
 *   persist     : ONE int8 mc op, then R fp16 mc ops back-to-back (no int8 between) — does priming persist?
 *   interleave  : R iterations of {int8 mc ; fp16 mc} — sustained interleaved stream (int8 precedes each fp16).
 *   persist-mm  : one int8 mc, then R fp16 ops on a SECOND (K,N) shape — priming generalizes across shapes?
 * argv[2]=R (default 20). fp16: A=B=1.0 => every output element must == (float)K. Reports per-op pass/fail
 * and the min output value on failure (systemic short-count shows a clean K-n*32). Board only. */
#define _GNU_SOURCE
#include "ork_npu.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>

static inline void civac(volatile void*p){ __asm__ volatile("dc civac,%0"::"r"(p):"memory"); }

/* run S int8 M=Mi matmuls through the mc doorbell (D-equivalent); returns 0 ok. If chk, verify every row's
 * last col == K (A=B=1 => dot=K) and set *okc = #correct tasks. */
static int run_i8(ork_npu*c,ork_w*wi,int S,int Mi,const int8_t*A,int32_t*O,int N,int K,int*okc){
    ork_mm_task_i8*t=malloc(sizeof *t*S);
    for(int i=0;i<S;i++){ t[i].w=wi; t[i].M=Mi; t[i].A=A; t[i].C=O+(size_t)i*Mi*N; }
    ork_dyn_chain*h=ork_dyn_begin_mc(c,S,t,0);
    if(!h){ free(t); if(okc)*okc=-1; return -1; }
    ork_dyn_end(h);
    if(okc){ int ok=0; for(int i=0;i<S;i++){ int allk=1; for(int m=0;m<Mi;m++){ volatile int32_t*d=(volatile int32_t*)(O+(size_t)i*Mi*N+(size_t)m*N+(N-1)); civac((void*)d); if(*d!=K){allk=0;break;} } if(allk)ok++; } *okc=ok; }
    free(t); return 0;
}
/* run S fp16 M=Mf matmuls; check every element == (float)K. Returns #tasks fully-correct. Fills *summary:
 * per-(task,row) last-col value classification — #bad rows, distinct wrong values (each = K-nblk*32 => nblk
 * K-blocks dropped), and whether ALL bad rows share one value (systemic) or vary (sporadic near-miss). */
static int run_f16(ork_npu*c,ork_w*wf,int S,int Mf,int K,int N,const ork_f16*A,float*O,char*summary){
    ork_mm_task_i8*t=malloc(sizeof *t*S);
    for(int i=0;i<S;i++){ t[i].w=wf; t[i].M=Mf; t[i].A=(const int8_t*)A; t[i].C=(int32_t*)(O+(size_t)i*Mf*N); }
    ork_dyn_chain*h=ork_dyn_begin_mc(c,S,t,0);
    if(!h){ free(t); if(summary)strcpy(summary,"begin_mc NULL"); return -1; }
    ork_dyn_end(h);
    int ok=0, nbad=0; float dv[8]; int dvn[8]; int ndv=0;   /* distinct wrong last-col values */
    for(int i=0;i<S;i++){ int allk=1;
        for(int m=0;m<Mf;m++){ volatile float*d=(volatile float*)(O+(size_t)i*Mf*N+(size_t)m*N+(N-1)); civac((void*)d);
            float v=*d; if(v<(float)K-1||v>(float)K+1){ allk=0; nbad++;
                int f=0; for(int k=0;k<ndv;k++) if(fabsf(dv[k]-v)<1){dvn[k]++;f=1;break;}
                if(!f&&ndv<8){dv[ndv]=v;dvn[ndv]=1;ndv++;} } }
        if(allk)ok++; }
    free(t);
    if(summary){ char*p=summary; p+=sprintf(p,"badrows=%d/%d",nbad,S*Mf);
        for(int k=0;k<ndv;k++){ int nblk=(int)lroundf(((float)K-dv[k])/32.0f); p+=sprintf(p," [%.0f=drop%d ×%d]",dv[k],nblk,dvn[k]); }
        if(ndv==1) sprintf(p," SYSTEMIC"); else if(ndv>1) sprintf(p," MIXED"); }
    return ok;
}

int main(int argc,char**argv){
    const char*mode=argc>1?argv[1]:"interleave";
    int R=argc>2?atoi(argv[2]):20;
    int S=16;
    setenv("ORK_DYN_F16","1",1);
    ork_npu*c=ork_npu_init(); if(!c){printf("init failed\n");return 1;}
    printf("mode=%s R=%d S=%d\n",mode,R);

    /* int8 weight+activation (D-like): K=512,N=512,M=4 */
    int Ki=512,Ni=512,Mi=4;
    int8_t*Bi=malloc((size_t)Ki*Ni); memset(Bi,1,(size_t)Ki*Ni);
    ork_w*wi=ork_i8_mm_pack(c,Ki,Ni,Bi); if(!wi){printf("i8 pack fail\n");return 1;}
    int8_t*Ai=(int8_t*)malloc((size_t)Mi*Ki); memset(Ai,1,(size_t)Mi*Ki);
    int32_t*Oi=(int32_t*)ork_dma_alloc(c,(size_t)S*Mi*Ni*4); if(!Oi){printf("Oi fail\n");return 1;}

    /* fp16 weight+activation (E-like): Kf=512,Nf=256,M=2 — packed LAZILY (see f16_ensure) so the pack order
     * relative to the int8 op can be controlled (ork_dyn_test packs fp16 AFTER the int8 sections). */
    int Kf=512,Nf=256,Mf=2; ork_f16 one=(ork_f16)1.0f;
    ork_w*wf=NULL; ork_f16*Af=NULL; float*Of=NULL;
    #define F16_ENSURE() do { if(!wf){ ork_f16*Bf=malloc((size_t)Kf*Nf*sizeof(ork_f16)); for(size_t i=0;i<(size_t)Kf*Nf;i++)Bf[i]=one; \
        wf=ork_f16_mm_pack(c,Kf,Nf,Bf); if(!wf){printf("f16 pack fail\n");return 1;} \
        Af=(ork_f16*)ork_dma_alloc(c,(size_t)Mf*Kf*sizeof(ork_f16)); for(int i=0;i<Mf*Kf;i++)Af[i]=one; \
        Of=(float*)ork_dma_alloc(c,(size_t)S*Mf*Nf*4); if(!Af||!Of){printf("f16 buf fail\n");return 1;} } } while(0)

    /* second fp16 shape for generalization: Kf2=1024,Nf2=512,M=3 — packed lazily in persist-mm */
    int Kf2=1024,Nf2=512,Mf2=3;
    ork_w*wf2=NULL; ork_f16*Af2=NULL; float*Of2=NULL;
    #define F16_2_ENSURE() do { if(!wf2){ ork_f16*Bf2=malloc((size_t)Kf2*Nf2*sizeof(ork_f16)); for(size_t i=0;i<(size_t)Kf2*Nf2;i++)Bf2[i]=one; \
        wf2=ork_f16_mm_pack(c,Kf2,Nf2,Bf2); if(!wf2){printf("f16 sh2 pack fail\n");return 1;} \
        Af2=(ork_f16*)ork_dma_alloc(c,(size_t)Mf2*Kf2*sizeof(ork_f16)); for(int i=0;i<Mf2*Kf2;i++)Af2[i]=one; \
        Of2=(float*)ork_dma_alloc(c,(size_t)S*Mf2*Nf2*4); if(!Af2||!Of2){printf("f16 sh2 buf fail\n");return 1;} } } while(0)

    int pass=0,fail=0; char sm[256];
    if(!strcmp(mode,"cold")){
        F16_ENSURE();
        for(int r=0;r<R;r++){ sm[0]=0; int ok=run_f16(c,wf,S,Mf,Kf,Nf,Af,Of,sm);
            printf("  f16 op %2d: %d/%d %s %s\n",r,ok,S,ok==S?"PASS":"FAIL",ok==S?"":sm); if(ok==S)pass++;else fail++; }
    } else if(!strcmp(mode,"persist")){
        int i8ok=0; if(run_i8(c,wi,S,Mi,Ai,Oi,Ni,Ni,&i8ok)||i8ok!=S){printf("i8 prime fail (ok=%d)\n",i8ok);return 1;}
        F16_ENSURE();   /* pack fp16 AFTER the int8 op (matches ork_dyn_test order) */
        printf("  [primed once with int8 mc; now R fp16 with NO int8 between]\n");
        for(int r=0;r<R;r++){ sm[0]=0; int ok=run_f16(c,wf,S,Mf,Kf,Nf,Af,Of,sm);
            printf("  f16 op %2d: %d/%d %s %s\n",r,ok,S,ok==S?"PASS":"FAIL",ok==S?"":sm); if(ok==S)pass++;else fail++; }
    } else if(!strcmp(mode,"persist-mm")){
        int i8ok=0; if(run_i8(c,wi,S,Mi,Ai,Oi,Ni,Ni,&i8ok)||i8ok!=S){printf("i8 prime fail (ok=%d)\n",i8ok);return 1;}
        F16_2_ENSURE();
        printf("  [primed once with int8 mc; now R fp16 on shape2 K=%d N=%d]\n",Kf2,Nf2);
        for(int r=0;r<R;r++){ sm[0]=0; int ok=run_f16(c,wf2,S,Mf2,Kf2,Nf2,Af2,Of2,sm);
            printf("  f16(sh2) op %2d: %d/%d %s %s\n",r,ok,S,ok==S?"PASS":"FAIL",ok==S?"":sm); if(ok==S)pass++;else fail++; }
    } else if(!strcmp(mode,"i8only")){   /* control: repeated int8 mc in ONE process — does int8 drift too? */
        for(int r=0;r<R;r++){ int i8ok=0; run_i8(c,wi,S,Mi,Ai,Oi,Ni,Ni,&i8ok);
            printf("  i8 op %2d: %d/%d %s\n",r,i8ok,S,i8ok==S?"PASS":"FAIL"); if(i8ok==S)pass++;else fail++; }
    } else { /* interleave — real int8 mc op immediately before EACH fp16 (re-prime every op) */
        for(int r=0;r<R;r++){ int i8ok=0; if(run_i8(c,wi,S,Mi,Ai,Oi,Ni,Ni,&i8ok)||i8ok!=S){printf("  iter %d: i8 FAIL (ok=%d/%d)\n",r,i8ok,S);fail++;continue;}
            F16_ENSURE(); sm[0]=0;
            int ok=run_f16(c,wf,S,Mf,Kf,Nf,Af,Of,sm);
            printf("  iter %2d: i8 ok, f16 %d/%d %s %s\n",r,ok,S,ok==S?"PASS":"FAIL",ok==S?"":sm); if(ok==S)pass++;else fail++; }
    }
    printf("RESULT mode=%s: fp16 PASS=%d FAIL=%d / %d\n",mode,pass,fail,R);
    ork_npu_free(c);
    return fail?2:0;
}
