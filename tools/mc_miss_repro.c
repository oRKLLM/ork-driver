/* Tight repro for the intermittent multi-core doorbell miss (test_stream_interleave flake). Runs the SAME
 * single<->stream interleave as examples/test_stream_interleave.c but for many more iters, stopping on the FIRST
 * miss so the [MC-DIAG] line (submit rc per core + HW dma_rw delta) fires and tells us the failure class:
 *   dma_rw delta == 0  => the NPU never dispatched the round (submit/dispatch problem)
 *   dma_rw delta  > 0  => it dispatched but never completed (pipeline/coherency/IRQ problem)
 * Board only. Build: make mc_miss_repro. Run: sudo env ORK_MM_TIMEOUT=2500 ORK_REPRO_ITERS=5000 ./mc_miss_repro
 */
#include "ork_npu.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <signal.h>
#include <unistd.h>

static volatile int g_iter = 0;
static void watchdog(int s){ (void)s; char m[64]; int n=snprintf(m,sizeof m,"\n*** mc_miss_repro WATCHDOG hung @iter=%d\n",g_iter); (void)!write(2,m,n); _exit(2); }

typedef struct { ork_w *w; int K, N; int8_t *B; int32_t *ref, *C; } WT;
static void mk(ork_npu *c, WT *t, int K, int N, const int8_t *A){
    t->K=K; t->N=N; t->B=malloc((size_t)K*N);
    for (size_t i=0;i<(size_t)K*N;i++) t->B[i]=(int8_t)((i*131u+7u)&0x7f);
    t->w=ork_mm_pack_i8(c,K,N,t->B); if(!t->w){ fprintf(stderr,"pack failed K=%d N=%d\n",K,N); exit(1); }
    t->C=calloc((size_t)N,4); t->ref=calloc((size_t)N,4);
    for (int n=0;n<N;n++){ int64_t s=0; for(int k=0;k<K;k++) s+=(int)A[k]*(int)t->B[(size_t)k*N+n]; t->ref[n]=(int32_t)s; }
}
static int check(const WT *t, const char *tag, int it){
    for (int n=0;n<t->N;n++) if (t->C[n]!=t->ref[n]){ fprintf(stderr,"MISMATCH %s @iter=%d n=%d got=%d want=%d\n",tag,it,n,t->C[n],t->ref[n]); return 1; }
    return 0;
}
int main(void){
    const char *e=getenv("ORK_REPRO_ITERS"); int iters=e?atoi(e):5000; if(iters<1)iters=5000;
    const char *ge=getenv("ORK_REPRO_GAP_US"); int gap=ge?atoi(ge):0;   /* usleep after each single-core op (test the retirement-race hypothesis) */
    signal(SIGALRM,watchdog); alarm(60);
    ork_npu *c=ork_npu_init(); if(!c){ printf("no NPU\n"); return 0; }
    if(getenv("ORK_REPRO_NC1") && atoi(getenv("ORK_REPRO_NC1"))){ ork_npu_set_core_budget(c,1); printf("  (NC1: core_budget forced to 1 -> single-core rounds)\n"); }
    if(getenv("ORK_REPRO_I4") && atoi(getenv("ORK_REPRO_I4"))){   /* TASK #4: stress the multi-M int4 doorbell route */
        int M=16,K=512,N=256; int8_t *A=malloc((size_t)M*K),*B=malloc((size_t)K*N); int32_t *C=malloc((size_t)M*N*4),*R=malloc((size_t)M*N*4);
        uint32_t s=7; for(int i=0;i<M*K;i++)A[i]=(int8_t)(((s=s*1103515245u+12345u)>>18&0xf))-8; for(int i=0;i<K*N;i++)B[i]=(int8_t)(((s=s*1103515245u+12345u)>>18&0xf))-8;
        for(int m=0;m<M;m++)for(int n=0;n<N;n++){long a=0;for(int k=0;k<K;k++)a+=(long)A[m*K+k]*B[k*N+n];R[m*N+n]=(int)a;}
        ork_w *w=ork_mm_pack_i4(c,K,N,B); if(!w){printf("i4 pack failed\n");return 1;}
        printf("  (I4 mode: %d iters of int4 M=%d K=%d N=%d via doorbell)\n", iters, M, K, N);
        for(int it=0; it<iters; it++){ g_iter=it;
            int rc=ork_mm_run_i4(c,w,M,A,C); int bad = rc?1:0;
            if(!bad) for(int i=0;i<M*N;i++) if(C[i]!=R[i]){ fprintf(stderr,"i4 MISMATCH @iter=%d [%d] got=%d want=%d\n",it,i,C[i],R[i]); bad=1; break; }
            if(bad){ fprintf(stderr,"*** i4 FIRST MISS at iter %d (rc=%d) ***\n",it,rc); ork_npu_free(c); return 1; }
            alarm(60); if(it && it%500==0){ printf("  ok through iter %d\n",it); fflush(stdout); } }
        printf("SURVIVED %d iters with NO miss (int4)\n", iters); ork_npu_free(c); return 0;
    }
    int8_t *A=calloc(2048,1); for(int i=0;i<2048;i++) A[i]=(int8_t)((i*7+3)&0x3f);
    WT o,q,k,v,g,u;
    mk(c,&o,2048,2048,A); mk(c,&q,2048,2048,A); mk(c,&k,2048,1024,A); mk(c,&v,2048,1024,A);
    mk(c,&g,2048,6144,A); mk(c,&u,2048,6144,A);
    ork_mm_task_i8 qkv[3]={{q.w,1,A,q.C},{k.w,1,A,k.C},{v.w,1,A,v.C}};
    ork_mm_task_i8 gu[2] ={{g.w,1,A,g.C},{u.w,1,A,u.C}};
    printf("mc_miss_repro: %d iters of [run_i8 | stream(qkv) | run_i8 | stream(gu)]\n", iters);
    int stream_only = getenv("ORK_REPRO_STREAM_ONLY") && atoi(getenv("ORK_REPRO_STREAM_ONLY"));
    if(stream_only) printf("  (STREAM-ONLY mode: just run_stream_i8(gu), no interleave)\n");
    for (int it=0; it<iters; it++){
        g_iter=it; int bad=0, rc;
        if(stream_only){
            if((rc=ork_mm_run_stream_i8(c,2,gu))){ fprintf(stderr,"stream gu rc=%d @%d\n",rc,it); bad=1; } bad|=check(&g,"g",it)|check(&u,"u",it);
            if(bad){ fprintf(stderr,"*** FIRST MISS at iter %d (of %d) — see [MC-DIAG] above ***\n", it, iters); ork_npu_free(c); return 1; }
            if(gap)usleep(gap);   /* between-rounds gap: does letting the NPU fully idle before the next submit prevent the miss? */
            alarm(60);            /* re-arm the per-iter watchdog (was missing here -> false 60s "hangs") */
            if(it && it%500==0){ printf("  ok through iter %d  dma_rw=%llu\n", it, (unsigned long long)ork_npu_dma_rw(c)); fflush(stdout); }
            continue;
        }
        if((rc=ork_mm_run_i8(c,o.w,1,A,o.C))){ fprintf(stderr,"run_i8 rc=%d @%d\n",rc,it); bad=1; } bad|=check(&o,"o",it); if(gap)usleep(gap);
        if((rc=ork_mm_run_stream_i8(c,3,qkv))){ fprintf(stderr,"stream qkv rc=%d @%d\n",rc,it); bad=1; } bad|=check(&q,"q",it)|check(&k,"k",it)|check(&v,"v",it);
        if((rc=ork_mm_run_i8(c,o.w,1,A,o.C))){ fprintf(stderr,"run_i8(2) rc=%d @%d\n",rc,it); bad=1; } bad|=check(&o,"o2",it); if(gap)usleep(gap);
        if((rc=ork_mm_run_stream_i8(c,2,gu))){ fprintf(stderr,"stream gu rc=%d @%d\n",rc,it); bad=1; } bad|=check(&g,"g",it)|check(&u,"u",it);
        alarm(60);
        if(bad){ fprintf(stderr,"*** FIRST MISS at iter %d (of %d) — see [MC-DIAG] above ***\n", it, iters); ork_npu_free(c); return 1; }
        if(it && it%500==0){ printf("  ok through iter %d\n", it); fflush(stdout); }
    }
    printf("SURVIVED %d iters with NO miss\n", iters);
    ork_npu_free(c); return 0;
}
