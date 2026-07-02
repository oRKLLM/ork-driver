// Regression: int8 decode interleaves single-core run_i8 (singletons -> last_dt=DT_I8) with
// run_stream_i8 groups (QKV, gate/up -> chain/stream marker). Both run int8 regcmd programs, so
// switching among them must NOT trigger an RKNPU_ACT_RESET (the reset is only for ENTERING int8 from
// fp16/int4/cold). A marker-keyed reset here fired a ~107ms hardware soft-reset at every matmul
// boundary -> 427ms/layer-iter (~12s/token, indistinguishable from a hang). See ORK_I8_LIVE in npu.c.
//
// This test guards BOTH failure modes at once:
//   (1) CORRECTNESS: every NPU output is compared bit-exact to a CPU int32 reference — so if skipping
//       the reset ever corrupted the int8->stream transition, this fails.
//   (2) PERF/HANG: a reintroduced reset-thrash (or a wedge) blows the per-iter time past a wide
//       threshold; a SIGALRM watchdog also bounds a true hang. Normal ~4ms/iter, thrashed ~427ms/iter.
#include "ork_npu.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <signal.h>
#include <unistd.h>
#include <time.h>

static double now_us(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec*1e6+t.tv_nsec/1e3; }

static volatile int   g_iter = 0;
static volatile const char *g_phase = "init";
static void watchdog(int sig){ (void)sig;
    const char m[] = "\n*** test_stream_interleave WATCHDOG: hung. iter=";
    write(2,m,sizeof m-1); char n[16]; int v=g_iter,i=15; n[i--]=0; if(!v)n[i--]='0'; while(v){n[i--]='0'+v%10;v/=10;}
    write(2,n+i+1,15-(i+1)); write(2," phase=",7); write(2,(const void*)g_phase,strlen((const char*)g_phase)); write(2,"\n",1);
    _exit(2);
}

typedef struct { ork_w *w; int K, N; int8_t *B; int32_t *ref; int32_t *C; } WT;
static void mk(ork_npu *c, WT *t, int K, int N, const int8_t *A){
    t->K=K; t->N=N; t->B=malloc((size_t)K*N);
    for (size_t i=0;i<(size_t)K*N;i++) t->B[i]=(int8_t)((i*131u+7u)&0x7f);
    t->w=ork_mm_pack_i8(c,K,N,t->B); if(!t->w){ fprintf(stderr,"pack failed K=%d N=%d\n",K,N); exit(1);}
    t->C=calloc((size_t)N,4); t->ref=calloc((size_t)N,4);
    for (int n=0;n<N;n++){ int64_t s=0; for(int k=0;k<K;k++) s+=(int)A[k]*(int)t->B[(size_t)k*N+n]; t->ref[n]=(int32_t)s; }
}
static int check(const WT *t, const char *tag, int it){
    for (int n=0;n<t->N;n++) if (t->C[n]!=t->ref[n]){ fprintf(stderr,"MISMATCH %s @iter=%d n=%d: got=%d want=%d\n",tag,it,n,t->C[n],t->ref[n]); return 1; }
    return 0;
}

int main(void){
    const int tokens=3, layers=28, total=tokens*layers;   // 84 interleaved iters, ~0.4s warm
    const double THRESH_US=50000.0;                        // fail > 50ms/iter (normal ~4ms, reset-thrash ~427ms)
    signal(SIGALRM,watchdog); alarm(60);

    ork_npu *c=ork_npu_init(); if(!c){ printf("no NPU (skipping)\n"); return 0; }
    int8_t *A=calloc(2048,1); for(int i=0;i<2048;i++) A[i]=(int8_t)((i*7+3)&0x3f);
    WT o,q,k,v,g,u;
    mk(c,&o,2048,2048,A);
    mk(c,&q,2048,2048,A); mk(c,&k,2048,1024,A); mk(c,&v,2048,1024,A);
    mk(c,&g,2048,6144,A); mk(c,&u,2048,6144,A);
    ork_mm_task_i8 qkv[3]={{q.w,1,A,q.C},{k.w,1,A,k.C},{v.w,1,A,v.C}};
    ork_mm_task_i8 gu[2] ={{g.w,1,A,g.C},{u.w,1,A,u.C}};

    int bad=0; double t0=now_us();
    for (int it=0; it<total; it++){
        g_iter=it;
        g_phase="run_i8(singleton)";  if(ork_mm_run_i8(c,o.w,1,A,o.C)){ fprintf(stderr,"run_i8 rc!=0 @%d\n",it); bad=1; }  bad|=check(&o,"o",it);
        g_phase="run_stream(QKV)";    { int rc=ork_mm_run_stream_i8(c,3,qkv); if(rc){ fprintf(stderr,"stream qkv rc=%d @%d\n",rc,it); bad=1; } } bad|=check(&q,"q",it)|check(&k,"k",it)|check(&v,"v",it);
        g_phase="run_i8(singleton2)"; if(ork_mm_run_i8(c,o.w,1,A,o.C)){ fprintf(stderr,"run_i8 rc!=0 @%d\n",it); bad=1; }  bad|=check(&o,"o2",it);
        g_phase="run_stream(gate/up)";{ int rc=ork_mm_run_stream_i8(c,2,gu);  if(rc){ fprintf(stderr,"stream gu rc=%d @%d\n",rc,it); bad=1; } } bad|=check(&g,"g",it)|check(&u,"u",it);
        alarm(60);
        if(bad){ fprintf(stderr,"FAIL: mismatch/error at iter %d\n",it); ork_npu_free(c); return 1; }
    }
    double us_iter=(now_us()-t0)/total;
    printf("interleave int8 single<->stream: %d iters bit-exact, %.0f us/iter\n", total, us_iter);
    if (us_iter > THRESH_US){ fprintf(stderr,"FAIL: %.0f us/iter > %.0f (reset-thrash regression? int8<->stream should not RESET)\n", us_iter, THRESH_US); ork_npu_free(c); return 1; }
    printf("STREAM INTERLEAVE OK\n");
    ork_npu_free(c); return 0;
}
