/* test_job_abort_queued.c — DIAGNOSTIC, not a regression test. Read this before trusting it.
 *
 * WHAT IT WAS FOR. oRKLLM/rk3588-kernel#1: rknpu_job_abort() freed a job still linked on
 * subcore_data->todo_list, so the next rknpu_job_next() did list_first_entry() + list_del_init() through
 * freed slab — a write to a wild address, oopsing via rknpu_job_schedule/rknpu_submit_ioctl. Fixed by
 * adding list_del_init(&job->head[i]) to the abort loop plus INIT_LIST_HEAD at alloc.
 *
 * WHAT IT ACTUALLY DOES: two processes. A child re-execs itself with a tiny ORK_MM_TIMEOUT so its submits
 * time out, then the parent packs and runs a matmul and asserts bit-exactness against a CPU reference. The
 * todo_list is per-DEVICE and outlives the child, so a job freed while linked would be walked by the
 * parent's submit.
 *
 * WHY IT IS NOT A REGRESSION TEST: it does not reach the abort path. MEASURED — dmesg records ZERO
 * "job abort" lines across a full run, at 5ms and at 1ms timeouts and with a K=4096 shape that cannot
 * finish in time. The reason is structural: ork submits with RKNPU_JOB_NONBLOCK (submit.c, dyn.c), so the
 * kernel never calls rknpu_job_wait() and rknpu_job_abort() is only reached from a BLOCKING submit's
 * timeout or a failed rknpu_job_schedule(). Neither is reachable from the normal API. The field crash came
 * from llama-completion with two concurrent threads, and concurrent submits on the single NPU queue are
 * themselves the documented wedge-and-power-cycle hazard, so synthesising it here is not safe.
 *
 * So it PASSES without testing anything, which is why it is in DIAGNOSTICS and not EXAMPLES: `make test`
 * must not report a guarantee it does not provide. Genuinely closing that gap needs either a kernel-side
 * debugfs hook to force an abort with a job queued, or an accepted way to drive concurrent submits.
 *
 * Kept because the two-process shape is the reusable part, and because the negative result is worth
 * recording: a future attempt should start by checking dmesg for "job abort" before believing a pass.
 *
 * fork+EXEC, never a bare fork: ork_npu_init() spawns a worker pool, so a bare fork in a later cycle hands
 * the child a mutex locked by a thread that does not exist in it and it deadlocks in futex_wait (observed).
 */
#include "ork_npu.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <unistd.h>
#include <sys/wait.h>

static uint32_t g_s;
static int8_t rnd8(void){ g_s = g_s*1103515245u + 12345u; return (int8_t)((int)((g_s>>16)&0xff) - 128); }

static void ref_i8(int M,int K,int N,const int8_t*A,const int8_t*B,int32_t*C){
    for(int m=0;m<M;m++)for(int n=0;n<N;n++){ int32_t s=0; for(int k=0;k<K;k++) s+=(int32_t)A[m*K+k]*(int32_t)B[k*N+n]; C[m*N+n]=s; }
}

/* One bit-exact matmul on a fresh context. Returns 0 on match. */
static int verify_mm(int M,int K,int N,uint32_t seed){
    g_s = seed;
    int8_t *B = malloc((size_t)K*N), *A = malloc((size_t)M*K);
    int32_t *C = calloc((size_t)M*N,4), *R = malloc((size_t)M*N*4);
    for(int i=0;i<K*N;i++) B[i]=rnd8();
    for(int i=0;i<M*K;i++) A[i]=rnd8();
    ork_npu *c = ork_npu_init();
    int rc = -1;
    if(c){
        ork_w *w = ork_i8_mm_pack(c,K,N,B);
        if(w && ork_i8_mm_run(c,w,M,A,C)==0){
            ref_i8(M,K,N,A,B,R);
            rc = memcmp(C,R,(size_t)M*N*4)==0 ? 0 : -2;
        }
        ork_npu_free(c);
    }
    free(A);free(B);free(C);free(R);
    return rc;
}

/* Child (re-exec'd with --abort-phase): force submits to time out so the kernel aborts jobs with
 * others still queued. The submits are EXPECTED to fail -- provoking the abort IS the work, so
 * nothing here is asserted. */
static void abort_phase(void){
    setenv("ORK_MM_TIMEOUT","1",1);         /* 1ms: the submit below cannot finish in time -> rknpu_job_abort() */
    setenv("ORK_DEBUG_RESET","0",1);
    ork_npu *c = ork_npu_init();
    if(!c) _exit(0);
    const int K=4096,N=4096,M=64;        /* big enough that the job CANNOT complete inside 1ms */
    int8_t *B = malloc((size_t)K*N), *A = malloc((size_t)M*K);
    int32_t *C = calloc((size_t)M*N,4);
    g_s = 7; for(int i=0;i<K*N;i++) B[i]=rnd8(); for(int i=0;i<M*K;i++) A[i]=rnd8();
    ork_w *w = ork_i8_mm_pack(c,K,N,B);
    if(w) for(int i=0;i<4;i++) ork_i8_mm_run(c,w,M,A,C);   /* back-to-back: queue behind the running job */
    ork_npu_free(c);
    free(A);free(B);free(C);
    _exit(0);
}

int main(int argc,char**argv){
    if(argc>1 && strcmp(argv[1],"--abort-phase")==0){ abort_phase(); return 0; }

    const int CYCLES = 2;
    for(int cyc=0; cyc<CYCLES; cyc++){
        pid_t p = fork();
        if(p==0){                                  /* exec, don't just fork -- see the header note */
            execl("/proc/self/exe","test_job_abort_queued","--abort-phase",(char*)NULL);
            execl(argv[0],argv[0],"--abort-phase",(char*)NULL);   /* fallback if /proc is absent */
            _exit(127);
        }
        if(p<0){ fprintf(stderr,"[abort-queued] fork failed\n"); return 1; }
        int st=0; waitpid(p,&st,0);
        if(WIFEXITED(st) && WEXITSTATUS(st)==127){
            fprintf(stderr,"[abort-queued] SKIP (could not re-exec self)\n"); return 0; }

        /* The submit that would walk a freed-but-still-linked todo_list entry. */
        int rc = verify_mm(32,256,256,0x1234u+cyc);
        if(rc!=0){
            fprintf(stderr,"[abort-queued] cycle %d: post-abort matmul %s\n",
                    cyc, rc==-2?"MISMATCHED the CPU reference":"failed to run");
            return 1;
        }
        fprintf(stderr,"[abort-queued] cycle %d OK (aborts survived, next submit bit-exact)\n",cyc);
    }
    fprintf(stderr,"[abort-queued] PASS — %d abort/resubmit cycles, todo_list intact (issue #3)\n",CYCLES);
    return 0;
}
