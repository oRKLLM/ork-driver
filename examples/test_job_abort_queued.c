/* test_job_abort_queued.c — REGRESSION TEST for oRKLLM/ork-driver#3: a job aborted while still
 * QUEUED must be unlinked from the kernel's per-core todo_list before it is freed.
 *
 * THE BUG (kernel, drivers/rknpu/rknpu_job.c). rknpu_job_abort() cleared subcore_data->job -- which
 * only retires the job that is RUNNING -- and then freed the job via rknpu_job_cleanup(). A job
 * aborted while still QUEUED stayed linked in subcore_data->todo_list, so the NEXT submit's
 * rknpu_job_next() did list_first_entry() + list_del_init() through freed slab: a WRITE to a wild
 * address (WnR=1), oopsing in rknpu_job_next <- rknpu_job_schedule <- rknpu_submit_ioctl.
 * Fixed by rknpu-job-list-uaf ("rknpu: unlink an aborted job from the core todo_lists before
 * freeing it"), which adds list_del_init(&job->head[i]) to the abort loop + INIT_LIST_HEAD at alloc.
 *
 * HOW THIS REPRODUCES IT WITHOUT THE DOMAIN SWITCH. In the field the aborts came from a concurrent
 * IOMMU domain switch (get_and_switch -> reap_all_cores -> timeout_clean). But the switch is only
 * what PROVOKED the aborts; the defect is in abort itself. A very short submit timeout provokes the
 * same aborts directly, with no domain juggling and no concurrent submits -- which matters, because
 * concurrent submits on the single NPU queue are themselves unsafe and would confound the test.
 *
 * SHAPE. Two phases, and they must be two PROCESSES:
 *   child  - ORK_MM_TIMEOUT tiny -> submits time out -> rknpu_job_abort() runs with jobs still
 *            queued behind the running one on the same core. (The timeout is cached in a static on
 *            first use, so it cannot be changed mid-process -- hence a separate process.)
 *            It is fork + EXEC, never a bare fork: ork_npu_init() spawns a worker pool, so a bare
 *            fork in a later cycle hands the child a mutex locked by a thread that does not exist
 *            in it, and the child deadlocks in futex_wait. exec resets the address space.
 *   parent - normal timeout -> packs and runs a matmul, asserting bit-exactness vs a CPU reference.
 *            The todo_list is per-DEVICE and outlives the child, so if the child freed a linked job
 *            this submit is the one that walks it.
 * Sequential, never concurrent: the child exits before the parent submits.
 *
 * FAILURE MODE IS LOUD. With the fix absent this can oops the kernel rather than fail cleanly, and
 * a panic here does not reboot (SMP: failed to stop secondary CPUs) -- it needs a power cycle. That
 * is inherent to regression-testing a kernel use-after-free from userspace: the honest signal is a
 * dead board. Kept bounded (few cycles, tiny shapes) so the exposure is as small as it can be.
 *
 * 0 = pass, nonzero = fail.
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
