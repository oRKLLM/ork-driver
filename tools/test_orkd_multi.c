/* ⚠ KNOWN-FRAGILE (2026-07-19): this fork()-based variant WEDGED the NPU on the board — each forked worker
 * auto-spawns/connects orkd from INSIDE a forked child, and that path (not the daemon's serialization) got
 * stuck (orkd D-state, unkillable). The CLEAN multi-consumer proof is tools/test_orkd_2conn.c (ONE process,
 * TWO connections, sequential init, no fork) — validated bit-exact. Keep this only as a documented negative
 * result for the fork+auto-spawn-from-a-forked-child path; do NOT run it expecting a pass.
 *
 * test_orkd_multi — MULTI-TENANT orkd proof: N independent CLIENT PROCESSES sharing ONE daemon.
 *
 * The whole reason orkd exists: the RK3588 NPU is single-stream — two processes submitting to /dev/dri
 * concurrently wedge the IOMMU. orkd owns the NPU and SERIALIZES submits (single poll loop), so many client
 * processes can share it safely. This forks N workers; each is its own orkd client (own connection, own
 * weight table) doing pack+run+verify loops with DISTINCT per-worker data, so any cross-client corruption
 * (daemon mixing up weights/outputs) or wedge shows as a mismatch / nonzero exit. First worker to init
 * auto-spawns orkd; the rest connect (orkd's flock keeps it a singleton). All must exit 0.
 *
 *   make test_orkd_multi && sudo env ORK_USE_ORKD=1 ORKD_BIN=$PWD/orkd ./test_orkd_multi [nclients] [iters]
 */
#include "ork_npu.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/wait.h>

/* one client process: pack a distinct int8 weight, run M×K·K×N matmuls, verify vs the CPU int32 reference. */
static int worker(int idx, int iters){
    ork_npu *c = ork_npu_init();               /* connects to (or auto-spawns) orkd under ORK_USE_ORKD */
    if(!c){ fprintf(stderr,"[w%d] init failed\n", idx); return 2; }
    const int M=8, K=512, N=64;
    int8_t *B=malloc((size_t)K*N), *A=malloc((size_t)M*K); int32_t *C=malloc((size_t)M*N*4), *R=malloc((size_t)M*N*4);
    if(!B||!A||!C||!R){ ork_npu_free(c); return 2; }
    unsigned g = 0x9e3779b9u ^ (unsigned)(idx*2654435761u);
    #define RND() ((int8_t)(((g=g*1103515245u+12345u)>>18&0x1f)-16))   /* [-16,15] */
    for(int i=0;i<K*N;i++) B[i]=RND();          /* weight distinct per worker */
    ork_w *w = ork_mm_pack_i8(c, K, N, B);
    int bad=0;
    for(int it=0; it<iters && !bad; it++){
        for(int i=0;i<M*K;i++) A[i]=RND();       /* activations distinct per worker+iter */
        for(int m=0;m<M;m++)for(int n=0;n<N;n++){ long a=0; for(int k=0;k<K;k++) a+=(long)A[m*K+k]*B[k*N+n]; R[m*N+n]=(int)a; }
        memset(C,0,(size_t)M*N*4);
        if(!w || ork_mm_run_i8(c, w, M, A, C)){ fprintf(stderr,"[w%d] run it=%d FAIL\n", idx, it); bad=1; break; }
        for(int i=0;i<M*N;i++) if(C[i]!=R[i]){ fprintf(stderr,"[w%d] it=%d MISMATCH [%d] %d!=%d\n", idx,it,i,C[i],R[i]); bad=1; break; }
    }
    if(w) ork_mm_free(c, w);
    ork_npu_free(c);
    free(B);free(A);free(C);free(R);
    #undef RND
    if(!bad) fprintf(stderr,"[w%d] OK (%d iters, own weight+A, verified)\n", idx, iters);
    return bad?1:0;
}

int main(int argc, char **argv){
    int nc = argc>1?atoi(argv[1]):3, iters = argc>2?atoi(argv[2]):8;
    if(nc<1)nc=1; if(nc>32)nc=32; if(iters<1)iters=1;
    setenv("ORK_USE_ORKD","1",1);               /* force client mode for every worker */
    fprintf(stderr,"[mt] forking %d concurrent orkd clients x %d iters (one shared daemon)\n", nc, iters);
    pid_t pid[32];
    for(int i=0;i<nc;i++){ pid[i]=fork(); if(pid[i]==0){ _exit(worker(i, iters)); } if(pid[i]<0){ perror("fork"); return 2; } }
    int bad=0;
    for(int i=0;i<nc;i++){ int st=0; waitpid(pid[i], &st, 0); if(!WIFEXITED(st)||WEXITSTATUS(st)){ bad++; fprintf(stderr,"[mt] worker %d bad (status %d)\n", i, st); } }
    printf("MULTI_TENANT_ORKD: %s — %d client processes x %d iters through one daemon%s\n",
           bad?"FAIL":"PASS", nc, iters, bad?"":", all correct, no wedge");
    return bad?1:0;
}
