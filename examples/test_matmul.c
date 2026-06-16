/* examples/test_matmul.c — validates the ork_npu library: one handle, resident weights reused across
 * many matmuls of varying M (the forward-pass access pattern). Builds vs CPU reference.
 *   cc -O2 -I. -o test_mm test_mm.c ork_npu.c && sudo ./test_mm */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include "ork_npu.h"
typedef ork_f16 f16;
static unsigned sd=12345; static int rnd(){sd=sd*1103515245+12345;return (sd>>16)%4;}

static int check(ork_npu*ctx,int M,int K,int N){
    f16*A=malloc((size_t)M*K*2),*B=malloc((size_t)K*N*2); float*C=malloc((size_t)M*N*4);
    for(size_t i=0;i<(size_t)M*K;i++)A[i]=(f16)rnd();
    for(size_t i=0;i<(size_t)K*N;i++)B[i]=(f16)rnd();
    ork_w*w=ork_mm_pack(ctx,K,N,B);
    if(!w){printf("pack failed %d,%d\n",K,N);return 1;}
    int bad=0;
    /* run the SAME resident weights for several M (decode then prefill), validate each */
    int Ms[]={1,1,4,M}; for(int t=0;t<4;t++){int m=Ms[t]; if(m>M)m=M;
        if(ork_mm_run(ctx,w,m,A,C)){printf("run failed\n");return 1;}
        for(int i=0;i<m;i++)for(int n=0;n<N;n++){float ref=0;for(int k=0;k<K;k++)ref+=(float)A[(size_t)i*K+k]*(float)B[(size_t)k*N+n]; if(C[(size_t)i*N+n]!=ref)bad++;}
    }
    printf("  %s MKN=%d,%d,%d (reused weights x4 runs) mism=%d\n",bad?"WRONG":"ok  ",M,K,N,bad);
    ork_w_free(w); free(A);free(B);free(C); return bad?1:0;
}
/* int8/w8a8: A,B int8 -> C int32 (exact integer reference). K%32, N%32. */
static int check_i8(ork_npu*ctx,int M,int K,int N){
    int8_t*A=malloc((size_t)M*K),*B=malloc((size_t)K*N); int32_t*C=malloc((size_t)M*N*4);
    for(size_t i=0;i<(size_t)M*K;i++)A[i]=(int8_t)(rnd()-1);
    for(size_t i=0;i<(size_t)K*N;i++)B[i]=(int8_t)(rnd()-1);
    ork_w*w=ork_mm_pack_i8(ctx,K,N,B);
    if(!w){printf("pack_i8 failed %d,%d\n",K,N);return 1;}
    int bad=0; int Ms[]={1,1,4,M};
    for(int t=0;t<4;t++){int m=Ms[t]; if(m>M)m=M;
        if(ork_mm_run_i8(ctx,w,m,A,C)){printf("run_i8 failed\n");return 1;}
        for(int i=0;i<m;i++)for(int n=0;n<N;n++){int32_t ref=0;for(int k=0;k<K;k++)ref+=(int)A[(size_t)i*K+k]*(int)B[(size_t)k*N+n]; if(C[(size_t)i*N+n]!=ref)bad++;}
    }
    printf("  %s MKN=%d,%d,%d int8 (reused weights x4 runs) mism=%d\n",bad?"WRONG":"ok  ",M,K,N,bad);
    ork_w_free(w); free(A);free(B);free(C); return bad?1:0;
}
static int check_chain_i8(ork_npu*ctx) {
    int S = 4;
    int Ms[] = {1, 2, 4, 3};
    int K = 256;
    int N = 64;
    
    int8_t *A[4] = {NULL};
    int8_t *B[4] = {NULL};
    int32_t *C[4] = {NULL};
    ork_w *w[4] = {NULL};
    ork_mm_task_i8 tasks[4];

    for (int i = 0; i < S; i++) {
        if (i == 0) {
            A[i] = ork_dma_alloc(ctx, (size_t)Ms[i] * K);
            C[i] = ork_dma_alloc(ctx, (size_t)Ms[i] * N * 4);
        } else {
            A[i] = malloc((size_t)Ms[i] * K);
            C[i] = malloc((size_t)Ms[i] * N * 4);
        }
        B[i] = malloc((size_t)K * N);
        
        for (size_t j = 0; j < (size_t)Ms[i] * K; j++) A[i][j] = (int8_t)(rnd() - 1);
        for (size_t j = 0; j < (size_t)K * N; j++) B[i][j] = (int8_t)(rnd() - 1);
        
        w[i] = ork_mm_pack_i8(ctx, K, N, B[i]);
        if (!w[i]) { printf("pack_chain_i8 failed %d\n", i); return 1; }
        
        tasks[i].w = w[i];
        tasks[i].M = Ms[i];
        tasks[i].A = A[i];
        tasks[i].C = C[i];
    }
    
    int bad = 0;
    if (ork_mm_run_chain_i8(ctx, S, tasks)) {
        printf("run_chain_i8 failed\n");
        bad = 1;
    } else {
        for (int i = 0; i < S; i++) {
            int m = Ms[i];
            for (int r = 0; r < m; r++) {
                for (int n = 0; n < N; n++) {
                    int32_t ref = 0;
                    for (int k = 0; k < K; k++) {
                        ref += (int)A[i][(size_t)r * K + k] * (int)B[i][(size_t)k * N + n];
                    }
                    if (C[i][(size_t)r * N + n] != ref) bad++;
                }
            }
        }
    }

    printf("  %s chained S=%d (varying M, independent matmuls) mism=%d\n", bad ? "WRONG" : "ok  ", S, bad);

    for (int i = 0; i < S; i++) {
        if (w[i]) ork_w_free(w[i]);
        if (i == 0) {
            ork_dma_free(ctx, A[i]);
            ork_dma_free(ctx, C[i]);
        } else {
            free(A[i]);
            free(C[i]);
        }
        free(B[i]);
    }
    return bad ? 1 : 0;
}

int main(void){
    int fail=0;
    /* fp16 and int8 in SEPARATE contexts: a model is one precision, and switching regcmd mode
     * (fp16<->int8) on a live context wedges the first submit in the new mode for ~6s (the NPU
     * mode is stateful — see the wiki). Each precision gets a fresh context here, which also
     * mirrors real usage. (ork_mm_run_i8 on fp16 weights still safely returns an error.) */
    ork_npu*ctx=ork_npu_init(); if(!ctx){printf("init failed (NPU?)\n");return 1;}
    fail|=check(ctx,128,512,128);
    fail|=check(ctx,256,4096,512);
    fail|=check(ctx,512,8192,128);
    fail|=check(ctx,64,11008,64);     /* non-power-of-2 K */
    fail|=check(ctx,32,2048,256);
    fail|=check(ctx,8,512,16384);     /* N-tiling: N>8192 (NPU output-width cap) */
    fail|=check(ctx,1,288,32000);     /* LM-head shape: non-pow2 K + N tiled into 4 slices */
    fail|=check(ctx,4,6144,2048);     /* non-pow2 K<=8192 — decode (M=1) single-submit boundary */
    ork_npu_free(ctx);
    ork_npu*c8=ork_npu_init(); if(!c8){printf("init failed (NPU?)\n");return 1;}
    fail|=check_i8(c8,128,512,128);
    fail|=check_i8(c8,256,4096,512);
    fail|=check_i8(c8,64,11008,32);   /* non-power-of-2 K (768 remainder fallback) */
    fail|=check_i8(c8,1,8192,512);    /* decode */
    fail|=check_i8(c8,8,1280,64);     /* 256 remainder slice */
    fail|=check_i8(c8,4,6144,2048);   /* non-pow2 K<=8192 — decode (M=1) single-submit boundary */
    fail|=check_chain_i8(c8);         /* verify chained matmuls / MoE API */
    ork_npu_free(c8);
    printf("%s\n",fail?"FAIL":"ALL OK");
    return fail?1:0;
}
