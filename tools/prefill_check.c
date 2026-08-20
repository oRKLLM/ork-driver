/* tools/prefill_check.c — find the NPU int8 PREFILL correctness bug (PPL 82 single / 172 multi vs CPU
 * 18.6). int8*int8 -> int32 is EXACT (no quant), so a correct NPU matmul must match the CPU int32
 * reference BIT-FOR-BIT. quant.c only validated K=2048 (pow2) single-core; this sweeps the REAL model
 * shapes (Qwen3-1.7B) at prefill M values, single- and multi-core, and reports the first mismatch.
 *
 *   make prefill_check && sudo ./prefill_check
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include "ork_npu.h"

static unsigned sd=7; static int8_t r8(void){sd=sd*1103515245+12345;return (int8_t)((int)((sd>>9)%255)-127);}

static long check(ork_npu*ctx,int M,int K,int N,int nc,long*maxerr){
    ork_npu_set_core_budget(ctx,nc);
    int8_t*A=malloc((size_t)M*K),*B=malloc((size_t)K*N); int32_t*C=malloc((size_t)M*N*4);
    for(size_t i=0;i<(size_t)M*K;i++)A[i]=r8();
    for(size_t i=0;i<(size_t)K*N;i++)B[i]=r8();
    ork_w*w=ork_i8_mm_pack(ctx,K,N,B); if(!w){printf("pack fail K=%d N=%d\n",K,N);exit(1);}
    if(ork_i8_mm_run(ctx,w,M,A,C)){printf("run fail\n");exit(1);}
    long bad=0,me=0;
    for(int m=0;m<M;m++)for(int n=0;n<N;n++){
        long ref=0; for(int k=0;k<K;k++) ref+=(long)A[(size_t)m*K+k]*B[(size_t)k*N+n];
        long got=C[(size_t)m*N+n]; if(got!=ref){bad++; long e=labs(got-ref); if(e>me)me=e;}
    }
    ork_w_free(w); free(A);free(B);free(C); *maxerr=me; return bad;
}

int main(void){
    setvbuf(stdout, NULL, _IONBF, 0);            // unbuffered: incremental output when redirected to a file
    ork_npu*ctx=ork_npu_init(); if(!ctx){printf("init failed\n");return 1;}
    // Qwen3-1.7B matmul shapes (K,N): q/o 2048x2048, kv 2048x1024, gate/up 2048x6144, down 6144x2048,
    // lm_head 2048x151936 (tested at a smaller wide-N proxy + the real one). Prefill M values + remainders.
    // full validation: real Qwen3-1.7B shapes (incl. non-pow2 K=6144 down_proj + huge-N lm_head),
    // partial-tile M values (96/193/255 = the formerly-broken remainders), single + multi core.
    struct{int K,N;}sh[]={ {2048,2048},{6144,2048},{2048,6144},{2048,1024},{2048,8192},{2048,151936} };
    int Ms[]={512,255,193,96};
    for(int nc=1;nc<=2;nc++){
        printf("===== %d-core =====\n", nc);
        for(unsigned s=0;s<sizeof sh/sizeof*sh;s++)
            for(unsigned mi=0;mi<sizeof Ms/sizeof*Ms;mi++){
                int M=Ms[mi],K=sh[s].K,N=sh[s].N; long me; long bad=check(ctx,M,K,N,nc,&me);
                printf("  M=%-4d K=%-5d N=%-6d : %s", M,K,N, bad?"MISMATCH":"ok");
                if(bad) printf(" (%ld/%d wrong = %.1f rows, maxerr=%ld)", bad,M*N,(double)bad/N, me);
                printf("\n");
            }
    }
    ork_npu_free(ctx);
    return 0;
}
