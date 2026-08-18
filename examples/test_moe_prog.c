/* examples/test_moe_prog.c — characterize the MoE per-PROGRAM cost (the 4M-tiny-program wall).
 *
 * Times ONE int4 expert matmul at the real qwen35moe shapes (gate/up K=2048 N=512, down K=512 N=2048),
 * sweeping M (32 = current pad, 16 = the actual routed count / padding-lever) and cores (1 vs 3). Reports
 * us/matmul and us/program (programs = NG*NC from the BCHAIN formula), so we can read off:
 *   - how long a program actually takes (us/program) vs its ~sub-us theoretical MAC compute => the overhead,
 *   - whether it's SERIAL: does 3-core give ~3x? does M=16 (half the NG) roughly halve the time?
 * Run ORK_BCH_DEBUG=1 to print the actual H/Wb/NC/NG. Probe (not a correctness gate).
 *   make test_moe_prog && sudo env ORK_BCH_DEBUG=1 ./test_moe_prog
 */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>
#include "ork_npu.h"

static double now_us(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec*1e6+t.tv_nsec/1e3; }
static void fill_i4(int8_t*p,size_t n,unsigned s){ for(size_t i=0;i<n;i++){ s=s*1103515245u+12345u; p[i]=(int8_t)((int)((s>>17)%15)-7); } }

/* programs for one BCHAIN int4 matmul: H=min(16,16384/K), Wb=(131072/K)&~63 (>=64), NG=ceil(M/H), NC=ceil(N/Wb). */
static int nprog(int M,int K,int N){ int H=16384/K; if(H>16)H=16; if(H<2)H=2; int Wb=(131072/K)&~63; if(Wb<64)Wb=64; if(Wb>N)Wb=N;
    int NG=(M+H-1)/H, NC=(N+Wb-1)/Wb; return NG*NC; }

static void bench(ork_npu*c,const char*tag,int K,int N,int M){
    int8_t*B=malloc((size_t)K*N); fill_i4(B,(size_t)K*N,7);
    int8_t*A=malloc((size_t)M*K); fill_i4(A,(size_t)M*K,9);
    int32_t*C=malloc((size_t)M*N*4);
    ork_w*w=ork_mm_pack_i4(c,K,N,B); if(!w){ printf("  %-14s pack fail\n",tag); free(A);free(B);free(C); return; }
    for(int nc=1;nc<=3;nc+=2){ ork_npu_set_core_budget(c,nc);
        ork_mm_run_i4(c,w,M,A,C);                                   /* warm */
        int iters=200; double best=1e18;
        for(int r=0;r<3;r++){ double t0=now_us(); for(int i=0;i<iters;i++) ork_mm_run_i4(c,w,M,A,C);
            double u=(now_us()-t0)/iters; if(u<best)best=u; }
        int P=nprog(M,K,N);
        printf("  %-14s M=%-3d %d-core: %8.1f us/matmul | %2d programs | %6.2f us/program\n",
               tag,M,nc,best,P,best/P);
    }
    ork_mm_free(c,w); free(A);free(B);free(C);
}

int main(void){
    ork_npu*c=ork_npu_init(); if(!c){printf("init failed\n");return 1;}
    printf("MoE per-program cost (qwen35moe expert shapes, int4 BCHAIN):\n");
    size_t sf0=ork_npu_sram_free(c);
    (void)sf0;
    printf("== gate/up  K=2048 N=512 (M=128: NG=16 => 15/16 tiles reuse the weight) ==\n");
    bench(c,"gate/up",2048,512,32); bench(c,"gate/up",2048,512,128); bench(c,"gate/up",2048,512,256);
    printf("== down     K=512  N=2048 (M=128: NG=8) ==\n"); bench(c,"down",512,2048,32); bench(c,"down",512,2048,128);
    ork_npu_free(c);
    return 0;
}
