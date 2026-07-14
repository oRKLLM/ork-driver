/* softmax_cost — HONEST on-NPU vs CPU softmax cost at attention scale, deciding whether a fused
 * softmax-on-NPU chain can beat the baseline's parallel CPU softmax. Uses float expf (what the
 * ggml-ork attention handler actually uses), and batches to attention scale (M=H*N rows) so the
 * NPU submit floor is amortized exactly as a fused chain would amortize it. ORK_SOFTMAX_NPU set
 * internally so ork_npu_softmax_f16 runs exp-on-NPU. BOARD: sudo ./softmax_cost */
#include "ork_npu.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>
static double now(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec*1e6+t.tv_nsec/1e3;}
/* single-thread float softmax — exactly the handler's per-row scale-free math (max/expf/sum/norm) */
static void cpu_softmax(const ork_f16*x,int M,int n,ork_f16*o){
    for(int m=0;m<M;m++){const ork_f16*r=x+(size_t)m*n; ork_f16*w=o+(size_t)m*n;
        float mx=(float)r[0]; for(int j=1;j<n;j++){float v=(float)r[j]; if(v>mx)mx=v;}
        float s=0; for(int j=0;j<n;j++)s+=expf((float)r[j]-mx);
        float inv=1.0f/s; for(int j=0;j<n;j++)w[j]=(ork_f16)(expf((float)r[j]-mx)*inv);}
}
int main(void){
    setenv("ORK_SOFTMAX_NPU","1",1);                 /* exp on NPU inside ork_npu_softmax_f16 */
    ork_npu*c=ork_npu_init(); if(!c)return 2;
    /* attention-scale batches: M = heads*seq (one attention op's whole softmax, all heads at once) */
    struct{int M,n;const char*tag;}cs[]={
        {256,256,"tiny"}, {6144,512,"pp512 (12h*512)"}, {12288,1024,"pp1024 (12h*1024)"}, {3072,256,"pp256 (12h*256)"} };
    for(unsigned i=0;i<sizeof(cs)/sizeof(cs[0]);i++){
        int M=cs[i].M,n=cs[i].n; size_t N=(size_t)M*n;
        ork_f16*x=malloc(N*2),*o=malloc(N*2);
        unsigned s=99+i; for(size_t j=0;j<N;j++){s=s*1103515245+12345; x[j]=(ork_f16)(((int)((s>>16)&0xff)-128)/32.0f);}
        ork_npu_softmax_f16(c,M,n,x,o);              /* warm */
        int R=5; double t0=now(); for(int r=0;r<R;r++)ork_npu_softmax_f16(c,M,n,x,o); double npu=(now()-t0)/R;
        cpu_softmax(x,M,n,o); t0=now(); for(int r=0;r<R;r++)cpu_softmax(x,M,n,o); double cpu=(now()-t0)/R;
        printf("%-18s M=%-6d n=%-5d : NPU %8.0f us (%.1f ns/el)  CPU-1thread %8.0f us (%.1f ns/el)  %.2fx %s\n",
               cs[i].tag,M,n,npu,npu*1e3/N,cpu,cpu*1e3/N,cpu/npu,(npu<cpu)?"NPU":"CPU");
        free(x);free(o);
    }
    ork_npu_free(c); return 0;
}
