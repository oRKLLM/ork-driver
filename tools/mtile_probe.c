/* tools/mtile_probe.c — test rkllm's large-M-per-submit M-tile mode vs ork's current mode.
 *
 * RE finding (KSLICE_RE_WIP.md): rkllm does NOT fine-slice K (full K=3584 every submit). Its lever
 * is running large M (up to 36) per task with 0x1010 held at 0x20 const, 0x1044=(K/64)*M,
 * 0x107c=4*M, 0x1040=0xb1-0xf*(ceil(M/8)-1). ork instead clamps the M-tile to R-1 (=15 at K=3584)
 * and inflates 0x1010. This sweeps M and runs BOTH modes (mode 0 = ork current, 1 = rkllm),
 * validating each vs a CPU int8 reference and timing the warm submit -> effective GOPS.
 *   make mtile_probe && sudo ./mtile_probe [K] [N]
 * Bit-exact-gated: a WRONG result is reported (not silently used). A wedge is recoverable.
 */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include "ork_npu.h"
static unsigned sd=12345; static int rnd(void){sd=sd*1103515245+12345;return (int)((sd>>16)%7)-3;}
int main(int argc,char**argv){
    int K=argc>1?atoi(argv[1]):3584, N=argc>2?atoi(argv[2]):128;
    ork_npu*c=ork_npu_init(); if(!c){printf("init failed (NPU?)\n");return 1;}
    printf("SoC %s  K=%d N=%d  (mode0=ork-current, mode1=rkllm-captured)\n",ork_npu_soc(c),K,N);
    int Ms[]={1,2,4,8,15,16,20,24,32,36}; int NM=sizeof(Ms)/sizeof(*Ms);
    int8_t*A=malloc((size_t)64*K),*B=malloc((size_t)K*N); int32_t*C=malloc((size_t)64*N*4);
    int32_t*ref=malloc((size_t)64*N*4);
    for(size_t j=0;j<(size_t)K*N;j++)B[j]=(int8_t)rnd();
    int anyfail=0;
    printf("%4s | %-30s | %-30s\n","M","mode0 (ork)","mode1 (rkllm)");
    for(int i=0;i<NM;i++){int M=Ms[i];
        for(size_t j=0;j<(size_t)M*K;j++)A[j]=(int8_t)rnd();
        for(int m=0;m<M;m++)for(int n=0;n<N;n++){int32_t s=0;for(int k=0;k<K;k++)s+=(int)A[(size_t)m*K+k]*(int)B[(size_t)k*N+n];ref[(size_t)m*N+n]=s;}
        char r0[64],r1[64];
        for(int mode=0;mode<2;mode++){
            double us=0; int rc=ork_npu_probe_mtile_i8(c,M,K,N,mode,A,B,C,&us);
            char*o=mode?r1:r0;
            if(rc==0){ long bad=0; for(size_t j=0;j<(size_t)M*N;j++)if(C[j]!=ref[j])bad++;
                double gops=us>0?(2.0*M*K*N)/(us*1e3):0;
                if(bad){snprintf(o,64,"WRONG (%ld/%d) %.0fus",bad,M*N,us);anyfail=1;}
                else snprintf(o,64,"OK %.0fus %.0f GOPS",us,gops);
            } else if(rc==-1) snprintf(o,64,"WEDGED");
            else snprintf(o,64,"bad-dims");
        }
        printf("%4d | %-30s | %-30s\n",M,r0,r1);
    }
    free(A);free(B);free(C);free(ref); ork_npu_free(c);
    printf("%s\n",anyfail?"NOTE: a mode produced WRONG output (see above)":"all completed runs bit-exact");
    return 0;
}
