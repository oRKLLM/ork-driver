/* tools/i16_matmul_test.c — first real int16 matmul via the fp16 run path + 0x100c=INT16 fuzz-override.
 *
 * int16 tiles are byte-identical layout to fp16, so ork_f16_mm_pack tiles int16 weights correctly. Flipping
 * 0x100c FP16(2)->INT16(1) via ork_f16_fuzz_add makes the CNA MAC integer. We feed small int16 A/B and
 * compare the NPU output to the CPU int16 reference C[m,n]=sum_k A[m,k]*B[k,n] (int32). This tells us
 * whether proc=1 gives a correct integer matmul with the fp16 output stage, or whether the output stage
 * needs the int8-style integer config (next step). Small values (no int32 overflow). Single-core.
 *   make i16_matmul_test && sudo ORK_NPU_MC=1 ./i16_matmul_test [M K N]   (board only)
 */
#define _GNU_SOURCE
#include "ork_npu.h"
#include <stdio.h>
#include <stdlib.h>
typedef ork_f16 f16;

int main(int argc,char**argv){
    setvbuf(stdout,NULL,_IONBF,0);
    int M=argc>3?atoi(argv[1]):8, K=argc>3?atoi(argv[2]):64, N=argc>3?atoi(argv[3]):32;
    ork_npu*c=ork_npu_init(); if(!c){printf("no board\n");return 0;}
    /* small int16 weights/acts so K-sum stays < int32 */
    int16_t*Bi=malloc((size_t)K*N*2),*Ai=malloc((size_t)M*K*2);
    for(int k=0;k<K;k++)for(int n=0;n<N;n++)Bi[(size_t)k*N+n]=(int16_t)(((k+n)%5)-2);   /* -2..2 */
    for(int m=0;m<M;m++)for(int k=0;k<K;k++)Ai[(size_t)m*K+k]=(int16_t)((m+k)%3);        /* 0..2 */
    /* CPU int16 reference (int32 accumulate) */
    int32_t*ref=malloc((size_t)M*N*4);
    for(int m=0;m<M;m++)for(int n=0;n<N;n++){ int32_t s=0; for(int k=0;k<K;k++) s+=(int32_t)Ai[(size_t)m*K+k]*(int32_t)Bi[(size_t)k*N+n]; ref[(size_t)m*N+n]=s; }

    ork_w*w=ork_f16_mm_pack(c,K,N,(const f16*)Bi); if(!w){printf("pack fail\n");return 2;}
    float*C=malloc((size_t)M*N*4); for(int i=0;i<M*N;i++)C[i]=-1e30f;
    /* int16 = fp16 2-byte INPUT geometry + INT16 precision + INT8 INTEGER OUTPUT stage (fp16's output stage
     * gave all-zeros — it reads an fp accumulator). Output-format regs from the int8 template (0x4010/0x4050/
     * 0x40c0 differ int8-vs-fp16 and synth doesn't set them). Env-overridable for on-board RE tuning. */
    ork_f16_fuzz_clear();
    ork_f16_fuzz_add(0x0201,0x100c,  getenv("ORK_I16_100c")?strtoul(getenv("ORK_I16_100c"),0,0):0x20000090u);
    ork_f16_fuzz_add(0x1001,0x4010,  getenv("ORK_I16_4010")?strtoul(getenv("ORK_I16_4010"),0,0):0x80000000u);
    ork_f16_fuzz_add(0x1001,0x4050,  getenv("ORK_I16_4050")?strtoul(getenv("ORK_I16_4050"),0,0):0x000007fcu);
    ork_f16_fuzz_add(0x1001,0x40c0,  getenv("ORK_I16_40c0")?strtoul(getenv("ORK_I16_40c0"),0,0):0x00000080u);
    int rc=ork_f16_mm_run(c,w,M,(const f16*)Ai,C);
    ork_f16_fuzz_clear();
    printf("i16 matmul M=%d K=%d N=%d rc=%d\n",M,K,N,rc);
    /* output stage is INTEGER: the device buffer holds int32, but ork_f16_mm_run's C is float* — reinterpret bits */
    int32_t*Ci=(int32_t*)C;
    int nbad=0;
    for(int i=0;i<M*N;i++){ if(Ci[i]!=ref[i]) nbad++; }
    printf("  ref[0..7]= %d %d %d %d %d %d %d %d\n", ref[0],ref[1],ref[2],ref[3],ref[4],ref[5],ref[6],ref[7]);
    printf("  npu[0..7]= %d %d %d %d %d %d %d %d\n", Ci[0],Ci[1],Ci[2],Ci[3],Ci[4],Ci[5],Ci[6],Ci[7]);
    printf("  mismatches=%d/%d -> %s\n", nbad,M*N, nbad==0?"INT16 MATMUL WORKS (bit-exact vs CPU int16)":"differs");
    ork_mm_free(c,w); ork_npu_free(c); free(Bi);free(Ai);free(ref);free(C);
    return 0;
}
