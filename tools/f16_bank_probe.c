/* tools/f16_bank_probe.c — does the fp16 fused-SiLU index EVER reach the UPPER LUT bank (idx>514)?
 * The int8 path uses idx=in+512 across BOTH banks (0..511 neg, 512..1023 pos); the fp16 path was found to
 * CLAMP positive acc at idx~514 (only the lower bank), which is why the calibration negates the gate. This
 * probe feeds acc across BOTH signs and prints the measured index per acc, so we can see the actual index
 * function and hunt (via the ORK_F16_C4060/C4064/ZA/C4108/C410C register knobs + idx_off/cfg4068 args) for a
 * config where positive acc spreads UP into 512..1023. If found: 2x usable LUT + no negate hack.
 *   make f16_bank_probe && sudo env ORK_F16_C4060=... ./f16_bank_probe [idx_off_hex] [cfg4068_hex]   (board)
 */
#define _GNU_SOURCE
#include "ork_npu.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define K 512
#define N 64
static ork_npu*c; static ork_w*w; static const int M=8;
static ork_f16 *A,*B; static float *C; static double acc[N];
static unsigned g_io,g_c4;
static int run(const int16_t*lut){ return ork_f16_mm_run_silu(c,w,M,A,C,0,g_io,g_c4,lut,1030); }

int main(int argc,char**argv){
    g_io = argc>1?(unsigned)strtoul(argv[1],0,0):0xffffc000u;
    g_c4 = argc>2?(unsigned)strtoul(argv[2],0,0):0x56391100u;
    c=ork_npu_init(); if(!c){printf("no board\n");return 0;}
    A=malloc((size_t)M*K*2); B=malloc((size_t)K*N*2); C=malloc((size_t)M*N*4);
    for(int i=0;i<M*K;i++)A[i]=(ork_f16)1.0f;
    /* acc[n] spans [-390,+390] symmetric: B[k,n]=acc/K so K*A*B = acc (A=1). Positive AND negative. */
    for(int n=0;n<N;n++){ acc[n]=12.2*(n-31.5); double b=acc[n]/(double)K; for(int k=0;k<K;k++)B[(size_t)k*N+n]=(ork_f16)b; }
    w=ork_f16_mm_pack(c,K,N,B); if(!w){printf("pack fail\n");return 2;}
    int16_t lut[1030];
    for(int i=0;i<1030;i++)lut[i]=1000; if(run(lut)){printf("run fail\n");return 1;} double o1=C[32];
    for(int i=0;i<1030;i++)lut[i]=3000; run(lut); double o2=C[32];
    double R=(o2-o1)/2000.0, bias=o1-R*1000.0;
    printf("io=0x%x c4068=0x%x  R=%.5f bias=%.2f  [C4060=%s C4064=%s ZA=%s C4108=%s C410C=%s]\n",
           g_io,g_c4,R,bias, getenv("ORK_F16_C4060")?:"def",getenv("ORK_F16_C4064")?:"def",
           getenv("ORK_F16_ZA")?:"def",getenv("ORK_F16_C4108")?:"def",getenv("ORK_F16_C410C")?:"def");
    if(fabs(R)<1e-9){ printf("  R~0 -> not R*LUT+bias under this config (out@1000=%.1f @3000=%.1f)\n",o1,o2); return 1; }
    for(int i=0;i<1030;i++)lut[i]=(int16_t)(i-512); run(lut);
    int mn=99999,mx=-99999,upper=0;
    for(int n=0;n<N;n++){ int idx=(int)lround((C[n]-bias)/R)+512; if(idx<mn)mn=idx; if(idx>mx)mx=idx; if(idx>520)upper++;
        if(n%4==0||fabs(acc[n])<20) printf("  acc=%+8.2f  idx=%d\n",acc[n],idx); }
    printf("SUMMARY: idx range [%d..%d]  upper-bank(idx>520) points=%d  %s\n",
           mn,mx,upper, upper>0?"*** UPPER BANK REACHED ***":"(lower bank only / clamped)");
    ork_mm_free(c,w); ork_npu_free(c); return 0;
}
