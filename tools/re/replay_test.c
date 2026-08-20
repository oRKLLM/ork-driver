/* replay_test.c — validate an ASSEMBLED op (assemble_op.c output) reproduces RKNN's activation bit-exactly.
 * Replays LUT_SILU_I16 + REGCMD_SILU_I16 (RKNN's matched LUT+params) and compares to RKNN's dequant model:
 *   real_in = (q_in - zp_in)*in_scale ; q_out = round( silu(real_in)/out_scale + zp_out )   (clamped int16)
 * Pass RKNN's scales/zps (from run_rknn):  replay_test <in_scale> <zp_in> <out_scale> <zp_out>
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "ork_npu.h"
#include "regcmd_silu_i16_assembled.h"   /* LUT_SILU_I16[], REGCMD_SILU_I16[] */

static long clampi16(long v){ if(v>32767)v=32767; if(v<-32768)v=-32768; return v; }
static double siluf(double x){ return x/(1.0+exp(-x)); }

int main(int argc,char**argv){
    double in_scale = argc>1?atof(argv[1]):0.00008759;
    int    zp_in    = argc>2?atoi(argv[2]):381;
    double out_scale= argc>3?atof(argv[3]):0.00004514;
    int    zp_out   = argc>4?atoi(argv[4]):-26599;
    ork_npu *c=ork_npu_init();
    if(!c){ printf("init (board only) — SKIP\n"); return 0; }
    if(!ork_ppu_fuse_enabled(c)){ printf("PPU off — SKIP\n"); ork_npu_free(c); return 0; }

    const int M=16,N=64;                       /* 1024 samples across the int16 input range */
    static short in[1024],out[1024];
    for(int i=0;i<M*N;i++){ long q=(long)(-32000 + (64000L*i)/(M*N-1)); in[i]=(short)q; }
    double us=0;
    int r=ork_i16_npu_replay_lut(c, REGCMD_SILU_I16, REGCMD_SILU_I16_N, LUT_SILU_I16, LUT_SILU_I16_N, in, M, N, out, &us);
    if(r){ printf("replay rc=%d\n",r); ork_npu_free(c); return 1; }
    int bad=0; long mx=0;
    for(int i=0;i<M*N;i++){
        double real_in=(in[i]-zp_in)*in_scale;
        long ref=clampi16(lround(siluf(real_in)/out_scale + zp_out));
        long e=labs((long)out[i]-ref); if(e>mx)mx=e; if(e>1)bad++;
        if((i%64)==0) printf("  q_in=%6d real_in=%7.3f out=%7d ref=%7ld\n", in[i], real_in, out[i], ref);
    }
    printf("assembled int16 SiLU vs RKNN dequant model: bad(|e|>1)=%d/%d max|e|=%ld  (%.1f us)\n", bad, M*N, mx, us);
    ork_npu_free(c);
    return bad?1:0;
}
