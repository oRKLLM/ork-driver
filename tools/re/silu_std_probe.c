/* silu_std_probe.c — RE/calibration harness for the standalone on-NPU SiLU op (ork_npu_probe_silu_std).
 * Phase 1 ("ramp"): load a ramp LUT (LUT[i]=i-512) and sweep the full int8 input range, printing the
 * transfer curve out(in) — this reveals the op's index math idx(in) so we can build a silu curve.
 * Phase 2 ("silu"): build a silu LUT for a chosen (in_scale,out_scale) under the measured index model and
 * validate the NPU output against a CPU silu reference.
 *
 *   cc -O2 -Iinclude -Isrc -pthread -o silu_std_probe tools/re/silu_std_probe.c src/npu.c src/soc.c \
 *       src/soc/rk3588.c src/soc/rk3576.c src/neon_activations.c -lm
 *   sudo ./silu_std_probe ramp <r_mult> <r_shift> <out_bias_hex> <idx_off_hex> <c4064_hex> <c4068_hex>
 *   sudo ./silu_std_probe silu <r_mult> <r_shift> <in_scale> <out_scale> <idx_off_hex> <c4064_hex> <c4068_hex>
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "ork_npu.h"

static int clampi8(long v){ if(v>127)v=127; if(v<-128)v=-128; return (int)v; }
static int clampi16(long v){ if(v>32767)v=32767; if(v<-32768)v=-32768; return (int)v; }
static double siluf(double x){ return x/(1.0+exp(-x)); }

int main(int argc, char**argv){
    if(argc<2){ printf("usage: %s ramp|silu ...\n",argv[0]); return 2; }
    ork_npu *c = ork_npu_init();
    if(!c){ printf("init failed (board only) — SKIP\n"); return 0; }
    if(!ork_ppu_fuse_enabled(c)){ printf("PPU not enabled — SKIP\n"); ork_npu_free(c); return 0; }

    const int M = getenv("PM")?atoi(getenv("PM")):8;
    const int N = getenv("PN")?atoi(getenv("PN")):64;
    static signed char in[8192]; static signed char out[8192];
    for(int i=0;i<M*N;i++) in[i]=(signed char)((i%256)-128);   /* sweep int8 range */
    static short lut[1030];

    if(!strcmp(argv[1],"ramp")){
        int r_mult = argc>2?atoi(argv[2]):0x4000, r_shift = argc>3?atoi(argv[3]):14;
        unsigned out_bias = argc>4?(unsigned)strtoul(argv[4],0,16):0;
        unsigned idx_off  = argc>5?(unsigned)strtoul(argv[5],0,16):0xffffc000;
        unsigned c4064    = argc>6?(unsigned)strtoul(argv[6],0,16):0;
        unsigned c4068    = argc>7?(unsigned)strtoul(argv[7],0,16):0;
        for(int i=0;i<1030;i++) lut[i]=(short)clampi16(i-512);   /* ramp: LUT[idx]=idx-512 */
        double us=0;
        int r = ork_npu_probe_silu_std(c,in,M,N,r_mult,r_shift,out_bias,idx_off,c4064,c4068,lut,1030,out,&us);
        printf("ramp rc=%d (%.1f us)  r=0x%x/2^%d out_bias=0x%x idx_off=0x%x c4064=0x%x c4068=0x%x\n",
               r,us,r_mult,r_shift,out_bias,idx_off,c4064,c4068);
        if(r){ ork_npu_free(c); return 1; }
        for(int i=0;i<M*N;i++) if((i&7)==0||in[i]==0) printf("  in=%4d -> out=%4d\n", in[i], out[i]);
        ork_npu_free(c); return 0;
    }

    if(!strcmp(argv[1],"silu")){
        /* ork-native self-calibrating standalone SiLU: (1) measure idx(v) with a ramp LUT (R=1,bias=0), then
         * (2) build LUT[idx(v)] = silu(v*in_scale)/out_scale/R_run and run, validating vs CPU silu. */
        double in_scale = argc>2?atof(argv[2]):0.125, out_scale = argc>3?atof(argv[3]):0.063;
        int r_mult = argc>4?atoi(argv[4]):0x4000, r_shift = argc>5?atoi(argv[5]):14;
        unsigned idx_off = argc>6?(unsigned)strtoul(argv[6],0,16):0xffffc000;
        unsigned c4064   = argc>7?(unsigned)strtoul(argv[7],0,16):0xffff7dc8;
        unsigned c4068   = argc>8?(unsigned)strtoul(argv[8],0,16):0x411c0800;
        double Rrun=(double)r_mult/(double)(1u<<r_shift), us=0;
        /* pass 1: ramp LUT (LUT[i]=i-512), R=0.5, bias=0 -> out=clamp(0.5*(idx-512)) -> idx(v)=2*out+512.
         * R=0.5 keeps out unclamped across the full int8 input range (idx~[245,762]); mapping is R-independent. */
        for(int i=0;i<1030;i++) lut[i]=(short)clampi16(i-512);
        for(int i=0;i<M*N;i++) in[i]=(signed char)((i%256)-128);
        if(ork_npu_probe_silu_std(c,in,M,N,0x2000,14,0,idx_off,c4064,c4068,lut,1030,out,&us)){ printf("calib wedged\n"); ork_npu_free(c); return 1; }
        int idxof[256]; for(int v=0;v<256;v++) idxof[v]=-1;
        for(int i=0;i<M*N;i++){ int v=(unsigned char)in[i]; int o=out[i];
            if(o>-127&&o<127) idxof[v]=2*o+512; }             /* skip saturated ends */
        /* pass 2: build silu curve at measured indices; interp gaps, hold ends */
        int set[1030]; for(int i=0;i<1030;i++){lut[i]=0;set[i]=0;}
        for(int vv=-128;vv<128;vv++){ int v=(unsigned char)vv; int idx=idxof[v]; if(idx<0||idx>1029)continue;
            double val=siluf(vv*in_scale)/out_scale/Rrun; lut[idx]=(short)clampi16(lround(val)); set[idx]=1; }
        int lo=-1,hi=-1; for(int i=0;i<1030;i++)if(set[i]){lo=i;break;} for(int i=1029;i>=0;i--)if(set[i]){hi=i;break;}
        for(int i=0;i<lo;i++)lut[i]=lut[lo]; for(int i=hi+1;i<1030;i++)lut[i]=lut[hi];
        for(int i=lo;i<=hi;i++){ if(set[i])continue; int a=i,b=i; while(a>lo&&!set[a])a--; while(b<hi&&!set[b])b++;
            lut[i]=(short)(lut[a]+(lut[b]-lut[a])*(i-a)/(b-a)); }
        /* pass 3: run with the built curve + chosen R, validate vs CPU silu */
        if(ork_npu_probe_silu_std(c,in,M,N,r_mult,r_shift,0,idx_off,c4064,c4068,lut,1030,out,&us)){ printf("run wedged\n"); ork_npu_free(c); return 1; }
        int bad=0; double mx=0;
        for(int i=0;i<M*N;i++){ double ref=siluf(in[i]*in_scale)/out_scale; double got=out[i];
            double e=fabs(got-ref); if(e>mx)mx=e; if(e>2)bad++;
            if((i%37)==0) printf("  in=%4d out=%4d ref=%6.1f\n",in[i],out[i],ref); }
        printf("silu rc=0 (%.1f us) R=%.4f in_scale=%.3f out_scale=%.3f  bad(|e|>2)=%d/%d max|e|=%.2f\n",
               us,Rrun,in_scale,out_scale,bad,M*N,mx);
        ork_npu_free(c); return bad?1:0;
    }
    printf("unknown mode\n"); ork_npu_free(c); return 2;
}
