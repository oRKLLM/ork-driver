/* tools/silu_f16_calib.c — calibrate the fp16 fused SiLU (ork_mm_run_f16_silu), same 2-pass scheme as
 * silu_native does for int8: (1) probe the output relation out = R*LUT[idx] + bias via flat LUTs; (2) recover
 * idx(gate) via a ramp LUT; (3) build the silu curve at those indices; (4) validate vs CPU silu. idx_off is
 * tunable (arg) because the fp16 program's index-centering differs from int8 (positives were landing out of
 * range). A=1 fp16, B[k,n]=b[n] -> gate[n]=K*b[n] sweeps ~[-8,8].
 *   make silu_f16_calib && sudo ./silu_f16_calib [idx_off_hex] [cfg4068_hex]     (board only)
 */
#define _GNU_SOURCE
#include "ork_npu.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

static double silu(double x){ return x/(1.0+exp(-x)); }
#define K 512
#define N 64
static ork_npu*c; static ork_w*w; static const int M=8;
static ork_f16 *A,*B; static float *C; static double gate[N];
static unsigned g_io, g_c4;
static int run(const int16_t*lut){ return ork_mm_run_f16_silu(c,w,M,A,C,0,g_io,g_c4,lut,1030); }

int main(int argc,char**argv){
    g_io = argc>1?(unsigned)strtoul(argv[1],0,0):0xffffc000u;
    g_c4 = argc>2?(unsigned)strtoul(argv[2],0,0):0x56391100u;
    c=ork_npu_init(); if(!c){printf("no board\n");return 0;}
    A=malloc((size_t)M*K*2); B=malloc((size_t)K*N*2); C=malloc((size_t)M*N*4);
    for(int i=0;i<M*K;i++)A[i]=(ork_f16)1.0f;
    /* SCALE the gate up by S so the (fixed, small) fp16 index gain spreads acc across the LUT; silu is
     * evaluated at acc/S (the true gate in silu's active range). S from argv[3] (default 38 -> acc~+-300). */
    double S = argc>3?atof(argv[3]):38.0;
    /* NEGATE the gate: feed acc = -true*S so POSITIVE true gates -> negative acc (the region that spreads);
     * negative true gates -> positive acc -> clamp ~0 (== silu(neg)). true = -acc/S = -gate/S. */
    for(int n=0;n<N;n++){ double tru=0.25*(n-32); double b=(-tru*S)/(double)K; for(int k=0;k<K;k++)B[(size_t)k*N+n]=(ork_f16)b; gate[n]=(double)K*b; }
    w=ork_mm_pack(c,K,N,B); if(!w){printf("pack fail\n");return 2;}
    int16_t lut[1030];

    /* (1) output relation: flat LUT V -> out = R*V + bias (independent of gate/idx) */
    for(int i=0;i<1030;i++)lut[i]=1000; if(run(lut)){printf("run fail\n");return 1;} double o1=C[32];
    for(int i=0;i<1030;i++)lut[i]=3000; run(lut); double o2=C[32];
    double R=(o2-o1)/2000.0, bias=o1-R*1000.0;
    printf("idx_off=0x%x cfg4068=0x%x  R=%.6f bias=%.3f\n",g_io,g_c4,R,bias);
    if(fabs(R)<1e-9){ printf("R~0 -> output not R*LUT+bias (model differs); out(V=1000)=%.2f out(3000)=%.2f\n",o1,o2); return 1; }

    /* (2) idx(gate): ramp LUT[i]=i-512 -> out = R*(idx-512)+bias -> idx = (out-bias)/R + 512 */
    for(int i=0;i<1030;i++)lut[i]=(int16_t)(i-512); run(lut);
    int idx[N], inrange=0, mn=99999,mx=-99999;
    for(int n=0;n<N;n++){ idx[n]=(int)lround((C[n]-bias)/R)+512; if(idx[n]>=0&&idx[n]<=1029)inrange++; if(idx[n]<mn)mn=idx[n]; if(idx[n]>mx)mx=idx[n]; }
    printf("idx(gate): range [%d..%d], %d/%d in [0,1029]  (idx@gate0=%d, idx@gate-8=%d, idx@gate+8=%d)\n",
           mn,mx,inrange,N,idx[32],idx[0],idx[N-1]);
    if(inrange<N){ printf("  -> not all gates index in-range; try a different idx_off (arg1)\n"); }

    /* (3) build silu curve at measured indices; (4) validate */
    double out_scale = silu(-gate[N-1]/S)/8000.0; if(out_scale<=0) out_scale=1e-3;
    int set[1030]; for(int i=0;i<1030;i++){lut[i]=0;set[i]=0;}
    for(int n=0;n<N;n++){ int i=idx[n]; if(i<0||i>1029)continue;
        double v=(silu(-gate[n]/S)/out_scale - bias)/R; long q=lround(v); if(q>32767)q=32767; if(q<-32768)q=-32768; lut[i]=(int16_t)q; set[i]=1; }
    int lo=-1,hi=-1; for(int i=0;i<1030;i++)if(set[i]){lo=i;break;} for(int i=1029;i>=0;i--)if(set[i]){hi=i;break;}
    if(lo<0){ printf("no indices set -> calibration failed\n"); return 1; }
    for(int i=0;i<lo;i++)lut[i]=lut[lo]; for(int i=hi+1;i<1030;i++)lut[i]=lut[hi];
    for(int i=lo;i<=hi;i++){ if(set[i])continue; int a=i,b=i; while(a>lo&&!set[a])a--; while(b<hi&&!set[b])b++;
        lut[i]=(int16_t)(lut[a]+(lut[b]-lut[a])*(i-a)/(b-a)); }
    run(lut);
    double se=0,em=0; for(int n=0;n<N;n++){ double ref=silu(-gate[n]/S); double got=C[n]*out_scale; double e=fabs(got-ref); se+=e; if(e>em)em=e; }
    printf("CALIB: mean|err|=%.4f max=%.4f  (silu range ~%.2f, out_scale=%.2e)\n", se/N, em, silu(-gate[N-1]/S), out_scale);
    for(int n=0;n<N;n+=8) printf("  gate=%7.3f  got=%8.4f  ref=%8.4f  idx=%d\n", gate[n], C[n]*out_scale, silu(-gate[n]/S), idx[n]);
    ork_mm_free(c,w); ork_npu_free(c);
    return 0;
}
