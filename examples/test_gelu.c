/* examples/test_gelu.c — validate the on-NPU GELU (ork_npu_gelu_i8/_i16) vs a CPU reference. GELU reuses the
 * same standalone SDP activation-LUT op as SiLU (the LUT holds the GELU curve), so int8 is bit-exact-class and
 * int16 is RKNN-class (few-LSB, scored vs output full-scale). Skips gracefully off-board / non-PPU SoC.
 *   make test_gelu && sudo ./test_gelu            (board only)
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "ork_npu.h"

#define MAXE (128*512)
static int clampi8(long v){ if(v>127)v=127; if(v<-128)v=-128; return (int)v; }
static double geluf(double x){ return 0.5*x*(1.0+erf(x*0.7071067811865476)); }
static double rsqrtf_(double x){ return x>1e-9 ? 1.0/sqrt(x) : 0.0; }

/* rsqrt (RMSNorm building block): positive inputs, out = clamp_i8(round(rsqrt(in*is)/os)) */
static int run_rsqrt_i8(ork_npu *c, int M, int N, double is, double os, int tol){
    static signed char in[MAXE], out[MAXE];
    for(int i=0;i<M*N;i++) in[i]=(signed char)(1+(i%126));   /* positive domain */
    double us=0;
    int r=ork_npu_rsqrt_i8(c,in,M,N,is,os,out,&us);
    if(r){ printf("  rsqrt i8 [%dx%-4d] FAIL (rc=%d)\n",M,N,r); return 1; }
    int mism=0,mx=0;
    for(int i=0;i<M*N;i++){ int ref=clampi8(lround(rsqrtf_(in[i]*is)/os)); int d=abs((int)out[i]-ref);
        if(d>tol){mism++; if(d>mx)mx=d;} }
    printf("  rsqrt i8 [%dx%-4d] is=%.4f os=%.4f %s mism=%d/%d max|err|=%d  (%.1f us)\n",M,N,is,os,mism?"FAIL":"ok  ",mism,M*N,mx,us);
    return mism?1:0;
}

static int run_i8(ork_npu *c, int M, int N, double is, double os, int tol){
    static signed char in[MAXE], out[MAXE];
    for(int i=0;i<M*N;i++) in[i]=(signed char)(((i*7)%256)-128);
    double us=0;
    int r=ork_npu_gelu_i8(c,in,M,N,is,os,out,&us);
    if(r){ printf("  i8  [%dx%-4d] is=%.4f FAIL (rc=%d)\n",M,N,is,r); return 1; }
    int mism=0,mx=0;
    for(int i=0;i<M*N;i++){ int ref=clampi8(lround(geluf(in[i]*is)/os)); int d=abs((int)out[i]-ref);
        if(d>tol){mism++; if(d>mx)mx=d;} }
    printf("  i8  [%dx%-4d] is=%.4f os=%.4f %s mism=%d/%d max|err|=%d  (%.1f us)\n",M,N,is,os,mism?"FAIL":"ok  ",mism,M*N,mx,us);
    return mism?1:0;
}
static int run_i16(ork_npu *c, int M, int N, double is, double os, double fs_tol){
    static short in[MAXE], out[MAXE];
    for(int i=0;i<M*N;i++) in[i]=(short)(-32768 + (int)((65535LL*((i*97)%(M*N)))/(M*N)));
    double us=0;
    int r=ork_npu_gelu_i16(c,in,M,N,is,os,out,&us);
    if(r){ printf("  i16 [%dx%-4d] FAIL (rc=%d)\n",M,N,r); return 1; }
    long mx=0,maxref=1;
    for(int i=0;i<M*N;i++){ long v=lround(geluf(in[i]*is)/os); if(labs(v)>maxref)maxref=labs(v); }
    long tol=2+(long)(fs_tol*maxref); int mism=0;
    for(int i=0;i<M*N;i++){ long ref=lround(geluf(in[i]*is)/os); if(ref>32767)ref=32767; if(ref<-32768)ref=-32768;
        long d=labs((long)out[i]-ref); if(d>mx)mx=d; if(d>tol)mism++; }
    printf("  i16 [%dx%-4d] %s mism=%d/%d max|err|=%ld (%.2f%% FS)  (%.1f us)\n",M,N,mism?"FAIL":"ok  ",mism,M*N,mx,100.0*mx/maxref,us);
    return mism?1:0;
}

int main(void){
    ork_npu *c = ork_npu_init();
    if(!c){ printf("ork_npu_init failed (board only) — SKIP\n"); return 0; }
    if(!ork_ppu_fuse_enabled(c)){ printf("on-NPU GELU not enabled on this SoC — SKIP\n"); ork_npu_free(c); return 0; }
    printf("on-NPU GELU (int8) vs CPU ref:\n");
    int fail=0;
    static const int shapes[][2] = { {8,64}, {16,64}, {8,128}, {32,256}, {1,16}, {4,512}, {8,4096} };
    for(unsigned s=0;s<sizeof(shapes)/sizeof(shapes[0]);s++) fail |= run_i8(c, shapes[s][0], shapes[s][1], 0.0625, 0.0625, 2);
    fail |= run_i8(c, 8, 64, 0.03125, 0.0625, 2);
    printf("on-NPU GELU (int16, full range) vs CPU ref:\n");
    static const int s16[][2] = { {8,64}, {16,64}, {8,128}, {4,512} };
    for(unsigned s=0;s<sizeof(s16)/sizeof(s16[0]);s++) fail |= run_i16(c, s16[s][0], s16[s][1], 0.0001, 0.0001, 0.0025);
    printf("on-NPU rsqrt (int8, RMSNorm building block) vs CPU ref:\n");
    fail |= run_rsqrt_i8(c, 8, 64, 0.5, 0.03125, 3);
    fail |= run_rsqrt_i8(c, 16, 64, 0.25, 0.0625, 3);
    printf("on-NPU exp (int8, softmax building block) vs CPU ref:\n");
    { static signed char in[512], out[512]; for(int i=0;i<512;i++) in[i]=(signed char)(-(i%64));  /* <=0, softmax domain */
      double us=0; int r=ork_npu_exp_i8(c,in,8,64,0.05,0.008,out,&us); int mism=0,mx=0;
      if(r) { printf("  exp i8 FAIL rc=%d\n",r); fail|=1; } else {
        for(int i=0;i<512;i++){ int ref=clampi8(lround(exp(in[i]*0.05)/0.008)); int d=abs((int)out[i]-ref); if(d>3){mism++;if(d>mx)mx=d;} }
        printf("  exp i8   [8x64  ] %s mism=%d/512 max|err|=%d  (%.1f us)\n",mism?"FAIL":"ok  ",mism,mx,us); fail|=mism?1:0; } }
    ork_npu_free(c);
    printf("%s\n", fail ? "FAIL" : "ALL OK");
    return fail;
}
