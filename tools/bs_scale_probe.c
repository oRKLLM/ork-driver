/* bs_scale_probe — validate on-NPU per-channel scale out[m][n]=a[m][n]*b[n], b[N] broadcast across rows
 * (ERDMA_DATA_MODE=0). int8 (ork_npu_mul_perchan_i8) + fp16 (ork_npu_mul_perchan_f16) at general geometry
 * vs CPU. BOARD: sudo ./bs_scale_probe */
#include "ork_npu.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
static int clamp8(int v){ return v>127?127:v<-128?-128:v; }
int main(void){
    ork_npu*c=ork_npu_init(); if(!c){printf("init failed\n");return 2;}
    if(!ork_ppu_fuse_enabled(c)){printf("PPU fuse not enabled — SKIP\n");ork_npu_free(c);return 0;}
    setenv("ORK_EW_TIMEOUT","2000",1);
    struct{int M,N;}cs[]={{8,64},{64,256},{128,512}};
    int fail=0;
    for(unsigned k=0;k<sizeof(cs)/sizeof(cs[0]);k++){
        int M=cs[k].M,N=cs[k].N;
        /* int8 */
        int8_t*a=malloc((size_t)M*N),*b=malloc(N),*out=malloc((size_t)M*N);
        for(int m=0;m<M;m++)for(int n=0;n<N;n++) a[(size_t)m*N+n]=(int8_t)(((m+n)%7)-3);
        for(int n=0;n<N;n++) b[n]=(int8_t)(n%4);
        int rc=ork_npu_mul_perchan_i8(c,a,b,M,N,0x4000,14,out,NULL);
        int bad=0; for(int m=0;m<M;m++)for(int n=0;n<N;n++){int e=clamp8((int)a[(size_t)m*N+n]*(int)b[n]); if(out[(size_t)m*N+n]!=e)bad++;}
        /* fp16 */
        ork_f16*fa=malloc((size_t)M*N*2),*fb=malloc(N*2),*fo=malloc((size_t)M*N*2);
        for(int m=0;m<M;m++)for(int n=0;n<N;n++) fa[(size_t)m*N+n]=(ork_f16)((((m+n)%7)-3)*0.5f);
        for(int n=0;n<N;n++) fb[n]=(ork_f16)((n%4)*0.25f);
        int rcf=ork_npu_mul_perchan_f16(c,fa,fb,M,N,fo,NULL);
        double maxerr=0; int badf=0; for(int m=0;m<M;m++)for(int n=0;n<N;n++){ float e=(float)fa[(size_t)m*N+n]*(float)fb[n]; float g=(float)fo[(size_t)m*N+n]; double er=fabs(g-e); if(er>maxerr)maxerr=er; if(er>0.02)badf++; }
        /* int16 (gain=1: mult=0x4000, shift=14 -> out=a*b) */
        int16_t*ia=malloc((size_t)M*N*2),*ib=malloc(N*2),*io=malloc((size_t)M*N*2);
        for(int m=0;m<M;m++)for(int n=0;n<N;n++) ia[(size_t)m*N+n]=(int16_t)(((m+n)%7)-3);
        for(int n=0;n<N;n++) ib[n]=(int16_t)(n%4);
        int rci=ork_npu_mul_perchan_i16(c,ia,ib,M,N,0x4000,14,io,NULL);
        int badi=0; for(int m=0;m<M;m++)for(int n=0;n<N;n++){ int e=(int)ia[(size_t)m*N+n]*(int)ib[n]; if(e>32767)e=32767; if(e<-32768)e=-32768; if(io[(size_t)m*N+n]!=e)badi++; }
        printf("M=%-4d N=%-4d : i8 %d/%d | f16 %d/%d (err %.4f) | i16 rc=%d %d/%d  %s\n",
               M,N,M*N-bad,M*N, M*N-badf,M*N,maxerr, rci,M*N-badi,M*N, (!rc&&!bad&&!rcf&&!badf&&!rci&&!badi)?"OK":"FAIL");
        if(rc||bad||rcf||badf||rci||badi)fail++;
        free(a);free(b);free(out);free(fa);free(fb);free(fo);free(ia);free(ib);free(io);
    }
    ork_npu_free(c);
    printf(fail?"\nRESULT: per-channel scale NOT clean\n":"\nRESULT: on-NPU per-channel scale (i8 + fp16) WORKS at general geometry\n");
    return fail?1:0;
}
