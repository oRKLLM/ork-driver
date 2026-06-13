/* tools/gguf_q4k.c — load a real GGUF Q4_K model, dequantize one weight tensor, and run it through
 * the NPU. GGUF Q4_K is W4A8 (int4 weight x int8 activation), which the NPU's same-precision MAC
 * can't do directly — so the adaptation is: dequant Q4_K -> fp32, requantize per-channel to int8,
 * run W8A8 on the NPU, and check vs the fp32 reference. Demonstrates the GGUF -> NPU path on real
 * model weights. (Q8_0 GGUF would map straight to int8 with no requant.)
 *   make gguf_q4k && sudo ./gguf_q4k /path/to/model.gguf
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include "ork_npu.h"
typedef ork_f16 f16;
static float h2f(uint16_t h){ f16 v; memcpy(&v,&h,2); return (float)v; }

/* ---- minimal GGUF v3 reader ---- */
static const uint8_t *G; static size_t GP;
static uint32_t ru32(void){ uint32_t v; memcpy(&v,G+GP,4); GP+=4; return v; }
static uint64_t ru64(void){ uint64_t v; memcpy(&v,G+GP,8); GP+=8; return v; }
static const char*rstr(uint64_t*len){ *len=ru64(); const char*s=(const char*)(G+GP); GP+=*len; return s; }
static size_t tsize(uint32_t t){ switch(t){case 0:case 1:case 7:return 1;case 2:case 3:return 2;
    case 4:case 5:case 6:return 4;case 10:case 11:case 12:return 8;default:return 0;} }
static void skipval(uint32_t t){
    if(t==8){ uint64_t l; rstr(&l); return; }
    if(t==9){ uint32_t et=ru32(); uint64_t n=ru64(); for(uint64_t i=0;i<n;i++){ if(et==8){uint64_t l;rstr(&l);} else GP+=tsize(et);} return; }
    GP+=tsize(t);
}

int main(int argc,char**argv){
    if(argc<2){printf("usage: %s model.gguf\n",argv[0]);return 1;}
    FILE*f=fopen(argv[1],"rb"); if(!f){perror("open");return 1;}
    fseek(f,0,SEEK_END); long sz=ftell(f); fseek(f,0,SEEK_SET);
    uint8_t*buf=malloc(sz); if(fread(buf,1,sz,f)!=(size_t)sz){printf("read fail\n");return 1;} fclose(f);
    G=buf; GP=0;
    if(ru32()!=0x46554747u){printf("not GGUF\n");return 1;}
    uint32_t ver=ru32(); uint64_t ntensor=ru64(), nkv=ru64();
    uint32_t align=32;
    for(uint64_t i=0;i<nkv;i++){ uint64_t kl; const char*k=rstr(&kl); uint32_t vt=ru32();
        if(kl==17&&!memcmp(k,"general.alignment",17)&&vt==4){ align=ru32(); } else skipval(vt); }
    printf("GGUF v%u: %llu tensors, %llu kv, align=%u\n",ver,(unsigned long long)ntensor,(unsigned long long)nkv,align);
    /* tensor infos: find a Q4_K (type 12) 2D tensor with usable dims */
    char tname[256]={0}; uint64_t off=0,ne0=0,ne1=0; int found=0;
    for(uint64_t i=0;i<ntensor;i++){
        uint64_t nl; const char*nm=rstr(&nl); uint32_t nd=ru32(); uint64_t d[4]={1,1,1,1};
        for(uint32_t j=0;j<nd&&j<4;j++) d[j]=ru64();
        uint32_t ty=ru32(); uint64_t o=ru64();
        if(!found && ty==12 && nd==2 && d[0]%256==0 && d[1]>=64){
            found=1; off=o; ne0=d[0]; ne1=d[1]; int c=nl<255?nl:255; memcpy(tname,nm,c); tname[c]=0;
        }
    }
    if(!found){printf("no usable Q4_K 2D tensor found\n");return 1;}
    size_t data0=(GP+align-1)/align*align;            /* tensor data starts here */
    printf("tensor '%s' Q4_K [ne0=%llu ne1=%llu]\n",tname,(unsigned long long)ne0,(unsigned long long)ne1);

    /* dequant Q4_K -> fp32 (ggml layout: super-block of 256 = d,dmin (fp16) + 12 scale bytes + 128 qs) */
    size_t nel=(size_t)ne0*ne1, nb=nel/256; float*wf=malloc(nel*4);
    const uint8_t*blk=G+data0+off;
    for(size_t b=0;b<nb;b++){ const uint8_t*x=blk+b*144;
        float d=h2f(*(const uint16_t*)x), dmin=h2f(*(const uint16_t*)(x+2));
        const uint8_t*sc=x+4,*q=x+16; float*y=wf+b*256; int is=0;
        for(int j=0;j<256;j+=64){
            uint8_t s1,m1,s2,m2;
            #define GSM(J,D,M) do{ if((J)<4){*(D)=sc[J]&63;*(M)=sc[(J)+4]&63;} \
              else{*(D)=(sc[(J)+4]&0xF)|((sc[(J)-4]>>6)<<4);*(M)=(sc[(J)+4]>>4)|((sc[(J)]>>6)<<4);} }while(0)
            GSM(is,&s1,&m1); GSM(is+1,&s2,&m2);
            float d1=d*s1,mm1=dmin*m1,d2=d*s2,mm2=dmin*m2;
            for(int l=0;l<32;l++)*y++=d1*(q[l]&0xF)-mm1;
            for(int l=0;l<32;l++)*y++=d2*(q[l]>>4)-mm2;
            q+=32; is+=2;
        }
    }
    /* matmul test: B[K][N]=wf[n*ne0+k]; requant per-channel int8; A random; W8A8 on NPU vs fp32 */
    int K=ne0, N=(ne1/32)*32; if(N>4096)N=4096; int M=4;
    signed char*B=malloc((size_t)K*N),*A=malloc((size_t)M*K); int32_t*C=malloc((size_t)M*N*4);
    float*Af=malloc((size_t)M*K*4),*bS=malloc((size_t)N*4),*aS=malloc((size_t)M*4);
    for(int n=0;n<N;n++){ float mx=1e-9f; for(int k=0;k<K;k++){float v=wf[(size_t)n*ne0+k];if(v<0)v=-v;if(v>mx)mx=v;}
        bS[n]=mx/127; for(int k=0;k<K;k++){int q=(int)lrintf(wf[(size_t)n*ne0+k]/bS[n]);if(q>127)q=127;if(q<-127)q=-127;B[(size_t)k*N+n]=q;} }
    unsigned sd=1; for(int m=0;m<M;m++){ float mx=1e-9f;
        for(int k=0;k<K;k++){sd=sd*1103515245+12345;Af[m*K+k]=((int)(sd>>9)%2001-1000)/1000.0f;float a=Af[m*K+k];if(a<0)a=-a;if(a>mx)mx=a;}
        aS[m]=mx/127; for(int k=0;k<K;k++){int q=(int)lrintf(Af[m*K+k]/aS[m]);if(q>127)q=127;if(q<-127)q=-127;A[m*K+k]=q;} }
    ork_npu*c=ork_npu_init(); if(!c){printf("npu init failed\n");return 1;}
    ork_w*w=ork_mm_pack_i8(c,K,N,B); if(!w){printf("pack failed (K=%d N=%d)\n",K,N);return 1;}
    int rc=ork_mm_run_i8(c,w,M,A,C); ork_w_free(w);
    if(rc){printf("run rc=%d\n",rc);return 1;}
    double se=0,sr=0; for(int m=0;m<M;m++)for(int n=0;n<N;n++){ double r=0; for(int k=0;k<K;k++)r+=(double)Af[m*K+k]*wf[(size_t)n*ne0+k];
        double deq=(double)aS[m]*bS[n]*C[m*N+n]; se+=(deq-r)*(deq-r); sr+=r*r; }
    printf("W8A8 on NPU vs fp32 (real Q4_K weights, M=%d K=%d N=%d): RMS rel err %.2f%%\n",M,K,N,100.0*sqrt(se/sr));
    ork_npu_free(c); return 0;
}
