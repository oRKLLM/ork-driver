/* tools/hadamard_real.c — Tier 4a, but on a REAL weight tensor (no NPU, pure CPU math).
 *
 * The synthetic hadamard_int4 test uses random-Gaussian data, which is the WORST case for quant
 * (no structure to exploit) — so it shows a ~13% int4 "floor" even when nothing is wrong. This tool
 * instead loads a real Q4_K weight tensor from a GGUF model, dequantizes it to fp32, and uses it as
 * the matmul weight B — real model weights, not random noise. Activations A are Gaussian with injected
 * outliers (real LLM activations are ~Gaussian with heavy-tailed outliers; that IS the thing Hadamard
 * tames). Measures the W4A4 / W4A8 / W8A8 matmul RMS error vs the fp32 reference, plain vs Hadamard.
 *
 *   make hadamard_real && ./hadamard_real /path/to/model.gguf [M] [N] [G] [outlier_scale]
 *
 * NOTE: matmul RMS is a DIAGNOSTIC, not the go/no-go. int4 error is unbiased and averages out across
 * a matmul and across residual-connected layers, so a "high" RMS can still give fine perplexity (this
 * is why QuaRot int4 models work). The real verdict needs perplexity from the wired-in pipeline.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>

typedef uint16_t f16b;
static float h2f(uint16_t h){ /* IEEE half -> float */
    uint32_t s=(h>>15)&1,e=(h>>10)&0x1f,m=h&0x3ff,out;
    if(e==0){ if(m==0)out=s<<31; else { e=127-15+1; while(!(m&0x400)){m<<=1;e--;} m&=0x3ff; out=(s<<31)|(e<<23)|(m<<13);} }
    else if(e==0x1f) out=(s<<31)|(0xff<<23)|(m<<13);
    else out=(s<<31)|((e-15+127)<<23)|(m<<13);
    float f; memcpy(&f,&out,4); return f;
}

/* ---- minimal GGUF v3 reader (same as gguf_q4k.c) ---- */
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

/* in-place fast Walsh-Hadamard transform, then *1/sqrt(n) -> orthonormal Q (n a power of 2) */
static void fwht_norm(float *v, int n){
    for(int len=1; len<n; len<<=1)
        for(int i=0;i<n;i+=len<<1)
            for(int j=i;j<i+len;j++){ float a=v[j], b=v[j+len]; v[j]=a+b; v[j+len]=a-b; }
    float s=1.0f/sqrtf((float)n);
    for(int i=0;i<n;i++) v[i]*=s;
}
/* per-group symmetric int quant+dequant of a length-K vector */
static void q_dequant_vec(const float *x, float *out, int K, int Gp, int bits){
    int lim=(1<<(bits-1))-1;
    for(int g=0; g<K; g+=Gp){
        float mx=1e-9f; for(int k=0;k<Gp;k++){ float a=fabsf(x[g+k]); if(a>mx)mx=a; }
        float s=mx/lim, inv=lim/mx;
        for(int k=0;k<Gp;k++){ int q=(int)lrintf(x[g+k]*inv); if(q>lim)q=lim; if(q<-lim)q=-lim; out[g+k]=q*s; }
    }
}
static double matmul_q_err(const float *A,const float *B,const float *Cref,int M,int K,int N,int Gp,int abits,int bbits,int rotate){
    float *Aq=malloc((size_t)M*K*4), *Bq=malloc((size_t)K*N*4), *ra=malloc((size_t)K*4), *rb=malloc((size_t)K*4);
    for(int m=0;m<M;m++){ const float *row=A+(size_t)m*K;
        if(rotate){ memcpy(ra,row,(size_t)K*4); fwht_norm(ra,K); q_dequant_vec(ra,Aq+(size_t)m*K,K,Gp,abits); }
        else q_dequant_vec(row,Aq+(size_t)m*K,K,Gp,abits); }
    for(int n=0;n<N;n++){
        for(int k=0;k<K;k++) rb[k]=B[(size_t)k*N+n];
        if(rotate) fwht_norm(rb,K);
        float *col=malloc((size_t)K*4); q_dequant_vec(rb,col,K,Gp,bbits);
        for(int k=0;k<K;k++) Bq[(size_t)k*N+n]=col[k]; free(col);
    }
    double num=0,den=0;
    for(int m=0;m<M;m++)for(int n=0;n<N;n++){
        double c=0; for(int k=0;k<K;k++) c+=(double)Aq[(size_t)m*K+k]*Bq[(size_t)k*N+n];
        double r=Cref[(size_t)m*N+n], e=c-r; num+=e*e; den+=r*r; }
    free(Aq);free(Bq);free(ra);free(rb);
    return den>0? sqrt(num/den):0;
}

int main(int argc,char**argv){
    if(argc<2){printf("usage: %s model.gguf [M] [N] [G] [outlier_scale]\n",argv[0]);return 1;}
    int M=argc>2?atoi(argv[2]):16, Nreq=argc>3?atoi(argv[3]):512, Gp=argc>4?atoi(argv[4]):128;
    float oscale=argc>5?atof(argv[5]):20.0f;
    FILE*f=fopen(argv[1],"rb"); if(!f){perror("open");return 1;}
    fseek(f,0,SEEK_END); long sz=ftell(f); fseek(f,0,SEEK_SET);
    uint8_t*buf=malloc(sz); if(fread(buf,1,sz,f)!=(size_t)sz){printf("read fail\n");return 1;} fclose(f);
    G=buf; GP=0;
    if(ru32()!=0x46554747u){printf("not GGUF\n");return 1;}
    ru32(); uint64_t ntensor=ru64(), nkv=ru64(); uint32_t align=32;
    for(uint64_t i=0;i<nkv;i++){ uint64_t kl; const char*k=rstr(&kl); uint32_t vt=ru32();
        if(kl==17&&!memcmp(k,"general.alignment",17)&&vt==4){ align=ru32(); } else skipval(vt); }
    char tname[256]={0}; uint64_t off=0,ne0=0,ne1=0; int found=0;
    for(uint64_t i=0;i<ntensor;i++){
        uint64_t nl; const char*nm=rstr(&nl); uint32_t nd=ru32(); uint64_t d[4]={1,1,1,1};
        for(uint32_t j=0;j<nd&&j<4;j++) d[j]=ru64();
        uint32_t ty=ru32(); uint64_t o=ru64();
        /* want any Q4_K 2D tensor with enough columns; K is truncated to a power of 2 below */
        if(!found && ty==12 && nd==2 && d[0]>=512 && d[1]>=64){
            found=1; off=o; ne0=d[0]; ne1=d[1]; int c=nl<255?nl:255; memcpy(tname,nm,c); tname[c]=0;
        }
    }
    if(!found){printf("no usable Q4_K 2D tensor found\n");return 1;}
    size_t data0=(GP+align-1)/align*align;
    /* plain Hadamard needs power-of-2 K — use the largest power-of-2 slice of the real channels */
    int K=1; while((uint64_t)(K<<1)<=ne0) K<<=1;
    int N=(int)ne1; if(N>Nreq)N=Nreq;
    printf("real weight '%s' Q4_K [K=%d N=%d], M=%d G=%d outliers x%.0f\n",tname,K,N,M,Gp,oscale);

    /* dequant Q4_K -> fp32 weights wf[ne1][ne0] (row n, col k) */
    size_t nel=(size_t)ne0*ne1, nb=nel/256; float*wf=malloc(nel*4);
    const uint8_t*blk=G+data0+off;
    for(size_t b=0;b<nb;b++){ const uint8_t*x=blk+b*144;
        float d=h2f(*(const uint16_t*)x), dmin=h2f(*(const uint16_t*)(x+2));
        const uint8_t*sc=x+4,*q=x+16; float*y=wf+b*256; int is=0;
        for(int j=0;j<256;j+=64){
            uint8_t s1,m1,s2,m2;
            #define GSM(J,D,Mn) do{ if((J)<4){*(D)=sc[J]&63;*(Mn)=sc[(J)+4]&63;} \
              else{*(D)=(sc[(J)+4]&0xF)|((sc[(J)-4]>>6)<<4);*(Mn)=(sc[(J)+4]>>4)|((sc[(J)]>>6)<<4);} }while(0)
            GSM(is,&s1,&m1); GSM(is+1,&s2,&m2);
            float d1=d*s1,mm1=dmin*m1,d2=d*s2,mm2=dmin*m2;
            for(int l=0;l<32;l++)*y++=d1*(q[l]&0xF)-mm1;
            for(int l=0;l<32;l++)*y++=d2*(q[l]>>4)-mm2;
            q+=32; is+=2;
        }
    }
    /* B[k][n] = wf[n*ne0+k] (real weights) */
    float*B=malloc((size_t)K*N*4);
    for(int n=0;n<N;n++)for(int k=0;k<K;k++) B[(size_t)k*N+n]=wf[(size_t)n*ne0+k];
    /* A: Gaussian + injected activation outliers */
    float*A=malloc((size_t)M*K*4); unsigned sd=1234;
    for(size_t i=0;i<(size_t)M*K;i++){ sd=sd*1103515245u+12345u; float u1=((sd>>9)+1)/(float)((1u<<23)+2);
        sd=sd*1103515245u+12345u; float u2=((sd>>9)+1)/(float)((1u<<23)+2);
        A[i]=sqrtf(-2*logf(u1))*cosf(6.2831853f*u2); }
    for(int k=0;k<K;k+=37) for(int m=0;m<M;m++) A[(size_t)m*K+k]*=oscale;
    float*Cref=malloc((size_t)M*N*4);
    for(int m=0;m<M;m++)for(int n=0;n<N;n++){ double s=0; for(int k=0;k<K;k++) s+=(double)A[(size_t)m*K+k]*B[(size_t)k*N+n]; Cref[(size_t)m*N+n]=s; }

    printf("matmul quant RMS rel err (REAL weights, synthetic-outlier activations):\n");
    printf("  W8A8 (int8/int8)            sanity : %.3f%%\n", 100*matmul_q_err(A,B,Cref,M,K,N,Gp,8,8,0));
    printf("  W4A4 (int4/int4)  plain            : %.3f%%\n", 100*matmul_q_err(A,B,Cref,M,K,N,Gp,4,4,0));
    printf("  W4A4 (int4/int4)  + Hadamard       : %.3f%%\n", 100*matmul_q_err(A,B,Cref,M,K,N,Gp,4,4,1));
    printf("  W4A8 (act8/wt4)   plain  [not HW]  : %.3f%%\n", 100*matmul_q_err(A,B,Cref,M,K,N,Gp,8,4,0));
    printf("  W4A8 (act8/wt4)   + Hadamard [n/HW]: %.3f%%\n", 100*matmul_q_err(A,B,Cref,M,K,N,Gp,8,4,1));
    free(buf);free(wf);free(A);free(B);free(Cref); return 0;
}
