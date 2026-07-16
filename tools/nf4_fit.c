/* nf4_fit — fit a model-specific NF4 codebook (16 levels) to the base model's actual weight distribution
 * via Lloyd-Max (k-means on 1-D). The generic NF4 levels assume a unit normal; a fitted codebook matches
 * the model's real per-channel-normalized weight distribution (the "make the LUT from the source model").
 *
 * Input: a raw file of f32 samples that are ALREADY per-channel-normalized to [-1,1] (extract with
 *        nf4_sample: per output channel, divide by absmax; sample a subset of weights). Or stdin.
 * Output: the 16 sorted levels as a C array (paste into ORK_NF4_LVL) + the fit vs generic-NF4 MSE.
 *   make nf4_fit && ./nf4_fit <normalized_samples.f32>   (CPU-only, board-safe; runs off-board too)
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

static const float NF4_GEN[16]={-1.0f,-0.6961928f,-0.5250731f,-0.3949175f,-0.2844414f,-0.1847734f,
    -0.0910500f,0.0f,0.0795803f,0.1609302f,0.2461123f,0.3379152f,0.4407098f,0.5626170f,0.7229568f,1.0f};

static double mse(const float*x,size_t n,const float*lv){
    double e=0; for(size_t i=0;i<n;i++){ float v=x[i],bd=1e30f; for(int j=0;j<16;j++){float d=lv[j]-v; d=d<0?-d:d; if(d<bd)bd=d;} e+=(double)bd*bd; } return e/n;
}
int main(int argc,char**argv){
    if(argc<2){ fprintf(stderr,"usage: %s <normalized_samples.f32> [iters=50]\n",argv[0]); return 1; }
    int iters=argc>2?atoi(argv[2]):50;
    FILE*f=fopen(argv[1],"rb"); if(!f){perror("open");return 1;}
    fseek(f,0,SEEK_END); long bytes=ftell(f); fseek(f,0,SEEK_SET);
    size_t n=bytes/4; float*x=malloc(bytes); if(fread(x,4,n,f)!=n){fprintf(stderr,"read fail\n");return 1;} fclose(f);
    printf("nf4_fit: %zu normalized samples\n", n);

    /* init levels: keep -1,0,+1 anchored (NF4 always includes them); init interior from generic NF4 */
    float lv[16]; memcpy(lv,NF4_GEN,sizeof lv);
    /* Lloyd-Max: assign to nearest, recompute centroid; keep endpoints/zero fixed (NF4 property) */
    double acc[16]; long cnt[16];
    for(int it=0;it<iters;it++){
        for(int j=0;j<16;j++){acc[j]=0;cnt[j]=0;}
        for(size_t i=0;i<n;i++){ float v=x[i]; int bj=0; float bd=1e30f;
            for(int j=0;j<16;j++){float d=lv[j]-v; d=d<0?-d:d; if(d<bd){bd=d;bj=j;}} acc[bj]+=v; cnt[bj]++; }
        for(int j=0;j<16;j++){ if(j==0){lv[j]=-1.0f;continue;} if(j==15){lv[j]=1.0f;continue;} if(j==7){lv[j]=0.0f;continue;}
            if(cnt[j]) lv[j]=(float)(acc[j]/cnt[j]); }
        /* keep sorted (Lloyd-Max on sorted 1-D data stays ordered, but guard) */
        for(int a=1;a<16;a++){float t=lv[a];int b=a-1;while(b>=0&&lv[b]>t){lv[b+1]=lv[b];b--;}lv[b+1]=t;}
    }
    printf("  fitted MSE  %.6f   vs generic-NF4 MSE %.6f  (%.2fx %s)\n",
           mse(x,n,lv), mse(x,n,NF4_GEN), mse(x,n,NF4_GEN)/mse(x,n,lv), mse(x,n,lv)<mse(x,n,NF4_GEN)?"BETTER":"worse");
    printf("static const float ORK_NF4_LVL[16]={");
    for(int j=0;j<16;j++) printf("%s%.7ff", j?",":"", lv[j]);
    printf("};\n");
    free(x); return 0;
}
