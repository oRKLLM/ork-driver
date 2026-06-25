/* examples/perplexity_i4.c
 * Evaluates perplexity of W4A4 Hadamard-rotated matmuls vs CPU-fp32 on stories15M.bin.
 * Hardcodes FWHT and int4 per-group quantization + NPU int4 submission.
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include "ork_npu.h"

typedef ork_f16 f16;

typedef struct { int dim,hidden,n_layers,n_heads,n_kv,vocab,seq; } Cfg;
typedef struct {
    const float *tok_emb;
    const float *rms_att, *rms_ffn;
    const float *wq,*wk,*wv,*wo;
    const float *w1,*w2,*w3;
    const float *rms_final;
    /* resident NPU matmul weights */
    ork_w **Wq,**Wk,**Wv,**Wo,**Wg,**Wd,**Wu; ork_w *Wcls;
    /* scales */
    float **Sq, **Sk, **Sv, **So, **Sg, **Sd, **Su; float *Scls;
} Weights;

/* RMSNorm / SwiGLU / softmax via the shared NEON kernels (src/neon_activations.c). */
#include "neon_activations.h"
static void rmsnorm(float*o,const float*x,const float*w,int n){ ork_rmsnorm_f32(o,x,w,n,1e-5f); }
static void softmax(float*x,int n){ ork_softmax_f32(x,n); }


/* pack W4A4 Hadamard */
static ork_w* pack_i4_hadamard(ork_npu* ctx, const float* w_raw, int OUT, int IN, int K_pad, int N_pad, int G, float** out_bS) {
    int use_hadamard = getenv("NO_HADAMARD") ? 0 : 1;
    float* Bf = calloc((size_t)K_pad * N_pad, sizeof(float));
    for(int n=0; n<OUT; n++) {
        for(int k=0; k<IN; k++) Bf[k*N_pad + n] = w_raw[n*IN + k];
    }
    if (use_hadamard) {
        float* col = malloc(K_pad * sizeof(float));
        for(int n=0; n<N_pad; n++) {
            for(int k=0; k<K_pad; k++) col[k] = Bf[k*N_pad + n];
            ork_fwht_norm(col, K_pad);
            for(int k=0; k<K_pad; k++) Bf[k*N_pad + n] = col[k];
        }
        free(col);
    }

    int Sk = K_pad / G;
    signed char* Bi = malloc((size_t)K_pad * N_pad);
    float* bS = malloc((size_t)Sk * N_pad * sizeof(float));
    for(int g=0; g<Sk; g++) {
        for(int n=0; n<N_pad; n++) {
            float mx = 1e-9f;
            for(int j=0; j<G; j++) {
                float b = Bf[(g*G + j)*N_pad + n];
                if(b < 0) b = -b;
                if(b > mx) mx = b;
            }
            bS[g*N_pad + n] = mx / 7.0f;
            for(int j=0; j<G; j++) {
                int q = (int)roundf(Bf[(g*G + j)*N_pad + n] / bS[g*N_pad + n]);
                if(q > 7) q = 7;
                if(q < -8) q = -8;
                Bi[(g*G + j)*N_pad + n] = (signed char)q;
            }
        }
    }
    free(Bf);
    ork_w* w = ork_mm_pack_i4_grouped(ctx, K_pad, N_pad, (int8_t*)Bi, G);
    free(Bi);
    *out_bS = bS;
    return w;
}

/* Run I4 Hadamard or CPU fp32 */
static void mv(ork_npu* ctx, ork_w* W, const float* bS, const float* wraw, int OUT, int IN, int K_pad, int N_pad, int G, const float* x, float* C, int useNPU) {
    int use_hadamard = getenv("NO_HADAMARD") ? 0 : 1;
    if(useNPU) {
        float* x_pad = calloc(K_pad, sizeof(float));
        for(int k=0; k<IN; k++) x_pad[k] = x[k];
        
        if (use_hadamard) ork_fwht_norm(x_pad, K_pad);

        int Sk = K_pad / G;
        signed char* Ai = malloc(K_pad);
        float* aS = malloc(Sk * sizeof(float));
        for(int g=0; g<Sk; g++) {
            float mx = 1e-9f;
            for(int j=0; j<G; j++) {
                float a = x_pad[g*G + j];
                if(a < 0) a = -a;
                if(a > mx) mx = a;
            }
            aS[g] = mx / 7.0f;
            if(aS[g] == 0) aS[g] = 1e-9f;
            for(int j=0; j<G; j++) {
                int q = (int)roundf(x_pad[g*G + j] / aS[g]);
                if(q > 7) q = 7;
                if(q < -8) q = -8;
                Ai[g*G + j] = (signed char)q;
            }
        }
        
        float* C_pad = calloc(N_pad, sizeof(float));
        ork_mm_run_i4_grouped(ctx, W, 1, (int8_t*)Ai, aS, bS, C_pad);

        for(int n=0; n<OUT; n++) C[n] = C_pad[n];

        free(x_pad);
        free(Ai);
        free(aS);
        free(C_pad);
    } else {
        // Pure CPU fp32 reference
        for(int n=0; n<OUT; n++) {
            float acc = 0;
            for(int k=0; k<IN; k++) acc += x[k] * wraw[(size_t)n*IN + k];
            C[n] = acc;
        }
    }
}

int main(int argc,char**argv){
    const char*path=argc>1?argv[1]:"stories15M.bin"; int NSTEP=argc>2?atoi(argv[2]):16;
    int fd=open(path,O_RDONLY); if(fd<0){printf("[ork] perplexity_i4: %s absent, skipping test gracefully.\n", path);return 0;}
    struct stat st; fstat(fd,&st);
    int32_t*hdr=mmap(NULL,st.st_size,PROT_READ,MAP_PRIVATE,fd,0); if(hdr==MAP_FAILED){perror("mmap");return 1;}
    Cfg c={hdr[0],hdr[1],hdr[2],hdr[3],hdr[4],abs(hdr[5]),hdr[6]};
    int dim=c.dim,L=c.n_layers,hid=c.hidden,V=c.vocab,hd=dim/c.n_heads,NH=c.n_heads;
    printf("model %s: dim=%d hidden=%d layers=%d heads=%d vocab=%d seq=%d head_dim=%d\n",path,dim,hid,L,NH,V,c.seq,hd);
    const float*p=(const float*)(hdr+7); Weights w; memset(&w,0,sizeof w);
    w.tok_emb=p; p+=(size_t)V*dim;
    w.rms_att=p; p+=(size_t)L*dim;
    w.wq=p; p+=(size_t)L*dim*dim;  w.wk=p; p+=(size_t)L*dim*dim;  w.wv=p; p+=(size_t)L*dim*dim;  w.wo=p; p+=(size_t)L*dim*dim;
    w.rms_ffn=p; p+=(size_t)L*dim;
    w.w1=p; p+=(size_t)L*hid*dim;  w.w2=p; p+=(size_t)L*dim*hid;  w.w3=p; p+=(size_t)L*hid*dim;
    w.rms_final=p;

    ork_npu*ctx=ork_npu_init(); if(!ctx){printf("init failed (NPU?)\n");return 1;}
    
    // Memory allocation for weights and scales
    w.Wq=malloc(L*sizeof(void*)); w.Sq=malloc(L*sizeof(void*));
    w.Wk=malloc(L*sizeof(void*)); w.Sk=malloc(L*sizeof(void*));
    w.Wv=malloc(L*sizeof(void*)); w.Sv=malloc(L*sizeof(void*));
    w.Wo=malloc(L*sizeof(void*)); w.So=malloc(L*sizeof(void*));
    w.Wg=malloc(L*sizeof(void*)); w.Sg=malloc(L*sizeof(void*));
    w.Wd=malloc(L*sizeof(void*)); w.Sd=malloc(L*sizeof(void*));
    w.Wu=malloc(L*sizeof(void*)); w.Su=malloc(L*sizeof(void*));
    
    int G = 128; // group size

    // Pad sizes for Hadamard & I4 NPU constraints
    int K_dim = 512, N_dim = 320;   // for dimxdim (288x288) -> 512x320
    int K_hid_in = 512, N_hid_out = 768; // for dimxhid (288x768) -> 512x768
    int K_hid_out = 1024, N_dim_out = 320; // for hidxdim (768x288) -> 1024x320
    int K_vcb = 512, N_vcb = ((V + 63)/64)*64; // for dimxvocab (288x32000) -> 512x32000

    printf("Packing W4A4 Hadamard weights...\n");
    for(int l=0;l<L;l++){
        w.Wq[l]=pack_i4_hadamard(ctx,w.wq+(size_t)l*dim*dim,dim,dim, K_dim,N_dim,G,&w.Sq[l]);
        w.Wk[l]=pack_i4_hadamard(ctx,w.wk+(size_t)l*dim*dim,dim,dim, K_dim,N_dim,G,&w.Sk[l]);
        w.Wv[l]=pack_i4_hadamard(ctx,w.wv+(size_t)l*dim*dim,dim,dim, K_dim,N_dim,G,&w.Sv[l]);
        w.Wo[l]=pack_i4_hadamard(ctx,w.wo+(size_t)l*dim*dim,dim,dim, K_dim,N_dim,G,&w.So[l]);
        w.Wg[l]=pack_i4_hadamard(ctx,w.w1+(size_t)l*hid*dim,hid,dim, K_hid_in,N_hid_out,G,&w.Sg[l]);
        w.Wu[l]=pack_i4_hadamard(ctx,w.w3+(size_t)l*hid*dim,hid,dim, K_hid_in,N_hid_out,G,&w.Su[l]);
        w.Wd[l]=pack_i4_hadamard(ctx,w.w2+(size_t)l*dim*hid,dim,hid, K_hid_out,N_dim_out,G,&w.Sd[l]);
    }
    w.Wcls=pack_i4_hadamard(ctx,w.tok_emb,V,dim, K_vcb,N_vcb,G,&w.Scls);

    float*Kc=calloc((size_t)L*c.seq*dim,4),*Vc=calloc((size_t)L*c.seq*dim,4);
    float *x=malloc(dim*4),*xn=malloc(dim*4),*q=malloc(dim*4),*kk=malloc(dim*4),*vv=malloc(dim*4),
          *att=malloc(dim*4),*o=malloc(dim*4),*g=malloc(hid*4),*uu=malloc(hid*4),*logN=malloc(V*4),*logC=malloc(V*4);
    
    int tok=1; /* BOS */
    double total_loss = 0;
    
    printf("Evaluating Perplexity on %d steps:\n", NSTEP);
    for(int pos=0;pos<NSTEP;pos++){
        memcpy(x,w.tok_emb+(size_t)tok*dim,dim*4);
        
        for(int run=0;run<2;run++){ 
            int useNPU=(run==0); float*xx=malloc(dim*4); memcpy(xx,x,dim*4);
            for(int l=0;l<L;l++){
                rmsnorm(xn,xx,w.rms_att+(size_t)l*dim,dim);
                mv(ctx,w.Wq[l],w.Sq[l],w.wq+(size_t)l*dim*dim,dim,dim, K_dim,N_dim,G,xn,q,useNPU);
                mv(ctx,w.Wk[l],w.Sk[l],w.wk+(size_t)l*dim*dim,dim,dim, K_dim,N_dim,G,xn,kk,useNPU);
                mv(ctx,w.Wv[l],w.Sv[l],w.wv+(size_t)l*dim*dim,dim,dim, K_dim,N_dim,G,xn,vv,useNPU);
                for(int i=0;i<dim;i+=2){
                    int hi=i%hd;float fr=1.0f/powf(10000.0f,(float)hi/hd),val=pos*fr,fcr=cosf(val),fci=sinf(val);
                    float a0=q[i],a1=q[i+1];q[i]=a0*fcr-a1*fci;q[i+1]=a0*fci+a1*fcr; 
                    float b0=kk[i],b1=kk[i+1];kk[i]=b0*fcr-b1*fci;kk[i+1]=b0*fci+b1*fcr;
                }
                memcpy(Kc+((size_t)l*c.seq+pos)*dim,kk,dim*4); memcpy(Vc+((size_t)l*c.seq+pos)*dim,vv,dim*4);
                float scale=1.0f/sqrtf((float)hd);
                for(int h=0;h<NH;h++){
                    float sc[512];
                    for(int j=0;j<=pos;j++){float dt=0;for(int e=0;e<hd;e++)dt+=q[h*hd+e]*Kc[((size_t)l*c.seq+j)*dim+h*hd+e];sc[j]=dt*scale;}
                    softmax(sc,pos+1);
                    for(int e=0;e<hd;e++){float ac=0;for(int j=0;j<=pos;j++)ac+=sc[j]*Vc[((size_t)l*c.seq+j)*dim+h*hd+e];att[h*hd+e]=ac;}
                }
                mv(ctx,w.Wo[l],w.So[l],w.wo+(size_t)l*dim*dim,dim,dim, K_dim,N_dim,G,att,o,useNPU);
                for(int i=0;i<dim;i++)xx[i]+=o[i];
                rmsnorm(xn,xx,w.rms_ffn+(size_t)l*dim,dim);
                mv(ctx,w.Wg[l],w.Sg[l],w.w1+(size_t)l*hid*dim,hid,dim, K_hid_in,N_hid_out,G,xn,g,useNPU);
                mv(ctx,w.Wu[l],w.Su[l],w.w3+(size_t)l*hid*dim,hid,dim, K_hid_in,N_hid_out,G,xn,uu,useNPU);
                ork_silu_mul_f32(g,uu,hid);   /* SwiGLU (NEON) */
                mv(ctx,w.Wd[l],w.Sd[l],w.w2+(size_t)l*dim*hid,dim,hid, K_hid_out,N_dim_out,G,g,o,useNPU);
                for(int i=0;i<dim;i++)xx[i]+=o[i];
            }
            rmsnorm(xn,xx,w.rms_final,dim);
            mv(ctx,w.Wcls,w.Scls,w.tok_emb,V,dim, K_vcb,N_vcb,G,xn,useNPU?logN:logC,useNPU);
            free(xx);
        }
        
        // Ground truth is the argmax of CPU fp32 reference logC
        int am=0; float best=logC[0]; for(int i=1;i<V;i++)if(logC[i]>best){best=logC[i];am=i;}
        
        // Calculate cross-entropy loss of I4 model on ground truth token
        softmax(logN, V); // modifies logN in-place to probabilities
        float prob = logN[am];
        if (prob < 1e-9f) prob = 1e-9f;
        float loss = -logf(prob);
        total_loss += loss;
        
        printf("  pos=%2d tok=%-5d -> ref_next=%-5d  I4_prob=%.4f  loss=%.4f\n",pos,tok,am,prob,loss);
        tok=am; // autoregress on ground truth
    }
    
    double perplexity = exp(total_loss / NSTEP);
    printf("\nTotal Perplexity over %d tokens: %.4f\n", NSTEP, perplexity);
    
    ork_npu_free(ctx);
    return 0;
}
