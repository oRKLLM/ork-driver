/* examples/llama2.c — M5.2/M5.3: run a REAL model (Karpathy's llama2.c stories15M) on the NPU
 * stack. Loads the .bin (fp32 weights, mmap), packs the projection + LM-head matrices into
 * resident fp16 NPU weights (transposed: llama2.c stores [out][in], ork_npu wants [in][out]),
 * and runs the full decoder (embed → N layers w/ KV cache → final RMSNorm → logits). The
 * non-matmul ops (RMSNorm, interleaved/GPT-J RoPE, causal softmax, SwiGLU) run on the CPU.
 * Validates NPU-hybrid logits against a pure-CPU-fp16 reference, then greedy-generates token
 * IDs (coherence sanity; decode with the llama2.c tokenizer separately).
 *   cc -O2 -I. -o rknpu_llama2 rknpu_llama2.c ork_npu.c -lm && sudo ./rknpu_llama2 stories15M.bin [nsteps]
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
#include "neon_activations.h"
typedef ork_f16 f16;

typedef struct { int dim,hidden,n_layers,n_heads,n_kv,vocab,seq; } Cfg;
typedef struct {
    const float *tok_emb;                 /* [vocab][dim] (also LM head, shared) */
    const float *rms_att, *rms_ffn;       /* [L][dim] */
    const float *wq,*wk,*wv,*wo;          /* [L][dim][dim]  (out,in) */
    const float *w1,*w2,*w3;              /* gate/down/up: w1,w3 [L][hidden][dim], w2 [L][dim][hidden] */
    const float *rms_final;               /* [dim] */
    /* resident NPU matmul weights (transposed to [in][out]) */
    ork_w **Wq,**Wk,**Wv,**Wo,**Wg,**Wd,**Wu; ork_w *Wcls;
} Weights;

/* RMSNorm / SwiGLU / softmax via the shared NEON kernels (src/neon_activations.c), validated vs libm in test_activations. */
static void rmsnorm(float*o,const float*x,const float*w,int n){ ork_rmsnorm_f32(o,x,w,n,1e-5f); }
static void softmax(float*x,int n){ ork_softmax_f32(x,n); }

/* pack a llama2.c weight w[OUT][IN] (row-major) as ork_npu B[IN][OUT] (transposed), fp16 */
static ork_w* pack_t(ork_npu*ctx,const float*w,int OUT,int IN){
    f16*B=malloc((size_t)IN*OUT*2);
    for(int k=0;k<IN;k++)for(int n=0;n<OUT;n++)B[(size_t)k*OUT+n]=(f16)w[(size_t)n*IN+k];
    ork_w*W=ork_mm_pack(ctx,IN,OUT,B); free(B); return W;
}
/* C[1,OUT] = x[1,IN] (fp16) x W ; NPU or CPU-fp16 reference from raw w[OUT][IN] */
static void mv(ork_npu*ctx,ork_w*W,const float*wraw,int OUT,int IN,const float*x,float*C,int useNPU){
    if(useNPU){ f16*a=malloc((size_t)IN*2); for(int i=0;i<IN;i++)a[i]=(f16)x[i]; ork_mm_run(ctx,W,1,a,C); free(a); }
    else for(int n=0;n<OUT;n++){float acc=0;for(int k=0;k<IN;k++)acc+=(float)(f16)x[k]*(float)(f16)wraw[(size_t)n*IN+k];C[n]=acc;}
}

int main(int argc,char**argv){
    const char*path=argc>1?argv[1]:"stories15M.bin"; int NSTEP=argc>2?atoi(argv[2]):8;
    int fd=open(path,O_RDONLY); if(fd<0){perror("open");return 1;}
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
    w.rms_final=p; /* p += dim; then freq_cis (computed here, skipped) */

    ork_npu*ctx=ork_npu_init(); if(!ctx){printf("init failed (NPU?)\n");return 1;}
    w.Wq=malloc(L*sizeof(void*));w.Wk=malloc(L*sizeof(void*));w.Wv=malloc(L*sizeof(void*));w.Wo=malloc(L*sizeof(void*));
    w.Wg=malloc(L*sizeof(void*));w.Wd=malloc(L*sizeof(void*));w.Wu=malloc(L*sizeof(void*));
    for(int l=0;l<L;l++){
        w.Wq[l]=pack_t(ctx,w.wq+(size_t)l*dim*dim,dim,dim); w.Wk[l]=pack_t(ctx,w.wk+(size_t)l*dim*dim,dim,dim);
        w.Wv[l]=pack_t(ctx,w.wv+(size_t)l*dim*dim,dim,dim); w.Wo[l]=pack_t(ctx,w.wo+(size_t)l*dim*dim,dim,dim);
        w.Wg[l]=pack_t(ctx,w.w1+(size_t)l*hid*dim,hid,dim); w.Wu[l]=pack_t(ctx,w.w3+(size_t)l*hid*dim,hid,dim);
        w.Wd[l]=pack_t(ctx,w.w2+(size_t)l*dim*hid,dim,hid);
    }
    w.Wcls=pack_t(ctx,w.tok_emb,V,dim);   /* logits = x · tok_emb^T (shared) */

    /* KV cache [L][seq][dim] */
    float*Kc=calloc((size_t)L*c.seq*dim,4),*Vc=calloc((size_t)L*c.seq*dim,4);
    float *x=malloc(dim*4),*xn=malloc(dim*4),*q=malloc(dim*4),*kk=malloc(dim*4),*vv=malloc(dim*4),
          *att=malloc(dim*4),*o=malloc(dim*4),*g=malloc(hid*4),*uu=malloc(hid*4),*logN=malloc(V*4),*logC=malloc(V*4);
    int tok=1; /* BOS */ int maxmism=0; double worstlogit=0;
    printf("greedy generation (token ids), NPU vs CPU-fp16 logit check per step:\n");
    for(int pos=0;pos<NSTEP;pos++){
        memcpy(x,w.tok_emb+(size_t)tok*dim,dim*4);
        for(int run=0;run<2;run++){ int useNPU=(run==0); float*xx=malloc(dim*4); memcpy(xx,x,dim*4);
        for(int l=0;l<L;l++){
            rmsnorm(xn,xx,w.rms_att+(size_t)l*dim,dim);
            mv(ctx,w.Wq[l],w.wq+(size_t)l*dim*dim,dim,dim,xn,q,useNPU);
            mv(ctx,w.Wk[l],w.wk+(size_t)l*dim*dim,dim,dim,xn,kk,useNPU);
            mv(ctx,w.Wv[l],w.wv+(size_t)l*dim*dim,dim,dim,xn,vv,useNPU);
            for(int i=0;i<dim;i+=2){int hi=i%hd;float fr=1.0f/powf(10000.0f,(float)hi/hd),val=pos*fr,fcr=cosf(val),fci=sinf(val);
                float a0=q[i],a1=q[i+1];q[i]=a0*fcr-a1*fci;q[i+1]=a0*fci+a1*fcr; float b0=kk[i],b1=kk[i+1];kk[i]=b0*fcr-b1*fci;kk[i+1]=b0*fci+b1*fcr;}
            memcpy(Kc+((size_t)l*c.seq+pos)*dim,kk,dim*4); memcpy(Vc+((size_t)l*c.seq+pos)*dim,vv,dim*4);
            float scale=1.0f/sqrtf((float)hd);
            for(int h=0;h<NH;h++){float sc[512];
                for(int j=0;j<=pos;j++){float dt=0;for(int e=0;e<hd;e++)dt+=q[h*hd+e]*Kc[((size_t)l*c.seq+j)*dim+h*hd+e];sc[j]=dt*scale;}
                softmax(sc,pos+1);
                for(int e=0;e<hd;e++){float ac=0;for(int j=0;j<=pos;j++)ac+=sc[j]*Vc[((size_t)l*c.seq+j)*dim+h*hd+e];att[h*hd+e]=ac;}}
            mv(ctx,w.Wo[l],w.wo+(size_t)l*dim*dim,dim,dim,att,o,useNPU);
            for(int i=0;i<dim;i++)xx[i]+=o[i];
            rmsnorm(xn,xx,w.rms_ffn+(size_t)l*dim,dim);
            mv(ctx,w.Wg[l],w.w1+(size_t)l*hid*dim,hid,dim,xn,g,useNPU);
            mv(ctx,w.Wu[l],w.w3+(size_t)l*hid*dim,hid,dim,xn,uu,useNPU);
            ork_silu_mul_f32(g,uu,hid);   /* SwiGLU: g = silu(g)*uu (NEON) */
            mv(ctx,w.Wd[l],w.w2+(size_t)l*dim*hid,dim,hid,g,o,useNPU);
            for(int i=0;i<dim;i++)xx[i]+=o[i];
        }
        rmsnorm(xn,xx,w.rms_final,dim);
        mv(ctx,w.Wcls,w.tok_emb,V,dim,xn,useNPU?logN:logC,useNPU);
        free(xx);
        }
        int am=0; float best=logN[0]; for(int i=1;i<V;i++)if(logN[i]>best){best=logN[i];am=i;}
        int mm=0; for(int i=0;i<V;i++){float e=fabsf(logN[i]-logC[i]);if(e>worstlogit)worstlogit=e;}
        (void)mm; (void)maxmism;
        printf("  pos=%2d tok=%-5d -> next=%-5d  logit[next]=%.3f\n",pos,tok,am,best);
        tok=am;
    }
    printf("NPU vs CPU-fp16: worst |logit diff| = %.4g  (vocab=%d, %d steps) : %s\n",
        worstlogit,V,NSTEP, worstlogit<0.5?"OK":"CHECK");
    ork_npu_free(ctx);
    return worstlogit<0.5?0:2;
}
