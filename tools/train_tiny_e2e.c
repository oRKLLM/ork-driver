/* train_tiny_e2e — the first END-TO-END training step (and 200 of them) of a Qwen3.5-shaped hybrid,
 * checked against MLX, with the matmuls optionally on the RK3588 NPU.
 *
 * Scope report section 7.4: every gradient path was validated in isolation; no end-to-end step had
 * run. This assembles them -- linear dX/dW, attention backward, RMSNorm / SwiGLU / CE backward, the GDN
 * layer ops (conv1d+silu, q/k rms-norm, decay, gated RMSNorm) and the chunkwise GDN backward -- into
 * one model and checks the WHOLE thing against an independent implementation:
 *
 *   qwen35-lora-moe/tools/tiny_e2e_reference.py writes a fixture: mlx-lm's own qwen3_5.TextModel at
 *   d=256 (GDN GDN GDN ATTN, byte vocab, tied embeddings), its init, the batches, the step-1 loss, the
 *   gradient of every parameter (on MLX GPU and MLX CPU), and a 200-step AdamW loss curve.
 *
 * The C model follows mlx-lm's conventions exactly (they define the model; the C side is the port):
 *   GDN q/k      (1/Dk) * rms_norm(q), Dk^-1/2 * rms_norm(k), eps 1e-6   -- rms_norm, NOT l2_norm
 *   decay        log g = -exp(A_log) softplus(a + dt_bias), beta = sigmoid(b)
 *   conv1d       depthwise causal, kernel 4, zero initial state, NO bias, then silu
 *   output       silu(z) * rms_norm(y, norm.weight)  per head
 *   attention    q_proj -> per head [query(64) | gate(64)], q/k RMSNorm with weights, rotary on the
 *                first 16 of 64 dims (non-traditional, theta 1e7, angles as MLX's CPU rope computes
 *                them), causal softmax, output * sigmoid(gate), GQA 4 query heads on 2 KV heads
 *   FFN          down(silu(gate(x)) * up(x)),  pre-norm RMSNorm blocks, final RMSNorm, tied lm_head
 *
 * Matmuls go through tiny_e2e_mm.h (CPU fp32 / CPU fp16-operand emulation / NPU), per class; the GDN
 * chunk matmuls through tiny_e2e_gdn.h, which tiny_e2e_gdn_check.c checks against gdn_chunk_bwd.c.
 * Everything else is CPU fp32 with fp32 master weights; AdamW is MLX's (bias correction, decoupled
 * weight decay applied as p *= 1 - lr*wd before the Adam step).
 *
 * The non-matmul backward formulas are those validated in bwd_ops_fp64_check.c (RMSNorm incl. the
 * r-coupling, SwiGLU, CE), attn_bwd_fp64_ref.c (softmax) and gdn_layer_bwd_check.c (conv, q/k norm,
 * decay, gated norm). Those files hold them as fixed-size self-tests over globals, so they cannot be
 * linked; the forms here are transcribed from them, and the fixture comparison is what checks the
 * transcription. Positive controls (--break) remove one term each and MUST fail the gradient check.
 *
 * Build:  CPU only   cc -O2 -Itools -o tte tools/train_tiny_e2e.c -lm   [-fopenmp]
 *         NPU        cc -O2 -DWITH_NPU -Iinclude -Isrc -Itools -o tte tools/train_tiny_e2e.c libork_npu.a -lm -pthread
 * Run:    tte FIXTURE [--lin cpu|emu|npu] [--att ...] [--gdn ...] [--scale 0|1] [--steps N]
 *             [--break none|rms|gdn_dk|silu] [--curve FILE] [--nocheck]
 */
#include "tiny_e2e_gdn.h"
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

/* ------------------------------------------------------------------ fixture */
#define MAXARR 512
typedef struct { char name[96]; int nd, dim[4]; size_t n, off; } arrinfo;
static arrinfo AR[MAXARR]; static int NAR; static float *BLOB;
static char HDR[64][2][64]; static int NHDR;

static const char *hget_opt(const char*k,const char*dflt){ for(int i=0;i<NHDR;i++) if(!strcmp(HDR[i][0],k)) return HDR[i][1]; return dflt; }
static const char *hget(const char*k){ for(int i=0;i<NHDR;i++) if(!strcmp(HDR[i][0],k)) return HDR[i][1];
    fprintf(stderr,"fixture header has no %s\n",k); exit(2); }
static const float *fx(const char*nm,size_t want,int must){
    for(int i=0;i<NAR;i++) if(!strcmp(AR[i].name,nm)){
        if(want && AR[i].n!=want){ fprintf(stderr,"%s: %zu floats, expected %zu\n",nm,AR[i].n,want); exit(2); }
        return BLOB+AR[i].off; }
    if(must){ fprintf(stderr,"fixture is missing %s\n",nm); exit(2); }
    return NULL;
}
static size_t fxn(const char*nm){ for(int i=0;i<NAR;i++) if(!strcmp(AR[i].name,nm)) return AR[i].n; return 0; }
static void load_fixture(const char*path){
    FILE*f=fopen(path,"rb"); if(!f){ perror(path); exit(2); }
    char line[512]; size_t tot=0;
    if(!fgets(line,sizeof line,f)||strncmp(line,"ORKTINYE2E 1",12)){ fprintf(stderr,"not ORKTINYE2E 1\n"); exit(2); }
    while(fgets(line,sizeof line,f)){
        if(!strncmp(line,"DATA",4)) break;
        if(!strncmp(line,"ARRAY ",6)){
            arrinfo*a=&AR[NAR++]; char*p=line+6; int used;
            sscanf(p,"%95s%n",a->name,&used); p+=used; a->nd=0; a->n=1;
            int d; while(sscanf(p,"%d%n",&d,&used)==1){ a->dim[a->nd++]=d; a->n*=d; p+=used; }
            a->off=tot; tot+=a->n;
        } else if(NHDR<64){ sscanf(line,"%63s %63s",HDR[NHDR][0],HDR[NHDR][1]); NHDR++; }
    }
    /* mmap, read-only: at 0.8B scale the fixture is 5.3 GB, 2.2 GB of it reference gradients read
     * once. Mapped, those stay reclaimable page cache instead of anonymous memory. The data offset is
     * not float-aligned (the ASCII header has arbitrary length), so parameters are COPIED into aligned
     * masters (pnew) and only read-only comparisons read through BLOB (aarch64/x86 tolerate unaligned
     * float loads on normal memory). */
    long off=ftell(f); fclose(f);
    int fd=open(path,O_RDONLY); struct stat st; if(fd<0||fstat(fd,&st)){ perror(path); exit(2); }
    if((size_t)st.st_size < (size_t)off+tot*4){ fprintf(stderr,"short fixture\n"); exit(2); }
    char *base=mmap(NULL,(size_t)st.st_size,PROT_READ,MAP_PRIVATE,fd,0);
    if(base==MAP_FAILED){ perror("mmap"); exit(2); }
    close(fd);
    BLOB=(float*)(void*)(base+off);
}

/* ------------------------------------------------------------------ model dims */
static int D,F,V,NL,HA,HKV,HD,GH,GDK,GDV,KC,FAI,ROPED,B,T,M,STEPS,CH=64;
static float EPS,THETA; static double LR,WD,B1,B2,AEPS;

/* ------------------------------------------------------------------ parameters */
typedef struct { char name[96]; size_t n; float *w,*g,*m,*v; } param;
static param PR[512]; static int NPR;
static param *pnew(const char*name){
    char k[128]; snprintf(k,sizeof k,"init.%s",name);
    param*p=&PR[NPR++]; snprintf(p->name,sizeof p->name,"%s",name);
    p->n=fxn(k); if(!p->n){ fprintf(stderr,"no parameter %s in fixture\n",name); exit(2); }
    p->w=malloc(p->n*4); if(!p->w){ fprintf(stderr,"oom\n"); exit(3); }
    memcpy(p->w,fx(k,0,1),p->n*4);   /* aligned fp32 master; the mapped fixture is read-only */
    p->g=calloc(p->n,4); p->m=calloc(p->n,4); p->v=calloc(p->n,4);
    return p;
}
typedef struct {
    int lin;
    param *ln1,*ln2,*wg,*wu,*wd;
    param *wqkv,*wz,*wb,*wa,*conv,*dtb,*alog,*gnorm,*wout;      /* GDN */
    param *wq,*wk,*wv,*wo,*qn,*kn;                              /* attention */
} layer;
static layer LY[64]; static param *EMB,*FNORM;

/* ------------------------------------------------------------------ activations */
typedef struct {
    float *x,*r1,*n1,*h,*r2,*n2,*fg,*fu,*fa;
    float *qkv,*pre,*co,*qr,*kr,*vv,*rq,*rk,*qh,*kh,*z,*bb,*aa,*beta,*logg,*y,*ry,*yn,*o,*ent;   /* GDN */
    float *qp,*kp,*vp,*qa,*ga,*rqa,*rka,*Q,*K,*Vh,*P,*O,*sg,*og;                                 /* attention */
} cache;
static cache CA[64];
static float *X0,*XL,*RF,*XF,*LOGITS,*DLOG;
static int *TOK,*TGT;

static int BRK_RMS=0, BRK_SILU=0;
/* r102 wall-clock split (seconds, reset per step). GDN = gm_fwd/gm_bwd wall incl. their matmuls;
 * SOFTMAX = attention softmax fwd + bwd; ELEM = norms, SwiGLU, conv, decay, gates, rope, gathers;
 * CE = logits softmax/loss/dlogits; OPT = AdamW. Matmul seconds come from tiny_e2e_mm.h. */
static double T_GDN, T_SOFTMAX, T_CE, T_OPT;     /* positive controls; gm_BRK covers the GDN Gram term */

static float *A(size_t n){ float*p=calloc(n?n:1,4); if(!p){fprintf(stderr,"oom\n");exit(3);} return p; }
static float sigm(float x){ return 1.0f/(1.0f+expf(-x)); }
static float siluf(float x){ return x*sigm(x); }
static float dsiluf(float x){ float s=sigm(x); return s*(1.0f+x*(1.0f-s)); }
static float softplusf(float x){ return x>0 ? x+log1pf(expf(-x)) : log1pf(expf(x)); }

/* ------------------------------------------------------------------ elementwise ops */
/* y = scale * w * x / r per row of length n; rinv[row] = 1/r. w may be NULL. */
static void rms_fwd(int rows,int n,const float*x,const float*w,float scale,float*y,float*rinv){
    for(int i=0;i<rows;i++){
        const float*a=x+(size_t)i*n; float*o=y+(size_t)i*n; float ss=0;
        for(int j=0;j<n;j++) ss+=a[j]*a[j];
        float ri=1.0f/sqrtf(ss/n+EPS); rinv[i]=ri;
        for(int j=0;j<n;j++) o[j]=scale*a[j]*ri*(w?w[j]:1.0f);
    }
}
/* bwd_ops_fp64_check.c's RMSNorm: dx = s*(w.dy*ri - x*ri^3*sum(dy.w.x)/n); dw += s*dy.x.ri */
static void rms_bwd(int rows,int n,const float*x,const float*w,const float*rinv,float scale,
                    const float*dy,float*dx,float*dw,int brk){
    for(int i=0;i<rows;i++){
        const float*a=x+(size_t)i*n,*d=dy+(size_t)i*n; float*o=dx+(size_t)i*n; float ri=rinv[i], dot=0;
        for(int j=0;j<n;j++) dot+=d[j]*(w?w[j]:1.0f)*a[j];
        float c = brk ? 0.f : ri*ri*ri*dot/n;
        for(int j=0;j<n;j++) o[j]=scale*((w?w[j]:1.0f)*d[j]*ri - a[j]*c);
        if(dw) for(int j=0;j<n;j++) dw[j]+=scale*d[j]*a[j]*ri;
    }
}
/* y = x W^T, W [out,in] (MLX nn.Linear layout) */
static void lin(int rows,int in,int out,const float*x,const param*W,float*y){ g_mm_class=MMC_LIN; mm_nt(rows,in,out,x,W->w,y); }
static float *g_lt; static size_t g_lt_n;
static void lin_bwd(int rows,int in,int out,const float*x,param*W,const float*dy,float*dx){
    g_mm_class=MMC_LIN;
    if(dx) mm_nn(rows,out,in,dy,W->w,dx);                         /* dX = dY W      */
    if((size_t)out*in>g_lt_n){ free(g_lt); g_lt=A((size_t)out*in); g_lt_n=(size_t)out*in; }
    g_mm_class=MMC_LIN;
    mm_tn(out,rows,in,dy,x,g_lt);                                 /* dW = dY^T X    */
    for(size_t i=0;i<(size_t)out*in;i++) W->g[i]+=g_lt[i];
}
static void addto(float*a,const float*b,size_t n){ for(size_t i=0;i<n;i++) a[i]+=b[i]; }

/* ------------------------------------------------------------------ FFN */
static void ffn_fwd(layer*L,cache*c,const float*in,float*out){
    lin(M,D,F,in,L->wg,c->fg); lin(M,D,F,in,L->wu,c->fu);
    for(size_t i=0;i<(size_t)M*F;i++) c->fa[i]=siluf(c->fg[i])*c->fu[i];
    lin(M,F,D,c->fa,L->wd,out);
}
static void ffn_bwd(layer*L,cache*c,const float*in,const float*dout,float*din){
    float *da=A((size_t)M*F), *dg=A((size_t)M*F), *du=A((size_t)M*F), *t=A((size_t)M*D);
    lin_bwd(M,F,D,c->fa,L->wd,dout,da);
    for(size_t i=0;i<(size_t)M*F;i++){
        dg[i]=da[i]*c->fu[i]*(BRK_SILU?1.0f:dsiluf(c->fg[i]));
        du[i]=da[i]*siluf(c->fg[i]);
    }
    lin_bwd(M,D,F,in,L->wg,dg,din);
    lin_bwd(M,D,F,in,L->wu,du,t); addto(din,t,(size_t)M*D);
    free(da); free(dg); free(du); free(t);
}

/* ------------------------------------------------------------------ GDN mixer */
static gm_scr *GW;
#ifdef GDN_ORACLE   /* debug: the fp64 recurrence from gdn_chunk_bwd.c (link tools/tiny_e2e_gdn_oracle.c) */
void gdn_oracle_fwd(int,int,int,int,const float*,const float*,const float*,const float*,const float*,float*);
void gdn_oracle_bwd(int,int,int,int,const float*,const float*,const float*,const float*,const float*,const float*,
                    float*,float*,float*,float*,float*);
#endif
static void gather_head(float*dst,const float*src,int b,int h,int nh,int dh){   /* [T,dh] from [M,nh*dh] */
    for(int t=0;t<T;t++) memcpy(dst+(size_t)t*dh, src+((size_t)(b*T+t)*nh+h)*dh, (size_t)dh*4);
}
static void scatter_head(float*dst,const float*src,int b,int h,int nh,int dh){
    for(int t=0;t<T;t++) memcpy(dst+((size_t)(b*T+t)*nh+h)*dh, src+(size_t)t*dh, (size_t)dh*4);
}
static void gather_sc(float*dst,const float*src,int b,int h,int nh){ for(int t=0;t<T;t++) dst[t]=src[(size_t)(b*T+t)*nh+h]; }
static void scatter_sc(float*dst,const float*src,int b,int h,int nh){ for(int t=0;t<T;t++) dst[(size_t)(b*T+t)*nh+h]=src[t]; }

static void gdn_fwd(layer*L,cache*c,const float*in,float*out){
    const int KD=GH*GDK, VD=GH*GDV, CD=2*KD+VD;
    lin(M,D,CD,in,L->wqkv,c->qkv); lin(M,D,VD,in,L->wz,c->z);
    lin(M,D,GH,in,L->wb,c->bb);    lin(M,D,GH,in,L->wa,c->aa);
    for(int b=0;b<B;b++) for(int t=0;t<T;t++) for(int ch=0;ch<CD;ch++){       /* conv1d + silu */
        float p=0;
        for(int r=0;r<KC;r++){ int u=t+r-(KC-1); if(u<0) continue;
            p+=L->conv->w[ch*KC+r]*c->qkv[(size_t)(b*T+u)*CD+ch]; }
        size_t ix=(size_t)(b*T+t)*CD+ch; c->pre[ix]=p; c->co[ix]=siluf(p);
    }
    for(int i=0;i<M;i++){
        memcpy(c->qr+(size_t)i*KD, c->co+(size_t)i*CD,       (size_t)KD*4);
        memcpy(c->kr+(size_t)i*KD, c->co+(size_t)i*CD+KD,    (size_t)KD*4);
        memcpy(c->vv+(size_t)i*VD, c->co+(size_t)i*CD+2*KD,  (size_t)VD*4);
    }
    float is=1.0f/sqrtf((float)GDK);
    rms_fwd(M*GH,GDK,c->qr,NULL,is*is,c->qh,c->rq);
    rms_fwd(M*GH,GDK,c->kr,NULL,is,c->kh,c->rk);
    for(int i=0;i<M;i++) for(int h=0;h<GH;h++){
        size_t ix=(size_t)i*GH+h;
        c->beta[ix]=sigm(c->bb[ix]);
        c->logg[ix]=-expf(L->alog->w[h])*softplusf(c->aa[ix]+L->dtb->w[h]);
    }
    float *q=A((size_t)T*GDK),*k=A((size_t)T*GDK),*v=A((size_t)T*GDV),*lg=A(T),*be=A(T),*y=A((size_t)T*GDV);
    int nc=(T+CH-1)/CH; size_t esz=(size_t)nc*GDV*GDK;
    for(int b=0;b<B;b++) for(int h=0;h<GH;h++){
        gather_head(q,c->qh,b,h,GH,GDK); gather_head(k,c->kh,b,h,GH,GDK); gather_head(v,c->vv,b,h,GH,GDV);
        gather_sc(lg,c->logg,b,h,GH); gather_sc(be,c->beta,b,h,GH);
#ifdef GDN_ORACLE
        gdn_oracle_fwd(T,GDK,GDV,CH,q,k,v,lg,be,y); (void)esz;
#else
        { double t0=now_s(); gm_fwd(GW,T,CH,q,k,v,lg,be,y,c->ent+(size_t)(b*GH+h)*esz); T_GDN+=now_s()-t0; }
#endif
        scatter_head(c->y,y,b,h,GH,GDV);
    }
    free(q);free(k);free(v);free(lg);free(be);free(y);
    rms_fwd(M*GH,GDV,c->y,L->gnorm->w,1.0f,c->yn,c->ry);                     /* gated RMSNorm */
    for(size_t i=0;i<(size_t)M*VD;i++) c->o[i]=siluf(c->z[i])*c->yn[i];
    lin(M,VD,D,c->o,L->wout,out);
}
static void gdn_bwd(layer*L,cache*c,const float*in,const float*dout,float*din){
    const int KD=GH*GDK, VD=GH*GDV, CD=2*KD+VD;
    float *dO=A((size_t)M*VD),*dz=A((size_t)M*VD),*dyn=A((size_t)M*VD),*dy=A((size_t)M*VD);
    lin_bwd(M,VD,D,c->o,L->wout,dout,dO);
    for(size_t i=0;i<(size_t)M*VD;i++){ dz[i]=dO[i]*c->yn[i]*dsiluf(c->z[i]); dyn[i]=dO[i]*siluf(c->z[i]); }
    rms_bwd(M*GH,GDV,c->y,L->gnorm->w,c->ry,1.0f,dyn,dy,L->gnorm->g,0);

    float *dqh=A((size_t)M*KD),*dkh=A((size_t)M*KD),*dvv=A((size_t)M*VD),*dlg=A((size_t)M*GH),*dbe=A((size_t)M*GH);
    float *q=A((size_t)T*GDK),*k=A((size_t)T*GDK),*v=A((size_t)T*GDV),*lg=A(T),*be=A(T),*gy=A((size_t)T*GDV);
    float *gq=A((size_t)T*GDK),*gk=A((size_t)T*GDK),*gv=A((size_t)T*GDV),*glg=A(T),*gbe=A(T);
    int nc=(T+CH-1)/CH; size_t esz=(size_t)nc*GDV*GDK;
    for(int b=0;b<B;b++) for(int h=0;h<GH;h++){
        gather_head(q,c->qh,b,h,GH,GDK); gather_head(k,c->kh,b,h,GH,GDK); gather_head(v,c->vv,b,h,GH,GDV);
        gather_sc(lg,c->logg,b,h,GH); gather_sc(be,c->beta,b,h,GH); gather_head(gy,dy,b,h,GH,GDV);
#ifdef GDN_ORACLE
        gdn_oracle_bwd(T,GDK,GDV,CH,q,k,v,lg,be,gy,gq,gk,gv,glg,gbe); (void)esz;
#else
        { double t0=now_s(); gm_bwd(GW,T,CH,q,k,v,lg,be,c->ent+(size_t)(b*GH+h)*esz,gy,gq,gk,gv,glg,gbe); T_GDN+=now_s()-t0; }
#endif
        scatter_head(dqh,gq,b,h,GH,GDK); scatter_head(dkh,gk,b,h,GH,GDK); scatter_head(dvv,gv,b,h,GH,GDV);
        scatter_sc(dlg,glg,b,h,GH); scatter_sc(dbe,gbe,b,h,GH);
    }
    free(q);free(k);free(v);free(lg);free(be);free(gy);free(gq);free(gk);free(gv);free(glg);free(gbe);

    float *dbb=A((size_t)M*GH),*daa=A((size_t)M*GH);                         /* decay */
    for(int i=0;i<M;i++) for(int h=0;h<GH;h++){
        size_t ix=(size_t)i*GH+h; float Ae=expf(L->alog->w[h]), u=c->aa[ix]+L->dtb->w[h];
        dbb[ix]=dbe[ix]*c->beta[ix]*(1.0f-c->beta[ix]);
        float dsp=dlg[ix]*(-Ae);                       /* log g = -A softplus(u) */
        daa[ix]=dsp*sigm(u);
        L->dtb->g[h]+=dsp*sigm(u);
        L->alog->g[h]+=dlg[ix]*(-Ae*softplusf(u));     /* d log g / d A_log = -A softplus(u) */
    }
    float is=1.0f/sqrtf((float)GDK);
    float *dqr=A((size_t)M*KD),*dkr=A((size_t)M*KD);
    rms_bwd(M*GH,GDK,c->qr,NULL,c->rq,is*is,dqh,dqr,NULL,0);
    rms_bwd(M*GH,GDK,c->kr,NULL,c->rk,is,dkh,dkr,NULL,0);
    float *dpre=A((size_t)M*CD),*dqkv=A((size_t)M*CD);
    for(int i=0;i<M;i++) for(int ch=0;ch<CD;ch++){
        size_t ix=(size_t)i*CD+ch;
        float dco = ch<KD ? dqr[(size_t)i*KD+ch] : ch<2*KD ? dkr[(size_t)i*KD+ch-KD] : dvv[(size_t)i*VD+ch-2*KD];
        dpre[ix]=dco*dsiluf(c->pre[ix]);
    }
    for(int b=0;b<B;b++) for(int t=0;t<T;t++) for(int ch=0;ch<CD;ch++){       /* conv bwd */
        float dp=dpre[(size_t)(b*T+t)*CD+ch];
        for(int r=0;r<KC;r++){ int u=t+r-(KC-1); if(u<0) continue;
            size_t iu=(size_t)(b*T+u)*CD+ch;
            L->conv->g[ch*KC+r]+=dp*c->qkv[iu];
            dqkv[iu]+=dp*L->conv->w[ch*KC+r]; }
    }
    float *t=A((size_t)M*D);
    lin_bwd(M,D,CD,in,L->wqkv,dqkv,din);
    lin_bwd(M,D,VD,in,L->wz,dz,t);  addto(din,t,(size_t)M*D);
    lin_bwd(M,D,GH,in,L->wb,dbb,t); addto(din,t,(size_t)M*D);
    lin_bwd(M,D,GH,in,L->wa,daa,t); addto(din,t,(size_t)M*D);
    free(dO);free(dz);free(dyn);free(dy);free(dqh);free(dkh);free(dvv);free(dlg);free(dbe);
    free(dbb);free(daa);free(dqr);free(dkr);free(dpre);free(dqkv);free(t);
}

/* ------------------------------------------------------------------ attention mixer */
static float *RCOS,*RSIN;     /* [T, ROPED/2] */
static void rope_tables(void){
    int h2=ROPED/2; RCOS=A((size_t)T*h2); RSIN=A((size_t)T*h2);
    float lb=log2f(THETA);
    for(int i=0;i<h2;i++){
        float inv=exp2f(-(float)i*lb/(float)h2);     /* MLX CPU rope: exp2(-d * log2(base)), d = i/(dims/2) */
        for(int t=0;t<T;t++){ float a=(float)t*inv; RCOS[t*h2+i]=cosf(a); RSIN[t*h2+i]=sinf(a); }
    }
}
static void rope(float*x,int t,int inv){                 /* in place on one head row of HD */
    int h2=ROPED/2;
    for(int i=0;i<h2;i++){
        float c=RCOS[t*h2+i], s=inv? -RSIN[t*h2+i] : RSIN[t*h2+i];
        float a=x[i], b=x[i+h2];
        x[i]=a*c-b*s; x[i+h2]=a*s+b*c;
    }
}
static void att_fwd(layer*L,cache*c,const float*in,float*out){
    const int QD=HA*HD, KVD=HKV*HD; float sc=1.0f/sqrtf((float)HD);
    lin(M,D,2*QD,in,L->wq,c->qp); lin(M,D,KVD,in,L->wk,c->kp); lin(M,D,KVD,in,L->wv,c->vp);
    for(int i=0;i<M;i++) for(int h=0;h<HA;h++){
        memcpy(c->qa+((size_t)i*HA+h)*HD, c->qp+(size_t)i*2*QD+(size_t)h*2*HD,    (size_t)HD*4);
        memcpy(c->ga+((size_t)i*HA+h)*HD, c->qp+(size_t)i*2*QD+(size_t)h*2*HD+HD, (size_t)HD*4);
    }
    float *qn=A((size_t)M*QD), *kn=A((size_t)M*KVD);
    rms_fwd(M*HA,HD,c->qa,L->qn->w,1.0f,qn,c->rqa);
    rms_fwd(M*HKV,HD,c->kp,L->kn->w,1.0f,kn,c->rka);
    for(int b=0;b<B;b++) for(int t=0;t<T;t++){
        size_t r=(size_t)b*T+t;
        for(int h=0;h<HA;h++){ float*d=c->Q+(((size_t)b*HA+h)*T+t)*HD; memcpy(d,qn+(r*HA+h)*HD,(size_t)HD*4); rope(d,t,0); }
        for(int j=0;j<HKV;j++){ float*d=c->K+(((size_t)b*HKV+j)*T+t)*HD; memcpy(d,kn+(r*HKV+j)*HD,(size_t)HD*4); rope(d,t,0);
                                memcpy(c->Vh+(((size_t)b*HKV+j)*T+t)*HD, c->vp+(r*HKV+j)*HD,(size_t)HD*4); }
    }
    free(qn); free(kn);
    float *o=A((size_t)T*HD);
    for(int b=0;b<B;b++) for(int h=0;h<HA;h++){
        int j=h/(HA/HKV);
        float *Q=c->Q+((size_t)b*HA+h)*T*HD, *K=c->K+((size_t)b*HKV+j)*T*HD, *Vv=c->Vh+((size_t)b*HKV+j)*T*HD;
        float *P=c->P+((size_t)b*HA+h)*T*T;
        g_mm_class=MMC_ATT; mm_nt(T,HD,T,Q,K,P);                                  /* S = Q K^T */
        double tsm=now_s();
        for(int i=0;i<T;i++){
            float*p=P+(size_t)i*T, mx=-INFINITY, s=0;
            for(int jj=0;jj<=i;jj++){ p[jj]*=sc; if(p[jj]>mx) mx=p[jj]; }
            for(int jj=0;jj<=i;jj++){ p[jj]=expf(p[jj]-mx); s+=p[jj]; }
            for(int jj=0;jj<=i;jj++) p[jj]/=s;
            for(int jj=i+1;jj<T;jj++) p[jj]=0.f;
        }
        T_SOFTMAX+=now_s()-tsm;
        g_mm_class=MMC_ATT; mm_nn(T,T,HD,P,Vv,o);                                 /* O = P V */
        for(int t=0;t<T;t++) memcpy(c->O+(((size_t)b*T+t)*HA+h)*HD, o+(size_t)t*HD,(size_t)HD*4);
    }
    free(o);
    for(size_t i=0;i<(size_t)M*QD;i++){ c->sg[i]=sigm(c->ga[i]); c->og[i]=c->O[i]*c->sg[i]; }
    lin(M,QD,D,c->og,L->wo,out);
}
static void att_bwd(layer*L,cache*c,const float*in,const float*dout,float*din){
    const int QD=HA*HD, KVD=HKV*HD; float sc=1.0f/sqrtf((float)HD);
    float *dog=A((size_t)M*QD),*dO=A((size_t)M*QD),*dga=A((size_t)M*QD);
    lin_bwd(M,QD,D,c->og,L->wo,dout,dog);
    for(size_t i=0;i<(size_t)M*QD;i++){ dO[i]=dog[i]*c->sg[i]; dga[i]=dog[i]*c->O[i]*c->sg[i]*(1.0f-c->sg[i]); }
    float *dQ=A((size_t)B*HA*T*HD),*dK=A((size_t)B*HKV*T*HD),*dV=A((size_t)B*HKV*T*HD);
    float *dO_h=A((size_t)T*HD),*dP=A((size_t)T*T),*tmp=A((size_t)T*HD);
    for(int b=0;b<B;b++) for(int h=0;h<HA;h++){
        int j=h/(HA/HKV);
        float *Q=c->Q+((size_t)b*HA+h)*T*HD, *K=c->K+((size_t)b*HKV+j)*T*HD, *Vv=c->Vh+((size_t)b*HKV+j)*T*HD;
        float *P=c->P+((size_t)b*HA+h)*T*T;
        for(int t=0;t<T;t++) memcpy(dO_h+(size_t)t*HD, dO+(((size_t)b*T+t)*HA+h)*HD,(size_t)HD*4);
        g_mm_class=MMC_ATT;
        mm_tn(T,T,HD,P,dO_h,tmp);  addto(dV+((size_t)b*HKV+j)*T*HD,tmp,(size_t)T*HD);   /* dV += P^T dO */
        mm_nt(T,HD,T,dO_h,Vv,dP);                                                     /* dP = dO V^T  */
        double tsm=now_s();
        for(int i=0;i<T;i++){                                    /* attn_bwd_fp64_ref: dS = P(dP-rowsum)scale */
            float*p=P+(size_t)i*T,*d=dP+(size_t)i*T, r=0;
            for(int jj=0;jj<=i;jj++) r+=d[jj]*p[jj];
            for(int jj=0;jj<T;jj++) d[jj] = jj<=i ? p[jj]*(d[jj]-r)*sc : 0.f;
        }
        T_SOFTMAX+=now_s()-tsm;
        g_mm_class=MMC_ATT;
        mm_nn(T,T,HD,dP,K,dQ+((size_t)b*HA+h)*T*HD);                                  /* dQ = dS K    */
        mm_tn(T,T,HD,dP,Q,tmp); addto(dK+((size_t)b*HKV+j)*T*HD,tmp,(size_t)T*HD);    /* dK += dS^T Q */
    }
    free(dO_h); free(dP); free(tmp);
    float *dqn=A((size_t)M*QD),*dkn=A((size_t)M*KVD),*dvp=A((size_t)M*KVD);
    for(int b=0;b<B;b++) for(int t=0;t<T;t++){
        size_t r=(size_t)b*T+t;
        for(int h=0;h<HA;h++){ float*s=dQ+(((size_t)b*HA+h)*T+t)*HD; rope(s,t,1); memcpy(dqn+(r*HA+h)*HD,s,(size_t)HD*4); }
        for(int j=0;j<HKV;j++){ float*s=dK+(((size_t)b*HKV+j)*T+t)*HD; rope(s,t,1); memcpy(dkn+(r*HKV+j)*HD,s,(size_t)HD*4);
                                memcpy(dvp+(r*HKV+j)*HD, dV+(((size_t)b*HKV+j)*T+t)*HD,(size_t)HD*4); }
    }
    float *dqa=A((size_t)M*QD),*dkp=A((size_t)M*KVD),*dqp=A((size_t)M*2*QD);
    rms_bwd(M*HA,HD,c->qa,L->qn->w,c->rqa,1.0f,dqn,dqa,L->qn->g,0);
    rms_bwd(M*HKV,HD,c->kp,L->kn->w,c->rka,1.0f,dkn,dkp,L->kn->g,0);
    for(int i=0;i<M;i++) for(int h=0;h<HA;h++){
        memcpy(dqp+(size_t)i*2*QD+(size_t)h*2*HD,    dqa+((size_t)i*HA+h)*HD,(size_t)HD*4);
        memcpy(dqp+(size_t)i*2*QD+(size_t)h*2*HD+HD, dga+((size_t)i*HA+h)*HD,(size_t)HD*4);
    }
    float *t=A((size_t)M*D);
    lin_bwd(M,D,2*QD,in,L->wq,dqp,din);
    lin_bwd(M,D,KVD,in,L->wk,dkp,t); addto(din,t,(size_t)M*D);
    lin_bwd(M,D,KVD,in,L->wv,dvp,t); addto(din,t,(size_t)M*D);
    free(dog);free(dO);free(dga);free(dQ);free(dK);free(dV);free(dqn);free(dkn);free(dvp);free(dqa);free(dkp);free(dqp);free(t);
}

/* ------------------------------------------------------------------ whole model */
static void alloc_model(void){
    char nm[96];
    EMB=pnew("model.embed_tokens.weight");
    for(int l=0;l<NL;l++){
        layer*L=&LY[l]; L->lin=((l+1)%FAI)!=0;
#define P_(field,suffix) do{ snprintf(nm,sizeof nm,"model.layers.%d.%s",l,suffix); L->field=pnew(nm);}while(0)
        if(L->lin){
            P_(conv,"linear_attn.conv1d.weight"); P_(wqkv,"linear_attn.in_proj_qkv.weight");
            P_(wz,"linear_attn.in_proj_z.weight"); P_(wb,"linear_attn.in_proj_b.weight");
            P_(wa,"linear_attn.in_proj_a.weight"); P_(dtb,"linear_attn.dt_bias"); P_(alog,"linear_attn.A_log");
            P_(gnorm,"linear_attn.norm.weight"); P_(wout,"linear_attn.out_proj.weight");
        } else {
            P_(wq,"self_attn.q_proj.weight"); P_(wk,"self_attn.k_proj.weight"); P_(wv,"self_attn.v_proj.weight");
            P_(wo,"self_attn.o_proj.weight"); P_(qn,"self_attn.q_norm.weight"); P_(kn,"self_attn.k_norm.weight");
        }
        P_(ln1,"input_layernorm.weight"); P_(ln2,"post_attention_layernorm.weight");
        P_(wg,"mlp.gate_proj.weight"); P_(wd,"mlp.down_proj.weight"); P_(wu,"mlp.up_proj.weight");
#undef P_
        cache*c=&CA[l]; size_t md=(size_t)M*D;
        c->x=A(md); c->r1=A(M); c->n1=A(md); c->h=A(md); c->r2=A(M); c->n2=A(md);
        c->fg=A((size_t)M*F); c->fu=A((size_t)M*F); c->fa=A((size_t)M*F);
        if(L->lin){
            int KD=GH*GDK, VD=GH*GDV, CD=2*KD+VD, nc=(T+CH-1)/CH;
            c->qkv=A((size_t)M*CD); c->pre=A((size_t)M*CD); c->co=A((size_t)M*CD);
            c->qr=A((size_t)M*KD); c->kr=A((size_t)M*KD); c->vv=A((size_t)M*VD);
            c->rq=A((size_t)M*GH); c->rk=A((size_t)M*GH); c->qh=A((size_t)M*KD); c->kh=A((size_t)M*KD);
            c->z=A((size_t)M*VD); c->bb=A((size_t)M*GH); c->aa=A((size_t)M*GH); c->beta=A((size_t)M*GH); c->logg=A((size_t)M*GH);
            c->y=A((size_t)M*VD); c->ry=A((size_t)M*GH); c->yn=A((size_t)M*VD); c->o=A((size_t)M*VD);
            c->ent=A((size_t)B*GH*nc*GDV*GDK);
        } else {
            int QD=HA*HD, KVD=HKV*HD;
            c->qp=A((size_t)M*2*QD); c->kp=A((size_t)M*KVD); c->vp=A((size_t)M*KVD);
            c->qa=A((size_t)M*QD); c->ga=A((size_t)M*QD); c->rqa=A((size_t)M*HA); c->rka=A((size_t)M*HKV);
            c->Q=A((size_t)B*HA*T*HD); c->K=A((size_t)B*HKV*T*HD); c->Vh=A((size_t)B*HKV*T*HD);
            c->P=A((size_t)B*HA*T*T); c->O=A((size_t)M*QD); c->sg=A((size_t)M*QD); c->og=A((size_t)M*QD);
        }
    }
    FNORM=pnew("model.norm.weight");
    X0=A((size_t)M*D); XL=A((size_t)M*D); RF=A(M); XF=A((size_t)M*D);
    LOGITS=A((size_t)M*V); DLOG=A((size_t)M*V); TOK=calloc(M,4); TGT=calloc(M,4);
    GW=gm_new(CH,GDK,GDV);
}

static double forward(int want_grad_buf){
    (void)want_grad_buf;
    size_t md=(size_t)M*D;
    for(int i=0;i<M;i++) memcpy(X0+(size_t)i*D, EMB->w+(size_t)TOK[i]*D,(size_t)D*4);
    const float *x=X0; float *mix=A(md), *ff=A(md);
    for(int l=0;l<NL;l++){
        layer*L=&LY[l]; cache*c=&CA[l];
        memcpy(c->x,x,md*4);
        rms_fwd(M,D,c->x,L->ln1->w,1.0f,c->n1,c->r1);
        if(L->lin) gdn_fwd(L,c,c->n1,mix); else att_fwd(L,c,c->n1,mix);
        for(size_t i=0;i<md;i++) c->h[i]=c->x[i]+mix[i];
        rms_fwd(M,D,c->h,L->ln2->w,1.0f,c->n2,c->r2);
        ffn_fwd(L,c,c->n2,ff);
        for(size_t i=0;i<md;i++) XL[i]=c->h[i]+ff[i];
        x=XL;                                           /* copied into the next layer's cache->x */
    }
    free(mix); free(ff);
    rms_fwd(M,D,XL,FNORM->w,1.0f,XF,RF);
    g_mm_class=MMC_LIN; mm_nt(M,D,V,XF,EMB->w,LOGITS);                          /* tied lm_head */
    double L=0, tce=now_s();
    #pragma omp parallel for reduction(+:L) schedule(static)
    for(int i=0;i<M;i++){
        float*z=LOGITS+(size_t)i*V, mx=-INFINITY; for(int v=0;v<V;v++) if(z[v]>mx) mx=z[v];
        double s=0; for(int v=0;v<V;v++) s+=exp((double)(z[v]-mx));
        double lse=mx+log(s); L+=lse-z[TGT[i]];
        float*d=DLOG+(size_t)i*V;
        for(int v=0;v<V;v++) d[v]=(float)(exp((double)z[v]-lse)/M);
        d[TGT[i]]-=1.0f/M;                                                       /* (p - y)/B */
    }
    T_CE+=now_s()-tce;
    return L/M;
}

static void backward(void){
    size_t md=(size_t)M*D;
    for(int p=0;p<NPR;p++) memset(PR[p].g,0,PR[p].n*4);
    float *dxf=A(md), *dx=A(md), *t=A(md), *dmix=A(md), *dn=A(md);
    g_mm_class=MMC_LIN;
    mm_nn(M,V,D,DLOG,EMB->w,dxf);                                               /* dXf = dlogits E */
    { float*dE=A((size_t)V*D); g_mm_class=MMC_LIN; mm_tn(V,M,D,DLOG,XF,dE); addto(EMB->g,dE,(size_t)V*D); free(dE); }
    rms_bwd(M,D,XL,FNORM->w,RF,1.0f,dxf,dx,FNORM->g,0);
    for(int l=NL-1;l>=0;l--){
        layer*L=&LY[l]; cache*c=&CA[l];
        /* out = h + ffn(norm2(h)); dh = dx + norm2'(ffn'(dx)) */
        ffn_bwd(L,c,c->n2,dx,dn);
        rms_bwd(M,D,c->h,L->ln2->w,c->r2,1.0f,dn,t,L->ln2->g,0);
        addto(dx,t,md);                                                          /* dx is now dh */
        if(L->lin) gdn_bwd(L,c,c->n1,dx,dmix); else att_bwd(L,c,c->n1,dx,dmix);
        rms_bwd(M,D,c->x,L->ln1->w,c->r1,1.0f,dmix,t,L->ln1->g, BRK_RMS && l==1);
        addto(dx,t,md);
    }
    for(int i=0;i<M;i++){ float*e=EMB->g+(size_t)TOK[i]*D, *d=dx+(size_t)i*D; for(int j=0;j<D;j++) e[j]+=d[j]; }
    free(dxf); free(dx); free(t); free(dmix); free(dn);
}

/* ------------------------------------------------------------------ AdamW (MLX semantics) */
static void adamw(int step){
    float lr=(float)LR, b1=(float)B1, b2=(float)B2, eps=(float)AEPS;
    float dec=1.0f-lr*(float)WD, omb1=(float)(1.0-B1), omb2=(float)(1.0-B2);
    float c1=(float)(LR/(1.0-pow(B1,step))), c2=(float)(1.0/sqrt(1.0-pow(B2,step)));
    double t0=now_s();
    for(int p=0;p<NPR;p++){
        param*q=&PR[p];
        #pragma omp parallel for schedule(static)
        for(size_t i=0;i<q->n;i++){
            float g=q->g[i], w=q->w[i]*dec;
            float m=b1*q->m[i]+omb1*g, v=b2*q->v[i]+omb2*g*g;
            q->m[i]=m; q->v[i]=v;
            q->w[i]=w-c1*m/(sqrtf(v)*c2+eps);
        }
    }
    T_OPT+=now_s()-t0;
}

/* ------------------------------------------------------------------ checks */
static double relerr(const float*a,const float*b,size_t n){
    double w=0,m=0; for(size_t i=0;i<n;i++){ double d=fabs((double)a[i]-b[i]); if(d>w) w=d; if(fabs(b[i])>m) m=fabs(b[i]); }
    return m>0? w/m : w;
}
static int nonfinite(void){ for(int p=0;p<NPR;p++) for(size_t i=0;i<PR[p].n;i++) if(!isfinite(PR[p].g[i])) return 1; return 0; }

static void set_batch(const float*stream,const float*offs,int bidx){
    for(int b=0;b<B;b++){ int o=(int)offs[(size_t)bidx*B+b];
        for(int t=0;t<T;t++){ TOK[b*T+t]=(int)stream[o+t]; TGT[b*T+t]=(int)stream[o+t+1]; } }
}
static float **snapshot_grads(void){
    float **s=malloc(NPR*sizeof*s);
    for(int p=0;p<NPR;p++){ s[p]=malloc(PR[p].n*4); memcpy(s[p],PR[p].g,PR[p].n*4); }
    return s;
}
static void free_snap(float**s){ for(int p=0;p<NPR;p++) free(s[p]); free(s); }

/* per-tensor comparison table vs MLX GPU + MLX CPU (+ optional C-side reference). Returns worst. */
static double grad_table(const char*title,float**ref2,const char*ref2name,int quiet){
    double worst=0,worstc=0,worst2=0; const char*wn="";
    const char *l2=hget_opt("GRAD2_LABEL","MLXcpu");     /* r102: the second MLX reference is chunk 32 */
    if(!quiet) printf("\n%s\n  %-52s %10s %10s %10s%s%s\n",title,"tensor","vs MLXgpu",l2,"max|g|",
                      ref2?"  vs ":"", ref2?ref2name:"");
    int nchk=0;
    for(int p=0;p<NPR;p++){
        char k[128]; snprintf(k,sizeof k,"grad.%s",PR[p].name); const float*gg=fx(k,PR[p].n,0);
        if(!gg) continue;                                /* r102 fixtures carry a spot-check subset */
        nchk++;
        snprintf(k,sizeof k,"gradcpu.%s",PR[p].name); const float*gc=fx(k,PR[p].n,1);
        double eg=relerr(PR[p].g,gg,PR[p].n), ec=relerr(PR[p].g,gc,PR[p].n), mg=0;
        for(size_t i=0;i<PR[p].n;i++) if(fabs(gg[i])>mg) mg=fabs(gg[i]);
        double e2=0;
        if(ref2){ double w=0,m=0; for(size_t i=0;i<PR[p].n;i++){ double d=fabs((double)PR[p].g[i]-ref2[p][i]); if(d>w)w=d; if(fabs(ref2[p][i])>m)m=fabs(ref2[p][i]); } e2=m>0?w/m:w; }
        if(eg>worst){ worst=eg; wn=PR[p].name; }
        if(ec>worstc) worstc=ec;
        if(e2>worst2) worst2=e2;
        if(!quiet){ printf("  %-52s %10.3e %10.3e %10.3e",PR[p].name,eg,ec,mg); if(ref2) printf("  %10.3e",e2); printf("\n"); }
    }
    printf("  WORST over %d tensors vs MLX gpu %.3e (%s)   vs %s %.3e", nchk, worst, wn, l2, worstc);
    if(ref2) printf("   vs %s %.3e", ref2name, worst2);
    printf("\n");
    return worst>worstc?worst:worstc;
}

static int parse_mode(const char*s){ return !strcmp(s,"npu")?MM_NPU : !strcmp(s,"emu")?MM_EMU : MM_CPU; }
static const char*MN[3]={"cpu","emu","npu"};

int main(int argc,char**argv){
    if(argc<2){ fprintf(stderr,"usage: %s FIXTURE [--lin|--att|--gdn cpu|emu|npu] [--scale 0|1] [--steps N] [--break none|rms|gdn_dk|silu] [--curve FILE] [--nocheck] [--quiet]\n",argv[0]); return 2; }
    const char *curve_path=NULL, *brk="none"; int steps=-1, nocheck=0, quiet=0, perturb=0, draws=1, srseed=1, noemu=0;
    for(int i=2;i<argc;i++){
        if(!strcmp(argv[i],"--lin")&&i+1<argc) g_mm_mode[MMC_LIN]=parse_mode(argv[++i]);
        else if(!strcmp(argv[i],"--att")&&i+1<argc) g_mm_mode[MMC_ATT]=parse_mode(argv[++i]);
        else if(!strcmp(argv[i],"--gdn")&&i+1<argc) g_mm_mode[MMC_GDN]=parse_mode(argv[++i]);
        else if(!strcmp(argv[i],"--all")&&i+1<argc){ int m=parse_mode(argv[++i]); for(int c=0;c<MMC_N;c++) g_mm_mode[c]=m; }
        else if(!strcmp(argv[i],"--scale")&&i+1<argc) g_mm_scale=atoi(argv[++i]);
        else if(!strcmp(argv[i],"--steps")&&i+1<argc) steps=atoi(argv[++i]);
        else if(!strcmp(argv[i],"--break")&&i+1<argc) brk=argv[++i];
        else if(!strcmp(argv[i],"--curve")&&i+1<argc) curve_path=argv[++i];
        else if(!strcmp(argv[i],"--nocheck")) nocheck=1;
        else if(!strcmp(argv[i],"--quiet")) quiet=1;
        else if(!strcmp(argv[i],"--perturb")&&i+1<argc) perturb=atoi(argv[++i]);
        else if(!strcmp(argv[i],"--draws")&&i+1<argc) draws=atoi(argv[++i]);     /* SR-averaged gradient check */
        else if(!strcmp(argv[i],"--srseed")&&i+1<argc) srseed=atoi(argv[++i]);
        else if(!strcmp(argv[i],"--cpufast")) g_cpu_fast=1;
        else if(!strcmp(argv[i],"--noemu")) noemu=1;        /* skip the EMU twin of an NPU step-1 check */   /* r102: vectorisable CPU nt kernel */
        else { fprintf(stderr,"unknown arg %s\n",argv[i]); return 2; }
    }
    if(!strcmp(brk,"rms")) BRK_RMS=1; else if(!strcmp(brk,"gdn_dk")) gm_BRK=2; else if(!strcmp(brk,"silu")) BRK_SILU=1;
    else if(strcmp(brk,"none")){ fprintf(stderr,"unknown --break %s\n",brk); return 2; }

    setvbuf(stdout,NULL,_IOLBF,0);       /* line-buffered: long runs log to files */
    load_fixture(argv[1]);
    D=atoi(hget("HIDDEN_SIZE")); F=atoi(hget("INTERMEDIATE_SIZE")); NL=atoi(hget("NUM_HIDDEN_LAYERS"));
    HA=atoi(hget("NUM_ATTENTION_HEADS")); HKV=atoi(hget("NUM_KEY_VALUE_HEADS")); HD=atoi(hget("HEAD_DIM"));
    V=atoi(hget("VOCAB_SIZE")); GH=atoi(hget("LINEAR_NUM_VALUE_HEADS")); GDK=atoi(hget("LINEAR_KEY_HEAD_DIM"));
    GDV=atoi(hget("LINEAR_VALUE_HEAD_DIM")); KC=atoi(hget("LINEAR_CONV_KERNEL_DIM")); FAI=atoi(hget("FULL_ATTENTION_INTERVAL"));
    ROPED=atoi(hget("ROPE_DIMS")); B=atoi(hget("B")); T=atoi(hget("T")); STEPS=atoi(hget("STEPS"));
    EPS=(float)atof(hget("RMS_EPS")); THETA=(float)atof(hget("ROPE_THETA"));
    LR=atof(hget("LR")); WD=atof(hget("WD")); B1=atof(hget("BETA1")); B2=atof(hget("BETA2")); AEPS=atof(hget("ADAM_EPS"));
    if(atoi(hget("LINEAR_NUM_KEY_HEADS"))!=GH || GDK!=GDV || atoi(hget("TIED"))!=1){ fprintf(stderr,"unsupported config\n"); return 2; }
    M=B*T; if(steps<0) steps=STEPS;
    printf("tiny e2e: d=%d ffn=%d vocab=%d layers=%d (GDN x%d + ATTN, interval %d)  B=%d T=%d M=%d  steps=%d\n",
           D,F,V,NL,NL-NL/FAI,FAI,B,T,M,steps);
    printf("  GDN heads %d dk=dv=%d chunk %d conv %d | ATTN %d q / %d kv heads, hd %d, rope %d dims theta %g\n",
           GH,GDK,CH,KC,HA,HKV,HD,ROPED,THETA);
    printf("  matmuls: linear=%s attention=%s gdn-chunk=%s  operand scaling=%s  break=%s\n",
           MN[g_mm_mode[MMC_LIN]],MN[g_mm_mode[MMC_ATT]],MN[g_mm_mode[MMC_GDN]],g_mm_scale?"pow2":"none",brk);
    printf("  AdamW lr=%g wd=%g betas=(%g,%g) eps=%g bias-correction   fixture host %s mlx %s  MLX gpu-vs-cpu grad floor %s\n",
           LR,WD,B1,B2,AEPS,hget("HOST"),hget("MLX"),hget("SELFCONSIST"));
    int use_npu=0; for(int c=0;c<MMC_N;c++) if(g_mm_mode[c]==MM_NPU) use_npu=1;
#ifdef WITH_NPU
    if(use_npu){ g_npu=ork_npu_init(); if(!g_npu){ fprintf(stderr,"ork_npu_init failed\n"); return 2; } }
#else
    if(use_npu){ fprintf(stderr,"--*-mode npu needs a -DWITH_NPU build\n"); return 2; }
#endif
    alloc_model(); rope_tables();
    if(perturb){
        /* --perturb S: multiply every init weight by (1 + u*2^-23), u in [-1,1] from seed S. A 1-ulp
         * change moves the true gradient by ~1e-7 relative but re-draws WHICH elements round which way
         * at every fp16 cast, so repeating the EMU run over seeds measures the SPREAD of the fp16
         * operand error -- the honest basis for an NPU bound (one EMU run is one draw). */
        unsigned st=2654435761u*(unsigned)perturb;
        for(int p=0;p<NPR;p++) for(size_t i=0;i<PR[p].n;i++){
            st=st*1103515245u+12345u; float u=(float)((int)((st>>8)&0xffff)-32768)/32768.0f;
            PR[p].w[i]*=1.0f+u*1.1920929e-07f; }
        printf("  init perturbed by 1 ulp, seed %d\n",perturb);
    }
    const float *trs=fx("train_bytes",0,1), *offs=fx("batch_offsets",(size_t)STEPS*B,1);
    const float *hos=fx("heldout_bytes",0,0), *hoo=fx("heldout_offsets",0,0); int nho=(int)fxn("heldout_offsets");
    const float *mcurve=fx("loss_curve",STEPS,0);       /* r102 fixtures have no curve / held-out / final */
    int fails=0;

    /* ---------------- step-1 checks ---------------- */
    set_batch(trs,offs,0);
    mm_stats_reset(); double t0=now_s();
    double L0=forward(1);
    double tf=now_s()-t0;
    if(!nocheck){
        double mlx0=atof(hget("LOSS0")), mlx0c=atof(hget("LOSS0_CPU"));
        printf("\nstep-1 loss  C %.9f   MLX gpu %.9f  cpu %.9f   |d| %.3e (rel %.3e)\n",L0,mlx0,mlx0c,fabs(L0-mlx0),fabs(L0-mlx0)/mlx0);
        printf("  forward vs MLX (max|d|/max|ref|): embed %.2e", relerr(X0,fx("hid.embed",(size_t)M*D,1),(size_t)M*D));
        for(int l=0;l<NL;l++){ char k[32]; snprintf(k,sizeof k,"hid.%d",l);
            const float*ref=fx(k,(size_t)M*D,0); if(!ref) continue;
            const float*mine = l+1<NL ? CA[l+1].x : XL;
            printf("  block%d %.2e",l,relerr(mine,ref,(size_t)M*D)); }
        const float*lref=fx("logits0",(size_t)M*V,0);
        if(lref) printf("  logits %.2e", relerr(LOGITS,lref,(size_t)M*V));
        printf("\n");
    }
    t0=now_s(); backward(); double tb=now_s()-t0;
    if(draws>1){
        /* mean gradient over `draws` stochastic-rounding seeds (tiny_e2e_mm.h, g_mm_sr) */
        g_mm_sr=1; float **acc=NULL;
        for(int d=0;d<draws;d++){
            g_sr_state=0x9E3779B97F4A7C15ull*(unsigned long long)(srseed*100003+d+1);
            forward(0); backward();
            if(nonfinite()){ printf("NON-FINITE gradient in SR draw %d\n",d); fails++; }
            if(!acc) acc=snapshot_grads(); else for(int p=0;p<NPR;p++) for(size_t i=0;i<PR[p].n;i++) acc[p][i]+=PR[p].g[i];
        }
        for(int p=0;p<NPR;p++) for(size_t i=0;i<PR[p].n;i++) PR[p].g[i]=acc[p][i]/draws;
        free_snap(acc); g_mm_sr=0;
        printf("gradient below = MEAN over %d stochastic-rounding draws (seed base %d)\n",draws,srseed);
    }
    printf("step-1 time: forward %.3f s, backward %.3f s\n",tf,tb);
    if(nonfinite()){ printf("NON-FINITE gradient at step 1\n"); fails++; }
    double worst=0;
    if(!nocheck){
        int any_f16=0; for(int c=0;c<MMC_N;c++) if(g_mm_mode[c]!=MM_CPU) any_f16=1;
        float **ref2=NULL;
        if(draws==1 && !noemu && (g_mm_mode[MMC_LIN]==MM_NPU||g_mm_mode[MMC_ATT]==MM_NPU||g_mm_mode[MMC_GDN]==MM_NPU)){
            /* the NPU bound is justified by the EMU twin: same operand rounding, CPU accumulation */
            float **npu=snapshot_grads(); int sv[MMC_N]; memcpy(sv,g_mm_mode,sizeof sv);
            for(int c=0;c<MMC_N;c++) if(g_mm_mode[c]==MM_NPU) g_mm_mode[c]=MM_EMU;
            forward(0); backward(); ref2=snapshot_grads();
            memcpy(g_mm_mode,sv,sizeof sv);
            for(int p=0;p<NPR;p++) memcpy(PR[p].g,npu[p],PR[p].n*4);
            free_snap(npu);
        }
        worst=grad_table("step-1 gradients, per tensor (max|C-ref| / max|ref|)",ref2,"EMU",quiet);
        if(ref2) free_snap(ref2);
        if(!any_f16){
            int pass=worst<=1e-4;
            printf("  CRITERION (a) cpu fp32 <= 1e-4: %s\n",pass?"PASS":"FAIL");
            if(!pass) fails++;
        }
    }
    printf("matmul split at step 1:");
    for(int c=0;c<MMC_N;c++) printf("  %s %ld calls %.3fs (npu %.3fs, %ld cpu-fallback, %ld slow-path K)",MMC_NAME[c],
        g_mm_calls[c],g_mm_sec[c],g_mm_npu_sec[c],g_mm_fallback[c],g_mm_slowpath[c]);
    printf("\n");
    if(steps==0){ printf("\n%s\n",fails?"FAIL":"done"); return fails!=0; }

    /* ---------------- training ---------------- */
    /* restore pristine init (the step-1 check ran forward/backward only; nothing was updated) */
    double hoL0=0;
    if(hos){ int nb=nho/B; float*o=malloc(B*4);
      for(int i=0;i<nb;i++){ for(int b=0;b<B;b++) o[b]=hoo[i*B+b]; set_batch(hos,o,0); hoL0+=forward(0); }
      hoL0/=nb; free(o);
      printf("\nheld-out loss at init: C %.6f  MLX %.6f\n",hoL0,atof(hget("HELDOUT0"))); }
    double *tstep=calloc(steps,8), *tsplit=calloc((size_t)steps*10,8);
    FILE*cf=curve_path?fopen(curve_path,"w"):NULL;
    if(cf) fprintf(cf,"step,loss_c,loss_mlx,sec,mm_lin,mm_att,mm_gdn,npu\n");
    double maxabs=0,maxrel=0, tsum=0, lastL=0; int at=0;
    for(int s=0;s<steps;s++){
        set_batch(trs,offs,s%STEPS);
        mm_stats_reset(); T_GDN=T_SOFTMAX=T_CE=T_OPT=0; double ts=now_s();
        double L=forward(1); backward(); lastL=L;
        if(nonfinite()){ printf("NON-FINITE gradient at step %d -- stopping\n",s+1); fails++; break; }
        adamw(s+1);
        double dt=now_s()-ts; tsum+=dt;
        double npu=g_mm_npu_sec[0]+g_mm_npu_sec[1]+g_mm_npu_sec[2];
        {   /* split: npu | host prep (mm wall - npu) for lin/att/gdn | GDN recurrence CPU (gm wall - gdn mm) |
             * softmax | CE | optimizer | rest (norms, SwiGLU, conv, decay, gates, rope, gathers) */
            double *q=tsplit+(size_t)s*10; tstep[s]=dt;
            q[0]=npu; q[1]=g_mm_sec[0]-g_mm_npu_sec[0]; q[2]=g_mm_sec[1]-g_mm_npu_sec[1]; q[3]=g_mm_sec[2]-g_mm_npu_sec[2];
            q[4]=T_GDN-g_mm_sec[2]; q[5]=T_SOFTMAX; q[6]=T_CE; q[7]=T_OPT;
            q[8]=dt-(g_mm_sec[0]+g_mm_sec[1]+T_GDN+T_SOFTMAX+T_CE+T_OPT);
            q[9]=g_mm_npu_sec[2];
            if(!mcurve) printf("  step %d  loss %.6f  %.3f s  [npu %.2f (gdn %.2f) | host prep lin %.2f att %.2f gdn %.2f | gdn-rec cpu %.2f | softmax %.2f | CE %.2f | adamw %.2f | rest %.2f]  fallbacks lin %ld\n",
                s+1,L,dt,q[0],q[9],q[1],q[2],q[3],q[4],q[5],q[6],q[7],q[8],g_mm_fallback[0]);
        }
        if(!mcurve){ fflush(stdout); continue; }
        double d=fabs(L-mcurve[s]), r=d/mcurve[s];
        if(d>maxabs){ maxabs=d; at=s+1; } if(r>maxrel) maxrel=r;
        if(cf) fprintf(cf,"%d,%.9f,%.9f,%.4f,%.4f,%.4f,%.4f,%.4f\n",s+1,L,mcurve[s],dt,g_mm_sec[0],g_mm_sec[1],g_mm_sec[2],npu);
        if(s<3 || (s+1)%20==0 || s==steps-1)
            printf("  step %4d  loss C %.6f  MLX %.6f  |d| %.2e   %.3f s/step (mm lin %.3f att %.3f gdn %.3f; npu %.3f)\n",
                   s+1,L,mcurve[s],d,dt,g_mm_sec[0],g_mm_sec[1],g_mm_sec[2],npu);
        fflush(stdout);
    }
    if(cf) fclose(cf);
    if(!mcurve){
        int n0 = steps>1 ? 1 : 0, n=steps-n0; double mu=0,sd=0,mn=1e30,mxx=0;
        for(int s=n0;s<steps;s++){ mu+=tstep[s]; if(tstep[s]<mn)mn=tstep[s]; if(tstep[s]>mxx)mxx=tstep[s]; } mu/=n;
        for(int s=n0;s<steps;s++) sd+=(tstep[s]-mu)*(tstep[s]-mu); sd=n>1?sqrt(sd/(n-1)):0;
        printf("\nsteps %d..%d (step 1 = warm-up excluded): %.3f s/step  sd %.3f  min %.3f max %.3f   %.1f tok/s\n",
               n0+1,steps,mu,sd,mn,mxx,M/mu);
        const char*nm[10]={"NPU repack+run","host prep linear","host prep attention","host prep gdn","GDN recurrence (CPU part)","attention softmax","cross-entropy","AdamW","rest (norms/SwiGLU/conv/gates/rope)","  of which NPU gdn"};
        for(int k=0;k<10;k++){ double a=0; for(int s=n0;s<steps;s++) a+=tsplit[(size_t)s*10+k]; a/=n;
            printf("  %-38s %8.3f s  %5.1f%%\n",nm[k],a,100*a/mu); }
#ifdef WITH_NPU
        if(use_npu){ int ns; double b=mm_pool_bytes(&ns); printf("resident fp16 scratch (IOVA pinned by this harness): %d shapes, %.1f MB\n",ns,b/1048576.0); }
#endif
        printf("\n%s\n",fails?"FAIL":"done"); return fails!=0;
    }
    double hoL=0;
    { int nb=nho/B; float*o=malloc(B*4);
      for(int i=0;i<nb;i++){ for(int b=0;b<B;b++) o[b]=hoo[i*B+b]; set_batch(hos,o,0); hoL+=forward(0); }
      hoL/=nb; free(o); }
    printf("\ncurve vs MLX over %d steps: max |d| %.3e (step %d), max rel %.3e\n",steps,maxabs,at,maxrel);
    printf("train loss: step 1 %.4f -> step %d C %.4f (MLX %.4f)   held-out: C %.6f  MLX %.6f\n",
           (double)mcurve[0],steps,lastL,(double)mcurve[steps-1],hoL,atof(hget("HELDOUT1")));
    printf("mean wall time %.3f s/step\n",tsum/steps);
    if(steps==STEPS){
        double ww=0,wmed[512]; int n=0; const char*wn="";
        for(int p=0;p<NPR;p++){ char k[128]; snprintf(k,sizeof k,"final.%s",PR[p].name);
            double e=relerr(PR[p].w,fx(k,PR[p].n,1),PR[p].n); wmed[n++]=e; if(e>ww){ww=e;wn=PR[p].name;} }
        for(int i=0;i<n;i++) for(int j=i+1;j<n;j++) if(wmed[j]<wmed[i]){ double t=wmed[i]; wmed[i]=wmed[j]; wmed[j]=t; }
        printf("final weights vs MLX: worst tensor %.3e (%s), median %.3e\n",ww,wn,wmed[n/2]);
    }
#ifdef WITH_NPU
    if(use_npu){ mm_npu_free_all(); ork_npu_free(g_npu); }
#endif
    printf("\n%s\n",fails?"FAIL":"done");
    return fails!=0;
}
