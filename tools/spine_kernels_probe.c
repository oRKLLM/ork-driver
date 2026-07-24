/* spine_kernels_probe — increment 3 foundation: the CPU "glue" kernels the heterogeneous spine's CPU worker
 * runs DIRECTLY on the resident activation (NOT delegated back to llama.cpp — that would re-introduce the
 * socket round-trip). Lean C (auto-vectorizes at -O2; explicit NEON later only where it pays). Plus the spine
 * PLACEMENT TABLE (op -> {NPU,CPU,EITHER}), the operational sibling of OPS_REGISTRY.
 * Validates each kernel vs a double-precision reference of its own formula (catches indexing/dtype bugs).
 *   make spine_kernels_probe && ./spine_kernels_probe          (no NPU needed — pure CPU kernels)
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
static uint32_t rng=0xC0FFEEu; static float frnd(void){ rng=rng*1664525u+1013904223u; return (int)(rng>>9)/4194304.0f-1.0f; } /* [-1,1) */

/* ---- the spine placement table: op kind -> execution unit ---- */
enum ork_op_kind { OP_RMSNORM, OP_ROPE_NEOX, OP_RESIDUAL, OP_ATTN_DECODE, OP_QKV, OP_OPROJ, OP_FFN, OP_NKIND };
enum ork_unit_pl { PL_CPU, PL_NPU, PL_EITHER };
static const struct { const char*name; enum ork_unit_pl unit; } OP_PLACEMENT[OP_NKIND] = {
    [OP_RMSNORM]     = {"rmsnorm",     PL_CPU},   /* tiny, domain-free */
    [OP_ROPE_NEOX]   = {"rope_neox",   PL_CPU},
    [OP_RESIDUAL]    = {"residual",    PL_CPU},
    [OP_ATTN_DECODE] = {"attn_decode", PL_CPU},   /* M=1 softmax·V — faster on CPU than the NPU submit overhead */
    [OP_QKV]         = {"qkv_proj",    PL_NPU},   /* weight-streaming, bandwidth-bound */
    [OP_OPROJ]       = {"o_proj",      PL_NPU},
    [OP_FFN]         = {"ffn",         PL_NPU},
};

/* ---- CPU glue kernels (fp32; the spine runs these on the resident activation) ---- */
/* RMSNorm with gain: y = x / sqrt(mean(x^2)+eps) * gain */
static void k_rmsnorm(const float*x, const float*gain, int D, float eps, float*y){
    float ss=0; for(int i=0;i<D;i++) ss+=x[i]*x[i]; float r=1.0f/sqrtf(ss/D+eps);
    for(int i=0;i<D;i++) y[i]=x[i]*r*gain[i];
}
/* RoPE NEoX: rotate the pair (x[i], x[i+D/2]) by theta_i = pos * base^(-2i/D), i in [0,D/2) */
static void k_rope_neox(float*x, int D, int pos, float base){
    int H=D/2; for(int i=0;i<H;i++){ float th=pos*powf(base,-2.0f*i/D); float cs=cosf(th),sn=sinf(th);
        float a=x[i], b=x[i+H]; x[i]=a*cs-b*sn; x[i+H]=a*sn+b*cs; }
}
/* residual: x += y */
static void k_residual(float*x, const float*y, int D){ for(int i=0;i<D;i++) x[i]+=y[i]; }
/* decode attention (M=1): out[d] = sum_j softmax_j(scale * q·k_j) * v_j[d]; K=[nkv,dk], V=[nkv,dv] row-major */
static void k_attn_decode(const float*q, const float*K, const float*V, int nkv, int dk, int dv, float scale, float*out){
    float*s=malloc((size_t)nkv*4); float mx=-1e30f;
    for(int j=0;j<nkv;j++){ float a=0; for(int d=0;d<dk;d++) a+=q[d]*K[(size_t)j*dk+d]; s[j]=a*scale; if(s[j]>mx)mx=s[j]; }
    float Z=0; for(int j=0;j<nkv;j++){ s[j]=expf(s[j]-mx); Z+=s[j]; }
    for(int d=0;d<dv;d++){ float acc=0; for(int j=0;j<nkv;j++) acc+=s[j]*V[(size_t)j*dv+d]; out[d]=acc/Z; }
    free(s);
}

static double maxerr(const float*a, const float*b, int n){ double m=0; for(int i=0;i<n;i++){ double d=fabs((double)a[i]-b[i]); if(d>m)m=d; } return m; }

int main(void){
    printf("spine_kernels_probe: CPU glue kernels + placement table\n");
    printf("  placement:"); for(int k=0;k<OP_NKIND;k++) printf(" %s=%s", OP_PLACEMENT[k].name, OP_PLACEMENT[k].unit==PL_CPU?"CPU":OP_PLACEMENT[k].unit==PL_NPU?"NPU":"EITHER"); printf("\n");
    int fail=0, D=2048;

    /* rmsnorm vs double ref */
    { float *x=malloc(D*4),*g=malloc(D*4),*y=malloc(D*4); for(int i=0;i<D;i++){x[i]=frnd()*4;g[i]=1+0.1f*frnd();}
      k_rmsnorm(x,g,D,1e-6f,y);
      double ss=0; for(int i=0;i<D;i++) ss+=(double)x[i]*x[i]; double r=1.0/sqrt(ss/D+1e-6);
      double me=0; for(int i=0;i<D;i++){ double ref=x[i]*r*g[i]; double d=fabs(ref-y[i]); if(d>me)me=d; }
      printf("  rmsnorm: max|err|=%.2e %s\n", me, me<1e-3?"OK":"FAIL"); if(me>=1e-3)fail=1; free(x);free(g);free(y); }

    /* rope neox: check it's a rotation (preserves the pair norm) + matches double ref */
    { float *x=malloc(D*4),*x0=malloc(D*4); for(int i=0;i<D;i++) x[i]=x0[i]=frnd()*2;
      k_rope_neox(x,D,37,10000.0f); int H=D/2; double me=0, nrm=0;
      for(int i=0;i<H;i++){ double th=37.0*pow(10000.0,-2.0*i/D); double cs=cos(th),sn=sin(th);
        double ra=x0[i]*cs-x0[i+H]*sn, rb=x0[i]*sn+x0[i+H]*cs; double d=fabs(ra-x[i]); if(d>me)me=d; d=fabs(rb-x[i+H]); if(d>me)me=d;
        double n0=x0[i]*x0[i]+x0[i+H]*x0[i+H], n1=(double)x[i]*x[i]+(double)x[i+H]*x[i+H]; if(fabs(n0-n1)>nrm)nrm=fabs(n0-n1); }
      printf("  rope_neox: max|err|=%.2e norm-drift=%.2e %s\n", me, nrm, (me<1e-3&&nrm<1e-2)?"OK":"FAIL"); if(me>=1e-3||nrm>=1e-2)fail=1; free(x);free(x0); }

    /* residual */
    { float *x=malloc(D*4),*y=malloc(D*4),*x0=malloc(D*4); for(int i=0;i<D;i++){x[i]=x0[i]=frnd();y[i]=frnd();}
      k_residual(x,y,D); double me=0; for(int i=0;i<D;i++){ double d=fabs((double)x0[i]+y[i]-x[i]); if(d>me)me=d; }
      printf("  residual: max|err|=%.2e %s\n", me, me<1e-5?"OK":"FAIL"); if(me>=1e-5)fail=1; free(x);free(y);free(x0); }

    /* attn_decode vs double ref */
    { int nkv=256, dk=128, dv=128; float scale=1.0f/sqrtf(dk);
      float *q=malloc(dk*4),*Kk=malloc((size_t)nkv*dk*4),*Vv=malloc((size_t)nkv*dv*4),*out=malloc(dv*4);
      for(int i=0;i<dk;i++) q[i]=frnd(); for(size_t i=0;i<(size_t)nkv*dk;i++) Kk[i]=frnd(); for(size_t i=0;i<(size_t)nkv*dv;i++) Vv[i]=frnd();
      k_attn_decode(q,Kk,Vv,nkv,dk,dv,scale,out);
      double *s=malloc((size_t)nkv*8),mx=-1e30,Z=0; for(int j=0;j<nkv;j++){ double a=0; for(int d=0;d<dk;d++)a+=(double)q[d]*Kk[(size_t)j*dk+d]; s[j]=a*scale; if(s[j]>mx)mx=s[j]; }
      for(int j=0;j<nkv;j++){ s[j]=exp(s[j]-mx); Z+=s[j]; } double me=0;
      for(int d=0;d<dv;d++){ double acc=0; for(int j=0;j<nkv;j++)acc+=s[j]*Vv[(size_t)j*dv+d]; double ref=acc/Z; double dd=fabs(ref-out[d]); if(dd>me)me=dd; }
      printf("  attn_decode: max|err|=%.2e %s\n", me, me<1e-4?"OK":"FAIL"); if(me>=1e-4)fail=1; free(q);free(Kk);free(Vv);free(out);free(s); }

    printf("%s\n", fail?"FAIL":"PASS — spine CPU glue kernels coherent; placement table defined");
    return fail;
}
