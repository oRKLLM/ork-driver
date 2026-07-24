/* spine_kernels.h — the CPU "glue" kernels the heterogeneous spine's CPU unit runs DIRECTLY on the resident
 * activation (norms, rope, M=1 attention, silu·glu, residual, int8 quant) + the cross-unit cache-coherency
 * helper (dc civac). Shared by the daemon's ORKD_LAYER executor and the spine probes so there is ONE copy.
 * Lean fp32 C (auto-vectorizes at -O2). See tools/spine_*_probe.c for standalone validation. */
#ifndef ORK_SPINE_KERNELS_H
#define ORK_SPINE_KERNELS_H
#include <stdint.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

/* dc civac = clean+invalidate to point-of-coherency. A cross-thread NPU->CPU handoff needs it on BOTH sides
 * (producer flush after the doorbell drains, consumer invalidate before the read) — ork_dyn_end's own sync only
 * covers the draining thread's cache. */
static inline void spine_civac1(volatile void*p){ __asm__ volatile("dc civac,%0"::"r"(p):"memory"); }
static inline void spine_civac_range(const void*b,size_t n){ for(size_t o=0;o<n;o+=64) spine_civac1((char*)b+o); __asm__ volatile("dsb ish":::"memory"); }

/* RMSNorm with gain: y = x / sqrt(mean(x^2)+eps) * gain  (in-place safe: ss computed before any write) */
static inline void spine_rmsnorm(const float*x,const float*g,int D,float eps,float*y){
    float ss=0; for(int i=0;i<D;i++) ss+=x[i]*x[i]; float r=1.0f/sqrtf(ss/D+eps);
    for(int i=0;i<D;i++) y[i]=x[i]*r*g[i]; }
/* RoPE NEoX: rotate the pair (x[i], x[i+D/2]) by theta_i = pos * base^(-2i/D), i in [0,D/2) */
static inline void spine_rope_neox(float*x,int D,int pos,float base){ int H=D/2;
    for(int i=0;i<H;i++){ float th=pos*powf(base,-2.0f*i/D),cs=cosf(th),sn=sinf(th),a=x[i],b=x[i+H]; x[i]=a*cs-b*sn; x[i+H]=a*sn+b*cs; } }
/* residual: x += y */
static inline void spine_residual(float*x,const float*y,int D){ for(int i=0;i<D;i++) x[i]+=y[i]; }
/* M=1 decode attention: o[d] = sum_j softmax_j(scale * q·k_j) * v_j[d]; K=[nkv,dk], V=[nkv,dv] row-major */
static inline void spine_attn(const float*q,const float*K,const float*V,int nkv,int dk,int dv,float scale,float*o){
    float*s=(float*)malloc((size_t)nkv*4); float mx=-1e30f;
    for(int j=0;j<nkv;j++){ float a=0; for(int d=0;d<dk;d++)a+=q[d]*K[(size_t)j*dk+d]; s[j]=a*scale; if(s[j]>mx)mx=s[j]; }
    float Z=0; for(int j=0;j<nkv;j++){ s[j]=expf(s[j]-mx); Z+=s[j]; }
    for(int d=0;d<dv;d++){ float acc=0; for(int j=0;j<nkv;j++)acc+=s[j]*V[(size_t)j*dv+d]; o[d]=acc/Z; } free(s); }
/* SwiGLU: out = silu(gate) * up, silu(x)=x*sigmoid(x) */
static inline void spine_silu_glu(const float*gate,const float*up,int n,float*o){
    for(int i=0;i<n;i++){ float g=gate[i]; o[i]=(g/(1.0f+expf(-g)))*up[i]; } }
/* per-tensor fp32->int8 quant; returns scale s so dequant = int32 / s */
static inline float spine_quant(const float*x,int n,int8_t*o){ float mx=1e-6f; for(int i=0;i<n;i++){ float a=fabsf(x[i]); if(a>mx)mx=a; }
    float s=127.0f/mx; for(int i=0;i<n;i++){ int q=(int)lrintf(x[i]*s); o[i]=(int8_t)(q>127?127:q<-127?-127:q); } return s; }

#endif
