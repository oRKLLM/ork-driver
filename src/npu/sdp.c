/* npu/sdp.c — the dtype-agnostic SDP substrate: the activation CURVES and the LUT machinery every
 * precision's SDP path builds on.
 *
 * The reference activations (silu/gelu/rsqrt/exp as plain doubles), the generic LUT builder that
 * calibrates the NPU's index(acc) mapping and fills a curve at those indices, and the SDP register
 * canonicaliser. i8, i16 and f16 all sit on top of this — it is shared, not owned by any of them,
 * which is why it is a peer module rather than living in one precision's file.
 *
 * Lifted verbatim from npu.c by the precision split (MODULARIZE_PLAN.md round 1). */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "ork_regs.h"
#include "regcmd_i8.h"
#include "npu/internal.h"
#include "npu/core.h"

/* #39 Path-1 CANONICAL OUTPUT-STAGE STATE-SETTER. The full-prefill sweep (tools/re full_sdp.py + full_regmap.py
 * over pf.dump, 120,923+ tiles) proved the output stage is ONE invariant config for every int8 fold matmul,
 * spread across TWO blocks: the DPU/SDP block 0x1001 AND the PDP/aux output-dims mirror block 0x801. In both,
 * every functional register holds a single value across ALL tiles; only the geometry registers vary with (M,N).
 * sdp_canon() returns that first-principles value for any 0x1001/0x801 register — no captured blob.
 * ork_npu_sdp_stamp() rewrites the value of EVERY 0x1001/0x801 register present in a REGCMD_I8_N regcmd to its
 * canonical value (leaving DST_BASE_ADDR 0x4020 for the caller's C IOVA — the only output-stage address), so a
 * proven-runnable fold skeleton whose 0x1001+0x801 blocks are zeroed gets its ENTIRE output stage rebuilt from
 * understood values. This is the "state-setter" a delta-encoded, register-inheriting big-M tile depends on (NVDLA
 * register-file persistence). surfadd = 0x40c0 SURFACE_ADD (128*M for matched small-M tiles; the burst-regime
 * value for big-M, e.g. 0x3000 at M=36 — see full_sdp.py's 0x40c0-by-M histogram). */
static uint32_t sdp_canon(unsigned reg,int M,int N,uint32_t surfadd){
    switch(reg){
        /* ── DPU/SDP output stage (block 0x1001) ── */
        case 0x4004: return 0xe;                                   /* DPU_S_POINTER */
        case 0x400c: return 0x1e4;                                 /* FEATURE_MODE_CFG */
        case 0x4010: return 0x80000000u;                           /* OUT_PRECISION: int32 accumulate-out (bit-31) */
        case 0x4024: return (uint32_t)(16*M);                      /* DST_SURF_STRIDE = 16*M */
        case 0x4030: return (uint32_t)(M-1);                       /* DATA_CUBE_WIDTH = M-1 */
        case 0x403c: return ((uint32_t)(N-1)<<16)|(uint32_t)(N-1); /* DST_N_DIMS = ((s-1)<<16)|(N-1), s=N */
        case 0x4040: return 0x53;                                  /* BS_CFG */
        case 0x4050: return 0x7fc;                                 /* BS_OW_CFG (int32 output row byte-stride) */
        case 0x4058: return (uint32_t)(N-1);                       /* DST_N2 = N-1 */
        case 0x405c: return (uint32_t)(M-1);                       /* WDMA_SIZE_1 width = M-1 */
        case 0x4060: return 0x53;                                  /* BN_CFG */
        case 0x4070: return 0x383;                                 /* EW_CFG */
        case 0x4078: return 1;                                     /* EW_CVT_SCALE = 1 */
        case 0x4084: return 1;                                     /* OUT_CVT_SCALE = 1 (identity requant) */
        case 0x40c0: return surfadd;                               /* SURFACE_ADD (M-fold config) */
        /* ── PDP/aux output-dims mirror (block 0x801): const + (M,N) geometry, no IOVA ── */
        case 0x3010: return 1;                                     /* PDP_R3010 (const 1) */
        case 0x3014: return (uint32_t)(M-1);                       /* PDP_OUT_M = M-1 */
        case 0x3018: return (uint32_t)(N-1);                       /* PDP_OUT_N = N-1 */
        default:     return 0;   /* every other 0x1001/0x801 reg is invariant-0 across the whole prefill:
                                  * 0x1001: 4014/4034/4038/4044/4048/404c/4054/4064/4068/406c/4074/407c/4080/4088/
                                  *         4090/4094/4098-40ac/40c4/4100-412c ; 0x801: 301c/3030 */
    }
}

int ork_npu_sdp_stamp(uint32_t *rc,int rn,int M,int N,uint32_t surfadd){
    if(!rc||rn<108||M<1||M>4096||(N%16)) return -2;
    int nset=0;
    for(int k=0;k+1<rn;k+=2){ unsigned reg=rc[k]&0xffff, blk=(rc[k+1]>>16)&0xffff;
        if((blk!=0x1001 && blk!=0x801) || reg==0x4020) continue;  /* both output-stage blocks; skip C-IOVA DST_BASE_ADDR */
        uint32_t v=sdp_canon(reg,M,N,surfadd);
        rc[k]=(reg)|((v&0xffff)<<16); rc[k+1]=(blk<<16)|((v>>16)&0xffff);  /* unified 16-bit/wide encode (as setr) */
        nset++; }
    return nset;
}

double orki_silu_f(double x){ return x/(1.0+exp(-x)); }

double orki_gelu_f(double x){ return 0.5*x*(1.0+erf(x*0.7071067811865476)); }   /* exact (erf) GELU */

double orki_rsqrt_f(double x){ return x>1e-9 ? 1.0/sqrt(x) : 0.0; }              /* rsqrt (RMSNorm) — positive domain */

double orki_exp_f(double x){ return exp(x); }                                    /* exp (softmax) */

int orki_chain_build_lut_fn(ork_npu*c, double(*fn)(double), double in_scale, double out_scale,
                              int r_mult, int r_shift, uint32_t cfg4068, int16_t *lut){
    const int K=512, N=64;
    signed char *A=malloc(K), *B=calloc(1,(size_t)K*N); int8_t *C=malloc(N);
    int16_t *ramp=malloc(1030*2); int *acc=malloc(N*sizeof(int)), *idx=malloc(N*sizeof(int));
    if(!A||!B||!C||!ramp||!acc||!idx){ free(A);free(B);free(C);free(ramp);free(acc);free(idx); return -1; }
    double R = (double)r_mult / (double)(1u<<r_shift);
    for(int k=0;k<K;k++)A[k]=1;
    int accmax=(int)(8.0/in_scale); int step=(2*accmax)/(N-1); if(step<1)step=1;
    for(int n=0;n<N;n++){ int T=-accmax+n*step; int b=T/K; for(int k=0;k<K;k++)B[k*N+n]=(signed char)(b+(k<(T-b*K)?1:0)); }
    for(int n=0;n<N;n++){ int a=0; for(int k=0;k<K;k++)a+=A[k]*B[k*N+n]; acc[n]=a; }
    /* pass 1: ramp LUT[i]=(i-512)*8 -> out = R*LUT[idx] -> idx = round(out/(R*8)) + 512 */
    for(int i=0;i<1030;i++){ int v=(i-512)*8; if(v>32767)v=32767; if(v<-32768)v=-32768; ramp[i]=(int16_t)v; }
    if(ork_npu_probe_i8_silu_cfg(c,1,K,N,A,B,r_mult,r_shift,0u,0xffffc000u,cfg4068,ramp,1030,C,0)){
        free(A);free(B);free(C);free(ramp);free(acc);free(idx); return -1; }
    for(int n=0;n<N;n++){ int i=(int)lround(C[n]/(R*8.0))+512; idx[n]=i; }
    /* pass 2: build ork's silu LUT at the measured indices; interp gaps, hold at ends */
    int *set=calloc(1030,sizeof(int)); for(int i=0;i<1030;i++)lut[i]=0;
    for(int n=0;n<N;n++){ int i=idx[n]; if(i<0||i>1029)continue;
        double v=fn(acc[n]*in_scale)/out_scale/R; long q=lround(v); if(q>32767)q=32767; if(q<-32768)q=-32768;
        lut[i]=(int16_t)q; set[i]=1; }
    int lo=-1,hi=-1; for(int i=0;i<1030;i++)if(set[i]){lo=i;break;} for(int i=1029;i>=0;i--)if(set[i]){hi=i;break;}
    if(lo<0){ free(A);free(B);free(C);free(ramp);free(acc);free(idx);free(set); return -1; }
    for(int i=0;i<lo;i++)lut[i]=lut[lo]; for(int i=hi+1;i<1030;i++)lut[i]=lut[hi];
    for(int i=lo;i<=hi;i++){ if(set[i])continue; int a=i,b=i; while(a>lo&&!set[a])a--; while(b<hi&&!set[b])b++;
        lut[i]=(int16_t)(lut[a]+(lut[b]-lut[a])*(i-a)/(b-a)); }
    free(A);free(B);free(C);free(ramp);free(acc);free(idx);free(set);
    return 0;
}

int orki_build_act_lut16(ork_npu *c,double(*f)(double),double in_scale,double out_scale,int16_t *lut){
    if(orki_silu_calibrate_idx16(c)) return -1;
    static double qsum[1030]; static int qn[1030];
    for(int k=0;k<1030;k++){ qsum[k]=0; qn[k]=0; }
    for(int s=0;s<SILU16_NS;s++){ int k=c->silu_idx16[s]; if(k<0||k>1029)continue; qsum[k]+=-32768.0+s*SILU16_QSTEP; qn[k]++; }
    int lo=-1,hi=-1;
    for(int k=0;k<1030;k++){ if(qn[k]){ if(lo<0)lo=k; hi=k;
        double q_in=qsum[k]/qn[k]; double val=f(q_in*in_scale)/out_scale; long q=lround(val);
        if(q>32767)q=32767; if(q<-32768)q=-32768; lut[k]=(int16_t)q; } else lut[k]=0; }
    if(lo<0) return -1;
    for(int k=0;k<lo;k++)lut[k]=lut[lo]; for(int k=hi+1;k<1030;k++)lut[k]=lut[hi];
    for(int k=lo;k<=hi;k++){ if(qn[k])continue; int a=k,b=k; while(a>lo&&!qn[a])a--; while(b<hi&&!qn[b])b++;
        lut[k]=(int16_t)(lut[a]+(lut[b]-lut[a])*(k-a)/(b-a)); }
    return 0;
}
int orki_silu_calibrate_idx(ork_npu *c){
    if(c->silu_idx_ok) return 0;
    const int M=4,N=64;                       /* 256 elems = each int8 value exactly once */
    int8_t in[256],out[256]; int16_t lut[1030];
    for(int i=0;i<256;i++) in[i]=(int8_t)(i-128);
    for(int i=0;i<1030;i++){ int v=i-512; if(v>32767)v=32767; if(v<-32768)v=-32768; lut[i]=(int16_t)v; }
    if(ork_npu_probe_silu_std(c,in,M,N,0x2000,14,0,ORK_SILU_IDXOFF,ORK_SILU_C4064,ORK_SILU_C4068,lut,1030,out,0)) return -1;
    for(int v=0;v<256;v++) c->silu_idx[v]=-1;
    for(int i=0;i<M*N;i++){ int v=(uint8_t)in[i]; int o=out[i]; if(o>-127&&o<127) c->silu_idx[v]=(short)(2*o+512); }
    c->silu_idx_ok=1; return 0;
}

void orki_silu_build_curve_biased(ork_npu *c,double(*f)(double),double in_scale,double out_scale,double bias,int16_t *lut){
    int set[1030]; for(int i=0;i<1030;i++){lut[i]=0;set[i]=0;}
    for(int vv=-128;vv<128;vv++){ int idx=c->silu_idx[(uint8_t)vv]; if(idx<0||idx>1029)continue;
        double val=f((vv-bias)*in_scale)/out_scale; long q=lround(val); if(q>32767)q=32767; if(q<-32768)q=-32768;
        lut[idx]=(int16_t)q; set[idx]=1; }
    int lo=-1,hi=-1; for(int i=0;i<1030;i++)if(set[i]){lo=i;break;} for(int i=1029;i>=0;i--)if(set[i]){hi=i;break;}
    if(lo<0)return; for(int i=0;i<lo;i++)lut[i]=lut[lo]; for(int i=hi+1;i<1030;i++)lut[i]=lut[hi];
    for(int i=lo;i<=hi;i++){ if(set[i])continue; int a=i,b=i; while(a>lo&&!set[a])a--; while(b<hi&&!set[b])b++;
        lut[i]=(int16_t)(lut[a]+(lut[b]-lut[a])*(i-a)/(b-a)); }
}

void orki_silu_build_curve(ork_npu *c,double(*f)(double),double in_scale,double out_scale,int16_t *lut){
    orki_silu_build_curve_biased(c,f,in_scale,out_scale,0.0,lut);   /* plain curve = no bias */
}

int orki_silu_calibrate_idx16(ork_npu *c){
    if(c->silu_idx16_ok) return 0;
    const int M=64,N=64;                      /* 4096 samples across the full int16 range (step 16) */
    static int16_t in[SILU16_NS],out[SILU16_NS]; int16_t lut[1030];
    for(int s=0;s<SILU16_NS;s++) in[s]=(int16_t)(-32768 + s*SILU16_QSTEP);
    for(int i=0;i<1030;i++){ int v=i-512; if(v>32767)v=32767; if(v<-32768)v=-32768; lut[i]=(int16_t)v; }
    /* NB: runs LAZILY on the first silu call — in the FFN chain that's right after a MULTI-CORE matmul,
     * so this pure-SDP probe hits the chain-context wedge (retry does NOT help — it wedges every attempt
     * even after soft-resets). See ork_npu_probe_silu_std_i16 (#35). Standalone it's clean. */
    if(ork_npu_probe_silu_std_i16(c,in,M,N,0x4000,14,0,ORK_SILU16_IDXOFF,ORK_SILU16_C4064,ORK_SILU16_C4068,lut,1030,out,0)) return -1;
    for(int s=0;s<SILU16_NS;s++){ int o=out[s]; c->silu_idx16[s]=(o>-490&&o<510)?(short)(o+512):(short)-32768; }
    c->silu_idx16_ok=1; return 0;
}
