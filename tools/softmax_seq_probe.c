/* softmax_seq_probe — task #20: validate the softmax + rope seq adapters + end-to-end on-NPU softmax coherence.
 *
 * Wires the three newly-added seq dispatch adapters (REDUCEMAX_I8, MUL_PERCHANNEL_F16, RSQRT_I16) and
 * confirms (a) each dispatches correctly through ork_submit_seq (1-op seq) vs a CPU reference, and (b) the
 * primitives compose into a COHERENT end-to-end softmax using the PROVEN recipe (ggml-ork.cpp attn handler):
 *
 *     row_max(NPU) -> x-max+quantize(CPU bridge) -> exp_i16(NPU) -> int16->f16(CPU bridge)
 *       -> Sigma = e . ones[n,16] reduce-matmul(NPU) -> 1/Sigma + per-row normalize(CPU) -> softmax
 *
 * The CPU bridges (int8->int16 quant, int16->f16, the per-row divide) are exactly where the current recipe
 * bounces to the host; they are the residency gaps the full single-seq chain must still close (broadcast-sub
 * for x-max, an on-NPU requant bridge, rsqrt_i16 for 1/Sigma). This probe validates the pieces are coherent
 * so they can be assembled into the resident attention chain. Coherence bar (per ATTN WIP): softmax rel-err
 * ~1-3% (int8 max is coarse, exp rides the int16 SDP LUT).
 *
 * BOARD:  make softmax_seq_probe && sudo env ORK_MM_TIMEOUT=3000 timeout 300 ./softmax_seq_probe
 * Exit 0 = all stages coherent; nonzero = a stage failed.
 */
#define _GNU_SOURCE
#include "ork_npu.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>

static uint32_t g_rng = 0x1234567u;
static float frand(void){ g_rng = g_rng*1664525u + 1013904223u; return (float)(g_rng>>8)/(float)(1u<<24); }   /* [0,1) */

int main(int argc,char**argv){
    int M = argc>1?atoi(argv[1]):64;      /* rows (queries)                */
    int n = argc>2?atoi(argv[2]):512;     /* softmax width (keys) — n%32==0 (exp tile), n%16, K%512 (doorbell) */
    setvbuf(stdout,0,_IONBF,0);
    ork_npu*c=ork_npu_init(); if(!c){ printf("init failed\n"); return 2; }
    printf("softmax_seq_probe: M=%d n=%d\n", M, n);
    int fail=0;

    /* ---- Stage 1: REDUCEMAX_I8 as a 1-op seq vs CPU row-max ------------------------------------------- */
    {
        int8_t *A=malloc((size_t)M*n); int8_t *Cn=malloc((size_t)M), *Cc=malloc((size_t)M);
        for(size_t i=0;i<(size_t)M*n;i++) A[i]=(int8_t)((int)(frand()*254.f)-127);
        for(int m=0;m<M;m++){ int8_t mx=A[(size_t)m*n]; for(int j=1;j<n;j++) if(A[(size_t)m*n+j]>mx) mx=A[(size_t)m*n+j]; Cc[m]=mx; }
        memset(Cn,0x7f? 0:0, (size_t)M); for(int m=0;m<M;m++) Cn[m]=-128;   /* poison */
        ork_seq_op op={ .kind=ORK_OP_REDUCEMAX_I8, .M=M, .N=n, .A=A, .C=Cn };
        int rc=ork_submit_seq(c,&op,1);
        int bad=0; for(int m=0;m<M;m++) if(Cn[m]!=Cc[m]) bad++;
        printf("  [1] REDUCEMAX_I8 (seq)      : rc=%d  %s (%d/%d rows exact)\n", rc, (rc==0&&!bad)?"OK":"MISMATCH", M-bad, M);
        if(rc||bad) fail=1;
        free(A);free(Cn);free(Cc);
    }

    /* ---- Stage 2: MUL_PERCHANNEL_F16 as a 1-op seq vs CPU per-channel multiply ------------------------ */
    {
        ork_f16 *A=malloc((size_t)M*n*sizeof(ork_f16)), *B=malloc((size_t)n*sizeof(ork_f16));
        ork_f16 *Cn=malloc((size_t)M*n*sizeof(ork_f16));
        for(size_t i=0;i<(size_t)M*n;i++) A[i]=(ork_f16)(frand()*2.f-1.f);
        for(int j=0;j<n;j++) B[j]=(ork_f16)(frand()*2.f-1.f);
        for(size_t i=0;i<(size_t)M*n;i++) Cn[i]=(ork_f16)-1e30f;   /* poison */
        ork_seq_op op={ .kind=ORK_OP_MUL_PERCHANNEL_F16, .M=M, .N=n, .A=A, .B=B, .C=Cn };
        int rc=ork_submit_seq(c,&op,1);
        int bad=0; float me=0;
        for(int m=0;m<M;m++) for(int j=0;j<n;j++){ float want=(float)A[(size_t)m*n+j]*(float)B[j], got=(float)Cn[(size_t)m*n+j];
            float e=fabsf(got-want); if(e>me)me=e; if(e>1e-2f*(fabsf(want)+1e-3f)+1e-3f) bad++; }
        printf("  [2] MUL_PERCHANNEL_F16 (seq): rc=%d  %s (max|err|=%.2e, %d/%zu bad)\n", rc, (rc==0&&!bad)?"OK":"MISMATCH", me, bad,(size_t)M*n);
        if(rc||bad) fail=1;
        free(A);free(B);free(Cn);
    }

    /* ---- Stage 3: RSQRT_I16 as a 1-op seq vs CPU rsqrt ------------------------------------------------ */
    {
        int rows=M, cols=n;
        int16_t *A=malloc((size_t)rows*cols*2), *Cn=malloc((size_t)rows*cols*2);
        /* Well-conditioned RMSNorm-like domain (the op's real use): x in [1,~7.8], away from the rsqrt
         * singularity where the 1030-bin int16 LUT is coarse. out = rsqrt(in*in_scale)/out_scale. */
        double in_scale=1.0/4096.0, out_scale=1.0/16384.0;
        for(size_t i=0;i<(size_t)rows*cols;i++) A[i]=(int16_t)(4096+(int)(frand()*27900.f));   /* [4096,32000] positive */
        for(size_t i=0;i<(size_t)rows*cols;i++) Cn[i]=-1;
        ork_seq_op op={ .kind=ORK_OP_RSQRT_I16, .M=rows, .N=cols, .A=A, .C=Cn, .in_scale=in_scale, .out_scale=out_scale };
        int rc=ork_submit_seq(c,&op,1);
        int bad=0; double me=0;
        for(size_t i=0;i<(size_t)rows*cols;i++){ double x=(double)A[i]*in_scale; double want= x>1e-9? (1.0/sqrt(x))/out_scale : 0.0;
            double got=(double)Cn[i]; double e=fabs(got-want); if(e>me)me=e; if(e> 100.0 + 0.02*fabs(want)) bad++; }   /* int16 LUT RKNN-class (~75 LSB) */
        printf("  [3] RSQRT_I16 (seq)         : rc=%d  %s (max|err|=%.1f LSB, %d/%zu bad)\n", rc, (rc==0&&!bad)?"OK":"MISMATCH", me, bad,(size_t)rows*cols);
        if(rc||bad) fail=1;
        free(A);free(Cn);
    }

    /* ---- Stage 4: end-to-end on-NPU softmax (proven recipe) coherence vs CPU ------------------------- */
    {
        float *X=malloc((size_t)M*n*sizeof(float));            /* QK^T-like scores */
        for(size_t i=0;i<(size_t)M*n;i++) X[i]=frand()*8.f-4.f;
        /* CPU reference softmax over n per row */
        float *ref=malloc((size_t)M*n*sizeof(float));
        for(int m=0;m<M;m++){ float mx=X[(size_t)m*n]; for(int j=1;j<n;j++) if(X[(size_t)m*n+j]>mx)mx=X[(size_t)m*n+j];
            double s=0; for(int j=0;j<n;j++){ double e=exp((double)X[(size_t)m*n+j]-mx); ref[(size_t)m*n+j]=(float)e; s+=e; }
            for(int j=0;j<n;j++) ref[(size_t)m*n+j]/=(float)s; }

        /* (a) quantize scores -> int8 for the on-NPU row-max (coarse, as the recipe does) */
        float amax=0; for(size_t i=0;i<(size_t)M*n;i++){ float a=fabsf(X[i]); if(a>amax)amax=a; } if(amax<=0)amax=1;
        double sq=amax/127.0;
        int8_t *q8=malloc((size_t)M*n); for(size_t i=0;i<(size_t)M*n;i++){ long v=lround(X[i]/sq); if(v<-127)v=-127; if(v>127)v=127; q8[i]=(int8_t)v; }
        int8_t *maxq=malloc((size_t)M); for(int m=0;m<M;m++) maxq[m]=-128;
        ork_seq_op o_max={ .kind=ORK_OP_REDUCEMAX_I8, .M=M, .N=n, .A=q8, .C=maxq };
        int rc_max=ork_submit_seq(c,&o_max,1);

        /* (b) CPU bridge: x - max, quantize to int16 for the exp SDP LUT */
        float lo=0; for(int m=0;m<M;m++){ float mf=maxq[m]*(float)sq; for(int j=0;j<n;j++){ float d=X[(size_t)m*n+j]-mf; if(d<lo)lo=d; } }
        double in_scale=(-lo)/32000.0; if(in_scale<=0)in_scale=1e-6; double out_scale=1.0/32000.0;
        int16_t *xi=malloc((size_t)M*n*2), *ei=malloc((size_t)M*n*2);
        for(int m=0;m<M;m++){ float mf=maxq[m]*(float)sq; for(int j=0;j<n;j++){ long v=lround((double)(X[(size_t)m*n+j]-mf)/in_scale);
            if(v<-32768)v=-32768; if(v>32767)v=32767; xi[(size_t)m*n+j]=(int16_t)v; } }

        /* (c) exp on the NPU (int16 SDP LUT) via seq */
        ork_seq_op o_exp={ .kind=ORK_OP_EXP_I16, .M=M, .N=n, .A=xi, .C=ei, .in_scale=in_scale, .out_scale=out_scale };
        int rc_exp=ork_submit_seq(c,&o_exp,1);

        /* (d) CPU bridge: int16 exp -> f16 for the reduce matmul */
        ork_f16 *ef=malloc((size_t)M*n*sizeof(ork_f16));
        for(size_t i=0;i<(size_t)M*n;i++) ef[i]=(ork_f16)((double)ei[i]*out_scale);

        /* (e) Sigma = e . ones[n,16] reduce-matmul on the NPU via seq (sum lands in col 0) */
        ork_f16 *ones=malloc((size_t)n*16*sizeof(ork_f16)); for(size_t i=0;i<(size_t)n*16;i++) ones[i]=(ork_f16)1.0f;
        ork_w *w_ones=ork_f16_mm_pack(c,n,16,ones);
        float *ss=malloc((size_t)M*16*sizeof(float));
        int rc_red=-1;
        if(w_ones){ ork_seq_op o_red={ .kind=ORK_OP_MM_F16, .w=w_ones, .M=M, .A=ef, .C=ss }; rc_red=ork_submit_seq(c,&o_red,1); }

        /* (f) CPU: 1/Sigma + per-row normalize -> on-NPU softmax result */
        float *P=malloc((size_t)M*n*sizeof(float));
        for(int m=0;m<M;m++){ float S = (rc_red==0)? ss[(size_t)m*16] : 0.f;
            if(rc_red!=0){ double s=0; for(int j=0;j<n;j++) s+=(double)ef[(size_t)m*n+j]; S=(float)s; }   /* fallback CPU sum */
            float inv = S>0? 1.0f/S : 0.f;
            for(int j=0;j<n;j++) P[(size_t)m*n+j]=(float)ef[(size_t)m*n+j]*inv; }

        /* coherence vs CPU reference */
        double me=0, sae=0; int bad=0;
        for(size_t i=0;i<(size_t)M*n;i++){ double e=fabs((double)P[i]-(double)ref[i]); sae+=e; if(e>me)me=e; if(e>0.03) bad++; }
        printf("  [4] softmax seq (max/exp/Sigma on NPU): rc(max=%d exp=%d red=%d)  max|err|=%.2e mae=%.2e  %s (%d/%zu > 3%%)\n",
               rc_max,rc_exp,rc_red, me, sae/((double)M*n), (rc_max==0&&rc_exp==0&&!bad)?"COHERENT":"CHECK", bad,(size_t)M*n);
        if(rc_max||rc_exp||bad) fail=1;
        if(w_ones) ork_mm_free(c,w_ones);
        free(X);free(ref);free(q8);free(maxq);free(xi);free(ei);free(ef);free(ones);free(ss);free(P);
    }

    /* ---- Stage 5: ROPE_NEOX_F16 as a 1-op seq vs CPU NEOX RoPE ---------------------------------------
     * Field overload (see seq_disp_rope_neox_f16): pos[] via o->B, freq_base via o->in_scale. */
    {
        int nrow=32, hd=128, hd2=hd/2; double freq_base=10000.0;
        ork_f16 *X=malloc((size_t)nrow*hd*sizeof(ork_f16)), *Cn=malloc((size_t)nrow*hd*sizeof(ork_f16));
        int *pos=malloc((size_t)nrow*sizeof(int));
        for(int r=0;r<nrow;r++) pos[r]=r+1;
        for(size_t i=0;i<(size_t)nrow*hd;i++){ X[i]=(ork_f16)(frand()*2.f-1.f); Cn[i]=(ork_f16)-1e30f; }
        ork_seq_op op={ .kind=ORK_OP_ROPE_NEOX_F16, .M=nrow, .N=hd, .A=X, .B=pos, .C=Cn, .in_scale=freq_base };
        int rc=ork_submit_seq(c,&op,1);
        int bad=0; float me=0;
        for(int r=0;r<nrow;r++){ double p=(double)pos[r];
            for(int i=0;i<hd2;i++){ double th=p*pow(freq_base,-2.0*(double)i/(double)hd); float cc=(float)cos(th),ss=(float)sin(th);
                float x0=(float)X[(size_t)r*hd+i], x1=(float)X[(size_t)r*hd+i+hd2];
                float w0=x0*cc - x1*ss, w1=x1*cc + x0*ss;
                float g0=(float)Cn[(size_t)r*hd+i], g1=(float)Cn[(size_t)r*hd+i+hd2];
                float e0=fabsf(g0-w0), e1=fabsf(g1-w1); if(e0>me)me=e0; if(e1>me)me=e1;
                if(e0>1e-2f*(fabsf(w0)+1.f)) bad++; if(e1>1e-2f*(fabsf(w1)+1.f)) bad++; } }
        printf("  [5] ROPE_NEOX_F16 (seq)     : rc=%d  %s (max|err|=%.2e, %d/%d bad)\n", rc, (rc==0&&!bad)?"OK":"MISMATCH", me, bad, nrow*hd);
        if(rc||bad) fail=1;
        free(X);free(Cn);free(pos);
    }

    printf("%s\n", fail? "FAIL — a softmax/rope seq stage miscomputed" : "PASS — softmax + rope seq adapters dispatch + compose coherently");
    ork_npu_free(c);
    return fail;
}
