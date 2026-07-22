/* orkd_seq_probe — task #20: validate the attention SDP ops routing through ORKD_SEQ Path B.
 *
 * Builds ONE ork_submit_seq call containing all six attention SDP ops (independent operands) —
 *   [RMSNORM_F16][ROPE_NEOX_F16][REDUCEMAX_I8][EXP_I16][RSQRT_I16][MUL_PERCHANNEL_F16]
 * and checks each output vs a CPU reference. This is exactly the batching win: N SDP ops collapse
 * into ONE round-trip. Run BOTH ways to prove the extension:
 *   direct (Path A):   sudo env ORK_MM_TIMEOUT=3000 ./orkd_seq_probe
 *   daemon (Path B):   sudo env ORK_USE_ORKD=1 ORKD_BIN=./orkd ORK_MM_TIMEOUT=3000 ./orkd_seq_probe
 * Under orkd the client sizes each op (npu.c Path B switch) and the daemon reconstructs + runs them
 * via its own Path A adapters; identical results prove the wire round-trip is faithful.
 *
 * Exit 0 = all six coherent; nonzero = a stage failed / miscomputed.
 */
#define _GNU_SOURCE
#include "ork_npu.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>

static uint32_t g_rng=0x51ab77u;
static float frand(void){ g_rng=g_rng*1664525u+1013904223u; return (float)(g_rng>>8)/(float)(1u<<24); }

int main(int argc,char**argv){
    int M=argc>1?atoi(argv[1]):32;    /* rows           */
    int N=argc>2?atoi(argv[2]):128;   /* width (%32/%16/%8) */
    setvbuf(stdout,0,_IONBF,0);
    ork_npu*c=ork_npu_init(); if(!c){ printf("init failed\n"); return 2; }
    int viaorkd = getenv("ORK_USE_ORKD") ? 1 : 0;
    printf("orkd_seq_probe: M=%d N=%d  path=%s\n", M, N, viaorkd?"ORKD_SEQ (Path B)":"direct (Path A)");
    int fail=0;
    double eps=1e-5, freq_base=10000.0;

    /* --- operands (independent per op) --- */
    ork_f16 *rn_x=malloc((size_t)M*N*2), *rn_g=malloc((size_t)N*2), *rn_o=malloc((size_t)M*N*2);
    ork_f16 *rp_x=malloc((size_t)M*N*2), *rp_o=malloc((size_t)M*N*2); int *rp_pos=malloc((size_t)M*sizeof(int));
    int8_t  *rm_a=malloc((size_t)M*N), *rm_o=malloc((size_t)M);
    int16_t *ex_a=malloc((size_t)M*N*2), *ex_o=malloc((size_t)M*N*2);
    int16_t *rs_a=malloc((size_t)M*N*2), *rs_o=malloc((size_t)M*N*2);
    ork_f16 *mp_a=malloc((size_t)M*N*2), *mp_b=malloc((size_t)N*2), *mp_o=malloc((size_t)M*N*2);
    for(size_t i=0;i<(size_t)M*N;i++){ rn_x[i]=(ork_f16)(frand()*2.f-1.f); rp_x[i]=(ork_f16)(frand()*2.f-1.f);
        rm_a[i]=(int8_t)((int)(frand()*254.f)-127); ex_a[i]=(int16_t)((int)(frand()*20000.f)-30000);
        rs_a[i]=(int16_t)(4096+(int)(frand()*27900.f)); mp_a[i]=(ork_f16)(frand()*2.f-1.f); }
    for(int j=0;j<N;j++){ rn_g[j]=(ork_f16)(0.5f+frand()); mp_b[j]=(ork_f16)(frand()*2.f-1.f); }
    for(int r=0;r<M;r++) rp_pos[r]=r+1;
    double ex_is=1.0/1024.0, ex_os=1.0/32000.0, rs_is=1.0/4096.0, rs_os=1.0/16384.0;
    /* poison outputs */
    for(size_t i=0;i<(size_t)M*N;i++){ rn_o[i]=(ork_f16)-1e30f; rp_o[i]=(ork_f16)-1e30f; ex_o[i]=-1; rs_o[i]=-1; mp_o[i]=(ork_f16)-1e30f; }
    for(int r=0;r<M;r++) rm_o[r]=-128;

    /* --- ONE seq with all six SDP ops --- */
    ork_seq_op ops[6] = {
        { .kind=ORK_OP_RMSNORM_F16,      .M=M, .N=N, .A=rn_x, .B=rn_g,  .C=rn_o, .in_scale=eps },
        { .kind=ORK_OP_ROPE_NEOX_F16,    .M=M, .N=N, .A=rp_x, .B=rp_pos,.C=rp_o, .in_scale=freq_base },
        { .kind=ORK_OP_REDUCEMAX_I8,     .M=M, .N=N, .A=rm_a,           .C=rm_o },
        { .kind=ORK_OP_EXP_I16,          .M=M, .N=N, .A=ex_a,           .C=ex_o, .in_scale=ex_is, .out_scale=ex_os },
        { .kind=ORK_OP_RSQRT_I16,        .M=M, .N=N, .A=rs_a,           .C=rs_o, .in_scale=rs_is, .out_scale=rs_os },
        { .kind=ORK_OP_MUL_PERCHANNEL_F16,.M=M,.N=N, .A=mp_a, .B=mp_b,  .C=mp_o },
    };
    int rc=ork_submit_seq(c,ops,6);
    printf("  ork_submit_seq(6 ops) rc=%d\n", rc);
    if(rc){ printf("FAIL — seq rc=%d (%s)\n", rc, viaorkd?"orkd Path B rejected/failed":"direct"); ork_npu_free(c); return 1; }

    /* --- validate each op vs CPU --- */
    int hd2=N/2, bad;
    /* rmsnorm */
    bad=0; double me=0; for(int r=0;r<M;r++){ double ms=0; for(int j=0;j<N;j++){ float v=(float)rn_x[(size_t)r*N+j]; ms+=(double)v*v; }
        float rr=1.0f/sqrtf((float)(ms/N)+(float)eps); for(int j=0;j<N;j++){ float w=(float)rn_x[(size_t)r*N+j]*rr*(float)rn_g[j], g=(float)rn_o[(size_t)r*N+j];
            double e=fabs(g-w); if(e>me)me=e; if(e>1e-2*(fabs(w)+1e-2)) bad++; } }
    printf("  [RMSNORM_F16]        max|err|=%.2e %s\n", me, bad?"MISMATCH":"OK"); if(bad)fail=1;
    /* rope */
    bad=0; me=0; for(int r=0;r<M;r++){ double p=rp_pos[r]; for(int i=0;i<hd2;i++){ double th=p*pow(freq_base,-2.0*i/(double)N); float cc=cos(th),ss=sin(th);
        float x0=rp_x[(size_t)r*N+i],x1=rp_x[(size_t)r*N+i+hd2]; float w0=x0*cc-x1*ss,w1=x1*cc+x0*ss;
        double e0=fabs((float)rp_o[(size_t)r*N+i]-w0), e1=fabs((float)rp_o[(size_t)r*N+i+hd2]-w1); if(e0>me)me=e0; if(e1>me)me=e1;
        if(e0>1e-2*(fabs(w0)+1.f))bad++; if(e1>1e-2*(fabs(w1)+1.f))bad++; } }
    printf("  [ROPE_NEOX_F16]      max|err|=%.2e %s\n", me, bad?"MISMATCH":"OK"); if(bad)fail=1;
    /* reducemax */
    bad=0; for(int r=0;r<M;r++){ int8_t mx=rm_a[(size_t)r*N]; for(int j=1;j<N;j++) if(rm_a[(size_t)r*N+j]>mx)mx=rm_a[(size_t)r*N+j]; if(rm_o[r]!=mx)bad++; }
    printf("  [REDUCEMAX_I8]       %s (%d/%d rows)\n", bad?"MISMATCH":"OK", M-bad, M); if(bad)fail=1;
    /* exp */
    bad=0; me=0; for(size_t i=0;i<(size_t)M*N;i++){ double x=(double)ex_a[i]*ex_is; double want=exp(x)/ex_os; if(want>32767)want=32767;
        double e=fabs((double)ex_o[i]-want); if(e>me)me=e; if(e>150+0.03*fabs(want))bad++; }
    printf("  [EXP_I16]            max|err|=%.0f LSB %s\n", me, bad?"MISMATCH":"OK"); if(bad)fail=1;
    /* rsqrt */
    bad=0; me=0; for(size_t i=0;i<(size_t)M*N;i++){ double x=(double)rs_a[i]*rs_is; double want=(1.0/sqrt(x))/rs_os;
        double e=fabs((double)rs_o[i]-want); if(e>me)me=e; if(e>100+0.02*fabs(want))bad++; }
    printf("  [RSQRT_I16]          max|err|=%.0f LSB %s\n", me, bad?"MISMATCH":"OK"); if(bad)fail=1;
    /* mul_perchan */
    bad=0; me=0; for(int r=0;r<M;r++) for(int j=0;j<N;j++){ float w=(float)mp_a[(size_t)r*N+j]*(float)mp_b[j], g=(float)mp_o[(size_t)r*N+j];
        double e=fabs(g-w); if(e>me)me=e; if(e>1e-2*(fabs(w)+1e-3)+1e-3)bad++; }
    printf("  [MUL_PERCHANNEL_F16] max|err|=%.2e %s\n", me, bad?"MISMATCH":"OK"); if(bad)fail=1;

    printf("%s\n", fail? "FAIL — an attention SDP op miscomputed via the seq"
                       : (viaorkd? "PASS — six attention SDP ops batched through ORKD_SEQ (one round-trip), all coherent"
                                 : "PASS — six attention SDP ops via ork_submit_seq (direct), all coherent"));
    ork_npu_free(c);
    return fail;
}
