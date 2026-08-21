/* tools/re/i16_mcap_probe.c — map the int16 SDP activation envelope (Tier 14 A #2).
 *
 * WHY. `orki_i16_act_lut` — the shared path behind ork_i16_npu_{exp,silu,gelu,rsqrt} — validates only
 *   if (M<1 || M>8192 || N<8 || N>8192 || (N&7)) return -2;
 * but the roadmap records NRMSE 2e-4 @ MR=8192 and **0.55-0.69 @ MR=6144/4096** — i.e. known-garbage
 * row counts sit INSIDE the accepted range, and the shape is non-monotonic. On-NPU softmax was made
 * opt-in to dodge it; the envelope itself was never mapped.
 *
 * WHERE THE LIMITS COULD COME FROM (`orki_set_mul_geom`, src/npu/i8/regcmd.c):
 *   M-1  -> RK_DPU_DATA_CUBE_WIDTH (0x4030, mask 0x1fff => 13 bits => M <= 8192)   <- the existing guard
 *   M-1  -> RK_DPU_WDMA_SIZE_1     (0x405c, WIDTH_WDMA[12:0]  => M <= 8192)
 *   N-1  -> RK_SDP_5014 / RK_DPU_DST_N_DIMS / RK_DPU_DST_N2
 *   M*16 -> RK_SDP_5040, RK_DPU_DST_SURF_STRIDE (0x4024)
 *   M*16 -> RK_DPU_SURFACE_ADD (0x40c0) ** SUSPECT ** — ork_regs.h calls this an element-size CONFIG
 *           ("NOT an IOVA: 0x20 int8, 0x80 int32, 0x400 M-fold"), not a stride. The capture was M=8,
 *           where M*16 == 0x80 exactly, so the M*16 formula may be a one-point over-generalisation.
 *
 * METHOD (the fp16 playbook, which found a non-monotonic envelope there):
 *   - REFERENCE = the same op run in M=8 chunks (the CAPTURED shape, so the geometry is the one the
 *     template was recorded at). exp/silu are ELEMENTWISE, so row m depends only on row m — a chunked
 *     reference is exact, not an approximation, and cannot inherit a large-M geometry error.
 *   - Cross-check that reference against a CPU double evaluation so a systematically-wrong reference
 *     cannot masquerade as agreement.
 *   - SCAN M UPWARD. Never bisect: fp16 proved this class of predicate can be non-monotonic, and
 *     bisection then lands on an arbitrary valid point.
 *   - Do NOT stop at the first failure — keep walking to detect recovery (the non-monotonic signature).
 *
 *   make i16_mcap_probe
 *   sudo tools/util/npu_guard.sh -- env ORK_MM_TIMEOUT=2000 ./i16_mcap_probe [N] [Mmax]
 */
#define _GNU_SOURCE
#include "ork_npu.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define REFM 8                     /* the captured cube width — the known-good reference shape */

static double nrmse(const short *got,const short *ref,size_t n){
    double num=0,den=0;
    for(size_t i=0;i<n;i++){ double d=(double)got[i]-(double)ref[i]; num+=d*d; den+=(double)ref[i]*(double)ref[i]; }
    if(den<=0) return num>0?1.0:0.0;
    return sqrt(num/den);
}
static int first_bad_row(const short *got,const short *ref,int M,int N,int tol){
    for(int m=0;m<M;m++) for(int n=0;n<N;n++){
        int d=got[(size_t)m*N+n]-ref[(size_t)m*N+n]; if(d<0)d=-d;
        if(d>tol) return m; }
    return -1;
}

int main(int argc,char**argv){
    int N     = argc>1?atoi(argv[1]):64;
    int Mmax  = argc>2?atoi(argv[2]):8192;
    const double IS=1.0/2048.0, OS=1.0/32767.0;   /* in/out fixed-point scales for exp */
    const int TOL=8;                              /* LSBs; a geometry break is orders of magnitude worse */

    ork_npu *c=ork_npu_init(); if(!c){ printf("no board\n"); return 0; }
    printf("i16_mcap_probe — SoC=%s  N=%d  ref=M%d chunks  tol=%d LSB\n",ork_npu_soc(c),N,REFM,TOL);

    short *in =malloc((size_t)Mmax*N*2), *out=malloc((size_t)Mmax*N*2), *ref=malloc((size_t)Mmax*N*2);
    if(!in||!out||!ref){ printf("OOM\n"); return 2; }
    /* deterministic, row-VARYING input in the LUT's useful band (constant rows would hide row errors) */
    for(int m=0;m<Mmax;m++) for(int n=0;n<N;n++)
        in[(size_t)m*N+n]=(short)(-3000 + ((m*37+n*11)%6000));

    /* ---- reference: same op, REFM rows at a time (the captured geometry) ---- */
    for(int m0=0;m0<Mmax;m0+=REFM){
        int mc=(Mmax-m0<REFM)?(Mmax-m0):REFM;
        if(ork_i16_npu_exp(c,in+(size_t)m0*N,mc,N,IS,OS,ref+(size_t)m0*N,NULL)){
            printf("reference failed at m0=%d\n",m0); return 2; } }

    /* ---- sanity: does that reference actually track exp() on the CPU? ---- */
    { double num=0,den=0;
      for(int n=0;n<N;n++){ double x=(double)in[n]*IS, e=exp(x)/OS;
          if(e>32767)e=32767; double d=(double)ref[n]-e; num+=d*d; den+=e*e; }
      printf("reference vs CPU exp (row 0): rel=%.2e  %s\n",
             den>0?sqrt(num/den):0.0, (den>0&&sqrt(num/den)<0.05)?"OK":"** REFERENCE SUSPECT **"); }

    printf("\n%-7s %-9s %-11s %-10s %s\n","M","M*16","nrmse","firstbad","verdict");
    int lastbad=0, recovered=0, nok=0, nbad=0;
    /* argv[3..] = an EXPLICIT ascending M list (for pinning a boundary); else the default ladder. */
    int explicit_n = argc>3 ? argc-3 : 0, ei = 0;
    for(int M=REFM; M<=Mmax; M = (explicit_n? (++ei<explicit_n? atoi(argv[3+ei]) : Mmax+1)
                                             : (M<64? M+8 : (M<512? M*2 : M+512)))){
        if(explicit_n && ei==0) M = atoi(argv[3]);
        if(M<1||M>Mmax) break;
        memset(out,0,(size_t)M*N*2);
        int rc=ork_i16_npu_exp(c,in,M,N,IS,OS,out,NULL);
        if(rc){ printf("%-7d %-9d %-11s %-10s rc=%d\n",M,M*16,"-","-",rc); continue; }
        double e=nrmse(out,ref,(size_t)M*N);
        int fb=first_bad_row(out,ref,M,N,TOL);
        const char *v = fb<0 ? "OK" : "MISMATCH";
        if(fb<0){ nok++; if(lastbad){ recovered=1; printf("   ^^^ RECOVERED after a failing M — envelope is NON-MONOTONIC\n"); } lastbad=0; }
        else { nbad++; lastbad=1; }
        printf("%-7d %-9d %-11.3e %-10d %s\n",M,M*16,e,fb,v);
    }
    printf("\nRESULT: %d ok, %d bad, non-monotonic=%s\n",nok,nbad,recovered?"YES":"no");
    printf("(guard today is M<=8192 from DATA_CUBE_WIDTH's 13-bit field; anything failing below that is unguarded)\n");
    ork_npu_free(c); free(in);free(out);free(ref);
    return nbad?1:0;
}
