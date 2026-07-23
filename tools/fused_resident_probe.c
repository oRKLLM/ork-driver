/* fused_resident_probe — task #20 (a): the fused activation, PACKED ONCE and RUN MANY resident. Proves the
 * calibration (LUT build + S-pack) is factored out of the per-call path (ork_mm_pack_f16_fused_act), so a
 * fused C=fn(A·B) matmul can live resident and replay with NO re-pack (ork_mm_run_f16_fused_act) — the
 * enabler for composing fused exp/rsqrt/silu inside a resident seq (per-call re-pack would defeat residency).
 *   R1/R2: exp(Q·K^T) on TWO different Q against the SAME resident weight -> validate both vs CPU exp.
 *   sudo env ORK_MM_TIMEOUT=3000 timeout 60 ./fused_resident_probe [M] [d] [n]
 */
#define _GNU_SOURCE
#include "ork_npu.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
static double myexp(double x, void *ctx){ (void)ctx; return exp(x); }
static uint32_t g=0x1abe11u; static float fr(void){ g=g*1664525u+1013904223u; return (float)(g>>8)/(float)(1u<<24); }
static int check(const char*tag,const float*C,const ork_f16*Q,const ork_f16*KT,int M,int d,int n){
    int bad=0; double me=0,mre=0;
    for(int m=0;m<M;m++)for(int j=0;j<n;j++){ double s=0; for(int k=0;k<d;k++) s+=(double)(float)Q[(size_t)m*d+k]*(float)KT[(size_t)k*n+j];
        double want=exp(s); double e=fabs(C[(size_t)m*n+j]-want); double re=e/(want+1e-3); if(e>me)me=e; if(re>mre)mre=re; if(re>0.03&&e>4e-3)bad++; }
    printf("  [%s] fused exp(Q·K^T): max|err|=%.3e maxrel=%.3e %s (%d/%d)\n",tag,me,mre,bad?"CHECK":"COHERENT",bad,M*n);
    return bad;
}
int main(int argc,char**argv){
    int M=argc>1?atoi(argv[1]):32, d=argc>2?atoi(argv[2]):128, n=argc>3?atoi(argv[3]):64;  /* n%16 */
    setvbuf(stdout,0,_IONBF,0);
    ork_npu*c=ork_npu_init(); if(!c){printf("init failed\n");return 2;}
    printf("fused_resident_probe: M=%d d=%d n=%d (pack-once fused exp, run-many resident)\n",M,d,n);
    ork_f16 *KT=malloc((size_t)d*n*sizeof(ork_f16));
    for(size_t i=0;i<(size_t)d*n;i++) KT[i]=(ork_f16)(-fr()*0.5f);     /* K^T<=0 */
    ork_f16 *Q1=malloc((size_t)M*d*sizeof(ork_f16)), *Q2=malloc((size_t)M*d*sizeof(ork_f16));
    for(size_t i=0;i<(size_t)M*d;i++){ Q1[i]=(ork_f16)(fr()*0.5f); Q2[i]=(ork_f16)(fr()*0.5f); }  /* Q>=0 => scores<=0 */
    /* widest score band over both Q (weight is calibrated once for the whole band) */
    float lo=0;
    for(int pass=0;pass<2;pass++){ const ork_f16*Q=pass?Q2:Q1;
        for(int m=0;m<M;m++)for(int j=0;j<n;j++){ double s=0; for(int k=0;k<d;k++) s+=(double)(float)Q[(size_t)m*d+k]*(float)KT[(size_t)k*n+j]; if(s<lo)lo=(float)s; } }
    printf("  score band [%.3f, 0]\n", lo);
    /* PACK ONCE: bake exp LUT + S-weight into the resident weight */
    ork_w *w=ork_mm_pack_f16_fused_act(c,d,n,KT,myexp,NULL,(double)lo-0.01,0.0);
    if(!w){ printf("FAIL pack_f16_fused_act -> NULL (PPU off / shape / mixed-sign)\n"); ork_npu_free(c); return 1; }
    /* RUN MANY: two different Q against the same resident weight, no re-pack */
    float *C1=malloc((size_t)M*n*4), *C2=malloc((size_t)M*n*4);
    int r1=ork_mm_run_f16_fused_act(c,w,M,Q1,C1);
    int r2=ork_mm_run_f16_fused_act(c,w,M,Q2,C2);
    printf("  run R1 rc=%d  R2 rc=%d (same resident weight, no re-pack)\n",r1,r2);
    int fail=(r1||r2)?1:0;
    if(!r1) fail|=check("R1",C1,Q1,KT,M,d,n);
    if(!r2) fail|=check("R2",C2,Q2,KT,M,d,n);
    printf("%s\n", fail?"FAIL":"PASS — fused exp packed once, run many resident (calibration factored out of the per-call path)");
    ork_mm_free(c,w); ork_npu_free(c);
    return fail;
}
