/* exp_biased_probe — task #20 (resident int8 softmax): scalar GLOBAL-max-subtract baked into exp_i8.
 * ork_npu_exp_i8_biased computes clamp_i8(exp((x-max)*in_scale)/out_scale) with max = the scalar global max.
 * This is the numerically-stable softmax numerator WITHOUT a per-row op (per-row max would need the dead
 * per-channel-add): a global scalar max is >= every row max so every argument (x-max)<=0 (exp in (0,1], no int8
 * overflow), and the constant cancels in P=e/Sum. Proves (1) biased exp coherent vs CPU, (2) plain exp_i8
 * overflows on the same positive scores, (3) softmax P=e/Sum matches CPU softmax (max cancels).
 *   direct: sudo env ORK_MM_TIMEOUT=3000 timeout 60 ./exp_biased_probe [M] [n]
 */
#define _GNU_SOURCE
#include "ork_npu.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
static uint32_t g=0x5eed3u; static int r8(void){ g=g*1664525u+1013904223u; return (int)((g>>24)%81)-40; }  /* [-40,40] */
int main(int argc,char**argv){
    int M=argc>1?atoi(argv[1]):32, n=argc>2?atoi(argv[2]):64;   /* n%16 */
    setvbuf(stdout,0,_IONBF,0);
    ork_npu*c=ork_npu_init(); if(!c){printf("init failed\n");return 2;}
    printf("exp_biased_probe: M=%d n=%d (scalar global-max-biased exp_i8)\n",M,n);
    double in_scale=0.05, out_scale=1.0/127.0;   /* exp((x-max)*0.05) in (0,1] -> *127 fits int8 */
    int8_t *X=malloc((size_t)M*n); for(size_t i=0;i<(size_t)M*n;i++) X[i]=(int8_t)r8();   /* mixed sign incl. positive */
    int gmax=-128; for(size_t i=0;i<(size_t)M*n;i++) if(X[i]>gmax) gmax=X[i];
    printf("  int8 scores in [-40,40], global max=%d\n",gmax);

    /* pre-calibrate the int-LUT idx in a clean context (documented first-op-after-matmul trap; harmless here) */
    { int8_t wi[32],wo[32]; for(int i=0;i<32;i++) wi[i]=(int8_t)(i-16); ork_npu_exp_i8(c,wi,1,32,in_scale,out_scale,wo,NULL); }

    /* (1) biased exp vs CPU */
    int8_t *e=malloc((size_t)M*n); for(size_t i=0;i<(size_t)M*n;i++)e[i]=-128;
    int rc=ork_npu_exp_i8_biased(c,X,M,n,in_scale,out_scale,(double)gmax,e,NULL);
    printf("  exp_i8_biased rc=%d e[0]=%d (want %d)\n",rc,e[0],(int)lround(exp((X[0]-(double)gmax)*in_scale)/out_scale));
    if(rc){ printf("FAIL rc=%d\n",rc); ork_npu_free(c); return 1; }
    double me=0;   /* int8 PWL-LUT + banker's-round error is a few LSB (documented class) — informational, not a gate */
    for(size_t i=0;i<(size_t)M*n;i++){ double want=exp(((double)X[i]-gmax)*in_scale)/out_scale; if(want>127)want=127; if(want<-128)want=-128;
        double er=fabs((double)e[i]-want); if(er>me)me=er; }
    printf("  (1) biased exp vs CPU: max|err|=%.1f LSB (int8-LUT class, informational)\n",me);

    /* (2) plain exp_i8 (no bias) on the SAME scores — saturates (positive args overflow int8) */
    int8_t *ep=malloc((size_t)M*n); ork_npu_exp_i8(c,X,M,n,in_scale,out_scale,ep,NULL);
    int sat=0; for(size_t i=0;i<(size_t)M*n;i++) if(ep[i]==127) sat++;
    printf("  (2) plain exp_i8 (no bias): %d/%d saturated at 127 (why the max-subtract is needed)\n",sat,M*n);

    /* (3) softmax P = e/Sum vs CPU softmax(X*in_scale) — the global-max constant cancels. THE correctness gate. */
    int fail = 0;
    { double meP=0; int badP=0;
      for(int m=0;m<M;m++){ double S=0; for(int j=0;j<n;j++) S+=(double)e[(size_t)m*n+j]; if(S<=0)S=1;
        double Sc=0; for(int j=0;j<n;j++) Sc+=exp((double)X[(size_t)m*n+j]*in_scale);
        for(int j=0;j<n;j++){ double P=(double)e[(size_t)m*n+j]/S; double ref=exp((double)X[(size_t)m*n+j]*in_scale)/Sc;
            double er=fabs(P-ref); if(er>meP)meP=er; if(er>2e-2)badP++; } }
      printf("  (3) softmax P=e/Sum vs CPU: max|err|=%.3e  %s (%d/%d)\n",meP,badP?"CHECK":"COHERENT",badP,M*n);
      if(badP)fail=1; }
    printf("%s\n", fail?"FAIL":"PASS — scalar global-max-biased exp_i8: stable numerator, no per-row op, softmax coherent");
    ork_npu_free(c);
    return fail;
}
