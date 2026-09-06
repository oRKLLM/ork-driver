/* aa_probe — A/A control: run the IDENTICAL configuration in every arm.
 *
 * mouter_bench measures arm 0 (base) then arms 1/2 (reordered/reuse) in sequence, and the wide-N rows came
 * out 1.01-1.035x. If that band is real it should VANISH here, because nothing differs between arms. If it
 * survives, it is position-in-sequence -- the first arm paying one-time costs (mode entry, first weight
 * DMA, page warm) that later arms inherit -- and every A/B number in that band is bias, not signal.
 *
 * This is not a shippable effect either way: production has no "first arm", so it already sits on the warm
 * side after the first call. The probe exists to size the bias so A/B results can be read honestly.
 */
#include <ork_npu.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
static double now(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec*1e6+t.tv_nsec/1e3; }
static unsigned s_=12345; static int rnd(void){ s_=s_*1103515245u+12345u; return (int)((s_>>16)&0xff)-128; }
int main(int argc,char**argv){
    int iters = argc>1?atoi(argv[1]):3;
    ork_npu *c = ork_npu_init(); if(!c){ printf("init failed\n"); return 1; }
    struct { int K,N,M; } sh[] = {{3584,18944,64},{3584,4608,64},{2048,8192,64},{2048,2048,64},{2048,2048,256}};
    printf("%-6s %-6s %-5s %-9s %-9s %-9s %s\n","K","N","M","arm0_us","arm1_us","arm2_us","arm0/arm1  arm0/arm2");
    for(size_t q=0;q<sizeof sh/sizeof sh[0];q++){
        int K=sh[q].K,N=sh[q].N,M=sh[q].M;
        int8_t *B=malloc((size_t)K*N),*A=malloc((size_t)M*K); int32_t *C=malloc((size_t)M*N*4);
        s_=999; for(size_t i=0;i<(size_t)K*N;i++)B[i]=rnd();
        s_=4242; for(size_t i=0;i<(size_t)M*K;i++)A[i]=rnd();
        ork_w *w=ork_i8_mm_pack(c,K,N,B);
        if(!w){ printf("%-6d %-6d pack failed\n",K,N); free(A);free(B);free(C); continue; }
        double us[3]; int ok=1;
        for(int arm=0;arm<3;arm++){                     /* identical every time -- no knob is ever set */
            if(ork_i8_mm_run(c,w,M,A,C)){ ok=0; break; }
            double t0=now(); for(int it=0;it<iters;it++) ork_i8_mm_run(c,w,M,A,C); us[arm]=(now()-t0)/iters;
        }
        if(!ok) printf("%-6d %-6d %-5d run failed\n",K,N,M);
        else printf("%-6d %-6d %-5d %-9.0f %-9.0f %-9.0f %.3fx      %.3fx\n",
                    K,N,M,us[0],us[1],us[2],us[0]/us[1],us[0]/us[2]);
        ork_mm_free(c,w); free(A); free(B); free(C);
    }
    ork_npu_free(c); return 0;
}
