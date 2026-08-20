/* tools/i4_probe.c — validate ork-driver's W4A4 (int4 A x int4 B -> int16 C) against a CPU
 * reference, using the captured RK3588 regcmd (REGCMD_I4, via synth_i4 / ork_i4_npu_probe, M=4).
 * The register program is the real librknnrt capture (tools/int4_capture.c), so the only unknown is
 * the native int4 TILE LAYOUT of A and B — we sweep candidate packings until the NPU output matches
 * the CPU int4xint4 reference. No librknnrt at runtime (we own both sides).
 *
 *   make i4_probe && sudo ./i4_probe              # sweep blayout x alayout
 *   sudo ./i4_probe <blayout> <alayout>           # focused single combo
 *   (extra args: reg val ... -> CNA 0x0201 overrides)
 */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include "ork_npu.h"
int main(int argc,char**argv){
    ork_npu*c=ork_npu_init(); if(!c){printf("init failed (NPU?)\n");return 1;}
    int M=4,K=64,N=64;                             /* M-tiling: each row is one M=1 task */
    if(getenv("ORK_M")) M=atoi(getenv("ORK_M"));
    if(getenv("ORK_K")) K=atoi(getenv("ORK_K"));
    if(getenv("ORK_N")) N=atoi(getenv("ORK_N"));
    signed char*A=malloc((size_t)M*K);             /* int4 A [M][K] */
    signed char*B=malloc((size_t)K*N);             /* int4 B [K][N] row-major */
    short*C=malloc((size_t)M*N*2);
    long*ref=malloc((size_t)M*N*sizeof(long));
    unsigned sd=12345;
    for(int i=0;i<M*K;i++){ sd=sd*1103515245+12345; A[i]=(signed char)((int)((sd>>17)%15)-7); }
    for(int i=0;i<K*N;i++){ sd=sd*1103515245+12345; B[i]=(signed char)((int)((sd>>17)%15)-7); }
    for(int m=0;m<M;m++)for(int n=0;n<N;n++){ long s=0; for(int k=0;k<K;k++) s+=(long)A[m*K+k]*B[k*N+n]; ref[m*N+n]=s; }

    uint32_t regs[4],vals[4]; int nov=0;
    int sweep = argc<3;
    int b0=0,b1=2,a0=0,a1=2;
    if(!sweep){ b0=atoi(argv[1]); b1=b0+1; a0=atoi(argv[2]); a1=a0+1;
        for(int i=3;i+1<argc && nov<4;i+=2){regs[nov]=(uint32_t)strtoul(argv[i],0,0);vals[nov]=(uint32_t)strtoul(argv[i+1],0,0);nov++;} }

    printf("W4A4 M=%d K=%d N=%d  ref C[0,0]=%ld  (documented native layout; sweeping nibble order)\n",M,K,N,ref[0]);
    int hits=0;
    for(int nb=b0;nb<b1;nb++)for(int na=a0;na<a1;na++){
        int rc=ork_i4_npu_probe(c,M,K,N,nb,na,nov,regs,vals,A,B,C);
        if(rc){ printf("  nibB=%d nibA=%d  rc=%d (wedge/abort)\n",nb,na,rc); continue; }
        long me=0; int n0=0; for(int i=0;i<M*N;i++){ long e=C[i]-ref[i]; if(e<0)e=-e; if(e>me){me=e;n0=i;} }
        int hit = me==0;
        printf("  nibB=%d nibA=%d  maxerr=%ld  C[%d]=%d ref=%ld %s\n",
               nb,na,me,n0,C[n0],ref[n0], hit?"*** MATCH ***":"");
        if(hit)hits++;
    }
    printf("%s\n", hits?"*** W4A4 VALIDATED — int4 matmul on NPU matches CPU ***":"no match — task[0] alone may be insufficient (capture task[1..3]); see ROADMAP");

    /* ---- full quant path: fp32 -> int4 (per-row A, per-channel B) -> NPU -> dequant vs fp32 ---- */
    if(hits){
        float*Af=malloc((size_t)M*K*4),*Bf=malloc((size_t)K*N*4),*sa=malloc((size_t)M*4),*sb=malloc((size_t)N*4);
        for(int i=0;i<M*K;i++){ sd=sd*1103515245+12345; Af[i]=((int)(sd>>9)%2001-1000)/1000.0f; }
        for(int i=0;i<K*N;i++){ sd=sd*1103515245+12345; Bf[i]=((int)(sd>>9)%2001-1000)/1000.0f; }
        for(int m=0;m<M;m++){ float mx=1e-9f; for(int k=0;k<K;k++){float a=Af[m*K+k];if(a<0)a=-a;if(a>mx)mx=a;} sa[m]=mx/7;
            for(int k=0;k<K;k++){ int q=(int)(Af[m*K+k]/sa[m]+(Af[m*K+k]>=0?0.5f:-0.5f)); if(q>7)q=7;if(q<-8)q=-8; A[m*K+k]=(signed char)q; } }
        for(int n=0;n<N;n++){ float mx=1e-9f; for(int k=0;k<K;k++){float b=Bf[k*N+n];if(b<0)b=-b;if(b>mx)mx=b;} sb[n]=mx/7;
            for(int k=0;k<K;k++){ int q=(int)(Bf[k*N+n]/sb[n]+(Bf[k*N+n]>=0?0.5f:-0.5f)); if(q>7)q=7;if(q<-8)q=-8; B[k*N+n]=(signed char)q; } }
        int rc=ork_i4_npu_probe(c,M,K,N,0,0,0,0,0,A,B,C);
        if(!rc){ double se=0,sr=0; for(int m=0;m<M;m++)for(int n=0;n<N;n++){
                double ref32=0; for(int k=0;k<K;k++) ref32+=(double)Af[m*K+k]*Bf[k*N+n];
                double deq=(double)sa[m]*sb[n]*C[m*N+n]; se+=(deq-ref32)*(deq-ref32); sr+=ref32*ref32; }
            printf("quant path (fp32->int4->NPU->dequant): RMS rel err = %.2f%%  (int4 is coarse; <%d%% expected)\n",
                   100.0*sqrt(se/sr), 15); }
        free(Af);free(Bf);free(sa);free(sb);
    }
    free(A);free(B);free(C);free(ref); ork_npu_free(c); return hits?0:2;
}
