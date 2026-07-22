/* i16out_fix_probe — root-cause + MAP the int16 matmul output stage (set_i16_out). Runs an int8 matmul with
 * the int16 output stage (ork_npu_probe_i16_out) using inputs crafted so acc(m,n)=n+32m is UNIQUE per (m,n),
 * so each raw-buffer int16 slot decodes to exactly one (m,n) -> reveals the output tile layout directly.
 * Test reg fixes via env (no recompile): ORK_MM_R4010/R4038/R4050/R40c0. ORK_MM_TIMEOUT bounds a stall. Board only.
 */
#include "ork_npu.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
int main(void){
    ork_npu *c=ork_npu_init(); if(!c){ printf("no board\n"); return 0; }
    enum { M=4, K=512, N=32 };
    const int mult=0x4000, shift=14;   /* identity: out = acc */
    static int8_t A[M*K], B[K*N];
    for(int i=0;i<M*K;i++) A[i]=0; for(int i=0;i<K*N;i++) B[i]=0;
    for(int m=0;m<M;m++){ A[m*K+0]=1; A[m*K+1]=(int8_t)m; }      /* acc(m,n) = B[0,n] + m*B[1,n] */
    for(int n=0;n<N;n++){ B[0*N+n]=(int8_t)n; B[1*N+n]=32; }     /* = n + 32m  (unique 0..127) */
    int16_t out[M*N]; for(int i=0;i<M*N;i++) out[i]=-1;
    double us=0;
    int rc=ork_npu_probe_i16_out(c,M,K,N,A,B,mult,shift,out,&us);
    printf("probe_i16_out rc=%d (0=ran,-1=wedge) %.0fus  M=%d N=%d\n", rc, us, M, N);
    if(rc==0){
        const int16_t *ob=out;
        /* dump each buffer slot -> the (m,n) its value decodes to (val=n+32m) */
        printf("  slot -> (m,n)  [first 2 rows of slots]:\n");
        for(int i=0;i<2*N;i++){ int v=ob[i]; if(v>=0&&v<128) printf("   [%3d]=%3d ->(%d,%2d)%s", i, v, v/32, v%32, (i%8==7)?"\n":""); else printf("   [%3d]=%4d ->(?)%s", i, v, (i%8==7)?"\n":""); }
        printf("\n");
        /* auto-fit: for each candidate layout, count how many (m,n) land where expected */
        struct { const char*name; } L[5]={{"linear m*N+n"},{"EWCUBEH_S (n/8)(M*16)+m16+(n%8)2"},{"colmaj n*M+m"},{"atom8-rowcube (m)(N*?)..."},{"n*M+m *? "}};
        int cnt[5]={0};
        for(int m=0;m<M;m++)for(int n=0;n<N;n++){ int ref=n+32*m;
            long off[5]={ (long)(m*N+n)*2, (n/8)*(M*16)+m*16+(n%8)*2, (long)(n*M+m)*2, (n/8)*(M*16)+m*16+(n%8)*2, (long)(n*M+m)*2 };
            for(int L_=0;L_<5;L_++){ long idx=off[L_]/2; if(idx<0||idx>=M*N*4) continue; if(ob[idx]==ref) cnt[L_]++; } }
        for(int L_=0;L_<3;L_++) printf("  layout[%s]: %d/%d\n", L[L_].name, cnt[L_], M*N);
    } else printf("  (stall/wedge — this reg config the NPU can't complete)\n");
    ork_npu_free(c);
    return rc==0?0:1;
}
