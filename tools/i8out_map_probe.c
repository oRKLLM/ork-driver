/* ===== DEPRECATED / QUARANTINED 2026-07-21 =====
 * Duplicate of the pre-session probes test_mm_i8_out8 / silu_bench (all drive ork_npu_probe_i8_out8). Not an
 * independent-validation source for the SDK supported-op derivation (origin/main examples only). Delete once
 * no longer referenced. Builds as a no-op stub unless ORK_DEPRECATED_PROBES is defined. */
#ifndef ORK_DEPRECATED_PROBES
#include <stdio.h>
int main(void){ fprintf(stderr,"[deprecated] i8out_map_probe: quarantined duplicate of test_mm_i8_out8 (build -DORK_DEPRECATED_PROBES to run)\n"); return 0; }
#else
/* i8out_map_probe — does the WORKING int8 matmul output stage (set_i8_out8) write int8-LINEAR or int8-CUBE?
 * Crafts acc(m,n)=n+32m (unique 0..127, fits int8), identity requant, reads C[M*N] int8 and tests both the
 * linear (m*N+n) and ORK_SEQCUBE ((n/16)*(M*16)+m*16+(n%16)) layouts. Answers whether the int16 output (which
 * is LINEAR) can reuse the int8 bridge's SiLU-read geometry. Board only.
 */
#include "ork_npu.h"
#include <stdio.h>
#include <stdlib.h>
int main(void){
    ork_npu *c=ork_npu_init(); if(!c){ printf("no board\n"); return 0; }
    enum { M=4, K=512, N=32 };
    static int8_t A[M*K], B[K*N];
    for(int i=0;i<M*K;i++) A[i]=0; for(int i=0;i<K*N;i++) B[i]=0;
    for(int m=0;m<M;m++){ A[m*K+0]=1; A[m*K+1]=(int8_t)m; }
    for(int n=0;n<N;n++){ B[0*N+n]=(int8_t)n; B[1*N+n]=32; }   /* acc(m,n)=n+32m */
    int8_t C[M*N]; for(int i=0;i<M*N;i++) C[i]=-1;
    double us=0;
    int rc=ork_npu_probe_i8_out8(c,M,K,N,A,B,0x4000,14,C,&us);   /* identity requant: out=acc */
    printf("probe_i8_out8 rc=%d %.0fus  M=%d N=%d\n", rc, us, M, N);
    if(rc==0){
        int lin=0, cube=0;
        for(int m=0;m<M;m++)for(int n=0;n<N;n++){ int ref=n+32*m;
            if(C[m*N+n]==ref) lin++;
            int co=(n/16)*(M*16)+m*16+(n%16); if(co>=0&&co<M*N && C[co]==ref) cube++; }
        printf("  layout[LINEAR m*N+n]: %d/%d\n", lin, M*N);
        printf("  layout[CUBE (n/16)*(M*16)+m*16+(n%%16)]: %d/%d\n", cube, M*N);
        printf("  VERDICT: int8 output stage writes %s\n",
               lin==M*N?"LINEAR (int16's compact-linear matches; mirror this SiLU-read geom for int16)":
               cube==M*N?"CUBE (int16 must also produce cube — the linear int16 fix won't feed the SiLU directly)":
               "neither cleanly — inspect");
    }
    ork_npu_free(c);
    return rc==0?0:1;
}
#endif /* ORK_DEPRECATED_PROBES */
