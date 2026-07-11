/* Phase-1 chained-FFN RE: does an int8 matmul emit CORRECT int16 via set_i16_out?
 * Test 1 (encoding, layout-independent): A=1,B=1,K=512 => every acc=512, identity requant
 *   => every int16 output must be 512. All elements equal, so layout can't hide a value error.
 * Test 2 (layout): B=identity-ish varying so outputs differ, dump the int16 grid to map the
 *   device write layout vs the row-major reference.
 * Sweep the encoding on-board with ORK_I16OUT_4010/4050/40c0 if Test 1 fails.
 * Build: make i16out_probe ; run: sudo ./i16out_probe [M] [K] [N] */
#include "ork_npu.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

/* IEEE half (uint16 bits) -> float */
static float h2f(uint16_t h){
    uint32_t s=(h>>15)&1, e=(h>>10)&0x1f, m=h&0x3ff, out;
    if(e==0){ if(m==0) out=s<<31; else { e=127-15+1; while(!(m&0x400)){m<<=1;e--;} m&=0x3ff; out=(s<<31)|(e<<23)|(m<<13);} }
    else if(e==0x1f) out=(s<<31)|0x7f800000|(m<<13);
    else out=(s<<31)|((e-15+127)<<23)|(m<<13);
    float f; memcpy(&f,&out,4); return f;
}

int main(int argc,char**argv){
    int M=argc>1?atoi(argv[1]):8;
    int K=argc>2?atoi(argv[2]):512;
    int N=argc>3?atoi(argv[3]):64;
    ork_npu*c=ork_npu_init();
    if(!c){ fprintf(stderr,"init failed\n"); return 2; }

    int8_t*A=malloc((size_t)M*K), *B=malloc((size_t)K*N);
    short  *C=malloc((size_t)M*N*sizeof(short));

    /* ---- Test 1: all-ones, identity requant ---- */
    memset(A,1,(size_t)M*K); memset(B,1,(size_t)K*N);
    memset(C,0,(size_t)M*N*sizeof(short));
    int f16 = getenv("ORK_MM_F16OUT")!=NULL;
    double us=0;
    int r=ork_npu_probe_i16_out(c,M,K,N,A,B,0x4000,14,C,&us);
    printf("=== Test1 all-ones (M=%d K=%d N=%d) mode=%s rc=%d %.0fus ===\n",M,K,N,f16?"fp16":"int16",r,us);
    if(r==0 && f16){
        int want=K; int nok=0; float mn=1e30f,mx=-1e30f;
        for(int i=0;i<M*N;i++){ float v=h2f((uint16_t)C[i]); if(v==(float)want)nok++; if(v<mn)mn=v; if(v>mx)mx=v; }
        printf("  want=%d : match=%d/%d  min=%.3f max=%.3f\n",want,nok,M*N,mn,mx);
        printf("  first 8 (fp16): "); for(int i=0;i<8&&i<M*N;i++)printf("%.2f ",h2f((uint16_t)C[i])); printf("\n");
        printf("  %s\n", nok==M*N ? "FP16 OUT_CVT OK *** (int8 matmul -> fp16!)" : (nok>0?"PARTIAL":"WRONG (sweep F16OUT regs)"));
    } else if(r==0){
        int want=K; int nok=0,nz=0; short mn=32767,mx=-32768;
        for(int i=0;i<M*N;i++){ short v=C[i]; if(v==want)nok++; if(v==0)nz++; if(v<mn)mn=v; if(v>mx)mx=v; }
        printf("  want=%d : match=%d/%d  zero=%d  min=%d max=%d\n",want,nok,M*N,nz,mn,mx);
        printf("  first 16: "); for(int i=0;i<16&&i<M*N;i++)printf("%d ",C[i]); printf("\n");
        printf("  %s\n", nok==M*N ? "ENCODING OK ***" : (nok>0? "PARTIAL (layout/stride?)":"ENCODING WRONG (sweep 4010/4050/40c0)"));
    } else printf("  (wedge/dim error rc=%d)\n",r);
    if(f16){ free(A);free(B);free(C); ork_npu_free(c); return 0; }  /* fp16: skip the int16-layout Test2 */

    /* ---- Test 2: varying, to map layout (only if Test1 produced nonzero data) ---- */
    if(r==0){
        for(int k=0;k<K;k++)for(int n=0;n<N;n++)B[(size_t)k*N+n]=(k==0)?(int8_t)(n+1):0; /* out[m,n]=A[m,0]*(n+1) */
        for(int m=0;m<M;m++)for(int k=0;k<K;k++)A[(size_t)m*K+k]=(k==0)?(int8_t)(m+1):0;  /* => out[m,n]=(m+1)*(n+1) */
        memset(C,0,(size_t)M*N*sizeof(short));
        r=ork_npu_probe_i16_out(c,M,K,N,A,B,0x4000,14,C,&us);
        printf("=== Test2 out[m,n]=(m+1)*(n+1) rc=%d ===\n",r);
        if(r==0){
            int nok=0;
            for(int m=0;m<M;m++)for(int n=0;n<N;n++){ int want=(m+1)*(n+1); if(C[(size_t)m*N+n]==want)nok++; }
            printf("  row-major match=%d/%d\n",nok,M*N);
            printf("  grid[m][0..7] (row-major read):\n");
            for(int m=0;m<M&&m<8;m++){ printf("   m=%d: ",m); for(int n=0;n<8&&n<N;n++)printf("%4d ",C[(size_t)m*N+n]); printf("  (want %d..%d)\n",(m+1)*1,(m+1)*8); }
        }
    }
    free(A);free(B);free(C); ork_npu_free(c);
    return 0;
}
