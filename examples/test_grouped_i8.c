/* test_grouped_i8 — grouped int8 (W4A8) on device must equal an independent CPU reference, INCLUDING Sn>1.
 *
 * The grouped path decomposes a matmul into one [G x Nc] submit per (N-slice, K-group) and stitches the
 * partials in fp32. Two things there are pure index arithmetic and therefore silently wrong when wrong:
 * the per-(ns,g) tile layout at load, and the scatter that puts each task's CONTIGUOUS [M x Nc] output back
 * into the right columns of [M x N]. A wrong scatter interleaves N-tiles and still produces a plausible
 * matmul -- no crash, no NaN, just a model that is quietly worse.
 *
 * Sn>1 is the case that matters: it is what a 27B needs (wide-N weights slice to Sn=2..3), and it is
 * exactly the case the int4 grouped path declined to implement. So the shapes below straddle nmax(8192).
 *
 * The reference is computed here rather than by calling the driver's own CPU kernel -- an independent
 * implementation, so a shared bug cannot agree with itself. Group sums run in the same order on both sides
 * (g ascending), so fp32 accumulation is deterministic and this asserts EXACT equality. Board only. */
#define _GNU_SOURCE
#include "ork_npu.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>

static uint32_t g_s = 0xC0FFEEu;
static int8_t r4(void){ g_s ^= g_s<<13; g_s ^= g_s>>17; g_s ^= g_s<<5; return (int8_t)(((int)(g_s & 0xf)) - 8); }

static int one(ork_npu *c, int M, int K, int N, int G, const char *tag){
    const int SG = K/G;
    int8_t *B  = malloc((size_t)K*N);          /* weight codes [k*N+n], int4 range */
    int8_t *A  = malloc((size_t)M*K);          /* activations, int8 range */
    float  *as = malloc((size_t)M*SG*sizeof(float));
    float  *bs = malloc((size_t)SG*N*sizeof(float));
    float  *Cd = malloc((size_t)M*N*sizeof(float));
    float  *Cr = malloc((size_t)M*N*sizeof(float));
    if(!B||!A||!as||!bs||!Cd||!Cr){ printf("  [%-9s] OOM\n", tag); return 1; }
    for(size_t i=0;i<(size_t)K*N;i++) B[i]=r4();
    for(size_t i=0;i<(size_t)M*K;i++) A[i]=(int8_t)((int)(r4())*7);      /* int8-range activations */
    for(int i=0;i<M*SG;i++) as[i]=0.001f*(float)((i%13)+1);
    for(int i=0;i<SG*N;i++) bs[i]=0.002f*(float)((i%7)+1);

    /* tiled int4 blob -> group-tiled int8-resident weight (the shipped load path) */
    size_t need = ork_i4_w_dump_cpu(c, K, N, B, NULL, 0);
    void  *blob = need ? malloc(need) : NULL;
    if(!blob || !ork_i4_w_dump_cpu(c, K, N, B, blob, need)){ printf("  [%-9s] dump FAIL\n", tag); return 1; }
    ork_w *w = ork_i4a8_mm_load_tiled(c, K, N, blob, need, G);
    if(!w){ printf("  [%-9s] M=%d K=%d N=%d G=%d: group-tiled LOAD returned NULL\n", tag,M,K,N,G); free(blob); return 1; }

    int rc = ork_i8_mm_run_grouped(c, w, M, A, as, bs, Cd);
    if(rc){ printf("  [%-9s] M=%d K=%d N=%d G=%d: run REFUSED rc=%d\n", tag,M,K,N,G,rc); ork_mm_free(c,w); free(blob); return 1; }

    /* independent reference: int32 within a group, one fp32 scale per (row,group,channel), g ascending */
    for(int m=0;m<M;m++){
        float *cr=Cr+(size_t)m*N;
        for(int n=0;n<N;n++) cr[n]=0.0f;
        for(int g=0;g<SG;g++){
            const float a=as[(size_t)m*SG+g];
            for(int n=0;n<N;n++){
                int32_t acc=0;
                for(int k=0;k<G;k++) acc += (int32_t)A[(size_t)m*K+(size_t)g*G+k] * (int32_t)B[(size_t)(g*G+k)*N+n];
                cr[n] += (float)acc * a * bs[(size_t)g*N+n];
            }
        }
    }
    size_t bad=0; double worst=0;
    for(size_t i=0;i<(size_t)M*N;i++){
        const double d=fabs((double)Cd[i]-(double)Cr[i]);
        if(d>worst) worst=d;
        if(Cd[i]!=Cr[i]) bad++;
    }
    const int Sn = (N + 8191)/8192;
    printf("  [%-9s] M=%-3d K=%-5d N=%-6d G=%-4d Sn=%d  %s (mismatch %zu/%zu, worst %.3g)\n",
           tag,M,K,N,G,Sn, bad?"FAIL":"exact", bad,(size_t)M*N, worst);
    ork_mm_free(c,w); free(blob);
    free(B);free(A);free(as);free(bs);free(Cd);free(Cr);
    return bad?1:0;
}

int main(void){
    ork_npu *c = ork_npu_init();
    if(!c){ printf("test_grouped_i8: no NPU\n"); return 1; }
    printf("test_grouped_i8: grouped W4A8 on device vs an independent CPU reference\n");
    int bad=0;
    /* G must be a multiple of 512 and <= 4096: each group is a full-K chain link with K=G, and the
     * 0x1040 K-reduction schedule is only valid there. G=32 is rejected on device (correctly) -- the
     * quality cost of coarsening is small, since group size barely matters once activations are int8. */
    bad |= one(c, 8, 2048, 4096,  512,  "Sn=1");     /* baseline: single N-slice */
    bad |= one(c, 8, 2048, 16384, 512,  "Sn=2");     /* > nmax: the case 27B needs */
    bad |= one(c, 8, 4096, 20480, 1024, "Sn=3");     /* three N-slices, coarser G */
    bad |= one(c, 4, 3072, 12288, 1024, "nonpow2K"); /* K not a power of two */
    ork_npu_free(c);
    printf("test_grouped_i8: %s\n", bad?"FAIL":"PASS");
    return bad?1:0;
}
