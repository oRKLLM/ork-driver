/* test_offline_load — an OFFLINE .orkpack load must recover the EXACT codes that were packed.
 *
 * ork_i4_mm_load has had an offline branch for a while; ork_i8_mm_load and ork_i4a8_mm_load did not, and
 * the absence was SILENT rather than unsupported: with no branch the loader ran on to bcreate(-1), returned
 * NULL, and the caller treated it as an ordinary pack MISS and quietly re-quantized the weight inline. Every
 * no-device run of a DT_I8 / DT_I4 pack was therefore scoring inline-packed weights, not the pack's — which
 * invalidated a day of offline screening before the pack-miss guard exposed it.
 *
 * An un-tiler can also be subtly wrong in ways no accuracy test catches (a swapped 32x32 block order, an
 * off-by-one page stride) and still produce plausible weights. So this asserts the strongest property
 * available with no NPU: dump with the CPU tiler, load it back offline, and memcmp the codes. Exact or fail.
 *
 * Shapes cover a single tile, a K-slice (K > KS), a wide-N slice (Sn > 1, exercising the Sn-major walk), and
 * a non-power-of-two K where the page-pad is not a whole tile. NO NPU NEEDED — pure CPU, both directions. */
#define _GNU_SOURCE
#include "ork_npu.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>

static uint32_t g = 246813579u;
static int8_t r8(void){ g ^= g<<13; g ^= g>>17; g ^= g<<5; return (int8_t)((int)(g & 0xff) - 128); }
static int8_t r4(void){ g ^= g<<13; g ^= g>>17; g ^= g<<5; return (int8_t)(((int)(g & 0xf)) - 8); }

/* int8 tier: CPU dump -> offline load -> codes must match byte for byte. */
static int one_i8(ork_npu *off, int K, int N, const char *tag){
    int8_t *B = malloc((size_t)K*N); if(!B){ printf("  [%-12s] OOM\n", tag); return 1; }
    for(size_t i=0;i<(size_t)K*N;i++) B[i]=r8();
    size_t need = ork_i8_w_dump_cpu(off, K, N, B, NULL, 0);
    if(!need){ printf("  [%-12s] i8 K=%d N=%d size query 0 FAIL\n",tag,K,N); free(B); return 1; }
    void *blob = malloc(need);
    if(!blob || !ork_i8_w_dump_cpu(off, K, N, B, blob, need)){ printf("  [%-12s] i8 dump FAIL\n",tag); free(B); free(blob); return 1; }
    ork_w *w = ork_i8_mm_load(off, K, N, blob, need);
    if(!w){ printf("  [%-12s] i8 K=%d N=%d OFFLINE LOAD RETURNED NULL — the bug this test exists for\n",tag,K,N);
            free(B); free(blob); return 1; }
    const int8_t *got = ork_w_codes(w);
    int bad = !got || memcmp(got, B, (size_t)K*N);
    printf("  [%-12s] i8   K=%-5d N=%-5d %s\n", tag, K, N, bad ? "MISMATCH FAIL" : "exact");
    ork_mm_free(off, w); free(B); free(blob);
    return bad ? 1 : 0;
}

/* i4a8 tier: the COMPACT container (hdr + bscale + nibbles). Build it with the CPU packer from f32 whose
 * values are already exact int4 codes, so the quantizer is an identity and the round-trip is exact. */
static int one_i4a8(ork_npu *off, int K, int N, const char *tag){
    int8_t *C = malloc((size_t)K*N);
    float  *f = malloc((size_t)K*N*sizeof(float));
    if(!C||!f){ printf("  [%-12s] OOM\n",tag); free(C); free(f); return 1; }
    /* The packer quantizes each channel with s = absmax/7, so the round trip is an IDENTITY only when the
     * channel's absmax is exactly 7 (then s = 1 and lrintf(code/1) == code). Codes are therefore drawn from
     * [-7,7] -- NOT [-8,7], since a single -8 makes absmax 8, s = 8/7, and every code in that channel comes
     * back rounded. Force one +7 per channel to pin absmax. Getting this wrong makes the test fail against
     * a correct loader, which is exactly what it did the first time. */
    for(size_t i=0;i<(size_t)K*N;i++){ int8_t v=r4(); if(v<-7) v=-7; C[i]=v; }
    for(int n=0;n<N;n++) C[(size_t)0*N+n] = 7;
    for(int n=0;n<N;n++) for(int k=0;k<K;k++) f[(size_t)n*K+k] = (float)C[(size_t)k*N+n];
    size_t need = ork_i4a8_pack_cpu_blob(off, K, N, f, NULL, 0, NULL, 0);
    void *blob = need ? malloc(need) : NULL;
    if(!blob || !ork_i4a8_pack_cpu_blob(off, K, N, f, NULL, 0, blob, need)){
        printf("  [%-12s] i4a8 blob build FAIL\n",tag); free(C); free(f); free(blob); return 1; }
    ork_w *w = ork_i4a8_mm_load(off, K, N, blob, need);
    if(!w){ printf("  [%-12s] i4a8 K=%d N=%d OFFLINE LOAD RETURNED NULL — the bug this test exists for\n",tag,K,N);
            free(C); free(f); free(blob); return 1; }
    const int8_t *got = ork_w_codes(w);
    int bad = 0;
    if(!got) bad = 1;
    else for(size_t i=0;i<(size_t)K*N && !bad;i++) if(got[i]!=C[i]) bad = 1;
    printf("  [%-12s] i4a8 K=%-5d N=%-5d %s\n", tag, K, N, bad ? "MISMATCH FAIL" : "exact");
    ork_mm_free(off, w); free(C); free(f); free(blob);
    return bad ? 1 : 0;
}

/* ork_i4a8_pack_cpu_blob_qerr must (a) emit a byte-identical blob to the plain packer -- a metric that
 * perturbs the pack is worse than no metric -- and (b) report a relative error that behaves. Checked
 * for both codebooks, with and without an imatrix. CPU only. */
static int one_qerr(ork_npu *off, int K, int N, int nf4, const char *tag){
    size_t nb=(size_t)K*N; float *f=malloc(nb*sizeof *f), *im=malloc((size_t)K*sizeof *im);
    int bad=0; void *a=NULL,*b=NULL,*e=NULL;
    if(!f||!im){ printf("  [%-12s] alloc FAIL\n",tag); bad=1; goto out; }
    { uint32_t s0=0x9e3779b9u ^ (uint32_t)(K*31+N);
      for(size_t i=0;i<nb;i++){ s0=s0*1103515245u+12345u; f[i]=(float)((int)((s0>>16)&0x7ff)-1024)/1024.0f; }
      for(int k=0;k<K;k++){ s0=s0*1103515245u+12345u; im[k]=(float)((s0>>16)&0x3ff)/1024.0f + 0.01f; } }

    size_t need=ork_i4a8_pack_cpu_blob(off,K,N,f,NULL,nf4,NULL,0);
    a=malloc(need); b=malloc(need); e=malloc(need);
    if(!a||!b||!e||!need){ printf("  [%-12s] size FAIL\n",tag); bad=1; goto out; }

    float q=-1.0f, q2=-1.0f, qim=-1.0f;
    if(!ork_i4a8_pack_cpu_blob     (off,K,N,f,NULL,nf4,a,need)      ||
       !ork_i4a8_pack_cpu_blob_qerr(off,K,N,f,NULL,nf4,b,need,&q)){ printf("  [%-12s] pack FAIL\n",tag); bad=1; goto out; }
    if(memcmp(a,b,need)){ printf("  [%-12s] BLOB DIFFERS with qerr requested\n",tag); bad=1; goto out; }
    if(!(q>0.0f && q<1.0f)){ printf("  [%-12s] qerr out of range: %g\n",tag,(double)q); bad=1; goto out; }

    /* deterministic, and NULL qerr_out must still be accepted */
    if(!ork_i4a8_pack_cpu_blob_qerr(off,K,N,f,NULL,nf4,e,need,&q2) || q2!=q){
        printf("  [%-12s] qerr not deterministic (%g vs %g)\n",tag,(double)q,(double)q2); bad=1; goto out; }
    if(!ork_i4a8_pack_cpu_blob_qerr(off,K,N,f,NULL,nf4,e,need,NULL)){ printf("  [%-12s] NULL qerr_out FAIL\n",tag); bad=1; goto out; }

    /* an imatrix reweights the sum, so the reported error should move */
    if(!ork_i4a8_pack_cpu_blob_qerr(off,K,N,f,im,nf4,e,need,&qim)){ printf("  [%-12s] imatrix pack FAIL\n",tag); bad=1; goto out; }
    if(!(qim>0.0f && qim<1.0f)){ printf("  [%-12s] imatrix qerr out of range: %g\n",tag,(double)qim); bad=1; goto out; }

    printf("  [%-12s] blob identical, qerr=%.5f  imatrix-weighted qerr=%.5f\n",tag,(double)q,(double)qim);
out:
    free(f); free(im); free(a); free(b); free(e);
    return bad;
}

int main(void){
    ork_npu *off = ork_npu_init_offline("rk3588");
    if(!off){ printf("test_offline_load: ork_npu_init_offline FAILED\n"); return 1; }
    printf("test_offline_load: offline .orkpack load must recover the packed codes exactly\n");
    int bad = 0;
    bad |= one_i8  (off, 1024, 1024, "single");
    bad |= one_i8  (off, 4096,  512, "K-slice");
    bad |= one_i8  (off, 1024, 8192, "wide-N");
    bad |= one_i8  (off, 1536,  256, "non-pow2");
    bad |= one_i4a8(off, 1024, 1024, "single");
    bad |= one_i4a8(off, 4096,  512, "K-slice");
    bad |= one_i4a8(off, 1024, 8192, "wide-N");
    bad |= one_i4a8(off, 1536,  256, "non-pow2");
    bad |= one_qerr (off, 1024,  256, 0, "qerr uniform");
    bad |= one_qerr (off, 1024,  256, 1, "qerr NF4");
    bad |= one_qerr (off, 1536,  128, 0, "qerr non-pow2");
    ork_npu_free(off);
    printf("test_offline_load: %s\n", bad ? "FAIL" : "PASS");
    return bad ? 1 : 0;
}
