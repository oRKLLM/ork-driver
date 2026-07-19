/* Path B: the NORMAL ork_npu API must give identical results whether it runs the NPU directly or transparently
 * routes through orkd (ORK_USE_ORKD=1). The example code is unchanged between the two modes — that's the whole
 * point of Path B. Covers int8 (ork_mm_pack_i8/run_i8) AND fp16 (ork_mm_pack/run). Board tool, not in `make test`
 * (orkd would contend with the direct-NPU examples).
 *   make orkd test_orkd_transparent
 *   sudo ./test_orkd_transparent                                        # direct
 *   sudo env ORK_USE_ORKD=1 ORKD_BIN=$PWD/orkd ./test_orkd_transparent  # routed through orkd
 */
#include "ork_npu.h"
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
static uint32_t g = 7;
static int8_t  r8(void){ g = g*1103515245u + 12345u; return (int8_t)(((g>>16)&0x7f) - 40); }
static int8_t  r4(void){ g = g*1103515245u + 12345u; return (int8_t)(((g>>18)&0xf) - 8); }              /* int4 [-8,7] */
static ork_f16 rf(void){ g = g*1103515245u + 12345u; return (ork_f16)((float)((int)((g>>17)&0xff) - 128) / 96.0f); }

static int one_i8(ork_npu *c, int M, int K, int N){
    int8_t *A = malloc((size_t)M*K), *B = malloc((size_t)K*N);
    int32_t *C = malloc((size_t)M*N*4), *ref = malloc((size_t)M*N*4);
    g = 7; for (int i=0;i<M*K;i++) A[i]=r8(); for (int i=0;i<K*N;i++) B[i]=r8();
    for (int m=0;m<M;m++) for (int n=0;n<N;n++){ long a=0; for (int k=0;k<K;k++) a+=(long)A[m*K+k]*B[k*N+n]; ref[m*N+n]=(int)a; }
    ork_w *w = ork_mm_pack_i8(c, K, N, B);
    int bad = 0;
    if (!w){ printf("  i8 pack FAIL M=%d K=%d N=%d\n", M, K, N); bad = 1; }
    else { int rc = ork_mm_run_i8(c, w, M, A, C); ork_mm_free(c, w);
        if (rc){ printf("  i8 run FAIL M=%d K=%d N=%d rc=%d\n", M, K, N, rc); bad = 1; }
        else for (int i=0;i<M*N;i++) if (C[i]!=ref[i]){ if (bad<3) printf("  i8 MISMATCH M=%d K=%d N=%d [%d] %d!=%d\n", M,K,N,i,C[i],ref[i]); bad++; } }
    if (!bad) printf("  ok i8  M=%d K=%d N=%d\n", M, K, N);
    free(A); free(B); free(C); free(ref);
    return bad ? 1 : 0;
}
static int one_f16(ork_npu *c, int M, int K, int N){
    ork_f16 *A = malloc((size_t)M*K*2), *B = malloc((size_t)K*N*2);
    float *C = malloc((size_t)M*N*4), *ref = malloc((size_t)M*N*4);
    g = 7; for (int i=0;i<M*K;i++) A[i]=rf(); for (int i=0;i<K*N;i++) B[i]=rf();
    for (int m=0;m<M;m++) for (int n=0;n<N;n++){ float a=0; for (int k=0;k<K;k++) a+=(float)A[m*K+k]*(float)B[k*N+n]; ref[m*N+n]=a; }
    ork_w *w = ork_mm_pack(c, K, N, B);
    int bad = 0;
    if (!w){ printf("  f16 pack FAIL M=%d K=%d N=%d\n", M, K, N); bad = 1; }
    else { int rc = ork_mm_run(c, w, M, A, C); ork_mm_free(c, w);
        if (rc){ printf("  f16 run FAIL M=%d K=%d N=%d rc=%d\n", M, K, N, rc); bad = 1; }
        else for (int i=0;i<M*N;i++){ float d = fabsf(C[i]-ref[i]), tol = fabsf(ref[i])*0.03f + 0.5f;
            if (d > tol){ if (bad<3) printf("  f16 MISMATCH M=%d K=%d N=%d [%d] %.3f!=%.3f\n", M,K,N,i,C[i],ref[i]); bad++; } } }
    if (!bad) printf("  ok f16 M=%d K=%d N=%d\n", M, K, N);
    free(A); free(B); free(C); free(ref);
    return bad ? 1 : 0;
}
static int one_i4(ork_npu *c, int M, int K, int N){
    int8_t *A = malloc((size_t)M*K), *B = malloc((size_t)K*N);
    int32_t *C = malloc((size_t)M*N*4), *ref = malloc((size_t)M*N*4);
    g = 7; for (int i=0;i<M*K;i++) A[i]=r4(); for (int i=0;i<K*N;i++) B[i]=r4();
    for (int m=0;m<M;m++) for (int n=0;n<N;n++){ long a=0; for (int k=0;k<K;k++) a+=(long)A[m*K+k]*B[k*N+n]; ref[m*N+n]=(int)a; }
    ork_w *w = ork_mm_pack_i4(c, K, N, B);
    int bad = 0;
    if (!w){ printf("  i4 pack FAIL M=%d K=%d N=%d\n", M, K, N); bad = 1; }
    else { int rc = ork_mm_run_i4(c, w, M, A, C); ork_mm_free(c, w);
        if (rc){ printf("  i4 run FAIL M=%d K=%d N=%d rc=%d\n", M, K, N, rc); bad = 1; }
        else for (int i=0;i<M*N;i++) if (C[i]!=ref[i]){ if (bad<3) printf("  i4 MISMATCH M=%d K=%d N=%d [%d] %d!=%d\n", M,K,N,i,C[i],ref[i]); bad++; } }
    if (!bad) printf("  ok i4  M=%d K=%d N=%d\n", M, K, N);
    free(A); free(B); free(C); free(ref);
    return bad ? 1 : 0;
}
/* ---- SDP ops (M=8,N=64: the standalone-op geometry). Tolerant refs: loose enough that the direct NPU output
 * passes; routed is the identical NPU op so it passes identically — a routing bug yields garbage/poison. ---- */
static int chk8(const char *tag, int M, int N, const int8_t *got, const int8_t *ref, int tol){
    int bad = 0; for (int i=0;i<M*N;i++){ int d = got[i]-ref[i]; if (d<0) d=-d; if (d>tol){ if (bad<3) printf("  %s MISMATCH [%d] %d!=%d\n", tag, i, got[i], ref[i]); bad++; } }
    if (!bad) printf("  ok %-9s M=%d N=%d\n", tag, M, N); return bad ? 1 : 0;
}
static int chkf(const char *tag, int M, int N, const ork_f16 *got, const float *ref){
    int bad = 0; for (int i=0;i<M*N;i++){ float d = fabsf((float)got[i]-ref[i]), t = fabsf(ref[i])*0.03f + 0.1f; if (d>t){ if (bad<3) printf("  %s MISMATCH [%d] %.3f!=%.3f\n", tag, i, (float)got[i], ref[i]); bad++; } }
    if (!bad) printf("  ok %-9s M=%d N=%d\n", tag, M, N); return bad ? 1 : 0;
}
static int8_t clip8(long v){ return (int8_t)(v>127?127:v<-128?-128:v); }
static int sdp_silu(ork_npu *c, int gelu){
    int M=8,N=64; int8_t *in=malloc(M*N),*out=malloc(M*N),*ref=malloc(M*N); double is=0.06,os=0.06;
    g=7; for (int i=0;i<M*N;i++) in[i]=r8();
    for (int i=0;i<M*N;i++){ double x=in[i]*is, y = gelu ? 0.5*x*(1.0+erf(x/1.4142135623730951)) : x/(1.0+exp(-x)); ref[i]=clip8(lround(y/os)); }
    int rc = gelu ? ork_npu_gelu_i8(c,in,M,N,is,os,out,NULL) : ork_npu_silu_i8(c,in,M,N,is,os,out,NULL);
    int bad = rc ? (printf("  %s run FAIL rc=%d\n", gelu?"gelu":"silu", rc),1) : chk8(gelu?"gelu_i8":"silu_i8",M,N,out,ref,2);
    free(in); free(out); free(ref); return bad;
}
static int sdp_ewmul_i8(ork_npu *c){
    int M=8,N=64,mult=16384,shift=14; int8_t *a=malloc(M*N),*b=malloc(M*N),*out=malloc(M*N),*ref=malloc(M*N);
    g=7; for (int i=0;i<M*N;i++){ a[i]=(int8_t)(((g=g*1103515245u+12345u)>>20)&0x7)-3; b[i]=(int8_t)(((g=g*1103515245u+12345u)>>20)&0x7)-3; }
    for (int i=0;i<M*N;i++) ref[i]=clip8(lround((long)a[i]*b[i]*mult/(double)(1<<shift)));
    int rc = ork_npu_ewmul_i8(c,a,b,M,N,mult,shift,out,NULL);
    int bad = rc ? (printf("  ewmul_i8 run FAIL rc=%d\n", rc),1) : chk8("ewmul_i8",M,N,out,ref,1);
    free(a); free(b); free(out); free(ref); return bad;
}
static int sdp_add_i8(ork_npu *c){
    int M=8,N=64; int8_t *a=malloc(M*N),*b=malloc(M*N),*out=malloc(M*N),*ref=malloc(M*N);
    g=7; for (int i=0;i<M*N;i++) a[i]=r8(); for (int i=0;i<M*N;i++) b[i]=r8();
    for (int i=0;i<M*N;i++) ref[i]=clip8((long)a[i]+b[i]);
    int rc = ork_npu_add_i8(c,a,b,M,N,1.0,1.0,1.0,out,NULL);
    int bad = rc ? (printf("  add_i8 run FAIL rc=%d\n", rc),1) : chk8("add_i8",M,N,out,ref,1);
    free(a); free(b); free(out); free(ref); return bad;
}
static int sdp_ewmul_f16(ork_npu *c){
    int M=8,N=64; ork_f16 *a=malloc(M*N*2),*b=malloc(M*N*2),*out=malloc(M*N*2); float *ref=malloc(M*N*4);
    g=7; for (int i=0;i<M*N;i++) a[i]=rf(); for (int i=0;i<M*N;i++) b[i]=rf();
    for (int i=0;i<M*N;i++) ref[i]=(float)a[i]*(float)b[i];
    int rc = ork_npu_ewmul_f16(c,a,b,M,N,out,NULL);
    int bad = rc ? (printf("  ewmul_f16 run FAIL rc=%d\n", rc),1) : chkf("ewmul_f16",M,N,out,ref);
    free(a); free(b); free(out); free(ref); return bad;
}
static int sdp_add_f16(ork_npu *c){
    int M=8,N=64; ork_f16 *a=malloc(M*N*2),*b=malloc(M*N*2),*out=malloc(M*N*2); float *ref=malloc(M*N*4);
    g=7; for (int i=0;i<M*N;i++) a[i]=rf(); for (int i=0;i<M*N;i++) b[i]=rf();
    for (int i=0;i<M*N;i++) ref[i]=(float)a[i]+(float)b[i];
    int rc = ork_npu_add_f16(c,a,b,M,N,out,NULL);
    int bad = rc ? (printf("  add_f16 run FAIL rc=%d\n", rc),1) : chkf("add_f16",M,N,out,ref);
    free(a); free(b); free(out); free(ref); return bad;
}
int main(void){
    ork_npu *c = ork_npu_init();
    if (!c){ printf("init FAIL\n"); return 1; }
    printf("mode: %s (cores=%d)\n", getenv("ORK_USE_ORKD") ? "orkd-client (routed)" : "direct", ork_npu_cores(c));
    int bad = 0;
    bad |= one_i8 (c, 8, 128, 64);
    bad |= one_i8 (c, 1, 64, 32);
    bad |= one_i8 (c, 64, 512, 64);
    bad |= one_i8 (c, 256, 512, 64);
    bad |= one_f16(c, 8, 128, 64);
    bad |= one_f16(c, 1, 64, 32);
    bad |= one_f16(c, 64, 512, 64);
    bad |= one_i4 (c, 8, 64, 64);
    bad |= one_i4 (c, 1, 64, 64);
    bad |= one_i4 (c, 32, 128, 64);
    bad |= sdp_silu(c, 0);
    bad |= sdp_silu(c, 1);
    bad |= sdp_ewmul_i8(c);
    bad |= sdp_ewmul_f16(c);
    bad |= sdp_add_i8(c);
    bad |= sdp_add_f16(c);
    ork_npu_free(c);
    printf("%s\n", bad ? "FAILED" : "ALL OK");
    return bad ? 1 : 0;
}
