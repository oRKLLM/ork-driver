// bmm_probe — differential test for ggml_backend_ork_bmm_fp16 (the batched/dynamic MUL_MAT path
// that ORK_ATTN claims for ne[2]>1 && M>1). Mimics the NON-flash attention decomposition that
// llama.cpp builds when flash_attn is disabled (which is what ork_ppl does):
//     QK^T : mul_mat(k[DK,kv,Hkv], q[DK,nb,H])   -> [kv,nb,H]
//     A.V  : mul_mat(vt[kv,DV,Hkv], p[kv,nb,H])  -> [DV,nb,H]
// Computes each on the CPU backend (oracle) and on the ORK backend and diffs.
//
// build: g++ -O2 -std=c++17 bmm_probe.cpp -o bmm_probe -I ~/llama.cpp/ggml/include \
//          -L ~/llama.cpp/build/bin -lggml -lggml-base -lggml-cpu -Wl,-rpath,$HOME/llama.cpp/build/bin
// run:   sudo timeout -s TERM 180 env ORK_ATTN=1 ./bmm_probe <K> <N> <M> <Hkv> <rk2>

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <string>

static uint32_t rs = 987654321u;
static float frand() { rs = rs*1664525u + 1013904223u; return ((rs>>8) & 0xffff) / 32768.0f - 1.0f; }

static void fill(ggml_tensor * t) {
    const size_t n = ggml_nelements(t);
    if (t->type == GGML_TYPE_F32) { std::vector<float> b(n); for (size_t i=0;i<n;i++) b[i]=frand(); ggml_backend_tensor_set(t,b.data(),0,n*4); }
    else                          { std::vector<ggml_fp16_t> b(n); for (size_t i=0;i<n;i++) b[i]=ggml_fp32_to_fp16(frand()); ggml_backend_tensor_set(t,b.data(),0,n*2); }
}

int main(int argc, char ** argv) {
    int K   = argc > 1 ? atoi(argv[1]) : 256;   // contraction
    int N   = argc > 2 ? atoi(argv[2]) : 512;   // src0->ne[1] (output cols)
    int M   = argc > 3 ? atoi(argv[3]) : 512;   // src1->ne[1] (rows)
    int Hkv = argc > 4 ? atoi(argv[4]) : 2;
    int rk2 = argc > 5 ? atoi(argv[5]) : 8;
    int H   = Hkv*rk2;
    printf("[bmm] K=%d N=%d M=%d H=%d Hkv=%d (broadcast r2=%d)\n", K, N, M, H, Hkv, rk2);

    ggml_backend_load_all();
    ggml_backend_t cpu = ggml_backend_cpu_init();
    ggml_backend_t ork = nullptr;
    for (size_t i = 0; i < ggml_backend_reg_count(); i++) {
        ggml_backend_reg_t r = ggml_backend_reg_get(i);
        if (std::string(ggml_backend_reg_name(r)) == "ORK") ork = ggml_backend_dev_init(ggml_backend_reg_dev_get(r, 0), nullptr);
    }
    if (!ork) { fprintf(stderr, "no ORK backend\n"); return 1; }

    ggml_init_params ip = { ggml_tensor_overhead()*64 + ggml_graph_overhead(), nullptr, true };
    ggml_context * ctx = ggml_init(ip);

    ggml_tensor * a = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, K, N, Hkv);   // "weight" side (K cache / V^T cache)
    ggml_tensor * b = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, K, M, H);     // activation side (Q / P)
    ggml_tensor * o = ggml_mul_mat(ctx, a, b);                             // -> [N, M, H]
    ggml_set_name(a,"a"); ggml_set_name(b,"b"); ggml_set_name(o,"o");

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, cpu);
    if (!buf) { fprintf(stderr, "alloc failed\n"); return 1; }
    fill(a); fill(b);

    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, o);

    if (ggml_backend_graph_compute(cpu, gf) != GGML_STATUS_SUCCESS) { fprintf(stderr,"cpu failed\n"); return 1; }
    const size_t no = ggml_nelements(o);
    std::vector<float> ref(no), got(no);
    ggml_backend_tensor_get(o, ref.data(), 0, no*4);
    memset(o->data, 0xCD, ggml_nbytes(o));

    ggml_status st = ggml_backend_graph_compute(ork, gf);
    printf("[bmm] ork status=%d\n", (int)st);
    ggml_backend_tensor_get(o, got.data(), 0, no*4);

    double se=0, sr=0; float worst=0; size_t wi=0, nbad=0, fb=(size_t)-1;
    for (size_t i=0;i<no;i++) {
        float x=ref[i], y=got[i], d=fabsf(x-y);
        se+=(double)d*d; sr+=(double)x*x;
        if (d>worst){worst=d;wi=i;}
        if (d > 5e-2f*(fabsf(x)+1e-2f)+5e-2f) { nbad++; if(fb==(size_t)-1) fb=i; }
    }
    auto coord=[&](size_t i,const char*t){ printf("[bmm] %s idx=%zu -> n=%d m=%d h=%d ref=%g got=%g\n",
        t,i,(int)(i%N),(int)((i/N)%M),(int)(i/((size_t)N*M)),ref[i],got[i]); };
    printf("[bmm] elems=%zu NRMSE=%.6g maxabs=%.6g out_of_tol=%zu (%.2f%%)\n", no, sqrt(se/(sr>0?sr:1)), (double)worst, nbad, 100.0*nbad/no);
    if (fb!=(size_t)-1) coord(fb,"FIRST BAD");
    coord(wi,"WORST   ");
    // ---- MECHANISM ORACLE ----
    // Hypothesis: bmm_fp16 fills B with  B[j] = src0(j%K, j/K)  (i.e. [N,K] order) while
    // ork_mm_repack_f16 reads B as [K,N] row-major (B[k*N+n]).  Emulate exactly that mis-index
    // for head 0 and see whether `got` matches it bit-for-bit-ish.  If yes, the bug is named.
    {
        std::vector<ggml_fp16_t> a16((size_t)K*N);
        std::vector<float>       b32((size_t)K*M);
        ggml_backend_tensor_get(a, a16.data(), 0, (size_t)K*N*2);          // head 0 slice
        ggml_backend_tensor_get(b, b32.data(), 0, (size_t)K*M*4);          // head 0 slice
        double se2=0, sr2=0; float w2=0; size_t wi2=0;
        // ggml stores element (i0,i1) of a [K,N] tensor at a16[i1*K + i0] (ne[0]-contiguous), i.e. the
        // raw buffer IS B^T in [N,K] row-major order. bmm_fp16 fills B[j] = src0(j%K, j/K) = a16[j],
        // i.e. it hands that raw [N,K] buffer to ork_mm_repack_f16, which reads it as B[k*N+n].
        std::vector<float> Bdrv((size_t)K*N);
        for (size_t idx=0; idx<(size_t)K*N; idx++) Bdrv[idx] = ggml_fp16_to_fp32(a16[idx]);   // Bdrv[k*N+n]
        for (int m=0;m<M;m++) for (int n=0;n<N;n++) {
            double acc=0; for (int k=0;k<K;k++) acc += (double)Bdrv[(size_t)k*N+n] * (double)b32[(size_t)m*K+k];
            size_t i=(size_t)m*N+n; double d=acc-got[i]; se2+=d*d; sr2+=(double)got[i]*got[i];
            if (fabs(d)>w2){w2=(float)fabs(d);wi2=i;}
        }
        printf("[bmm] MECHANISM(head0) emulated-transposed-B vs got: NRMSE=%.6g maxabs=%.6g -> %s\n",
               sqrt(se2/(sr2>0?sr2:1)), (double)w2, (sqrt(se2/(sr2>0?sr2:1)) < 1e-2) ? "MATCHES (transposed-B confirmed)" : "no match");
        (void)wi2;
    }
    printf("[bmm] %s\n", nbad==0 ? "PASS" : "FAIL");
    ggml_free(ctx); ggml_backend_buffer_free(buf); ggml_backend_free(ork); ggml_backend_free(cpu);
    return nbad==0?0:2;
}
