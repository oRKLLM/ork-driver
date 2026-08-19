// fa_probe — differential test for ggml_backend_ork_flash_attn_ext.
// Builds ONE GGML_OP_FLASH_ATTN_EXT node with model-realistic shapes, computes it on the CPU
// backend (oracle) and on the ORK backend, and diffs elementwise. Reports the first/worst
// divergence with its (dv, head, query) coordinates so a layout/mask/GQA bug names itself.
//
// build (on the board):
//   g++ -O2 -std=c++17 fa_probe.cpp -o fa_probe \
//       -I ~/llama.cpp/ggml/include -L ~/llama.cpp/build/bin \
//       -lggml -lggml-base -Wl,-rpath,$HOME/llama.cpp/build/bin
// run:
//   ORK_ATTN=1 ORK_ATTN_SM_CPU=1 ORK_ATTN_CPU=3 sudo -E ./fa_probe 256 256 2 8 512 64

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

static uint32_t rs = 1234567u;
static float frand() { rs = rs*1664525u + 1013904223u; return ((rs>>8) & 0xffff) / 32768.0f - 1.0f; }

static void fill_f32(ggml_tensor * t, float amp) {
    const size_t n = ggml_nelements(t);
    std::vector<float> b(n);
    for (size_t i = 0; i < n; i++) b[i] = amp*frand();
    ggml_backend_tensor_set(t, b.data(), 0, n*sizeof(float));
}
static void fill_f16(ggml_tensor * t, float amp) {
    const size_t n = ggml_nelements(t);
    std::vector<ggml_fp16_t> b(n);
    for (size_t i = 0; i < n; i++) b[i] = ggml_fp32_to_fp16(amp*frand());
    ggml_backend_tensor_set(t, b.data(), 0, n*sizeof(ggml_fp16_t));
}

int main(int argc, char ** argv) {
    int DK  = argc > 1 ? atoi(argv[1]) : 256;
    int DV  = argc > 2 ? atoi(argv[2]) : 256;
    int Hkv = argc > 3 ? atoi(argv[3]) : 2;
    int rk2 = argc > 4 ? atoi(argv[4]) : 8;
    int kv  = argc > 5 ? atoi(argv[5]) : 512;
    int nb  = argc > 6 ? atoi(argv[6]) : 64;
    int H   = Hkv*rk2;
    // llama.cpp pads the mask's token dim; mimic it so nb != mask->ne[1].
    int mrows = ((nb + 63)/64)*64;

    printf("[probe] DK=%d DV=%d H=%d Hkv=%d kv=%d nb=%d (mask rows=%d)\n", DK, DV, H, Hkv, kv, nb, mrows);

    ggml_backend_load_all();

    ggml_backend_t cpu = ggml_backend_cpu_init();
    if (!cpu) { fprintf(stderr, "no cpu backend\n"); return 1; }

    ggml_backend_t ork = nullptr;
    for (size_t i = 0; i < ggml_backend_reg_count(); i++) {
        ggml_backend_reg_t r = ggml_backend_reg_get(i);
        if (std::string(ggml_backend_reg_name(r)) == "ORK") {
            ork = ggml_backend_dev_init(ggml_backend_reg_dev_get(r, 0), nullptr);
        }
    }
    if (!ork) { fprintf(stderr, "no ORK backend registered\n"); return 1; }
    printf("[probe] ork backend: %s\n", ggml_backend_name(ork));

    ggml_init_params ip = { /*mem_size*/ ggml_tensor_overhead()*64 + ggml_graph_overhead(), nullptr, true };
    ggml_context * ctx = ggml_init(ip);

    ggml_tensor * q = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, DK, nb, H,   1);
    ggml_tensor * k = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, DK, kv, Hkv, 1);
    ggml_tensor * v = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, DV, kv, Hkv, 1);
    ggml_tensor * m = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, kv, mrows, 1, 1);
    ggml_set_name(q, "q"); ggml_set_name(k, "k"); ggml_set_name(v, "v"); ggml_set_name(m, "mask");

    ggml_tensor * out = ggml_flash_attn_ext(ctx, q, k, v, m, 1.0f/sqrtf((float)DK), 0.0f, 0.0f);
    ggml_flash_attn_ext_set_prec(out, GGML_PREC_F32);
    ggml_set_name(out, "out");

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, cpu);
    if (!buf) { fprintf(stderr, "alloc failed\n"); return 1; }

    fill_f32(q, 1.0f);
    fill_f16(k, 1.0f);
    fill_f16(v, 1.0f);
    // causal mask over the last nb positions of a kv-length cache, llama.cpp style:
    // row i is allowed to see cells 0 .. (kv-nb+i); padded rows are fully masked.
    {
        std::vector<ggml_fp16_t> mb((size_t)kv*mrows);
        const float ninf = -INFINITY;
        for (int i = 0; i < mrows; i++)
            for (int j = 0; j < kv; j++) {
                bool ok = (i < nb) && (j <= kv - nb + i);
                mb[(size_t)i*kv + j] = ggml_fp32_to_fp16(ok ? 0.0f : ninf);
            }
        ggml_backend_tensor_set(m, mb.data(), 0, mb.size()*sizeof(ggml_fp16_t));
    }

    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out);

    // ---- oracle: CPU backend ----
    if (ggml_backend_graph_compute(cpu, gf) != GGML_STATUS_SUCCESS) { fprintf(stderr, "cpu compute failed\n"); return 1; }
    const size_t no = ggml_nelements(out);
    std::vector<float> ref(no), got(no);
    ggml_backend_tensor_get(out, ref.data(), 0, no*sizeof(float));

    memset(out->data, 0xCD, ggml_nbytes(out));   // poison so a non-write shows up

    // ---- under test: ORK backend ----
    ggml_status st = ggml_backend_graph_compute(ork, gf);
    printf("[probe] ork graph_compute status=%d\n", (int)st);
    ggml_backend_tensor_get(out, got.data(), 0, no*sizeof(float));

    // ---- diff ----
    double se = 0, sr = 0; float worst = 0; size_t worst_i = 0; size_t nbad = 0, first_bad = (size_t)-1;
    for (size_t i = 0; i < no; i++) {
        float a = ref[i], b = got[i];
        float d = fabsf(a-b);
        se += (double)d*d; sr += (double)a*a;
        if (d > worst) { worst = d; worst_i = i; }
        float tol = 5e-3f*(fabsf(a)+1e-3f) + 5e-3f;
        if (d > tol) { nbad++; if (first_bad == (size_t)-1) first_bad = i; }
    }
    auto coord = [&](size_t i, const char * tag) {
        int d = (int)(i % DV), h = (int)((i / DV) % H), mm = (int)((i / ((size_t)DV*H)) % nb);
        printf("[probe] %s idx=%zu -> dv=%d head=%d query=%d  ref=%g got=%g\n", tag, i, d, h, mm, ref[i], got[i]);
    };
    printf("[probe] elems=%zu  NRMSE=%.6g  maxabs=%.6g  out_of_tol=%zu (%.2f%%)\n",
           no, sqrt(se/(sr>0?sr:1)), (double)worst, nbad, 100.0*nbad/no);
    if (first_bad != (size_t)-1) coord(first_bad, "FIRST BAD");
    coord(worst_i, "WORST   ");

    // Per-head and per-query NRMSE, to name a GQA / layout bug directly.
    printf("[probe] per-head NRMSE:");
    for (int h = 0; h < H; h++) {
        double e=0, r=0;
        for (int mm = 0; mm < nb; mm++) for (int d = 0; d < DV; d++) {
            size_t i = ((size_t)mm*H + h)*DV + d; double dd = ref[i]-got[i]; e += dd*dd; r += (double)ref[i]*ref[i];
        }
        printf(" h%d=%.3g", h, sqrt(e/(r>0?r:1)));
    }
    printf("\n[probe] per-query NRMSE (first 8, last 8):");
    for (int mm = 0; mm < nb; mm++) {
        if (mm >= 8 && mm < nb-8) continue;
        double e=0, r=0;
        for (int h = 0; h < H; h++) for (int d = 0; d < DV; d++) {
            size_t i = ((size_t)mm*H + h)*DV + d; double dd = ref[i]-got[i]; e += dd*dd; r += (double)ref[i]*ref[i];
        }
        printf(" q%d=%.3g", mm, sqrt(e/(r>0?r:1)));
    }
    printf("\n[probe] %s\n", nbad == 0 ? "PASS" : "FAIL");

    ggml_free(ctx);
    ggml_backend_buffer_free(buf);
    ggml_backend_free(ork);
    ggml_backend_free(cpu);
    return nbad == 0 ? 0 : 2;
}
