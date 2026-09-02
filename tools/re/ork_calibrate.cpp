/* ork_calibrate — measure the CPU/NPU routing threshold END-TO-END and store it in a .orkpack.
 *
 * WHY THIS IS A TOOL AND NOT PART OF PACKING. The threshold is where the NPU's fixed submit floor stops
 * dominating, and the built-in default (8) is measured but on one board with one model. The obvious place
 * to calibrate is at pack time, inside the backend — and that was tried and does not work. A per-shape
 * microbenchmark (time one matmul on each path, per (K,N)) MISSES the dominant cost at small M, because
 * that cost is graph-level, not per-node: declining every node yields 1 graph split, accepting yields 133,
 * and those 132 backend boundaries with their tensor copies are invisible to any single-matmul harness.
 * Measured, thresholds from that harness regressed M<=8 by up to 1.52x end-to-end. Charging the real
 * per-node dequant (the one genuinely asymmetric term) did not close the gap.
 *
 * The only instrument that measures the real quantity is a real forward pass, and only a CALLER can run
 * one, because only the caller controls the batch size. Hence a tool.
 *
 * WHAT IT MEASURES. For each M on a ladder, decode a batch of M tokens twice — once with routing forced to
 * the CPU, once to the NPU — and take the first M where the NPU wins and KEEPS winning at the next rung.
 * Requiring it to hold twice stops a single noisy sample setting a threshold the shape cannot sustain; the
 * CPU path has its own blocking thresholds (it changes character at M=32), so the ratio is not monotone
 * and a bisection would be wrong.
 *
 * The result is ONE number for the whole model, not per-shape: an end-to-end run cannot attribute its time
 * to a single (K,N), and storing per-shape values it did not measure would be a lie in the file format.
 *
 * PROVENANCE. What gets stored is stamped with the machine state it was measured under (CPU max clock, big
 * cores, governors, kernel) and is IGNORED on load by any machine that no longer matches. A pack outlives
 * the state it was calibrated under: raising this board's A76s 2304->2400 MHz silently invalidated every
 * earlier calibration, which is exactly the failure the stamp prevents.
 *
 *   ork_calibrate <model.gguf|pack.orkpack> [reps=3]
 *
 * Exits 0 on success (threshold written), nonzero otherwise.
 */
#include "llama.h"
#include "ggml-ork.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <sys/stat.h>

extern "C" void ggml_backend_ork_set_min_m(int) __attribute__((weak));
extern "C" bool ggml_backend_ork_write_calib(const char *, int) __attribute__((weak));
extern "C" bool ggml_backend_ork_extract_gguf(const char *, const char *) __attribute__((weak));

static const int LADDER[] = { 1, 2, 4, 8, 16, 24, 32, 48, 64 };
static const int NL = (int) (sizeof LADDER / sizeof LADDER[0]);

/* One timed decode of an M-token batch. Returns microseconds, or -1 on failure. */
static double decode_us(llama_context * ctx, int M, int reps) {
    std::vector<llama_token> toks((size_t) M, 1);
    double best = 1e30;
    for (int r = 0; r < reps; r++) {
        llama_memory_clear(llama_get_memory(ctx), true);
        llama_batch b = llama_batch_init(M, 0, 1);
        b.n_tokens = M;
        for (int i = 0; i < M; i++) { b.token[i]=toks[i]; b.pos[i]=i; b.n_seq_id[i]=1; b.seq_id[i][0]=0; b.logits[i]=(i==M-1); }
        const int64_t t0 = llama_time_us();
        const int rc = llama_decode(ctx, b);
        const double d = (double) (llama_time_us() - t0);
        llama_batch_free(b);
        if (rc != 0) return -1;
        if (d < best) best = d;
    }
    return best;
}

int main(int argc, char ** argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s <model.gguf|pack.orkpack> [reps=3]\n", argv[0]); return 2; }
    if (!ggml_backend_ork_set_min_m || !ggml_backend_ork_write_calib) {
        fprintf(stderr, "ork_calibrate: this build has no ork backend calibration API\n"); return 2;
    }
    const int reps = argc > 2 ? atoi(argv[2]) : 3;

    /* Accept a pack directly, as ork_ppl does: extract the sparse gguf beside it and point the backend at
     * the pack. The pack is what we are going to write the answer into, so remember its path. */
    std::string model = argv[1], pack;
    if (model.size() > 8 && model.compare(model.size()-8, 8, ".orkpack") == 0) {
        pack = model;
        const std::string gguf = pack + ".gguf";
        setenv("ORK_ORKPACK_PATH", pack.c_str(), 0);
        setenv("ORK_SOURCE_IS_STUB", "1", 1);
        struct stat st;
        if (stat(gguf.c_str(), &st) != 0 &&
            !(ggml_backend_ork_extract_gguf && ggml_backend_ork_extract_gguf(pack.c_str(), gguf.c_str()))) {
            fprintf(stderr, "ork_calibrate: %s carries no embedded metadata — pass the source .gguf\n", pack.c_str());
            return 2;
        }
        model = gguf;
    } else if (const char * p = getenv("ORK_ORKPACK_PATH")) {
        pack = p;
    }
    if (pack.empty()) { fprintf(stderr, "ork_calibrate: no .orkpack to write to (pass one, or set ORK_ORKPACK_PATH)\n"); return 2; }

    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 99;
    llama_model * m = llama_model_load_from_file(model.c_str(), mp);
    if (!m) { fprintf(stderr, "ork_calibrate: cannot load %s\n", model.c_str()); return 2; }
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = 512; cp.n_batch = 64; cp.n_ubatch = 64;
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;

    /* ONE CONTEXT PER ROUTING MODE, not one for the whole sweep. ggml's scheduler decides which backend
     * owns each node ONCE, when the graph is planned at context creation (sched_reserve); calling
     * set_min_m() afterwards changes nothing, because the plan is already fixed. The first version of this
     * tool did exactly that and measured the same plan twice -- CPU and NPU times agreed to within 0.05% at
     * every M, and the "winner" was pure noise. Re-planning means recreating the context, which is cheap
     * because the model stays loaded. */
    double cpu_us[NL], npu_us[NL];
    for (int i = 0; i < NL; i++) { cpu_us[i] = npu_us[i] = -1; }

    for (int mode = 0; mode < 2; mode++) {
        ggml_backend_ork_set_min_m(mode == 0 ? (1 << 20) : 1);      // 0 = force CPU, 1 = force NPU
        llama_context * c = llama_init_from_model(m, cp);
        if (!c) { fprintf(stderr, "ork_calibrate: cannot create context\n"); llama_model_free(m); return 2; }
        for (int i = 0; i < NL; i++) {
            const int M = LADDER[i];
            if (M > (int) cp.n_batch) break;
            const double d = decode_us(c, M, reps);
            if (d < 0) { fprintf(stderr, "  decode failed at M=%d\n", M); break; }
            (mode == 0 ? cpu_us : npu_us)[i] = d;
        }
        llama_free(c);
    }
    ggml_backend_ork_set_min_m(-1);

    fprintf(stderr, "ork_calibrate: %d rep(s) per point (min-of-reps)\n", reps);
    fprintf(stderr, "  %-5s %12s %12s   %s\n", "M", "cpu(us)", "npu(us)", "winner");
    int first = 0, chosen = 0;
    for (int i = 0; i < NL && !chosen; i++) {
        if (cpu_us[i] < 0 || npu_us[i] < 0) break;
        const double c = cpu_us[i], n = npu_us[i];
        fprintf(stderr, "  %-5d %12.0f %12.0f   %s (%.2fx)\n", LADDER[i], c, n,
                n < c ? "NPU" : "CPU", n < c ? c / n : n / c);
        /* Must hold for two consecutive rungs: the CPU path has its own blocking thresholds (it changes
         * character at M=32), so the ratio is not monotone and one noisy sample must not set the answer. */
        if (n < c) { if (first == 0) first = LADDER[i]; else chosen = first; }
        else first = 0;
    }

    llama_model_free(m); llama_backend_free();

    if (!chosen) { fprintf(stderr, "ork_calibrate: no sustained crossover found — leaving the pack alone\n"); return 1; }
    fprintf(stderr, "ork_calibrate: crossover at M>=%d\n", chosen);
    return ggml_backend_ork_write_calib(pack.c_str(), chosen) ? 0 : 1;
}
