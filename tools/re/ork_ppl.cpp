/* ork_ppl.cpp — teacher-forced perplexity through the SAME backend/env path as ork_bench.
 *
 *   ork_ppl <model.gguf> <text> [window=512] [ubatch=512]
 *
 * WHY THIS EXISTS RATHER THAN llama-perplexity. Quality has to be measured on the path that is actually
 * under test, and with a batch shape the NPU survives. `llama-perplexity` defaults to n_batch=2048,
 * n_seq=4 -> M=2048 -> wide colsplit -> RKNPU_SUBMIT timeouts and self-heal thrash that can wedge the
 * kernel NPU driver (recover with `sudo reboot`, then re-pin governors — they reset). So ubatch is a
 * first-class argument here and defaults to 512, and the orkpack resolution below is copied from
 * ork_bench so a run measures the tier you think it measures.
 *
 * WHAT IT COMPUTES. Non-overlapping windows of `window` tokens. Each window is one llama_decode with
 * logits enabled at EVERY position, KV cleared between windows, so window w's score never depends on
 * w-1 — the standard chunked estimator, and the reason the number is comparable across runs of
 * different length. Within a window, logits at position i predict token i+1; position 0 is unscorable
 * (no left context) so a W-token window contributes W-1 scored tokens. Reported PPL = exp(mean NLL)
 * over all scored tokens, log-sum-exp in double so a 250k-entry vocab does not lose the tail.
 *
 * MEMORY. All-position logits cost window * n_vocab * 4 B. Qwen3.5's vocab is ~248k, so the 512
 * default is ~508 MB — fine on a 32 GB board, but drop `window` on a smaller one rather than wondering
 * why the allocator failed.
 *
 * READING THE RESULT. PPL is only comparable between runs that share model, text, window AND
 * tokenizer. Compare tiers by changing ONLY the env (ORK_QUANT / ORK_MIXED_W4A4 / ...) and holding the
 * rest fixed; an absolute value means little on its own. The tier actually in force is printed by the
 * backend's own banner ("ork-driver <ver> (W8A8 / W4A4+Had)") — read it, do not assume.
 */
#include "llama.h"
#include "ggml-backend.h"
#include "ggml-ork.h"
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <unistd.h>
#include <sys/stat.h>
#ifdef _OPENMP
#include <omp.h>

#endif

/* ork residence query (Tier 15 stage 3). WEAK + extern "C": a build without ggml-ork links fine and simply
 * skips the report. These MUST be at file scope with C linkage — declaring them inside a function in C++
 * mangles the names, the weak symbols resolve to null, and the report silently never prints (observed). */
extern "C" void ggml_backend_ork_residence(int *, size_t *) __attribute__((weak));
extern "C" int  ggml_backend_ork_residence_report(void) __attribute__((weak));
extern "C" int  ggml_backend_ork_preload(void) __attribute__((weak));
extern "C" bool ggml_backend_ork_extract_gguf(const char *, const char *) __attribute__((weak));


/* How many threads should the ggml threadpool get?
 *
 * NOT a hardcoded 4. That number came from RK3588, where it is right for a specific reason: the SoC is
 * heterogeneous (4x A76 + 4x A55) and ggml's threadpool is BARRIER-synchronised, so a little core that
 * takes 3x as long to finish its slice holds every big core waiting at the barrier — using all 8 is
 * measurably slower than using the 4 fast ones. But the same constant on a homogeneous 16-core host
 * leaves 12 cores idle, which is what it did here.
 *
 * So derive it: group the CPUs by their maximum frequency and take the size of the FASTEST group. On a
 * heterogeneous part that is the big-core count (4 on RK3588); on a homogeneous one every CPU is in one
 * group and it degrades to the total, which is the right answer when the cores are interchangeable.
 * Falls back to the online CPU count where cpufreq is unavailable. ORK_PPL_THREADS overrides. */
static int ork_ppl_threads(void){
    if (const char* e = getenv("ORK_PPL_THREADS")) { int v = atoi(e); if (v > 0) return v; }
    const long n = sysconf(_SC_NPROCESSORS_ONLN);
    if (n < 1) return 4;
    long best = -1, count = 0, known = 0;
    for (long i = 0; i < n; i++) {
        char path[128];
        snprintf(path, sizeof path, "/sys/devices/system/cpu/cpu%ld/cpufreq/cpuinfo_max_freq", i);
        FILE* f = fopen(path, "r");
        if (!f) continue;
        long khz = 0;
        if (fscanf(f, "%ld", &khz) == 1 && khz > 0) {
            known++;
            if (khz > best) { best = khz; count = 1; }
            else if (khz == best) count++;
        }
        fclose(f);
    }
    if (known == n && count > 0) return (int) count;   /* every CPU reported: trust the fastest-group size */
    return (int) n;                                     /* no cpufreq (VM, container): cores are interchangeable */
}

int main(int argc, char** argv){
    if (argc < 3){
        fprintf(stderr,"usage: %s <model.gguf> <text> [window=512] [ubatch=512] [maxwin=0]\n"
                       "  window : tokens per scored chunk (KV cleared between chunks)\n"
                       "  ubatch : PIN THIS. large M wedges the NPU; 512 is the validated default\n"
                       "  maxwin : cap on scored windows (0 = whole text). BOUND THIS for A/B runs — an\n"
                       "           unbounded run on a long text outlives its `timeout`, and a reaped\n"
                       "           ork_ppl can leave a zombie holding the NPU (costs a reboot).\n", argv[0]);
        return 2;
    }
    const char* model_path = argv[1];
    /* A .orkpack is now a complete artifact (Tier 15 stage 2): it embeds the GGUF header/KV and every
     * tensor the pack does not own. Accept one directly — extract the loadable (sparse) gguf beside it on
     * first use, and point the backend at the pack. Beats making the caller carry, and keep in sync, two
     * files that must match. The extracted gguf is a derived cache: delete it and it comes back. */
    std::string _mp_hold;
    if (strlen(model_path) > 8 && strcmp(model_path + strlen(model_path) - 8, ".orkpack") == 0) {
        const std::string pack = model_path, gguf = pack + ".gguf";
        if (!getenv("ORK_ORKPACK_PATH")) setenv("ORK_ORKPACK_PATH", pack.c_str(), 0);
        /* The extracted gguf has HOLES where the pack owns tensors, so any op ggml computes itself would
         * read zeros. Tell the backend, so it refuses to decline a pack-owned matmul. Detecting this in
         * buffer_set_tensor does NOT work — that hook never fires for model weights (they live in CPU
         * buffers, not ORK_Weights), which is the same reason the eager-load hook was unreachable. */
        setenv("ORK_SOURCE_IS_STUB", "1", 1);
        struct stat st;
        if (stat(gguf.c_str(), &st) != 0) {
            if (!ggml_backend_ork_extract_gguf || !ggml_backend_ork_extract_gguf(pack.c_str(), gguf.c_str())) {
                fprintf(stderr, "[ork_ppl] %s carries no embedded metadata — pass the source .gguf instead\n",
                        pack.c_str());
                return 2;
            }
        }
        _mp_hold = gguf; model_path = _mp_hold.c_str();
        fprintf(stderr, "[ork_ppl] pack-native: %s (+ extracted %s)\n", pack.c_str(), _mp_hold.c_str());
    }
    int W  = argc>3 ? atoi(argv[3]) : 512;
    int UB = argc>4 ? atoi(argv[4]) : 512;
    int MAXW = argc>5 ? atoi(argv[5]) : 0;
    if (W  < 2)  { fprintf(stderr,"window must be >= 2 (position 0 is unscorable)\n"); return 2; }
    if (UB < 1)  UB = W;
    if (UB > W)  UB = W;

    FILE* f=fopen(argv[2],"rb"); if(!f){ fprintf(stderr,"cannot open %s\n",argv[2]); return 2; }
    fseek(f,0,SEEK_END); long fn=ftell(f); fseek(f,0,SEEK_SET);
    std::vector<char> txt(fn+1); size_t rd=fread(txt.data(),1,fn,f); txt[rd]=0; fclose(f);

    llama_backend_init();

    /* Orkpack resolution — same rule as ork_bench so the measured tier is the intended one. A non-default
     * ORK_QUANT yields different tile content and so a distinct .q<N> pack; the default keeps the bare
     * name the backend derives on its own. We do NOT build a missing pack here (that is ork_bench's job):
     * building would run the forward in WRITE mode, which is a different path from the one being scored. */
    std::string orkpack;
    if (const char* pp = getenv("ORK_ORKPACK_PATH")) orkpack = pp;
    else {
        std::string base = model_path;
        if (base.size() > 5 && base.compare(base.size()-5, 5, ".gguf") == 0) base.resize(base.size()-5);
        const char* q = getenv("ORK_QUANT");
        bool nondefault = q && *q && q[0] != '8';
        orkpack = base + (nondefault ? (std::string(".q") + q[0]) : std::string("")) + ".orkpack";
        if (nondefault) setenv("ORK_ORKPACK_PATH", orkpack.c_str(), 1);
    }
    if (ggml_backend_ork_orkpack_valid(orkpack.c_str()))
        fprintf(stderr,"[ork_ppl] orkpack: %s (READ mode)\n", orkpack.c_str());
    else
        fprintf(stderr,"[ork_ppl] WARNING: no valid orkpack at %s — the backend will pack inline (WRITE mode,\n"
                       "[ork_ppl]          M=1) which is NOT the path you are scoring. Run ork_bench once first.\n",
                orkpack.c_str());

    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 0;   // ggml-ork intercepts MUL_MAT; offloading would send it to Vulkan instead
    llama_model* model = llama_model_load_from_file(model_path, mp);
    if(!model){ fprintf(stderr,"model load FAILED\n"); return 1; }
    const llama_vocab* vocab = llama_model_get_vocab(model);
    const int n_vocab = llama_vocab_n_tokens(vocab);

    std::vector<llama_token> toks(rd + 8);
    int nt = llama_tokenize(vocab, txt.data(), (int)rd, toks.data(), (int)toks.size(), true, true);
    if(nt<=0){ fprintf(stderr,"tokenize failed (%d)\n",nt); return 1; }
    toks.resize(nt);
    if (nt < W){ fprintf(stderr,"text is %d tokens, shorter than window %d — lower the window\n", nt, W); return 2; }
    int nwin = nt / W;
    if (MAXW > 0 && nwin > MAXW) nwin = MAXW;   // paired A/B needs identical, bounded work per arm

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx    = W + 8;
    cp.n_batch  = W;
    cp.n_ubatch = UB;
    const int nthr = ork_ppl_threads();
    cp.n_threads = nthr; cp.n_threads_batch = nthr;
    cp.flash_attn_type = getenv("ORK_PPL_FA") ? LLAMA_FLASH_ATTN_TYPE_ENABLED : LLAMA_FLASH_ATTN_TYPE_DISABLED;
    /* No logits_all in this API — per-position output is requested via batch.logits[i] below. */
    llama_context* ctx = llama_init_from_model(model, cp);
    if(!ctx){ fprintf(stderr,"ctx init FAILED\n"); return 1; }

    fprintf(stderr,"[ork_ppl] %d tokens, %d window(s) of %d%s, ubatch=%d, threads=%d, vocab=%d (logits %.0f MiB/window)\n",
            nt, nwin, W, (MAXW > 0 ? " [capped]" : ""), UB, nthr, n_vocab, (double)W*n_vocab*4/1048576.0);

    llama_batch b = llama_batch_init(W, 0, 1);

    /* PRELOAD before the clock (Tier 15 stage 3). Weights are otherwise materialised by the first matmul
     * that touches them, so the load lands inside whatever is being timed: on 27B G=512 a single-window
     * run reported 220 s "scored" that was almost entirely weight load. Preload is OP-LESS -- it runs no
     * graph and issues no submits -- so it needs no warmup decode. (A warmup was the earlier approach and
     * is kept behind ORK_PPL_WARM for comparison; a 1-token warmup is useless here anyway, since M==1 is
     * declined to the CPU and materialises nothing.) */
    if (ggml_backend_ork_preload && !getenv("ORK_PPL_NOPRELOAD")) {
        const int64_t tp = llama_time_us();
        const int n = ggml_backend_ork_preload();
        fprintf(stderr, "[ork_ppl] preload: %d weights in %.1f s (excluded from the scored time below)\n",
                n, (llama_time_us() - tp) / 1e6);
    }
    if (getenv("ORK_PPL_WARM")) {
        const int64_t tw = llama_time_us();
        b.n_tokens = W;
        for (int i = 0; i < W; ++i){ b.token[i] = toks[i]; b.pos[i] = i; b.n_seq_id[i] = 1; b.seq_id[i][0] = 0; b.logits[i] = 0; }
        b.logits[W-1] = 1;
        if (llama_decode(ctx, b) != 0) fprintf(stderr, "[ork_ppl] warmup decode failed (continuing)\n");
        llama_memory_clear(llama_get_memory(ctx), true);
        fprintf(stderr, "[ork_ppl] warmup: %.1f s (excluded)\n", (llama_time_us() - tw) / 1e6);
    }
    if (ggml_backend_ork_residence) {
        int nw = 0; size_t rb = 0;
        ggml_backend_ork_residence(&nw, &rb);
        fprintf(stderr, "[ork_ppl] resident: %d weights, %.2f GiB before the clock starts\n",
                nw, rb / (1024.0*1024.0*1024.0));
    }
    if (ggml_backend_ork_residence_report) ggml_backend_ork_residence_report();

    double nll = 0.0; long scored = 0; int64_t t0 = llama_time_us();

    for (int w = 0; w < nwin; ++w){
        const llama_token* chunk = toks.data() + (size_t)w*W;
        llama_memory_clear(llama_get_memory(ctx), true);   // windows are independent by construction
        b.n_tokens = W;
        for (int i = 0; i < W; ++i){
            b.token[i]    = chunk[i];
            b.pos[i]      = i;
            b.n_seq_id[i] = 1;
            b.seq_id[i][0]= 0;
            b.logits[i]   = 1;                              // need every position to score i -> i+1
        }
        if (llama_decode(ctx, b) != 0){
            fprintf(stderr,"[ork_ppl] decode FAILED on window %d/%d (W=%d ubatch=%d)\n", w+1, nwin, W, UB);
            llama_batch_free(b); llama_free(ctx); llama_model_free(model); llama_backend_free(); return 3;
        }
        /* Scoring is ~250k exp() per position and is a third of a run's wall clock. Positions are
         * independent, so fan them out; the reduction keeps the sum deterministic ENOUGH for A/B (each
         * position's own log-sum-exp is computed identically, only the outer accumulation reorders, and
         * that is fp-associativity noise many orders below the differences under test). */
        double wnll = 0.0; int wscored = 0; int bad = -1;
        #pragma omp parallel for schedule(static) reduction(+:wnll,wscored)
        for (int i = 0; i < W - 1; ++i){
            const float* lg = llama_get_logits_ith(ctx, i);
            if (!lg){ bad = i; continue; }
            /* log_softmax in double: subtract the max before exp so a 250k vocab cannot overflow, and
             * the tail still contributes (a float accumulator loses it). */
            double mx = lg[0];
            for (int v = 1; v < n_vocab; ++v) if (lg[v] > mx) mx = lg[v];
            double sum = 0.0;
            for (int v = 0; v < n_vocab; ++v) sum += exp((double)lg[v] - mx);
            wnll += (mx + log(sum)) - (double)lg[chunk[i+1]];   // -log p(next)
            ++wscored;
        }
        if (bad >= 0){ fprintf(stderr,"[ork_ppl] no logits at %d — batch.logits[i] not honoured?\n", bad); return 3; }
        nll += wnll; scored += wscored;
        if (nwin > 1) fprintf(stderr,"[ork_ppl] window %d/%d  running PPL = %.4f\n", w+1, nwin, exp(nll/scored));
    }
    double secs = (llama_time_us()-t0)/1e6;

    printf("[ork_ppl] PPL = %.4f   (mean NLL %.6f over %ld scored tokens, %d x %d-token windows, ubatch=%d, %.1f s)\n",
           exp(nll/scored), nll/scored, scored, nwin, W, UB, secs);

    llama_batch_free(b);
    llama_free(ctx); llama_model_free(model); llama_backend_free();
    return 0;
}
