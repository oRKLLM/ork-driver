/* ork_bench.cpp — minimal open-stack perf harness, the mirror of rkllm_bench.cpp. Drives the llama.cpp C
 * API directly (ggml-ork backend intercepts MUL_MAT, so ORK_FFN_CHAIN etc. apply) so we control EXACTLY:
 * the prefill batch size (n_ubatch — the lever that decides single- vs multi-core prefill), warmup, and
 * timing — none of which llama-bench exposes. Reports prefill + decode tok/s from one clock, same shape as
 * the rkllm side, for a true apples-to-apples comparison.
 *   ork_bench <model.gguf> <promptfile> [P=128] [G=64] [ubatch=P]
 * Env: everything the ggml-ork backend reads (ORK_FFN_CHAIN, ORK_PERSIST, ORK_PROFILE, ...).
 *
 * DFlash mode (a benchmark OPERATING MODE — the canonical harness for DFlash speculative decode, so it
 * shares this file's NPU engagement + timing rather than an ad-hoc harness):
 *   ORK_DFLASH_DRAFT=<draft.gguf> [ORK_DFLASH_BLOCK=16] ork_bench <target.gguf> <promptfile> [P] [G]
 * Runs the skip-ahead loop (draft a block -> verify all B on the target in ONE M=B forward -> accept the
 * longest greedy prefix + bonus -> commit), reporting acceptance-length tau + decode tok/s. CRITICAL: like
 * the normal path it loads with n_gpu_layers=0 so weights engage the NPU via ggml-ork MUL_MAT interception,
 * NOT the Mali GPU (n_gpu_layers>0 offloads to Vulkan when it is in the runtime -> ~14x slower).
 */
#include "llama.h"
#include "ggml-backend.h"
#include "ggml-ork.h"
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

// DFlash extension API (exported extern "C" from libllama; see src/llama-ext.h).
extern "C" {
    void            llama_set_embeddings_nextn      (llama_context *, bool value, bool masked);
    float *         llama_get_embeddings_nextn      (llama_context *);
    void            llama_set_embeddings_layer_inp  (llama_context *, uint32_t lid, bool value);
    float *         llama_get_embeddings_layer_inp  (llama_context *, uint32_t lid);
    void            llama_set_dflash_context        (llama_context *, const float * buf, int32_t len, const int32_t * pos);
    const int32_t * llama_model_target_layer_ids    (const llama_model *);
    uint32_t        llama_model_target_layer_ids_n  (const llama_model *);
}

static llama_token argmax(const float* v, int n){ int b=0; float m=v[0]; for(int i=1;i<n;i++) if(v[i]>m){m=v[i];b=i;} return b; }

// Decode `n` tokens at the given positions in sequence `seq` (all outputs). Raw-batch (no common/).
static bool decode_batch(llama_context* c, const llama_token* tk, const int32_t* pos, int n, int seq){
    llama_batch b = llama_batch_init(n, 0, 1);
    b.n_tokens = n;
    for(int i=0;i<n;i++){ b.token[i]=tk[i]; b.pos[i]=pos[i]; b.n_seq_id[i]=1; b.seq_id[i][0]=seq; b.logits[i]=1; }
    int rc = llama_decode(c, b);
    llama_batch_free(b);
    return rc==0;
}

// ── DFlash speculative-decode benchmark mode ─────────────────────────────────────────────────────
static int run_dflash_bench(const char* target_path, const char* draft_path,
                            const std::vector<char>& ptxt, size_t rd, int P, int G){
    int block_size = 16; if(const char* e=getenv("ORK_DFLASH_BLOCK")) block_size=atoi(e);

    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 0;   // NPU via ggml-ork MUL_MAT interception, NOT Mali/Vulkan offload
    // EXCLUDE the Vulkan/Mali GPU device: with Vulkan in the runtime it both (a) grabs the DFlash encode
    // ops and aborts (ggml-vulkan descriptor_set assert) and (b) is where n_gpu_layers>0 would offload
    // (~14x slower). Keep CPU + the ggml-ork ACCEL device (the NPU MUL_MAT interceptor).
    static std::vector<ggml_backend_dev_t> g_devs;
    for (size_t i = 0; i < ggml_backend_dev_count(); i++) {
        ggml_backend_dev_t dv = ggml_backend_dev_get(i);
        const char* nm = ggml_backend_dev_name(dv);
        if (nm && (strstr(nm,"Vulkan") || strstr(nm,"Mali"))) continue;
        g_devs.push_back(dv);
    }
    g_devs.push_back(nullptr);
    if (g_devs.size() > 1) mp.devices = g_devs.data();
    llama_model* mtgt = llama_model_load_from_file(target_path, mp);
    if(!mtgt){ fprintf(stderr,"target load FAILED\n"); return 1; }
    const llama_vocab* vocab = llama_model_get_vocab(mtgt);
    const int n_vocab = llama_vocab_n_tokens(vocab);

    std::vector<llama_token> inp(rd+8);
    int n_prompt = llama_tokenize(vocab, ptxt.data(), (int)rd, inp.data(), (int)inp.size(), true, true);
    if(n_prompt<=0){ fprintf(stderr,"tokenize failed\n"); return 1; }
    if(P>0 && n_prompt>P) n_prompt=P; inp.resize(n_prompt);

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = n_prompt + G + block_size + 64;
    cp.n_batch = cp.n_ubatch = (uint32_t)(n_prompt > block_size ? n_prompt : block_size) + 1;
    cp.n_threads = cp.n_threads_batch = 4;
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    llama_context* ctgt = llama_init_from_model(mtgt, cp);
    if(!ctgt){ fprintf(stderr,"target ctx FAILED\n"); return 1; }

    // draft (borrows target tok_embd/output via ctx_other), NPU too
    llama_model* mdft = llama_model_load_from_file(draft_path, mp);
    if(!mdft){ fprintf(stderr,"draft load FAILED\n"); return 1; }
    llama_context_params cpd = llama_context_default_params();
    // The draft encoder precomputes context over the WHOLE prefill (n_prompt rows) in one llama_encode,
    // so its ubatch must cover n_prompt — not just the block. Size it like the target ctx (max of prompt
    // and block, +1) or the encoder asserts (n_ubatch >= n_tokens) on any prompt longer than the block.
    cpd.n_ctx = cp.n_ctx; cpd.n_batch = cpd.n_ubatch = (uint32_t)(n_prompt > block_size ? n_prompt : block_size) + 1;
    cpd.n_threads = cpd.n_threads_batch = 4;
    cpd.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    cpd.ctx_other = ctgt;
    llama_context* cdft = llama_init_from_model(mdft, cpd);
    if(!cdft){ fprintf(stderr,"draft ctx FAILED\n"); return 1; }
    llama_set_embeddings_nextn(cdft, true, false);

    const int32_t   n_embd_tgt = llama_model_n_embd(mtgt);
    const int32_t   n_embd_dec = llama_model_n_embd(mdft);
    const int32_t*  tlids      = llama_model_target_layer_ids(mdft);
    const uint32_t  n_tlids    = llama_model_target_layer_ids_n(mdft);
    const int32_t   n_embd_enc = (int32_t)n_tlids * n_embd_tgt;
    const llama_token mask_tok = llama_vocab_mask(llama_model_get_vocab(mdft));
    if(mask_tok<0){ fprintf(stderr,"draft has no mask token\n"); return 1; }
    for(uint32_t k=0;k<n_tlids;k++) llama_set_embeddings_layer_inp(ctgt, (uint32_t)tlids[k], true);
    fprintf(stderr,"[ORK BENCH dflash] target_layers=%u n_embd_enc=%d mask=%d block=%d, n_gpu_layers=0 (NPU)\n",
            n_tlids, n_embd_enc, mask_tok, block_size);

    std::vector<float> ctx_g; std::vector<int32_t> ctx_pos; std::vector<float> feat; std::vector<uint8_t> ckpt;
    auto grab = [&](int32_t n, int32_t roff){
        for(uint32_t k=0;k<n_tlids;k++){ const float* L=llama_get_embeddings_layer_inp(ctgt,(uint32_t)tlids[k]); if(!L) return false;
            for(int32_t i=0;i<n;i++) memcpy(feat.data()+(size_t)(roff+i)*n_embd_enc+(size_t)k*n_embd_tgt, L+(size_t)i*n_embd_tgt, (size_t)n_embd_tgt*sizeof(float)); }
        return true; };
    auto encode_append = [&](const float* f, int32_t n, int32_t base){
        llama_batch e = { n, nullptr, const_cast<float*>(f), nullptr, nullptr, nullptr, nullptr };
        if(llama_encode(cdft,e)!=0) return false;
        const float* g = llama_get_embeddings_nextn(cdft); if(!g) return false;
        size_t off=ctx_g.size(); ctx_g.resize(off+(size_t)n*n_embd_dec); memcpy(ctx_g.data()+off,g,(size_t)n*n_embd_dec*sizeof(float));
        for(int32_t i=0;i<n;i++) ctx_pos.push_back(base+i); return true; };

    // prefill target + build initial context (this is the WARMUP: an M>1 forward warms the NPU path)
    { std::vector<int32_t> pp(n_prompt); for(int i=0;i<n_prompt;i++) pp[i]=i;
      if(!decode_batch(ctgt, inp.data(), pp.data(), n_prompt, 0)){ fprintf(stderr,"prompt decode FAILED\n"); return 1; } }
    feat.assign((size_t)n_prompt*n_embd_enc,0.0f);
    if(!grab(n_prompt,0)||!encode_append(feat.data(),n_prompt,0)) return 1;

    int32_t n_ctx_tok=n_prompt; llama_token anchor=inp[n_prompt-1];
    llama_token t0=argmax(llama_get_logits_ith(ctgt,n_prompt-1),n_vocab);
    int64_t n_gen=0,n_cycles=0,acc_sum=0,tgt_fwd=0; bool eog=false;

    auto t_start = std::chrono::steady_clock::now();
    while(n_gen<G && !eog){
        // 1) draft a block on the draft (clear its KV; context passed out-of-band)
        llama_memory_seq_rm(llama_get_memory(cdft),0,-1,-1);
        llama_set_dflash_context(cdft, ctx_g.data(), (int32_t)ctx_pos.size(), ctx_pos.data());
        std::vector<llama_token> bt(block_size+1); std::vector<int32_t> bp(block_size+1);
        bt[0]=anchor; bp[0]=n_ctx_tok-1;
        for(int j=0;j<block_size;j++){ bt[j+1]=mask_tok; bp[j+1]=n_ctx_tok+j; }
        if(!decode_batch(cdft,bt.data(),bp.data(),block_size+1,0)){ fprintf(stderr,"draft FAILED\n"); break; }
        std::vector<llama_token> d(block_size);
        for(int j=0;j<block_size;j++) d[j]=argmax(llama_get_logits_ith(cdft,j+1),n_vocab);

        // 2) checkpoint + verify all B on the target in ONE forward (M=B, grouped -> NPU)
        size_t sz=llama_state_seq_get_size(ctgt,0); ckpt.resize(sz); llama_state_seq_get_data(ctgt,ckpt.data(),sz,0);
        { std::vector<int32_t> vp(block_size); for(int j=0;j<block_size;j++) vp[j]=n_ctx_tok+j;
          if(!decode_batch(ctgt,d.data(),vp.data(),block_size,0)){ fprintf(stderr,"verify FAILED\n"); break; } }
        tgt_fwd++;
        int acc=0; while(acc<block_size){ llama_token tj=(acc==0)?t0:argmax(llama_get_logits_ith(ctgt,acc-1),n_vocab); if(d[acc]!=tj) break; acc++; }
        llama_token bonus=(acc==0)?t0:argmax(llama_get_logits_ith(ctgt,acc-1),n_vocab);

        // 3) roll back + commit [accepted..., bonus] (M=acc+1) + extract their hiddens
        llama_state_seq_set_data(ctgt,ckpt.data(),sz,0);
        std::vector<llama_token> ct(acc+1); std::vector<int32_t> cpp(acc+1);
        for(int j=0;j<acc;j++){ ct[j]=d[j]; cpp[j]=n_ctx_tok+j; } ct[acc]=bonus; cpp[acc]=n_ctx_tok+acc;
        if(!decode_batch(ctgt,ct.data(),cpp.data(),acc+1,0)){ fprintf(stderr,"commit FAILED\n"); break; }
        tgt_fwd++;
        feat.assign((size_t)(acc+1)*n_embd_enc,0.0f);
        if(!grab(acc+1,0)||!encode_append(feat.data(),acc+1,n_ctx_tok)) break;
        t0=argmax(llama_get_logits_ith(ctgt,acc),n_vocab);

        for(int j=0;j<acc;j++){ if(llama_vocab_is_eog(vocab,d[j])) eog=true; }
        if(llama_vocab_is_eog(vocab,bonus)) eog=true;
        n_gen+=acc+1; acc_sum+=acc; n_cycles++; anchor=bonus; n_ctx_tok+=acc+1;
    }
    double dt=std::chrono::duration<double>(std::chrono::steady_clock::now()-t_start).count();
    double mean_acc = n_cycles ? (double)acc_sum/n_cycles : 0.0;
    fprintf(stderr,"\n[ORK BENCH dflash] cycles=%lld mean_accept=%.2f/%d | tau=%.2f tok/cycle | decode %lld tok in %.2fs = %.2f tok/s (target forwards=%lld -> %.2f tok/forward)\n",
            (long long)n_cycles, mean_acc, block_size, mean_acc+1.0, (long long)n_gen, dt, dt>0?n_gen/dt:0.0,
            (long long)tgt_fwd, tgt_fwd?(double)n_gen/tgt_fwd:0.0);
    llama_free(cdft); llama_model_free(mdft); llama_free(ctgt); llama_model_free(mtgt); llama_backend_free();
    return 0;
}

int main(int argc, char** argv){
    if (argc < 3){ fprintf(stderr,"usage: %s <model.gguf> <promptfile> [P=128] [G=64] [ubatch=P]\n  DFlash mode: ORK_DFLASH_DRAFT=<draft.gguf> %s <target.gguf> <promptfile> [P] [G]\n",argv[0],argv[0]); return 2; }
    const char* model_path = argv[1];
    int P = argc>3 ? atoi(argv[3]) : 128;
    int G = argc>4 ? atoi(argv[4]) : 64;
    int UB = argc>5 ? atoi(argv[5]) : P;

    FILE* f=fopen(argv[2],"rb"); if(!f){ fprintf(stderr,"cannot open %s\n",argv[2]); return 2; }
    fseek(f,0,SEEK_END); long fn=ftell(f); fseek(f,0,SEEK_SET);
    std::vector<char> ptxt(fn+1); size_t rd=fread(ptxt.data(),1,fn,f); ptxt[rd]=0; fclose(f);

    llama_backend_init();

    // GUARD: refuse to run without a prebuilt .orkpack — JIT-packing weights on first use pads the
    // measured prefill (weight resolve/pack counted in the timed forward), so a run without ORK_PERSIST
    // reports misleading numbers. Require ORK_PERSIST to point at an EXISTING orkpack; ORK_ALLOW_JIT=1
    // overrides (for the initial build-the-orkpack run, which legitimately JIT-packs then caches).
    if (!getenv("ORK_ALLOW_JIT")) {
        const char* pp = getenv("ORK_PERSIST");
        if (!pp || !*pp) {
            fprintf(stderr, "[ork_bench] ERROR: ORK_PERSIST=<model.orkpack> is required (refusing to report JIT-pack-padded runtimes).\n"
                            "            Set ORK_PERSIST to a prebuilt orkpack, or ORK_ALLOW_JIT=1 to build/JIT-pack.\n");
            return 3;
        }
        FILE* pf = fopen(pp, "rb");
        if (!pf) {
            fprintf(stderr, "[ork_bench] ERROR: orkpack not found at '%s' (ORK_PERSIST). Build it first (ORK_ALLOW_JIT=1) or fix the path.\n", pp);
            return 3;
        }
        fclose(pf);
        fprintf(stderr, "[ork_bench] orkpack: %s\n", pp);
    }

    // TEST HOOK (harness only): exercise the product load-config API instead of env knobs.
    //   ORK_BENCH_CFG=int16    -> set_load_config(dflash, silu_int8_fused=false) [DEFAULT: int16 coherent]
    //   ORK_BENCH_CFG=int8fused -> ...(silu_int8_fused=true) [int8 fully fused through-and-through]
    //   ORK_BENCH_DFLASH=1 sets dflash on.
    if (const char* cfg = getenv("ORK_BENCH_CFG")) {
        bool int8fused = strcmp(cfg,"int8fused")==0;
        bool dflash    = getenv("ORK_BENCH_DFLASH")!=nullptr;
        ggml_backend_ork_set_load_config(dflash, int8fused);
        fprintf(stderr,"[ork_bench] set_load_config(dflash=%d, silu=%s)\n", dflash, int8fused?"int8-fused":"int16-coherent");
    }

    // DFlash operating mode: a co-resident draft speculates a block, the target verifies M=B on the NPU.
    if (const char* draft = getenv("ORK_DFLASH_DRAFT")) {
        int rc = run_dflash_bench(model_path, draft, ptxt, rd, P, G);
        return rc;
    }

    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 0;   // ggml-ork is a MUL_MAT accelerator backend, engaged automatically (no offload)
    llama_model* model = llama_model_load_from_file(model_path, mp);
    if(!model){ fprintf(stderr,"model load FAILED\n"); return 1; }
    const llama_vocab* vocab = llama_model_get_vocab(model);
    int n_vocab = llama_vocab_n_tokens(vocab);

    // tokenize prompt, clamp to P
    std::vector<llama_token> toks(fn+8);
    int nt = llama_tokenize(vocab, ptxt.data(), (int)rd, toks.data(), (int)toks.size(), true, true);
    if(nt<=0){ fprintf(stderr,"tokenize failed (%d)\n",nt); return 1; }
    if(nt>P) nt=P;    // use first P tokens as the prefill batch
    toks.resize(nt);

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx   = P + G + 64;
    cp.n_batch = P > UB ? P : UB;
    cp.n_ubatch= UB;
    cp.n_threads = 4; cp.n_threads_batch = 4;
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;  // ggml-ork is a MUL_MAT accelerator; the fused FA
                                                          // op isn't on its path (auto-FA -> graph compute -1)
    llama_context* ctx = llama_init_from_model(model, cp);
    if(!ctx){ fprintf(stderr,"ctx init FAILED\n"); return 1; }

    // WARMUP: a PREFILL-SHAPED batch (M>1, so supports_op routes it to the NPU — M=1 stays on CPU and would
    // NOT warm the prefill path) to trigger lazy weight pack/import + wcache population + NPU warm, so the
    // timed prefill measures STEADY STATE. A 1-token (M=1) warmup left the timed prefill cold — it paid the
    // one-time weight-load from the .orkpack (~0.8s on Qwen3-1.7B int4), under-reporting warm prefill ~24%.
    { int wn = nt < 32 ? nt : 32; llama_batch wb=llama_batch_get_one(toks.data(), wn); if(llama_decode(ctx,wb)!=0){ fprintf(stderr,"warmup decode FAILED\n"); return 3; } }
    llama_memory_clear(llama_get_memory(ctx), true);

    // PREFILL: decode the P-token prompt as one batch (ubatch UB), time it.
    int64_t t0 = llama_time_us();
    llama_batch pb = llama_batch_get_one(toks.data(), nt);
    int rc = llama_decode(ctx, pb);
    int64_t t1 = llama_time_us();
    if(rc!=0){ fprintf(stderr,"prefill decode FAILED rc=%d (P=%d ubatch=%d)\n",rc,P,UB); return 3; }
    double pf_s = (t1-t0)/1e6, pf_tps = nt/pf_s;

    // DECODE: greedily generate G tokens, one decode each, time. Detokenize + print each piece so the run
    // doubles as a COHERENCY SMOKE TEST — a tiny prompt ("Hi, how are you?") + a few tokens instantly shows
    // whether the (fused) path produces sane text, no 5-minute benchmark needed.
    fprintf(stderr,"[ORK BENCH] response: ");
    llama_token cur = argmax(llama_get_logits_ith(ctx,-1), n_vocab);
    int64_t t2 = llama_time_us(); int gen=0; char piece[256];
    for(; gen<G; gen++){
        int np = llama_token_to_piece(vocab, cur, piece, sizeof piece, 0, true);
        if(np>0){ fwrite(piece,1,np,stderr); }
        if(llama_vocab_is_eog(vocab,cur)){ gen++; break; }
        llama_batch db = llama_batch_get_one(&cur,1);
        if(llama_decode(ctx,db)!=0){ fprintf(stderr,"\ndecode step %d FAILED\n",gen); break; }
        cur = argmax(llama_get_logits_ith(ctx,-1), n_vocab);
    }
    int64_t t3 = llama_time_us();
    double dc_s = (t3-t2)/1e6, dc_tps = gen>0 ? gen/dc_s : 0;

    fprintf(stderr,"\n[ORK BENCH] prefill: %d tok in %.1f ms = %.2f tok/s (ubatch=%d) | decode: %d tok in %.1f ms = %.2f tok/s\n",
            nt, pf_s*1e3, pf_tps, UB, gen, dc_s*1e3, dc_tps);

    llama_free(ctx); llama_model_free(model); llama_backend_free();
    return 0;
}
