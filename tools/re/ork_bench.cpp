/* ork_bench.cpp — minimal open-stack perf harness, the mirror of rkllm_bench.cpp. Drives the llama.cpp C
 * API directly (ggml-ork backend intercepts MUL_MAT, so ORK_FFN_CHAIN etc. apply) so we control EXACTLY:
 * the prefill batch size (n_ubatch — the lever that decides single- vs multi-core prefill), warmup, and
 * timing — none of which llama-bench exposes. Reports prefill + decode tok/s from one clock, same shape as
 * the rkllm side, for a true apples-to-apples comparison.
 *   ork_bench <model.gguf> <promptfile> [P=128] [G=64] [ubatch=P]
 * Env: everything the ggml-ork backend reads (ORK_FFN_CHAIN, ORK_PERSIST, ORK_PROFILE, ...).
 */
#include "llama.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

static llama_token argmax(const float* v, int n){ int b=0; float m=v[0]; for(int i=1;i<n;i++) if(v[i]>m){m=v[i];b=i;} return b; }

int main(int argc, char** argv){
    if (argc < 3){ fprintf(stderr,"usage: %s <model.gguf> <promptfile> [P=128] [G=64] [ubatch=P]\n",argv[0]); return 2; }
    const char* model_path = argv[1];
    int P = argc>3 ? atoi(argv[3]) : 128;
    int G = argc>4 ? atoi(argv[4]) : 64;
    int UB = argc>5 ? atoi(argv[5]) : P;

    FILE* f=fopen(argv[2],"rb"); if(!f){ fprintf(stderr,"cannot open %s\n",argv[2]); return 2; }
    fseek(f,0,SEEK_END); long fn=ftell(f); fseek(f,0,SEEK_SET);
    std::vector<char> ptxt(fn+1); size_t rd=fread(ptxt.data(),1,fn,f); ptxt[rd]=0; fclose(f);

    llama_backend_init();
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

    // WARMUP: one full prefill+decode of a tiny batch to trigger lazy weight-pack/import + NPU warm, so the
    // timed passes measure steady state (the exact trap the AGENTS doc warns raw llama-cli falls into).
    { llama_token w=toks[0]; llama_batch wb=llama_batch_get_one(&w,1); if(llama_decode(ctx,wb)!=0){ fprintf(stderr,"warmup decode FAILED\n"); return 3; } }
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
