/* ork_ppl.cpp — teacher-forced perplexity harness for the open stack, the quality companion to ork_bench.
 * Drives the llama.cpp C API directly (the ggml-ork backend intercepts MUL_MAT, so ORK_MIXED_DISPATCH /
 * ORK_QUANT / ORK_OFF all apply; the .orkpack is auto-derived from the model), so any q4/int4/int8 change gets a PPL
 * next to the ork_bench tok/s number — on the SAME model, SAME text, one clock. Perplexity is teacher-
 * forced: decode the whole token window in one pass with per-position logits, PPL = exp(mean NLL) over the
 * next-token targets. A window >= 32 tokens exercises the NPU prefill path (M = window >= the M-threshold).
 *   ork_ppl <model.gguf> <textfile> [window=256] [ubatch=window]
 * Compare: `ORK_OFF=1 ork_ppl …` (CPU/native baseline) vs plain `ork_ppl …` (the ork path; a MoE model
 * auto-selects experts-NF4-on-CPU + int8 attn on NPU). Pass ubatch to bound M (e.g. 512) — a large
 * ubatch (2048, llama-perplexity's default) drives wide colsplit into RKNPU_SUBMIT timeouts.
 */
#include "llama.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>

int main(int argc, char** argv){
    if (argc < 3){ fprintf(stderr,"usage: %s <model.gguf> <textfile> [window=256] [ubatch=window]\n",argv[0]); return 2; }
    const char* model_path = argv[1];
    int W  = argc>3 ? atoi(argv[3]) : 256;
    int UB = argc>4 ? atoi(argv[4]) : W;
    if (W < 2) W = 2;

    FILE* f=fopen(argv[2],"rb"); if(!f){ fprintf(stderr,"cannot open %s\n",argv[2]); return 2; }
    fseek(f,0,SEEK_END); long fn=ftell(f); fseek(f,0,SEEK_SET);
    std::vector<char> txt(fn+1); size_t rd=fread(txt.data(),1,fn,f); txt[rd]=0; fclose(f);

    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 0;   // ggml-ork is a MUL_MAT accelerator, engaged automatically
    llama_model* model = llama_model_load_from_file(model_path, mp);
    if(!model){ fprintf(stderr,"model load FAILED\n"); return 1; }
    const llama_vocab* vocab = llama_model_get_vocab(model);
    int n_vocab = llama_vocab_n_tokens(vocab);

    std::vector<llama_token> toks(fn+8);
    int nt = llama_tokenize(vocab, txt.data(), (int)rd, toks.data(), (int)toks.size(), true, false);
    if(nt<2){ fprintf(stderr,"tokenize failed / text too short (%d)\n",nt); return 1; }
    if(nt>W) nt=W;
    toks.resize(nt);

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx    = nt + 8;
    cp.n_batch  = nt > UB ? nt : UB;
    cp.n_ubatch = UB;
    cp.n_threads = 4; cp.n_threads_batch = 4;
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;   // ggml-ork path (as ork_bench)
    llama_context* ctx = llama_init_from_model(model, cp);
    if(!ctx){ fprintf(stderr,"ctx init FAILED\n"); return 1; }

    // Teacher-forced batch: all tokens, per-position logits requested.
    llama_batch batch = llama_batch_init(nt, 0, 1);
    batch.n_tokens = nt;
    for(int i=0;i<nt;i++){
        batch.token[i]=toks[i]; batch.pos[i]=i;
        batch.n_seq_id[i]=1; batch.seq_id[i][0]=0; batch.logits[i]=1;
    }
    if(llama_decode(ctx, batch)!=0){ fprintf(stderr,"decode FAILED (window=%d ubatch=%d)\n",nt,UB); return 3; }

    // PPL = exp( mean_i -log softmax(logits_i)[tok_{i+1}] ), i=0..nt-2 (skip position 0's BOS target quirk? we
    // score every next-token target from position 0). Numerically stable log-softmax (max-subtract).
    double nll=0.0; int scored=0;
    for(int i=0;i<nt-1;i++){
        const float* lg = llama_get_logits_ith(ctx, i);
        if(!lg) continue;
        float mx=lg[0]; for(int v=1;v<n_vocab;v++) if(lg[v]>mx) mx=lg[v];
        double sum=0.0; for(int v=0;v<n_vocab;v++) sum+=exp((double)(lg[v]-mx));
        int tgt=toks[i+1];
        double logp=(double)(lg[tgt]-mx)-log(sum);
        nll += -logp; scored++;
    }
    double mnll = scored? nll/scored : 0.0;
    double ppl = exp(mnll);
    const char* off = getenv("ORK_OFF");
    fprintf(stderr,"[ORK PPL] %s | window=%d scored=%d | mean NLL=%.4f | PPL=%.4f%s\n",
            model_path, nt, scored, mnll, ppl, (off&&off[0]&&off[0]!='0')?"  (ORK_OFF: CPU baseline)":"");

    llama_batch_free(batch);
    llama_free(ctx); llama_model_free(model); llama_backend_free();
    return 0;
}
