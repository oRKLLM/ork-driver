/* rkllm_bench.cpp — closed-baseline perf runner for the ork-driver-vs-rkllm comparison. Drives the public
 * librkllmrt API, feeds a prompt file (so prefill token count is realistic, ~pp256), generates ntok, and
 * prints the runtime's own RKLLMPerfStat: prefill tok/s + decode (generate) tok/s. Same model on both
 * runtimes is the AGENTS benchmark rule; this reports the .rkllm side. Documented rkllm.h API only.
 *   g++ -O2 -I. -o rkllm_bench rkllm_bench.cpp -L. -lrkllmrt ; ./rkllm_bench <model.rkllm> <ntok> <promptfile>
 */
#include <cstdint>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include "rkllm.h"

static int cb(RKLLMResult* r, void* u, LLMCallState s){ (void)u;
    if (r && r->text) fputs(r->text, stdout);
    if (s == RKLLM_RUN_FINISH || s == RKLLM_RUN_ERROR) {
        if (r) {
            RKLLMPerfStat* p = &r->perf;
            double pf = p->prefill_time_ms > 0 ? p->prefill_tokens * 1000.0 / p->prefill_time_ms : 0;
            double gf = p->generate_time_ms > 0 ? p->generate_tokens * 1000.0 / p->generate_time_ms : 0;
            fprintf(stderr, "\n[RKLLM PERF] prefill: %d tok in %.1f ms = %.2f tok/s | decode: %d tok in %.1f ms = %.2f tok/s | mem %.0f MB\n",
                    p->prefill_tokens, p->prefill_time_ms, pf, p->generate_tokens, p->generate_time_ms, gf, p->memory_usage_mb);
        }
        fputc('\n', stdout);
    }
    return 0;
}

int main(int argc, char** argv){
    if (argc < 4){ fprintf(stderr,"usage: %s <model.rkllm> <ntok> <promptfile>\n",argv[0]); return 2; }
    int ntok = atoi(argv[2]);
    FILE* f = fopen(argv[3], "rb"); if(!f){ fprintf(stderr,"cannot open prompt %s\n",argv[3]); return 2; }
    fseek(f,0,SEEK_END); long n=ftell(f); fseek(f,0,SEEK_SET);
    char* prompt=(char*)malloc(n+1); size_t rd=fread(prompt,1,n,f); prompt[rd]=0; fclose(f);

    RKLLMParam p = rkllm_createDefaultParam();
    p.model_path      = argv[1];
    p.max_new_tokens  = ntok;
    p.max_context_len = 2048;
    p.skip_special_token = true;
    LLMHandle h = NULL;
    if (rkllm_init(&h, &p, cb) != 0){ fprintf(stderr,"rkllm_init FAILED\n"); return 1; }
    RKLLMInput in; memset(&in,0,sizeof in);
    in.role = "user"; in.input_type = RKLLM_INPUT_PROMPT; in.prompt_input = prompt;
    RKLLMInferParam ip; memset(&ip,0,sizeof ip); ip.mode = RKLLM_INFER_GENERATE; ip.keep_history = 0;
    int r = rkllm_run(h, &in, &ip, NULL);
    fprintf(stderr,"[rkllm_bench] rc=%d\n", r);
    rkllm_destroy(h);
    return 0;
}
