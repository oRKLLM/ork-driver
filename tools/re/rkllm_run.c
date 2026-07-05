/* tools/re/rkllm_run.c — minimal clean-room runner that drives the public librkllmrt API to run one short
 * inference, so the LD_PRELOAD regcmd probe can observe the hardware register-commands rkllm issues for the
 * FFN/SiLU. Our own code using the documented rkllm.h API (interoperability); no proprietary code reproduced.
 *   gcc -O2 -I. -o rkllm_run rkllm_run.c -L. -lrknnrt ... (link librkllmrt); run: ./rkllm_run <model.rkllm> [ntok]
 */
#include <cstdint>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include "rkllm.h"

static int cb(RKLLMResult* r, void* u, LLMCallState s){ (void)u;
    if (r && r->text) fputs(r->text, stdout);
    if (s == RKLLM_RUN_FINISH || s == RKLLM_RUN_ERROR) fputc('\n', stdout);
    return 0;
}

int main(int argc, char** argv){
    if (argc < 2){ fprintf(stderr,"usage: %s <model.rkllm> [ntok]\n",argv[0]); return 2; }
    RKLLMParam p = rkllm_createDefaultParam();
    p.model_path      = argv[1];
    p.max_new_tokens  = argc>2 ? atoi(argv[2]) : 1;
    p.max_context_len = 256;
    p.skip_special_token = true;
    LLMHandle h = NULL;
    if (rkllm_init(&h, &p, cb) != 0){ fprintf(stderr,"rkllm_init FAILED\n"); return 1; }
    RKLLMInput in; memset(&in,0,sizeof in);
    in.role = "user"; in.input_type = RKLLM_INPUT_PROMPT; in.prompt_input = "Hi";
    RKLLMInferParam ip; memset(&ip,0,sizeof ip); ip.mode = RKLLM_INFER_GENERATE; ip.keep_history = 0;
    int r = rkllm_run(h, &in, &ip, NULL);
    fprintf(stderr,"\n[rkllm_run] rkllm_run rc=%d\n", r);
    rkllm_destroy(h);
    return 0;
}
