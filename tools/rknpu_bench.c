/* tools/rknpu_bench.c — benchmark Rockchip's closed `librkllmrt` on a .rkllm model, for an
 * apples-to-apples comparison against ork-driver's own bench (examples/bench.c). It dlopen's
 * librkllmrt at RUNTIME (no build dependency) and reads the runtime's own perf stats, so the
 * library is found on the board, not shipped here. Contains NO Rockchip code — the RKLLM ABI
 * structs are reverse-engineered (same status as the regcmd headers). Like tools/regcmd_capture.c,
 * it needs the proprietary runtime present at run time only; it is not part of the ork-driver build.
 *
 *   make rknpu_bench   # or: gcc -O2 -o rknpu_bench tools/rknpu_bench.c -ldl
 *   sudo ./rknpu_bench <librkllmrt.so> <model.rkllm> [new_tokens]
 * e.g. ./rknpu_bench /var/lib/orkllm/runtimes/librkllmrt-aarch64-v1.2.3.so Qwen3-1.7B-w8a8.rkllm 128
 * Prints librkllmrt's prefill/decode tok/s; compare with `make bench` at the model's config. */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <dlfcn.h>
#include <stdbool.h>
enum LLMCallState { RKLLM_RUN_NORMAL=0, RKLLM_RUN_WAITING=1, RKLLM_RUN_FINISH=2, RKLLM_RUN_ERROR=3 };
enum RKLLMInputType { RKLLM_INPUT_PROMPT=0, RKLLM_INPUT_TOKEN=1, RKLLM_INPUT_EMBED=2, RKLLM_INPUT_MULTIMODAL=3 };
enum RKLLMInferMode { RKLLM_INFER_GENERATE=0, RKLLM_INFER_GET_LAST_HIDDEN_LAYER=1, RKLLM_INFER_GET_LOGITS=2 };
typedef struct { int32_t base_domain_id; int8_t embed_flash; int8_t enabled_cpus_num; uint32_t enabled_cpus_mask; uint8_t n_batch; int8_t use_cross_attn; uint8_t reserved[104]; } RKLLMExtendParam;
typedef struct { const char* model_path; int32_t max_context_len; int32_t max_new_tokens; float top_k; int32_t n_keep; float top_p; float temperature; float repeat_penalty; float frequency_penalty; float presence_penalty; int32_t mirostat; float mirostat_tau; float mirostat_eta; bool skip_special_token; bool is_async; bool enable_thinking; const char* img_start; const char* img_end; const char* img_content; RKLLMExtendParam extend_param; bool use_gpu; } RKLLMParam;
typedef struct { float* embed; size_t n_tokens; } RKLLMEmbedInput;
typedef struct { int32_t* input_ids; size_t n_tokens; } RKLLMTokenInput;
typedef struct { const char* prompt; float* image_embed; size_t n_image_tokens; size_t n_image; size_t image_width; size_t image_height; } RKLLMMultiModelInput;
typedef union { const char* prompt_input; RKLLMEmbedInput embed_input; RKLLMTokenInput token_input; RKLLMMultiModelInput multimodal_input; } RKLLMInputUnion;
typedef struct { const char* role; bool enable_thinking; enum RKLLMInputType input_type; RKLLMInputUnion input_data; } RKLLMInput;
typedef struct { const char* lora_adapter_name; } RKLLMLoraParam;
typedef struct { int save_prompt_cache; const char* prompt_cache_path; } RKLLMPromptCacheParam;
typedef struct { enum RKLLMInferMode mode; RKLLMLoraParam* lora_params; RKLLMPromptCacheParam* prompt_cache_params; int keep_history; } RKLLMInferParam;
typedef struct { float* hidden_states; int embd_size; int num_tokens; } RKLLMResultLastHiddenLayer;
typedef struct { float* logits; int vocab_size; int num_tokens; } RKLLMResultLogits;
typedef struct { float prefill_time_ms; int prefill_tokens; float generate_time_ms; int generate_tokens; float memory_usage_mb; } RKLLMPerfStat;
typedef struct { const char* text; int token_id; RKLLMResultLastHiddenLayer last_hidden_layer; RKLLMResultLogits logits; RKLLMPerfStat perf; } RKLLMResult;
typedef void* RKLLM_Handle_t;
typedef int (*cb_t)(RKLLMResult*, void*, enum LLMCallState);
typedef int (*init_t)(RKLLM_Handle_t*, RKLLMParam*, cb_t);
typedef int (*run_t)(RKLLM_Handle_t, RKLLMInput*, RKLLMInferParam*, void*);
typedef int (*destroy_t)(RKLLM_Handle_t);

static int g_gen=0;
static int cb(RKLLMResult* r, void* u, enum LLMCallState st){ (void)u;
    if(st==RKLLM_RUN_NORMAL){ g_gen++; if(r->text) fputs(r->text,stdout); }
    else if(st==RKLLM_RUN_FINISH){
        printf("\n--- librkllmrt perf ---\n");
        printf("prefill: %d tok in %.1f ms = %.2f tok/s\n", r->perf.prefill_tokens, r->perf.prefill_time_ms,
            r->perf.prefill_tokens*1000.0/r->perf.prefill_time_ms);
        printf("decode : %d tok in %.1f ms = %.2f tok/s\n", r->perf.generate_tokens, r->perf.generate_time_ms,
            r->perf.generate_tokens*1000.0/r->perf.generate_time_ms);
        printf("mem: %.0f MB\n", r->perf.memory_usage_mb);
    } else if(st==RKLLM_RUN_ERROR) fprintf(stderr,"RKLLM_RUN_ERROR\n");
    return 0;
}
int main(int argc,char**argv){
    if(argc<3){fprintf(stderr,"usage: %s <librkllmrt.so> <model.rkllm> [new_tokens]\n",argv[0]);return 1;}
    int newtok=argc>3?atoi(argv[3]):128;
    void* h=dlopen(argv[1],RTLD_NOW); if(!h){fprintf(stderr,"dlopen: %s\n",dlerror());return 1;}
    init_t rkllm_init=dlsym(h,"rkllm_init"); run_t rkllm_run=dlsym(h,"rkllm_run"); destroy_t rkllm_destroy=dlsym(h,"rkllm_destroy");
    if(!rkllm_init||!rkllm_run){fprintf(stderr,"dlsym failed\n");return 1;}
    RKLLMParam p; memset(&p,0,sizeof p);
    p.model_path=argv[2]; p.max_context_len=512; p.max_new_tokens=newtok;
    p.top_k=40; p.top_p=0.9f; p.temperature=0.8f; p.repeat_penalty=1.1f; p.mirostat_tau=5.0f; p.mirostat_eta=0.1f;
    p.skip_special_token=1; p.use_gpu=1;   /* match oRKLLM addon */
    p.img_start=""; p.img_end=""; p.img_content="";   /* librkllmrt derefs these — must not be NULL */
    p.extend_param.base_domain_id=0; p.extend_param.embed_flash=1; p.extend_param.n_batch=1;
    p.extend_param.use_cross_attn=0; p.extend_param.enabled_cpus_num=4; p.extend_param.enabled_cpus_mask=0xf0;
    RKLLM_Handle_t hd=NULL;
    printf("loading %s ...\n",argv[2]); fflush(stdout);
    if(rkllm_init(&hd,&p,cb)){fprintf(stderr,"rkllm_init failed\n");return 1;}
    RKLLMInput in; memset(&in,0,sizeof in); in.input_type=RKLLM_INPUT_PROMPT;   /* role left NULL, as the addon does */
    in.input_data.prompt_input="Once upon a time, in a land far away,";
    RKLLMInferParam ip; memset(&ip,0,sizeof ip); ip.mode=RKLLM_INFER_GENERATE; ip.keep_history=0;
    rkllm_run(hd,&in,&ip,NULL);
    if(rkllm_destroy) rkllm_destroy(hd);
    return 0;
}
