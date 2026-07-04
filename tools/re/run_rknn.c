#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "rknn_api.h"
int main(int argc, char** argv){
    if(argc<2){ printf("usage: %s model.rknn\n", argv[0]); return 2; }
    FILE* f=fopen(argv[1],"rb"); if(!f){ printf("open fail\n"); return 1; }
    fseek(f,0,SEEK_END); long sz=ftell(f); fseek(f,0,SEEK_SET);
    void* model=malloc(sz); if(fread(model,1,sz,f)!=(size_t)sz){printf("read fail\n");return 1;} fclose(f);
    rknn_context ctx=0;
    int ret=rknn_init(&ctx, model, (uint32_t)sz, 0, NULL);
    if(ret<0){ printf("rknn_init failed %d\n", ret); return 1; }
    printf("init OK\n");
    rknn_input_output_num ion; memset(&ion,0,sizeof ion);
    rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, &ion, sizeof ion);
    printf("n_input=%u n_output=%u\n", ion.n_input, ion.n_output);
    rknn_input ins[8]; memset(ins,0,sizeof ins);
    for(uint32_t k=0;k<ion.n_input;k++){
        rknn_tensor_attr ia; memset(&ia,0,sizeof ia); ia.index=k;
        rknn_query(ctx, RKNN_QUERY_INPUT_ATTR, &ia, sizeof ia);
        uint32_t insz = ia.size ? ia.size : ia.n_elems;
        void* inbuf=calloc(1, insz);
        for(uint32_t i=0;i<insz;i++)((signed char*)inbuf)[i]=(signed char)(((i+k*3)*7)%13-6);
        ins[k].index=k; ins[k].type=ia.type; ins[k].fmt=ia.fmt; ins[k].buf=inbuf; ins[k].size=insz;
        printf("input %u: n_elems=%u size=%u type=%d fmt=%d scale=%.8f zp=%d\n", k, ia.n_elems, ia.size, ia.type, ia.fmt, ia.scale, ia.zp);
    }
    for(uint32_t k=0;k<ion.n_output;k++){
        rknn_tensor_attr oa; memset(&oa,0,sizeof oa); oa.index=k;
        rknn_query(ctx, RKNN_QUERY_OUTPUT_ATTR, &oa, sizeof oa);
        printf("output %u: n_elems=%u size=%u type=%d fmt=%d scale=%.8f zp=%d\n", k, oa.n_elems, oa.size, oa.type, oa.fmt, oa.scale, oa.zp);
    }
    ret=rknn_inputs_set(ctx, ion.n_input, ins);
    printf("inputs_set ret=%d\n", ret);
    printf("=== RUN ===\n"); fflush(stdout);
    ret=rknn_run(ctx, NULL);
    printf("run ret=%d\n", ret);
    rknn_destroy(ctx);
    return 0;
}
