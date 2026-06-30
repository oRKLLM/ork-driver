/* tools/int4_type_probe.c — which int4-WEIGHT matmul types does RK3588 accept, and is the weight
 * buffer half-size (native int4, NO software inflate)? Answers: is there a no-inflate W4A8/W4A16 path?
 * Just rknn_matmul_create per type + report io sizes (B.size==K*N/2 => packed int4 resident). No run.
 *   gcc -O2 -I~/rknn_sdk -o int4_type_probe tools/int4_type_probe.c -L~/rknn_sdk -lrknnrt -lm
 *   LD_LIBRARY_PATH=~/rknn_sdk ./int4_type_probe
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "rknn_matmul_api.h"

static void probe(int type, const char*name, int M,int K,int N){
    rknn_matmul_ctx ctx; rknn_matmul_info info; rknn_matmul_io_attr io;
    memset(&info,0,sizeof info); memset(&io,0,sizeof io);
    info.M=M; info.K=K; info.N=N; info.type=(rknn_matmul_type)type;
    int ret=rknn_matmul_create(&ctx,&info,&io);
    if(ret<0){ printf("  type %-2d %-32s create ret=%d  (NOT supported)\n", type, name, ret); return; }
    long bfull=(long)K*N;
    printf("  type %-2d %-32s ret=0  A=%u B=%u C=%u  B/(K*N)=%.2f -> %s\n",
           type, name, io.A.size, io.B.size, io.C.size, (double)io.B.size/bfull,
           io.B.size <= bfull*0.6 ? "PACKED int4 (no inflate)" : "int8-width weight");
    rknn_matmul_destroy(ctx);
}

int main(void){
    int M=8,K=512,N=512;
    printf("RK3588 int4/int8 matmul type support (M=%d K=%d N=%d):\n",M,K,N);
    probe(2,  "INT8_MM_INT8_TO_INT32",   M,K,N);   /* int8 reference: B=K*N */
    probe(11, "INT8_MM_INT4_TO_INT32",   M,K,N);   /* W4A8 int8 act x int4 wt -> int32 */
    probe(15, "INT8_MM_INT4_TO_FLOAT16", M,K,N);   /* W4A8 -> fp16 */
    probe(7,  "FLOAT16_MM_INT4_TO_FLOAT32",M,K,N); /* W4A16 */
    probe(8,  "FLOAT16_MM_INT4_TO_FLOAT16",M,K,N); /* W4A16 */
    probe(10, "INT4_MM_INT4_TO_INT16",   M,K,N);   /* W4A4 (ork's current int4) */
    return 0;
}
