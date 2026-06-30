/* tools/pgquant_capture.c — does the RK3588 NPU do per-K-group B dequant in ONE submit? (API lead #2)
 *
 * Drives the closed RKNN matmul API with B_quant_type = per-channel(1) vs per-group(2) at the same
 * shape, runs the matmul `runs` times. Run under strace counting the SUBMIT ioctl (DRM nr 0x41 ->
 * low16 0x6441): submits/run = (count@runsB - count@runsA)/(runsB-runsA), isolating run() from setup.
 *   per-group submits == per-channel (1) -> HW-native per-group -> int4 unlock (decode scale regs next)
 *   per-group submits ~ K/group_size      -> librknnrt fakes it -> no HW lever
 * Build (board): gcc -O2 -I~/rknn_sdk -o pgquant_capture tools/pgquant_capture.c -L~/rknn_sdk -lrknnrt -lm
 * Run:           LD_LIBRARY_PATH=~/rknn_sdk ./pgquant_capture M K N B_quant_type group_size runs
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "rknn_matmul_api.h"

int main(int argc,char**argv){
    int M   = argc>1?atoi(argv[1]):8;
    int K   = argc>2?atoi(argv[2]):512;
    int N   = argc>3?atoi(argv[3]):512;
    int bqt = argc>4?atoi(argv[4]):1;     /* 1=per-channel, 2=per-group */
    int gs  = argc>5?atoi(argv[5]):128;
    int runs= argc>6?atoi(argv[6]):1;

    rknn_matmul_ctx ctx; rknn_matmul_info info; rknn_matmul_io_attr io;
    memset(&info,0,sizeof info); memset(&io,0,sizeof io);
    info.M=M; info.K=K; info.N=N;
    info.type=RKNN_INT8_MM_INT8_TO_INT32;
    info.B_quant_type=bqt;
    if(bqt==2) info.group_size=gs;

    int ret=rknn_matmul_create(&ctx,&info,&io);
    printf("create ret=%d  (M=%d K=%d N=%d B_quant_type=%d group_size=%d)\n",
           ret,M,K,N,bqt,bqt==2?gs:0);
    if(ret<0) return 1;

    rknn_tensor_mem* Am=rknn_create_mem(ctx, io.A.size);
    rknn_tensor_mem* Bm=rknn_create_mem(ctx, io.B.size);
    rknn_tensor_mem* Cm=rknn_create_mem(ctx, io.C.size);
    if(!Am||!Bm||!Cm){ printf("create_mem failed\n"); return 1; }
    memset(Am->virt_addr,1,io.A.size); memset(Bm->virt_addr,1,io.B.size);
    rknn_matmul_set_io_mem(ctx,Am,&io.A);
    rknn_matmul_set_io_mem(ctx,Bm,&io.B);
    rknn_matmul_set_io_mem(ctx,Cm,&io.C);

    int ng   = (bqt==2)?(K/gs):1;
    int slen = N*ng;                          /* per-channel: N ; per-group: N*(K/G) */
    float* scale=malloc((size_t)slen*sizeof(float)); for(int i=0;i<slen;i++) scale[i]=0.01f;
    int32_t* zp =malloc((size_t)slen*sizeof(int32_t)); for(int i=0;i<slen;i++) zp[i]=0;
    rknn_quant_params qp; memset(&qp,0,sizeof qp);
    qp.scale=scale; qp.scale_len=slen; qp.zp=zp; qp.zp_len=slen;
    int qret=rknn_matmul_set_quant_params(ctx,&qp);
    printf("set_quant_params ret=%d  (scale_len=%d, ng=%d)\n",qret,slen,ng);

    for(int i=0;i<runs;i++){ int r=rknn_matmul_run(ctx); if(r) printf("run %d ret=%d\n",i,r); }
    printf("ran %d matmul(s)  io.A.size=%u io.B.size=%u io.C.size=%u\n",runs,io.A.size,io.B.size,io.C.size);
    return 0;
}
