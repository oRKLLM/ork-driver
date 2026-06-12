/* tools/int4_capture.c — drive librknnrt's rknn_matmul so the regcmd_capture shim can dump a real
 * INT4 regcmd, the reference for adding an int4 path to ork_mm_*. RE-time only: needs Rockchip's
 * librknnrt + rknn_matmul_api.h/rknn_api.h (download separately — not in this repo, same status as
 * rkllm_bench). Values are dummy; we want the register program + the int4 weight tile layout.
 *
 *   cc -O2 -I<rknn_include> int4_capture.c -L<rknn_lib> -l:librknnrt.so -o int4cap
 *   ./int4cap                                  # SWEEP: find a type-10 (int4xint4) param combo that creates
 *   sudo env LD_LIBRARY_PATH=<lib> LD_PRELOAD=./regcmd_capture.so ./int4cap 10 1 1 0   # capture the winner
 *   # args: <type> <B_layout 0/1> <B_quant 0=layer/1=chan/2=group> <group_size>
 *
 * STATUS (2026-06, librknnrt 2.3.2 on RK3588): types 7/8/11 (anything with fp16/int8 activation x
 * int4) are "unsupported matmul dtype in this platform". But type 10 (RKNN_INT4_MM_INT4_TO_INT16,
 * W4A4) is NOT refused — it returned -5 (RKNN_ERR_PARAM_INVALID), i.e. SUPPORTED but mis-configured.
 * invisiofficial/rk-llama.cpp confirms W4A4 runs on RK3588. So this sweep hunts the right
 * B_layout/B_quant/group_size to make type-10 create succeed, then runs it under the shim to capture
 * the int4 regcmd. The int4 WEIGHT packing + CNA bit-width regs are shared with w4a16 (only the
 * activation/output path differs, which we already have from the fp16<->int8 delta). Then diff vs
 * ork-driver's REGCMD_I8 -> synth_i4. See ROADMAP.md. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include "rknn_api.h"
#include "rknn_matmul_api.h"

static int try_create(int type,int M,int K,int N,int layout,int bq,int gs,rknn_matmul_ctx*ctx,rknn_matmul_io_attr*io){
    rknn_matmul_info info; memset(&info,0,sizeof info);
    info.M=M; info.K=K; info.N=N; info.type=type;
    info.B_layout=(int16_t)layout; info.B_quant_type=(int16_t)bq;
    if(bq==2) info.group_size=gs;
    memset(io,0,sizeof *io);
    return rknn_matmul_create(ctx,&info,io);
}

int main(int argc,char**argv){
    int M=4,K=64,N=64;                       /* RK3588 int4: K%32, N%64 */

    if(argc>=4){                             /* focused: create+RUN one combo (for capture under shim) */
        int type=atoi(argv[1]),layout=atoi(argv[2]),bq=atoi(argv[3]),gs=argc>4?atoi(argv[4]):32;
        if(argc>5) K=atoi(argv[5]); if(argc>6) N=atoi(argv[6]);   /* K,N override to probe reg scaling */
        rknn_matmul_ctx ctx; rknn_matmul_io_attr io;
        int r=try_create(type,M,K,N,layout,bq,gs,&ctx,&io);
        printf("type=%d layout=%d bq=%d gs=%d -> create=%d\n",type,layout,bq,gs,r);
        if(r){printf("create failed %d\n",r);return 1;}
        printf("io A:%u B:%u C:%u bytes   (int4 B should be ~K*N/2=%d)\n",io.A.size,io.B.size,io.C.size,K*N/2);
        rknn_tensor_mem *A=rknn_create_mem(ctx,io.A.size),*B=rknn_create_mem(ctx,io.B.size),*C=rknn_create_mem(ctx,io.C.size);
        if(!A||!B||!C){printf("mem failed\n");return 1;}
        memset(A->virt_addr,0,io.A.size); memset(B->virt_addr,0x11,io.B.size); memset(C->virt_addr,0,io.C.size);
        size_t sl=(size_t)N*(K/ (gs?gs:1)) + N + 1;       /* over-allocate: covers layer/chan/group */
        float*sc=malloc(sl*4); for(size_t i=0;i<sl;i++)sc[i]=1.0f;
        rknn_quant_params qp; memset(&qp,0,sizeof qp); qp.scale=sc; qp.scale_len=(int32_t)sl;
        rknn_matmul_set_quant_params(ctx,&qp);
        rknn_matmul_set_io_mem(ctx,A,&io.A); rknn_matmul_set_io_mem(ctx,B,&io.B); rknn_matmul_set_io_mem(ctx,C,&io.C);
        printf("run=%d\n",rknn_matmul_run(ctx));
        rknn_matmul_destroy(ctx); return 0;
    }

    /* SWEEP: which type-10 (int4xint4) config does this librknnrt accept? */
    int type=argc>1?atoi(argv[1]):RKNN_INT4_MM_INT4_TO_INT16;
    printf("sweep type=%d M=%d K=%d N=%d  (B_layout x B_quant x group_size)\n",type,M,K,N);
    int gss[]={0,16,32,64};
    for(int layout=0;layout<2;layout++) for(int bq=0;bq<3;bq++){
        int ngs = (bq==2)?4:1;
        for(int gi=0;gi<ngs;gi++){
            int gs=gss[gi]; rknn_matmul_ctx ctx; rknn_matmul_io_attr io;
            int r=try_create(type,M,K,N,layout,bq,gs,&ctx,&io);
            if(r==0){ printf("  layout=%d bq=%d gs=%-2d -> *** CREATE OK ***  io A:%u B:%u C:%u (K*N/2=%d)\n",
                             layout,bq,gs,io.A.size,io.B.size,io.C.size,K*N/2); rknn_matmul_destroy(ctx); }
            else      printf("  layout=%d bq=%d gs=%-2d -> %d\n",layout,bq,gs,r);
        }
    }
    return 0;
}
