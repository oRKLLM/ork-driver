/* tools/int4_capture.c — drive librknnrt's rknn_matmul so the regcmd_capture shim can dump the
 * INT4 (w4a16) regcmd, the reference for adding an int4 path to ork_mm_*. RE-time only: needs
 * Rockchip's librknnrt + rknn_matmul_api.h/rknn_api.h (download separately — not in this repo, same
 * status as rkllm_bench). Values are dummy; we want the register program + the int4 weight layout.
 *
 *   cc -O2 -I<rknn_include> int4_capture.c -L<rknn_lib> -l:librknnrt.so -o int4cap
 *   sudo env LD_LIBRARY_PATH=<rknn_lib> LD_PRELOAD=./regcmd_capture.so ./int4cap [matmul_type]
 *   # matmul_type: 7=f16xINT4>f32 (w4a16), 8=f16xi4>f16, 10=i4xi4>i16, 11=i8xi4>i32 (default 7)
 *
 * STATUS (2026-06): librknnrt 2.3.2's rknn_matmul rejects ALL int4 types on RK3588 ("unsupported
 * matmul dtype in this platform"; int8 type=2 works, validating this capture pipeline). So the int4
 * reference must instead be captured from **librkllmrt running a w4a16 .rkllm** (the LLM runtime
 * does int4) via tools/rkllm_bench under the same LD_PRELOAD=regcmd_capture.so. Then diff the int4
 * regcmd vs ork-driver's int8 (REGCMD_I8): the 4-bit weight packing, the CNA bit-width reg, and how
 * per-group/channel fp16 scales are applied are the unknowns to decode. See ROADMAP.md. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include "rknn_api.h"
#include "rknn_matmul_api.h"
int main(int argc,char**argv){
    int M=4,K=64,N=64;                       /* RK3588 int4: K%32, N%64 */
    rknn_matmul_ctx ctx; rknn_matmul_info info; memset(&info,0,sizeof info);
    info.M=M; info.K=K; info.N=N; info.type=argc>1?atoi(argv[1]):RKNN_FLOAT16_MM_INT4_TO_FLOAT32;
    info.B_quant_type=RKNN_QUANT_TYPE_PER_CHANNEL_SYM;
    rknn_matmul_io_attr io; memset(&io,0,sizeof io);
    int r=rknn_matmul_create(&ctx,&info,&io);
    if(r){printf("create failed %d (type %d unsupported in this librknnrt?)\n",r,info.type);return 1;}
    printf("io A:%u B:%u C:%u bytes\n",io.A.size,io.B.size,io.C.size);
    rknn_tensor_mem *A=rknn_create_mem(ctx,io.A.size),*B=rknn_create_mem(ctx,io.B.size),*C=rknn_create_mem(ctx,io.C.size);
    if(!A||!B||!C){printf("mem failed\n");return 1;}
    memset(A->virt_addr,0,io.A.size); memset(B->virt_addr,0x11,io.B.size); memset(C->virt_addr,0,io.C.size);
    float*sc=malloc((size_t)N*4); for(int i=0;i<N;i++)sc[i]=1.0f;
    rknn_quant_params qp; memset(&qp,0,sizeof qp); qp.scale=sc; qp.scale_len=N;
    rknn_matmul_set_quant_params(ctx,&qp);
    rknn_matmul_set_io_mem(ctx,A,&io.A); rknn_matmul_set_io_mem(ctx,B,&io.B); rknn_matmul_set_io_mem(ctx,C,&io.C);
    printf("run=%d\n",rknn_matmul_run(ctx));
    rknn_matmul_destroy(ctx); return 0;
}
