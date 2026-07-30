/* rknn_reuse_cap — drive librknnrt's int8 matmul (B set ONCE, run N times on the same ctx) so runs 2+ REUSE the
 * resident weight. Under the rknpu_dump.so LD_PRELOAD shim this captures the vendor's regcmds: if librknnrt sets
 * WEIGHT_REUSE (0x1040 bit13) / FC_SKIP (0x1060) on the reuse runs, we get the cross-tile weight-retention recipe.
 * Shape matches our fold (K=3584 N=1216) so the regcmds are directly comparable.
 *   cc -I<rknn_sdk> -o rknn_reuse_cap rknn_reuse_cap.c -L<rknn_sdk> -lrknnrt
 *   sudo env LD_PRELOAD=$PWD/rknpu_dump.so LD_LIBRARY_PATH=<rknn_sdk> ./rknn_reuse_cap [M] [runs]
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "rknn_matmul_api.h"
int main(int argc,char**argv){
    int M=argc>1?atoi(argv[1]):36, K=3584, N=1216, runs=argc>2?atoi(argv[2]):4;
    rknn_matmul_ctx ctx; rknn_matmul_info info; rknn_matmul_io_attr io;
    memset(&info,0,sizeof info); memset(&io,0,sizeof io);
    info.M=M; info.K=K; info.N=N; info.type=RKNN_INT8_MM_INT8_TO_INT32; info.AC_layout=0;
    if(rknn_matmul_create(&ctx,&info,&io)){ fprintf(stderr,"rknn_matmul_create FAILED (M=%d K=%d N=%d)\n",M,K,N); return 1; }
    rknn_tensor_mem*Am=rknn_create_mem(ctx,io.A.size),*Bm=rknn_create_mem(ctx,io.B.size),*Cm=rknn_create_mem(ctx,io.C.size);
    memset(Am->virt_addr,1,io.A.size); memset(Bm->virt_addr,1,io.B.size);
    rknn_matmul_set_io_mem(ctx,Am,&io.A); rknn_matmul_set_io_mem(ctx,Bm,&io.B); rknn_matmul_set_io_mem(ctx,Cm,&io.C);
    for(int i=0;i<runs;i++){ fprintf(stderr,"=== RKNN_MATMUL_RUN %d (B set once, K=%d N=%d M=%d) ===\n",i,K,N,M); rknn_matmul_run(ctx); }
    rknn_destroy_mem(ctx,Am); rknn_destroy_mem(ctx,Bm); rknn_destroy_mem(ctx,Cm); rknn_matmul_destroy(ctx);
    fprintf(stderr,"=== DONE (%d runs) ===\n",runs); return 0;
}
