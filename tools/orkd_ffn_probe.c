/* orkd_ffn_probe — validate the COALESCED FFN inner routed THROUGH orkd (ORKD_FFN / orkd_ffn_i8).
 * Mirrors chain_gu_silu_probe's ORK_GSILU_FFN5 (which passes on the DIRECT NPU) but runs the whole
 * SwiGLU [gate->silu->up->glu->down] as ONE daemon-side chain submit against 3 resident weights.
 * All-ones weights [512,512] + all-ones A -> deterministic: gate=up=32, silu=glu~61, down=glu*512=31232.
 *   make orkd orkd_ffn_probe && sudo env ORKD_BIN=$PWD/orkd ./orkd_ffn_probe
 */
#include "orkd_client.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
static double siluf(double x){ return x/(1.0+exp(-x)); }

int main(void){
    setvbuf(stdout,NULL,_IONBF,0);
    orkd_conn *c = orkd_connect();
    if(!c){ fprintf(stderr,"connect/spawn FAILED\n"); return 1; }
    printf("connected: client_id=%u npu_cores=%u\n", orkd_client_id(c), orkd_soc_cores(c));

    enum { M=8, K=512, Nff=512, Kd=512 };
    static int8_t W[K*Nff]; for(int i=0;i<K*Nff;i++) W[i]=1;   /* all-ones weight, reused for gate/up/down */
    uint64_t gid=orkd_pack_i8(c,K,Nff,W), uid=orkd_pack_i8(c,K,Nff,W), did=orkd_pack_i8(c,Nff,Kd,W);
    if(!gid||!uid||!did){ printf("pack FAILED g=%llu u=%llu d=%llu\n",(unsigned long long)gid,(unsigned long long)uid,(unsigned long long)did); orkd_disconnect(c); return 1; }
    printf("packed gate=%llu up=%llu down=%llu\n",(unsigned long long)gid,(unsigned long long)uid,(unsigned long long)did);

    static int8_t A[M*K]; for(int i=0;i<M*K;i++) A[i]=1;
    static int32_t out[M*Kd]; memset(out,0,sizeof out);
    double is = 3.0/32.0, os = siluf(3.0)/60.0;
    int rc = orkd_ffn_i8(c, gid, uid, did, M, K, Nff, Kd,
                         0x4000,18, 0x4000,18, 0x4000,19, is, os, A, out);
    printf("orkd_ffn_i8 rc=%d\n", rc);
    if(rc==0){
        int want = 31232;   /* glu(~61) * Nff(512) */
        int nd=0, dmn=out[0], dmx=out[0];
        for(int i=0;i<M*Kd;i++){ int v=out[i]; if(abs(v-want)<=512)nd++; if(v<dmn)dmn=v; if(v>dmx)dmx=v; }
        printf("  [down] int32~%d: %d/%d  range[%d,%d]\n", want, nd, M*Kd, dmn, dmx);
        printf("  VERDICT: %s\n", nd==M*Kd ? "COALESCED FFN THROUGH ORKD WORKS (one socket round-trip, one submit)"
                                           : "down wrong -- see range");
    } else printf("  VERDICT: %s\n", rc==-1?"error/wedge":"error");
    orkd_free_weight(c,gid); orkd_free_weight(c,uid); orkd_free_weight(c,did);
    orkd_disconnect(c);
    return rc==0 ? 0 : 1;
}
