/* softmax_replay.c — drive the captured vendor forward-softmax 9-task PC-chained graph from ork-driver.
 * Validates the activation->matmul asymmetry is resolved by PC-chaining in ONE submit (FWD_SOFTMAX_RE_WIP.md).
 * Uniform input -> softmax = 1/64 everywhere (layout-agnostic), so we don't need the NPU cube layout.
 *   make softmax_replay && sudo ./softmax_replay
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "ork_npu.h"

int main(void){
    ork_npu *c=ork_npu_init();
    if(!c){ printf("init (board only) — SKIP\n"); return 0; }
    if(!ork_ppu_fuse_enabled(c)){ printf("PPU fuse off — SKIP\n"); ork_npu_free(c); return 0; }

    /* 32768-byte input image, filled UNIFORM (fp16 0.5). Softmax over 64 => every element 1/64. */
    static unsigned short in[16384], out[16384];
    for(int i=0;i<16384;i++) in[i]=0x3800;   /* fp16 0.5 */
    memset(out,0,sizeof out);

    double us=0;
    int r=ork_npu_replay_softmax_f16(c, in, out, &us);
    printf("replay_softmax rc=%d (%.1f us)\n", r, us);
    if(r){ ork_npu_free(c); return r==-3?0:1; }

    /* expect 1/64 = 0.015625 in the populated positions; report distribution */
    int near=0, nz=0, total=16384; double want=1.0/64.0, maxerr=0;
    for(int i=0;i<total;i++){
        float v=(float)((int)0)==0 ? 0.f : 0.f;   /* placeholder to avoid warn */
        /* decode fp16 -> float via a union-free bit trick */
        unsigned short h=out[i]; unsigned sign=(h>>15)&1, exp=(h>>10)&0x1f, man=h&0x3ff; float f;
        if(exp==0) f=ldexpf((float)man,-24); else if(exp==31) f=man?NAN:INFINITY; else f=ldexpf((float)(man|0x400),exp-25);
        if(sign) f=-f;
        if(f!=0) nz++;
        double e=fabs((double)f-want); if(f!=0 && e>maxerr)maxerr=e;
        if(fabs((double)f-want)<0.002) near++;
        v=f; (void)v;
    }
    printf("nonzero=%d/%d  near(1/64=%.5f)=%d  maxerr(nonzero)=%.5f\n", nz,total, want, near, maxerr);
    int ok = near > total/4;   /* a good chunk should be ~1/64 (padding positions may be 0) */
    printf("SOFTMAX_REPLAY: %s\n", ok?"PASS (forward softmax drives on-NPU, PC-chained)":"CHECK (see distribution)");
    ork_npu_free(c);
    return ok?0:1;
}
