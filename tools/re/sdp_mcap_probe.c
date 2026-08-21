/* tools/re/sdp_mcap_probe.c — measure the M ceiling of the int8 and fp16 SDP ops.
 *
 * WHY. 21 SDP-family guards all carry `M <= 8192`. That number was never measured: it is
 * RK_DPU_DATA_CUBE_WIDTH's 13-bit field (mask 0x1fff => M-1 <= 8191) read as a capability. On the
 * int16 path the hardware actually stops at 8176 — M=8184 returns errno=110 (submit TIMEOUT) plus a
 * self-healing reset, i.e. the guard was admitting a FAULT-GENERATING shape (fixed in b351c14,
 * ORK_SDP_MAXM). The remaining 13 sites (8 int8, 5 fp16) share the same geometry patcher
 * (orki_set_mul_geom) and are LIKELY subject to the same ceiling — but "likely" is what produced
 * every bound this campaign has had to correct, so: measure them.
 *
 * METHOD (identical to the fp16/int16 work, so the numbers are comparable):
 *   - REFERENCE = the same op in M=8 chunks (the CAPTURED cube shape). These ops are ELEMENTWISE, so
 *     row m depends only on row m: a chunked reference is exact and cannot inherit a large-M geometry
 *     error the way a single large-M reference would.
 *   - SCAN M UPWARD, never bisect (this predicate class is non-monotonic on fp16 K=128).
 *   - Do NOT stop at the first failure — keep walking, so a recovery is visible.
 *   - Compare BIT-EXACTLY: int8/fp16 SDP is deterministic, so any difference at all is the envelope.
 *
 * Ops covered (two per dtype, to catch a per-op rather than per-dtype ceiling):
 *   int8: ork_i8_npu_ewmul, ork_i8_npu_add        (N%16)
 *   fp16: ork_f16_npu_ewmul, ork_f16_npu_add      (N%8)
 *
 *   make sdp_mcap_probe
 *   sudo tools/util/npu_guard.sh -- env ORK_MM_TIMEOUT=2000 ./sdp_mcap_probe [N] [Mmax]
 */
#define _GNU_SOURCE
#include "ork_npu.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Reference chunk size. Was 8 (the captured cube width) — correct but RUINOUS: an 8183-row reference
 * then needs ~1023 allocate/submit/free cycles PER OP, which fragments the IOVA space until bcreate
 * starts failing ([iova@fail], total live=11 MiB) and the box hard-wedges. That cost two power cycles.
 * These ops are now MEASURED bit-exact from M=8 to 7680+, so a 1024-row chunk is equally trustworthy
 * and needs ~8 allocations instead of ~1023. Keep this comfortably inside the proven-good range. */
#define REFM 1024

static int  g_N;
static int8_t   *i8a,*i8b,*i8o,*i8r;
static ork_f16  *f16a,*f16b,*f16o,*f16r;

/* each runner does ONE call of its op over M rows, writing into `dst` */
/* r0 = FIRST INPUT ROW. Essential: a reference chunk must compute the rows it stands for, otherwise
 * every chunk recomputes rows 0..REFM-1 and the reference is chunk 0 repeated (a probe bug that
 * reported a bogus "ceiling of 8" on the first run of this tool). */
static int run_i8_ewmul (ork_npu *c,int M,int r0,void *dst){ return ork_i8_npu_ewmul (c,i8a+(size_t)r0*g_N,i8b+(size_t)r0*g_N,M,g_N,1,0,(int8_t*)dst,NULL); }
static int run_i8_add   (ork_npu *c,int M,int r0,void *dst){ return ork_i8_npu_add   (c,i8a+(size_t)r0*g_N,i8b+(size_t)r0*g_N,M,g_N,1.0,1.0,1.0,(int8_t*)dst,NULL); }
static int run_f16_ewmul(ork_npu *c,int M,int r0,void *dst){ return ork_f16_npu_ewmul(c,f16a+(size_t)r0*g_N,f16b+(size_t)r0*g_N,M,g_N,(ork_f16*)dst,NULL); }
static int run_f16_add  (ork_npu *c,int M,int r0,void *dst){ return ork_f16_npu_add  (c,f16a+(size_t)r0*g_N,f16b+(size_t)r0*g_N,M,g_N,(ork_f16*)dst,NULL); }

struct op { const char *name; int (*fn)(ork_npu*,int,int,void*); int esz; void **out; void **ref; };

static int differs(const void *a,const void *b,size_t bytes){ return memcmp(a,b,bytes)!=0; }
static int first_bad_row(const char *got,const char *ref,int M,int N,int esz){
    for(int m=0;m<M;m++) if(memcmp(got+(size_t)m*N*esz, ref+(size_t)m*N*esz, (size_t)N*esz)) return m;
    return -1;
}

int main(int argc,char**argv){
    g_N      = argc>1?atoi(argv[1]):64;     /* 64 satisfies both N%16 (int8) and N%8 (fp16) */
    int Mmax = argc>2?atoi(argv[2]):8192;
    if(g_N%16){ printf("N must be %%16 so int8 and fp16 share it\n"); return 2; }

    ork_npu *c=ork_npu_init(); if(!c){ printf("no board\n"); return 0; }
    printf("sdp_mcap_probe — SoC=%s  N=%d  ref=M%d chunks  (int16 ceiling for reference: 8176)\n",
           ork_npu_soc(c),g_N,REFM);

    size_t n=(size_t)Mmax*g_N;
    i8a=malloc(n); i8b=malloc(n); i8o=malloc(n); i8r=malloc(n);
    f16a=malloc(n*2); f16b=malloc(n*2); f16o=malloc(n*2); f16r=malloc(n*2);
    if(!i8a||!i8b||!i8o||!i8r||!f16a||!f16b||!f16o||!f16r){ printf("OOM\n"); return 2; }
    for(size_t i=0;i<n;i++){
        i8a[i]=(int8_t)((i*7)%61-30); i8b[i]=(int8_t)((i*13)%41-20);
        f16a[i]=(ork_f16)(0.5f+0.001f*(float)((i*7)%17)); f16b[i]=(ork_f16)(0.25f+0.001f*(float)((i*13)%11)); }

    void *o_i8=i8o,*r_i8=i8r,*o_f=f16o,*r_f=f16r;
    struct op ops[]={
        {"i8_ewmul",  run_i8_ewmul,  1, &o_i8,&r_i8},
        {"i8_add",    run_i8_add,    1, &o_i8,&r_i8},
        {"f16_ewmul", run_f16_ewmul, 2, &o_f, &r_f },
        {"f16_add",   run_f16_add,   2, &o_f, &r_f },
    };

    /* SDP_OP=<substr> runs only matching ops. Needed to pin per-dtype boundaries WITHOUT submitting
     * a known-bad shape: int8 dies at 8184 while fp16 survives it, so one shared M list would wedge
     * the core on the int8 ops before the fp16 ones ran. Every over-ceiling SDP submit leaves core 0
     * stuck at 100% and costs a reboot, so probe only points expected to PASS, bounded below the
     * already-known-bad value. */
    const char *only = getenv("SDP_OP");
    for(size_t oi=0; oi<sizeof ops/sizeof*ops; oi++){
        struct op *op=&ops[oi];
        if(only && !strstr(op->name,only)) continue;
        printf("\n=== %s (esz=%d) ===\n",op->name,op->esz);
        char *ref=(char*)*op->ref, *out=(char*)*op->out;
        /* chunked reference at the captured shape */
        int bad=0;
        for(int m0=0;m0<Mmax && !bad;m0+=REFM){
            int mc=(Mmax-m0<REFM)?(Mmax-m0):REFM;
            /* the runners always read from offset 0, so build the reference by copying the
             * chunk result into place after running it on the leading rows */
            if(op->fn(c,mc,m0,ref+(size_t)m0*g_N*op->esz)) bad=1;
        }
        if(bad){ printf("  reference failed — op unsupported here, skipping\n"); continue; }
        /* ref[m] is now the op applied to input row m (chunk m0 read rows m0..), so a large-M run
         * compares row-for-row against it directly. */
        printf("  %-7s %-9s %-10s %s\n","M","M*16","firstbad","verdict");
        int lastbad=0,recovered=0,nok=0,nbad=0,maxok=0;
        /* argv[3..] = explicit ascending M list (to pin a boundary the ladder steps over) */
        int exn = argc>3 ? argc-3 : 0, ei = 0;
        for(int M=(exn?atoi(argv[3]):REFM); M<=Mmax;
                M=(exn ? (++ei<exn ? atoi(argv[3+ei]) : Mmax+1)
                       : (M<64?M+8:(M<1024?M*2:M+512)))){
            memset(out,0,(size_t)M*g_N*op->esz);
            int rc=op->fn(c,M,0,out);
            if(rc){ printf("  %-7d %-9d %-10s rc=%d\n",M,M*16,"-",rc); lastbad=1; nbad++; continue; }
            /* every REFM-row block must equal the reference's first block */
            int fb=first_bad_row(out,ref,M,g_N,op->esz);
            if(fb<0){ nok++; maxok=M; if(lastbad){recovered=1; printf("    ^^^ RECOVERED — NON-MONOTONIC\n");} lastbad=0;
                      printf("  %-7d %-9d %-10s OK\n",M,M*16,"-"); }
            else    { nbad++; lastbad=1; printf("  %-7d %-9d %-10d MISMATCH\n",M,M*16,fb); }
            (void)differs; (void)first_bad_row;
        }
        printf("  -> %s: max ok M=%d  (%d ok, %d bad, non-monotonic=%s)\n",
               op->name,maxok,nok,nbad,recovered?"YES":"no");
    }
    ork_npu_free(c);
    return 0;
}
