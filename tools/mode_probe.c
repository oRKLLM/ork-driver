/* mode_probe — RK3588 NPU mode/state-transition matrix (SSM_ON_NPU §8 "NPU mode-switch handling").
 *
 * Characterizes which ordered op-A -> op-B transitions across SEPARATE submits wedge the NPU (errno=110),
 * and the minimal interposed action that makes them safe. Reproduces the test_ssd_chunk_npu mode-2 wedge
 * (fp16 CumBA `bmm` after an int16/fp16 SDP op) and the mode-1 survivor (exp -> bmm) in a controlled 2-op
 * probe, then sweeps {matmul dtype} x {SDP op} both directions.
 *
 * Ops (public validated primitives only):
 *   MM_F16   ork_bmm_fp16   (1,M=64,K=64,N=16)   -- the exact CumBA shape that wedges mid-sequence
 *   MM_I8    ork_i8_bmm     (1,M=64,K=64,N=64)
 *   EXP_I16  ork_i16_npu_exp   (SDP LUT, int16)
 *   SILU_I16 ork_i16_npu_silu  (SDP LUT, int16)
 *   EWMUL16  ork_i16_npu_ewmul (SDP 2-in, int16)
 *   EWMULF16 ork_f16_npu_ewmul (SDP 2-in, fp16)
 *   ADD_F16  ork_f16_npu_add   (SDP 2-in, fp16)
 *  (int16 MATMUL is not a public RK3588 primitive — matmul is int8/fp16/int4; the int16 datapath is
 *   SDP-only. Noted, not swept.)
 *
 * Per ordered pair we warm B's dtype, run A (the contaminator), then run B (the victim) as a SEPARATE
 * submit, under three interposed actions: none / mode_invalidate (clear cached last_dt/warmed, no HW
 * reset) / mode_reset (explicit ACT_RESET + invalidate). Reports rc, wall (a wedge ~= the submit timeout),
 * and the MINIMAL action that keeps B correct+non-wedged. A wedge is rc!=0 AND wall ~= ORK_MM/EW_TIMEOUT.
 *
 * BOARD SAFETY: run with short timeouts so a wedge self-terminates, e.g.
 *   sudo env ORK_MM_TIMEOUT=2500 ORK_EW_TIMEOUT=2500 ./mode_probe
 * After every op that may wedge we ork_npu_mode_reset + re-warm + health-check before continuing.
 * argv: [Aidx [Bidx]] runs a single pair (indices below); no args = the scoped sweep.
  *
 * !! BOARD-KILLER — DO NOT USE AS A ROUTINE GATE (2026-08-22) !!
 * This probe triggers a KERNEL OOPS in the rknpu DRM driver and takes the whole board down (no ping,
 * not just SSH); it cost three power cycles to characterise. The pair GELU_I8 -> MM_F16 completes but
 * burns the full ORK_MM_TIMEOUT and emits ~14 `RKNPU: soft reset`; the NEXT transition's buffer
 * realloc then issues a GEM destroy and the driver faults walking a corrupted scatter-gather table:
 *     virt_to_folio <- sg_kfree <- __sg_free_table <- sg_free_table
 *     <- rknpu_gem_object_destroy <- rknpu_gem_free_object <- rknpu_gem_destroy_ioctl
 * Neither pair reproduces alone (`mode_probe 8 0` and `mode_probe 8 1` are both safe); it needs the
 * sequence, so run SINGLE PAIRS (`mode_probe <a> <b>`) rather than a sweep. After the Oops the kernel
 * is tainted and dies on the next activity, even a read-only one. Driver bug, not an ork-driver one —
 * write-up + capture recipe: wiki "Exp-2026-08-22 mode_probe Kernel Oops".
 */
#include "ork_npu.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

enum { OP_MM_F16=0, OP_MM_I8, OP_EXP_I16, OP_SILU_I16, OP_EWMUL_I16, OP_EWMUL_F16, OP_ADD_F16,
       /* SDP-family ops added 2026-07-21 to extend the transition campaign (matmul-weight/perchan/softmax/
        * reshape/rope ops don't fit this simple A->B harness): */
       OP_SILU_I8, OP_GELU_I8, OP_GELU_I16, OP_RSQRT_I8, OP_EXP_I8, OP_EWMUL_I8, OP_ADD_I8, OP_ADD_I16, NOP };
static const char *OPNAME[NOP]={"MM_F16","MM_I8","EXP_I16","SILU_I16","EWMUL_I16","EWMUL_F16","ADD_F16",
       "SILU_I8","GELU_I8","GELU_I16","RSQRT_I8","EXP_I8","EWMUL_I8","ADD_I8","ADD_I16"};
static int is_mm(int op){ return op==OP_MM_F16 || op==OP_MM_I8; }

static double now_ms(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec*1e3 + t.tv_nsec/1e6; }

/* ---- individual ops. Return 0/ok (and, for the matmuls, correctness-checked), nonzero = fail/wedge. --- */
#define MM_M 64
#define MM_K 64
#define MM_NF16 16
#define MM_NI8  64
#define SDP_M 8
#define SDP_N 64

static int op_mm_f16(ork_npu *c){
    static ork_f16 *A=NULL,*B=NULL; static float *C=NULL;
    if(!A){ A=malloc((size_t)MM_M*MM_K*2); B=malloc((size_t)MM_K*MM_NF16*2); C=malloc((size_t)MM_M*MM_NF16*4);
        for(int i=0;i<MM_M*MM_K;i++) A[i]=(ork_f16)0.01f;
        for(int i=0;i<MM_K*MM_NF16;i++) B[i]=(ork_f16)(1.0f/MM_K); }
    if(ork_bmm_fp16(c,1,MM_M,MM_K,MM_NF16,A,B,C)) return 1;
    /* expected C[m][n] = sum_k 0.01*(1/64) = 0.01 */
    double worst=0; for(int i=0;i<MM_M*MM_NF16;i++){ double e=fabs(C[i]-0.01); if(e>worst)worst=e; }
    return worst>2e-2 ? 2 : 0;                    /* 2 = ran but wrong output */
}
static int op_mm_i8(ork_npu *c){
    static int8_t *A=NULL,*B=NULL; static int32_t *C=NULL;
    if(!A){ A=malloc((size_t)MM_M*MM_K); B=malloc((size_t)MM_K*MM_NI8); C=malloc((size_t)MM_M*MM_NI8*4);
        memset(A,1,(size_t)MM_M*MM_K); memset(B,1,(size_t)MM_K*MM_NI8); }
    if(ork_i8_bmm(c,1,MM_M,MM_K,MM_NI8,A,B,C)) return 1;
    for(int i=0;i<MM_M*MM_NI8;i++) if(C[i]!=MM_K) return 2;    /* sum_k 1*1 = 64 */
    return 0;
}
static int op_exp_i16(ork_npu *c){
    static int16_t *in=NULL,*out=NULL;
    if(!in){ in=malloc((size_t)SDP_M*SDP_N*2); out=malloc((size_t)SDP_M*SDP_N*2);
        for(int i=0;i<SDP_M*SDP_N;i++) in[i]=(int16_t)(-1000+i); }
    return ork_i16_npu_exp(c,in,SDP_M,SDP_N,30.0/30000.0,1.0/30000.0,out,NULL)?1:0;
}
static int op_silu_i16(ork_npu *c){
    static int16_t *in=NULL,*out=NULL;
    if(!in){ in=malloc((size_t)SDP_M*SDP_N*2); out=malloc((size_t)SDP_M*SDP_N*2);
        for(int i=0;i<SDP_M*SDP_N;i++) in[i]=(int16_t)(-2000+8*i); }
    return ork_i16_npu_silu(c,in,SDP_M,SDP_N,4.0/32768.0,1.0/32768.0,out,NULL)?1:0;
}
static int op_ewmul_i16(ork_npu *c){
    static int16_t *a=NULL,*b=NULL,*o=NULL;
    if(!a){ a=malloc((size_t)SDP_M*SDP_N*2); b=malloc((size_t)SDP_M*SDP_N*2); o=malloc((size_t)SDP_M*SDP_N*2);
        for(int i=0;i<SDP_M*SDP_N;i++){ a[i]=(int16_t)(100+i); b[i]=(int16_t)(200-i); } }
    return ork_i16_npu_ewmul(c,a,b,SDP_M,SDP_N,16384,14,o,NULL)?1:0;
}
static int op_ewmul_f16(ork_npu *c){
    static ork_f16 *a=NULL,*b=NULL,*o=NULL;
    if(!a){ a=malloc((size_t)SDP_M*SDP_N*2); b=malloc((size_t)SDP_M*SDP_N*2); o=malloc((size_t)SDP_M*SDP_N*2);
        for(int i=0;i<SDP_M*SDP_N;i++){ a[i]=(ork_f16)0.5f; b[i]=(ork_f16)0.25f; } }
    return ork_f16_npu_ewmul(c,a,b,SDP_M,SDP_N,o,NULL)?1:0;
}
static int op_add_f16(ork_npu *c){
    static ork_f16 *a=NULL,*b=NULL,*o=NULL;
    if(!a){ a=malloc((size_t)SDP_M*SDP_N*2); b=malloc((size_t)SDP_M*SDP_N*2); o=malloc((size_t)SDP_M*SDP_N*2);
        for(int i=0;i<SDP_M*SDP_N;i++){ a[i]=(ork_f16)0.5f; b[i]=(ork_f16)0.25f; } }
    return ork_f16_npu_add(c,a,b,SDP_M,SDP_N,o,NULL)?1:0;
}
/* --- SDP-family ops added 2026-07-21 (int8 activations + int8/int16 elementwise; same SDP_MxN shape) --- */
static int op_silu_i8(ork_npu *c){ static int8_t *in=NULL,*out=NULL;
    if(!in){ in=malloc(SDP_M*SDP_N); out=malloc(SDP_M*SDP_N); for(int i=0;i<SDP_M*SDP_N;i++) in[i]=(int8_t)(-64+i%128); }
    return ork_i8_npu_silu(c,in,SDP_M,SDP_N,4.0/128,1.0/128,out,NULL)?1:0; }
static int op_gelu_i8(ork_npu *c){ static int8_t *in=NULL,*out=NULL;
    if(!in){ in=malloc(SDP_M*SDP_N); out=malloc(SDP_M*SDP_N); for(int i=0;i<SDP_M*SDP_N;i++) in[i]=(int8_t)(-64+i%128); }
    return ork_i8_npu_gelu(c,in,SDP_M,SDP_N,4.0/128,1.0/128,out,NULL)?1:0; }
static int op_gelu_i16(ork_npu *c){ static int16_t *in=NULL,*out=NULL;
    if(!in){ in=malloc((size_t)SDP_M*SDP_N*2); out=malloc((size_t)SDP_M*SDP_N*2); for(int i=0;i<SDP_M*SDP_N;i++) in[i]=(int16_t)(-2000+8*i); }
    return ork_i16_npu_gelu(c,in,SDP_M,SDP_N,4.0/32768.0,1.0/32768.0,out,NULL)?1:0; }
static int op_rsqrt_i8(ork_npu *c){ static int8_t *in=NULL,*out=NULL;
    if(!in){ in=malloc(SDP_M*SDP_N); out=malloc(SDP_M*SDP_N); for(int i=0;i<SDP_M*SDP_N;i++) in[i]=(int8_t)(1+i%126); }   /* positive for rsqrt */
    return ork_i8_npu_rsqrt(c,in,SDP_M,SDP_N,4.0/128,1.0/128,out,NULL)?1:0; }
static int op_exp_i8(ork_npu *c){ static int8_t *in=NULL,*out=NULL;
    if(!in){ in=malloc(SDP_M*SDP_N); out=malloc(SDP_M*SDP_N); for(int i=0;i<SDP_M*SDP_N;i++) in[i]=(int8_t)(-127+i%128); }
    return ork_i8_npu_exp(c,in,SDP_M,SDP_N,4.0/128,1.0/128,out,NULL)?1:0; }
static int op_ewmul_i8(ork_npu *c){ static int8_t *a=NULL,*b=NULL,*o=NULL;
    if(!a){ a=malloc(SDP_M*SDP_N); b=malloc(SDP_M*SDP_N); o=malloc(SDP_M*SDP_N); for(int i=0;i<SDP_M*SDP_N;i++){ a[i]=(int8_t)(10+i%100); b[i]=(int8_t)(5+i%50); } }
    return ork_i8_npu_ewmul(c,a,b,SDP_M,SDP_N,16384,14,o,NULL)?1:0; }
static int op_add_i8(ork_npu *c){ static int8_t *a=NULL,*b=NULL,*o=NULL;
    if(!a){ a=malloc(SDP_M*SDP_N); b=malloc(SDP_M*SDP_N); o=malloc(SDP_M*SDP_N); for(int i=0;i<SDP_M*SDP_N;i++){ a[i]=(int8_t)(10+i%100); b[i]=(int8_t)(5+i%50); } }
    return ork_i8_npu_add(c,a,b,SDP_M,SDP_N,1.0/128,1.0/128,1.0/128,o,NULL)?1:0; }
static int op_add_i16(ork_npu *c){ static int16_t *a=NULL,*b=NULL,*o=NULL;
    if(!a){ a=malloc((size_t)SDP_M*SDP_N*2); b=malloc((size_t)SDP_M*SDP_N*2); o=malloc((size_t)SDP_M*SDP_N*2); for(int i=0;i<SDP_M*SDP_N;i++){ a[i]=(int16_t)(100+i); b[i]=(int16_t)(200-i); } }
    return ork_i16_npu_add(c,a,b,SDP_M,SDP_N,1.0/32768.0,1.0/32768.0,1.0/32768.0,o,NULL)?1:0; }
static int run_op(ork_npu *c,int op){
    switch(op){
        case OP_MM_F16:    return op_mm_f16(c);
        case OP_MM_I8:     return op_mm_i8(c);
        case OP_EXP_I16:   return op_exp_i16(c);
        case OP_SILU_I16:  return op_silu_i16(c);
        case OP_EWMUL_I16: return op_ewmul_i16(c);
        case OP_EWMUL_F16: return op_ewmul_f16(c);
        case OP_ADD_F16:   return op_add_f16(c);
        case OP_SILU_I8:   return op_silu_i8(c);
        case OP_GELU_I8:   return op_gelu_i8(c);
        case OP_GELU_I16:  return op_gelu_i16(c);
        case OP_RSQRT_I8:  return op_rsqrt_i8(c);
        case OP_EXP_I8:    return op_exp_i8(c);
        case OP_EWMUL_I8:  return op_ewmul_i8(c);
        case OP_ADD_I8:    return op_add_i8(c);
        case OP_ADD_I16:   return op_add_i16(c);
    }
    return -1;
}
/* timed op: fills *ms with wall; returns rc. */
static int timed(ork_npu *c,int op,double *ms){ double t0=now_ms(); int rc=run_op(c,op); *ms=now_ms()-t0; return rc; }

/* health probe: a known-good matmul of each dtype must run + be correct. 0 = healthy. */
static int health(ork_npu *c){
    ork_npu_mode_reset(c);
    double ms; int r1=timed(c,OP_MM_I8,&ms); ork_npu_mode_reset(c); int r0=timed(c,OP_MM_F16,&ms);
    return (r0||r1)?1:0;
}

/* Run one ordered pair under a given interposed action; returns 0 if B ran correct + fast (safe). */
enum { FIX_NONE=0, FIX_INVAL, FIX_RESET };
static const char *FIXNAME[3]={"none","invalidate","reset"};
static int run_pair(ork_npu *c,int A,int B,int fix,double *msA,double *msB,int *rcA,int *rcB){
    /* establish a clean warm state for B's dtype (so the ONLY variable is the A->B transition) */
    ork_npu_mode_reset(c);
    double d; run_op(c,B); run_op(c,B);         /* warm B twice (clears fresh-buffer stale) */
    *rcA=timed(c,A,msA);                          /* op A: the contaminator */
    if(fix==FIX_INVAL) ork_npu_mode_invalidate(c);
    else if(fix==FIX_RESET) ork_npu_mode_reset(c);
    *rcB=timed(c,B,msB);                          /* op B: the victim, SEPARATE submit */
    (void)d;
    return *rcB!=0;
}

int main(int argc,char**argv){
    ork_npu *c=ork_npu_init();
    if(!c){ fprintf(stderr,"[mode_probe] no NPU — skipping\n"); return 0; }
    fprintf(stderr,"[mode_probe] SoC=%s cores=%d\n",ork_npu_soc(c),ork_npu_cores(c));
    if(health(c)){ fprintf(stderr,"[mode_probe] board NOT healthy at start — aborting\n"); ork_npu_free(c); return 2; }

    /* ONE-SHOT clean test: `mode_probe A B fix` (fix 0=none 1=invalidate 2=reset). Runs the single A->B
     * transition with exactly that interposed action in THIS fresh process (starts healthy), then reports.
     * This is the ONLY correct way to test whether an interposed reset PREVENTS the wedge — the multi-fix
     * loop below contaminates: once fix=none wedges, the exp/silu wedge is unrecoverable in-process, so a
     * later fix=reset trial never gets a clean board. Use one fresh process per (pair,fix). */
    if(argc>=4){
        int A=atoi(argv[1]),B=atoi(argv[2]),fx=atoi(argv[3]);
        if(A<0||A>=NOP||B<0||B>=NOP||fx<0||fx>FIX_RESET){ fprintf(stderr,"bad args\n"); ork_npu_free(c); return 2; }
        double msA,msB; int rcA,rcB; int wedged=run_pair(c,A,B,fx,&msA,&msB,&rcA,&rcB);
        printf("ONESHOT %s -> %s  fix=%s : A(rc=%d,%.1fms) B(rc=%d,%.1fms) -> %s\n",
               OPNAME[A],OPNAME[B],FIXNAME[fx],rcA,msA,rcB,msB, wedged?"WEDGE":"SAFE");
        ork_npu_mode_reset(c); ork_npu_free(c); return wedged?1:0;
    }

    /* scoped sweep: SDP->matmul (the hazard direction) + matmul->SDP, plus matmul->matmul baselines. */
    int As[64],Bs[64],np=0;
    if(argc==3){ As[0]=atoi(argv[1]); Bs[0]=atoi(argv[2]); np=1; }
    else if(argc==2){ int a=atoi(argv[1]); for(int b=0;b<NOP;b++){As[np]=a;Bs[np]=b;np++;} }
    else {
        /* every op -> each matmul (SDP->MM hazard + MM->MM baselines) */
        for(int a=0;a<NOP;a++){ As[np]=a; Bs[np]=OP_MM_F16; np++; As[np]=a; Bs[np]=OP_MM_I8; np++; }
        /* each matmul -> each SDP (the safe direction per the wiki; confirm) */
        for(int b=OP_EXP_I16;b<NOP;b++){ As[np]=OP_MM_F16; Bs[np]=b; np++; As[np]=OP_MM_I8; Bs[np]=b; np++; }
    }

    printf("\n  %-10s -> %-10s | %-22s | minimal-safe-action\n","A","B","none/inval/reset (rcB,ms)");
    printf("  ------------------------------------------------------------------------------------\n");
    int nwedge=0;
    for(int i=0;i<np;i++){
        int A=As[i],B=Bs[i]; if(A<0||A>=NOP||B<0||B>=NOP) continue;
        int minfix=-1; char cell[3][40];
        for(int fx=FIX_NONE;fx<=FIX_RESET;fx++){
            double msA,msB; int rcA,rcB;
            int wedged=run_pair(c,A,B,fx,&msA,&msB,&rcA,&rcB);
            snprintf(cell[fx],sizeof cell[fx],"%s(%d,%.0f)",wedged?"WEDGE":"ok",rcB,msB);
            if(rcB) nwedge++;
            /* recover before the next trial regardless */
            ork_npu_mode_reset(c);
            if(health(c)){ fprintf(stderr,"[mode_probe] board unhealthy after %s->%s fix=%s — attempting reset+continue\n",
                                    OPNAME[A],OPNAME[B],FIXNAME[fx]); ork_npu_mode_reset(c); health(c); }
            if(!wedged){ if(minfix<0)minfix=fx; break; }  /* short-circuit: first passing action is the minimal one */
            if(minfix<0 && fx==FIX_RESET) minfix=99;      /* wedges even after reset */
        }
        const char *m = minfix==FIX_NONE?"none (safe)": minfix==FIX_INVAL?"INVALIDATE (re-warm)":
                        minfix==FIX_RESET?"ACT_RESET": "WEDGES EVEN AFTER RESET";
        printf("  %-10s -> %-10s | none=%-12s | %s\n",OPNAME[A],OPNAME[B],cell[0],m);
        fflush(stdout);
    }
    printf("\n  (total wedged submits observed: %d — kernel soft-resets + recovers each)\n",nwedge);
    ork_npu_free(c);
    return 0;
}
