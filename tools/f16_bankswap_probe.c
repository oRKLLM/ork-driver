/* f16_bankswap_probe — is the fp16 mcap=4 at K=3584 a HARDWARE capacity or an unprogrammed register?
 *
 * Claim under test (from source, src/npu/f16/regcmd.c:173-177): the sched=0 branch of orki_f16_synth
 * never writes RK_CNA_CBUF_CON0 at all —
 *     if(sched){ ... orki_setrn(rc,...,RK_CNA_CBUF_CON0,v); }
 *     else     { orki_setrn(rc,...,RK_CNA_CONV_CON2,16*(mc+1)); }
 * so at non-power-of-two K the bank split inherits the TEMPLATE default in regcmd_array_4x32x16.h,
 * word 0x00b11040 → value 0xb1 → DATA_BANK=1, WEIGHT_BANK=11. One data bank is 32768 B, a K=3584 fp16
 * row is 7168 B, so 4.57 rows fit → the observed mcap=4. That is the template's allocation, not the
 * silicon's: the CBUF has 12 banks at K=3584 like anywhere else, 11 are just assigned to weights.
 *
 * sched=1's affine formula is broken this far out (K=3584 → scale=14 → base=(int)(177-195)=-18) and its
 * clamp `if(v<0x1b)v=0x1b` rescues it to 0x1b → DATA_BANK=11, WEIGHT_BANK=1 — 0xb1 nibble-swapped, and
 * 11*32768/7168 = 50.28 → mc=50, exactly the 180224/K figure. So the prediction is sharp:
 *
 *     at K=3584, force 0x1040=0x1b and mc<=50  =>  BIT-EXACT and ~4-6x faster.
 *
 * THREE ARMS, and the middle one is the point. A numerics check with no configuration that MUST fail is
 * vacuous — this harness has already produced one false pass (operands as multiples of 2^-9 accumulate
 * exactly in fp32, so every shape reported 0.00e+00), and this repo produced another last month (a 2-bit
 * LCG whose period equalled mcap*K made consecutive M-tiles bit-identical, and eight hypotheses were
 * fitted to a pattern that did not exist). So:
 *     A  default            mcap 4,  banks 0xb1  -> reference, known good
 *     B  mcap 50, NO swap   banks 0xb1           -> MUST MISCOMPUTE (positive control)
 *     C  mcap 50, swap      banks 0x1b           -> hypothesis: bit-exact vs A
 * If B does not fail, the checker is broken and C's pass means nothing.
 *
 * SAFETY: 0x1b is the CLAMP FLOOR. Writing exactly 0x1b is safe; forcing BELOW it hangs the submit, which
 * is the IOMMU-wedge path on a shared board. This probe never writes below 0x1b. Run under npu_guard with
 * a timeout regardless.
 */
#include <ork_npu.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#define CNA_BLK 0x201
#define REG_CBUF_CON0 0x1040
#define REG_CONV_CON2 0x1010
#define BANKS_SWAPPED 0x1b   /* DATA_BANK=11, WEIGHT_BANK=1  — the sched=1 clamp value */

/* The two registers are written AS A PAIR in the sched=1 branch, and the first attempt at this test moved
 * only the bank split — leaving CONV_CON2 at the sched=0 value 16*(mc+1) = 816 while telling the CBUF it
 * had 11 data banks. Inconsistent schedule vs split, and the submit HUNG (sentinel never landed, soft
 * reset num 6, orphaned job holding core 0, board needed a reboot). A real sched=1 emission computes
 *     R = pow2_floor(cbuf/K) ; rows = min(mc+1, R) ; CONV_CON2 = 16*rows
 * which at K=3584, cbuf=57344 gives R=16, rows=16, CONV_CON2=256. Note rows caps at R regardless of mc,
 * so mc=50 is multi-pass — that is normal, not a problem: the validated K=1024/mcap=176 case is equally
 * multi-pass (rows=32 against mc=176) and emits exactly 0x1b itself. */
static int conv_con2_for(int K,int mc,int cbuf){
    int R=cbuf/K; if(R<1)R=1; { int rp2=1; while(rp2*2<=R) rp2*=2; R=rp2; }
    int rows=(mc+1<R)?(mc+1):R; return 16*rows;
}

static double now_us(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec*1e6+t.tv_nsec/1e3; }
static unsigned s_=4242; static float rnd(void){ s_=s_*1103515245u+12345u; return (float)((int)((s_>>13)&0x7fff)-16384)/16384.0f; }

static void set_mtile(int v){ static char b[32]; snprintf(b,sizeof b,"%d",v); setenv("ORK_F16_MTILE",b,1); }

int main(int argc,char**argv){
    int M = argc>1?atoi(argv[1]):1024;
    int K = argc>2?atoi(argv[2]):3584;
    int N = argc>3?atoi(argv[3]):1024;
    int mc_hi = argc>4?atoi(argv[4]):50;
    ork_npu *c = ork_npu_init(); if(!c){ printf("init failed\n"); return 1; }

    printf("K=%d N=%d M=%d   1 bank holds %d rows, 11 banks hold %d rows\n",
           K,N,M, 32768/(K*2), 11*32768/(K*2));
    ork_f16 *A=malloc((size_t)M*K*2), *B=malloc((size_t)K*N*2);
    float *Ca=malloc((size_t)M*N*4), *Cb=malloc((size_t)M*N*4), *Cc=malloc((size_t)M*N*4);
    if(!A||!B||!Ca||!Cb||!Cc){ printf("alloc failed\n"); return 1; }
    for(size_t i=0;i<(size_t)M*K;i++) A[i]=(ork_f16)rnd();
    for(size_t i=0;i<(size_t)K*N;i++) B[i]=(ork_f16)rnd();

    ork_w *w=ork_f16_mm_pack(c,K,N,B); if(!w){ printf("pack failed\n"); return 1; }
    double ta,tb,tc; size_t nel=(size_t)M*N;

    /* ---- A: default. mcap 4, template banks 0xb1. The reference. ---- */
    ork_f16_fuzz_clear(); unsetenv("ORK_F16_MTILE");
    if(ork_f16_mm_run(c,w,M,A,Ca)){ printf("arm A failed\n"); return 1; }
    ta=now_us(); ork_f16_mm_run(c,w,M,A,Ca); ta=now_us()-ta;

    /* ---- B: POSITIVE CONTROL. Raise the tile WITHOUT moving the banks -> must be wrong. ---- */
    ork_f16_fuzz_clear(); set_mtile(mc_hi);
    if(ork_f16_mm_run(c,w,M,A,Cb)){ printf("arm B failed\n"); return 1; }
    tb=now_us(); ork_f16_mm_run(c,w,M,A,Cb); tb=now_us()-tb;

    /* ---- C: the hypothesis. Raise the tile AND move the banks AND make the grain count agree. ---- */
    int cc2 = conv_con2_for(K,mc_hi,57344);
    printf("arm C will emit CBUF_CON0=0x%02x, CONV_CON2=%d (sched=0 would have used %d)\n",
           BANKS_SWAPPED, cc2, 16*(mc_hi+1));
    ork_f16_fuzz_clear();
    ork_f16_fuzz_add(CNA_BLK,REG_CBUF_CON0,BANKS_SWAPPED);
    ork_f16_fuzz_add(CNA_BLK,REG_CONV_CON2,cc2);
    set_mtile(mc_hi);
    if(ork_f16_mm_run(c,w,M,A,Cc)){ printf("arm C failed (submit error / hang)\n"); return 1; }
    tc=now_us(); ork_f16_mm_run(c,w,M,A,Cc); tc=now_us()-tc;
    ork_f16_fuzz_clear(); unsetenv("ORK_F16_MTILE");

    long db=0,dc=0; double mb=0,mc_=0;
    for(size_t i=0;i<nel;i++){
        if(Cb[i]!=Ca[i]){ db++; double d=fabs(Cb[i]-Ca[i]), r=fabs(Ca[i])>1e-6?fabs(Ca[i]):1e-6; if(d/r>mb)mb=d/r; }
        if(Cc[i]!=Ca[i]){ dc++; double d=fabs(Cc[i]-Ca[i]), r=fabs(Ca[i])>1e-6?fabs(Ca[i]):1e-6; if(d/r>mc_)mc_=d/r; }
    }
    double g=2.0*M*K*N;
    printf("\n%-42s %10s %10s %12s %s\n","arm","us","GFLOP/s","differing","max relerr vs A");
    printf("%-42s %10.0f %10.1f %12s %s\n","A default      mcap 4,  banks 0xb1",ta,g/(ta*1e3),"-","(reference)");
    printf("%-42s %10.0f %10.1f %12ld %.2e\n","B control      mcap 50, banks 0xb1",tb,g/(tb*1e3),db,mb);
    printf("%-42s %10.0f %10.1f %12ld %.2e\n","C hypothesis   mcap 50, banks 0x1b",tc,g/(tc*1e3),dc,mc_);

    printf("\n");
    if(db==0){ printf("VOID — the positive control (B) did NOT miscompute. The checker cannot see a wrong\n"
                      "       answer here, so C's result carries no information. Do not report C.\n");
               ork_mm_free(c,w); ork_npu_free(c); return 3; }
    printf("control OK: raising the tile without moving the banks corrupts %ld/%zu outputs.\n",db,nel);
    if(dc==0) printf("RESULT: bit-exact with the bank swap at mcap=%d, %.2fx faster than default.\n"
                     "        mcap=4 was an unprogrammed register, not a hardware capacity.\n", mc_hi, ta/tc);
    else      printf("RESULT: the bank swap did NOT restore correctness (%ld differing). The 11-bank split\n"
                     "        is not sufficient at this K — something else also needs programming.\n", dc);
    ork_mm_free(c,w); free(A);free(B);free(Ca);free(Cb);free(Cc);
    ork_npu_free(c); return 0;
}
