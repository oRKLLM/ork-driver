/* tools/re/i8_mcap_probe.c — does the int8 M-tile cap under-report, like fp16's did?
 *
 * BACKGROUND. The fp16 sweep (tools/re/f16_mcap_probe, f16_k128_probe) showed 0x1040
 * (RK_CNA_CBUF_CON0) is a CBUF **bank split**: v = (WEIGHT_BANK<<4)|DATA_BANK with
 * DBNK+WBNK == 12 invariant, walking one bank at a time toward data-heavy as M grows. The
 * ceiling is where WEIGHT_BANK hits 1, i.e. 11 data banks. Measured bank size = 32 KB:
 *     fp16 (2 B/elem): M_max = 11*32768/(2K) = 180224/K   -> 704@256, 352@512, 176@1024 (exact)
 * int8 is 1 B/elem and uses the same register with scale=K/512, so the model predicts
 *     int8 M_max = 11*32768/K = 360448/K
 * Corroboration already in-tree: npu.c's sched=0 comment records int8 K=256 going garbage past
 * 32768/256 = 128 rows = exactly ONE 32 KB bank; and i8/regcmd.c's ORK_R1040 comment cites
 * rknn's CAPTURED 0x75 -> DBNK=5, WBNK=7, sum 12.
 *
 * WHAT THIS TESTS. AGENTS.md states the int8 cap is mg_max*64 and that mc+1 miscomputes "at
 * every K". mg_max uses INTEGER division, so it truncates; the two disagree where slope does not
 * divide (base-0x1b):
 *     K      mg_max*64   360448/K
 *     512    704         704        (agree — nothing to learn)
 *     1024   320         352        (+10%)
 *     2048   128         176        (+37%)
 *     3072    64         117        (+83%)
 *     4096    64          88        (+37%)
 *
 * METHOD. ORK_MCAP forces the int8 M-tile (npu.c, full-K M-scheduler path: needs K%512==0,
 * K<=4096, M>1, w->Bf). Set ORK_MCAP=M so the whole M is ONE program, then compare BIT-EXACTLY
 * against a CPU int32 reference — int8 matmul is exact in int32, so any difference is real.
 * ORK_MCAP's getenv is cached in a static, so this probe does ONE (K,M) per process; drive the
 * sweep from a shell loop. Scan M UPWARD — never bisect, the fp16 work proved the predicate can
 * be non-monotonic.
 *
 *   make i8_mcap_probe
 *   for M in ...; do sudo env ORK_MCAP=$M ./i8_mcap_probe <K> $M; done
 */
#define _GNU_SOURCE
#include "ork_npu.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc,char**argv){
    int K=argc>1?atoi(argv[1]):1024;
    int M=argc>2?atoi(argv[2]):320;
    int N=argc>3?atoi(argv[3]):64;
    if(K%512 || K>4096 || M<2){ printf("K must be %%512 and <=4096, M>1\n"); return 2; }

    /* label this M with the bank split orki_i8_synth will program (scale=K/512 for int8) */
    double scale=(double)K/512.0;
    int base=(int)(177.0-15.0*(scale-1.0)), slope=(int)(15.0*scale);
    int mg=(M+63)/64; if(mg<1)mg=1;
    int v=base-slope*(mg-1); int sat=(v<0x1b); if(sat) v=0x1b;
    int mg_max = base>=0x1b ? (base-0x1b)/slope+1 : 0;
    int pred_mg64 = mg_max*64, pred_bank = 360448/K;

    ork_npu *c=ork_npu_init(); if(!c){ printf("no board\n"); return 0; }

    int8_t *A=malloc((size_t)M*K), *B=malloc((size_t)K*N);
    int32_t *C=malloc((size_t)M*N*4), *ref=malloc((size_t)M*N*4);
    if(!A||!B||!C||!ref){ printf("OOM\n"); return 2; }
    for(int m=0;m<M;m++) for(int k=0;k<K;k++) A[(size_t)m*K+k]=(int8_t)(((m*7+k*3)%17)-8);
    for(int k=0;k<K;k++) for(int n=0;n<N;n++)  B[(size_t)k*N+n]=(int8_t)(((k+n*5)%11)-5);

    /* CPU int32 reference — EXACT, so the comparison is bit-exact (no tolerance) */
    for(int m=0;m<M;m++) for(int n=0;n<N;n++){ int32_t a=0;
        for(int k=0;k<K;k++) a+=(int32_t)A[(size_t)m*K+k]*(int32_t)B[(size_t)k*N+n];
        ref[(size_t)m*N+n]=a; }

    ork_w *w=ork_i8_mm_pack(c,K,N,B);
    if(!w){ printf("pack fail\n"); return 2; }
    memset(C,0,(size_t)M*N*4);
    int rc=ork_i8_mm_run(c,w,M,A,C);

    int firstbad=-1, nbad=0;
    if(!rc) for(int m=0;m<M && firstbad<0;m++) for(int n=0;n<N;n++)
        if(C[(size_t)m*N+n]!=ref[(size_t)m*N+n]){ firstbad=m; break; }
    if(!rc) for(size_t i=0;i<(size_t)M*N;i++) if(C[i]!=ref[i]) nbad++;

    printf("K=%-5d M=%-5d mg=%-3d v=0x%02x DBNK=%-2d WBNK=%-2d %-4s | mg64=%-4d bank=%-4d | %s",
           K,M,mg,v,v&0xf,(v>>4)&0xf,sat?"SAT":"-",pred_mg64,pred_bank,
           rc? "RC-FAIL" : (firstbad<0? "OK (bit-exact)":"MISMATCH"));
    if(rc) printf(" rc=%d",rc);
    else if(firstbad>=0) printf(" firstbad_row=%d bad_elems=%d/%d",firstbad,nbad,M*N);
    printf("\n");

    ork_mm_free(c,w); free(A);free(B);free(C);free(ref); ork_npu_free(c);
    return rc? 2 : (firstbad<0?0:1);
}
