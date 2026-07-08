/* tools/i4_multim_fuzz.c — INT4 multi-M K-schedule fuzzer (RE), state-saving + wedge-resumable.
 *
 * Resurrects the fuzzer that discovered native int4 multi-M (Exp-2026-06-19) — but rewritten to drive the
 * REAL synth_i4 in src/npu.c through the public ork_i4_fuzz_add/clear + ork_npu_probe_i4_mm hooks (the old
 * copy duplicated npu.c internals and silently drifted out of sync). Re-pointed at the REMAINING wall
 * (Exp-2026-07-07): native batch mode is bit-exact at the capture K=64 (mregs=0x1f, reg 0x1040 left at its
 * base), but at production K (>=512) only row 0 computes. The int4 K-reduction schedule (reg 0x1040) and/or
 * a companion CNA/CBUF batch-activation-budget reg must be re-derived as a function of (K, mc). This sweeps them.
 *
 * SCORE. Drives synth_i4's multi-M path (mc>1 => mc_phys=2*mc, 0x405c=0, stride-2 output) and counts how many
 * LOGICAL rows match a CPU int4 reference under the stride-2 layout (logical row m at physical row 2m). The
 * number to beat is the baseline row-count with no override; a HIT is any (block,reg,val) that computes MORE
 * rows. K<=512 keeps the int16 datapath output exact, so "rows matched" is a true bit-exact signal.
 *
 * ROBUSTNESS. Fuzzing the regcmd can wedge the NPU/kernel. State is persisted so a power-cycle / SIGINT-timeout
 * resumes exactly where it stopped:
 *   i4_fuzz_state.txt      "<done> <inflight>"  done = candidates finished; inflight = index being attempted
 *                          now (-1 if none). On restart, a still-set inflight means that candidate wedged ->
 *                          its reg is blacklisted and the sweep resumes after it.
 *   i4_fuzz_blacklist.txt  one "<blk> <reg>" per line; blacklisted regs are skipped forever.
 *   i4_fuzz_hits.txt       appended HIT log (reg + rows matched).
 * Run UNDER an external timeout so a hang self-terminates cleanly (NEVER SIGKILL an in-flight submit):
 *   make i4_multim_fuzz && while :; do sudo timeout -s INT 120 ./i4_multim_fuzz 512 8 sched || sleep 1;
 *                                     grep -q '^DONE' i4_fuzz_state.txt && break; done
 * (loop auto-resumes across wedges; rm i4_fuzz_state.txt to start a fresh sweep.)
 *
 *   ./i4_multim_fuzz [K=512] [M=8] [mode=sched]
 *     sched : sweep reg 0x1040 (block 0x201) over 0x000..0x1ff — the prime suspect (K-reduction schedule).
 *     regs  : sweep EVERY register present in the captured REGCMD_I4 x a value palette (finds an unknown
 *             row-count/budget reg). Essential K/N/addr/M-count regs are skipped (overriding them is noise).
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include "ork_npu.h"
#include "regcmd_i4.h"      /* REGCMD_I4[] / REGCMD_I4_N — read-only, to enumerate present regs in "regs" mode */

#define NN 64              /* single 64-wide N-block: the known-good granularity (skips the 2D N-surface) */
/* [-2,2] keeps the int16 datapath output exact even at K=4096 (max |sum| = 4*K < 32767), so rows-matched
 * stays a true bit-exact signal at production K (a [-7,7] range overflows int16 past K~512 and masks rows). */
static unsigned sd=12345; static int8_t r4(void){sd=sd*1103515245+12345;return (int8_t)((int)((sd>>10)%5)-2);}

struct cand { uint32_t blk, reg, val; uint32_t blk2, reg2, val2; };   /* blk2!=0 => 2-reg combo */

/* regs that are K/N/address/M-count derived — overriding them just breaks the matmul, not informative. */
static int essential(uint32_t b,uint32_t o){
    if(b==0x201) return (o==0x1024||o==0x1030||o==0x1034||o==0x1044||o==0x1070||o==0x1088||o==0x1110||
                         o==0x1038||o==0x100c||o==0x1080||o==0x1020||o==0x1084||o==0x102c);
    if(b==0x1001)return (o==0x4020||o==0x403c||o==0x4058||o==0x4010||o==0x4050||o==0x4024||o==0x405c);
    if(b==0x801) return (o==0x3018||o==0x3010);
    return 0;
}
/* DANGEROUS regs — writing them HARD-wedges the bus (kernel job-timeout can't recover; needs a power-cycle).
 * The 0x201/0x11xx CNA scratch region (all zero in the capture) is the known hazard: 0x1104=0x80 hard-wedged
 * the board 2026-07-07 (Exp log). Skip the whole 0x1100..0x1184 block. Extend as more are found. */
static int dangerous(uint32_t b,uint32_t o){ return (b==0x201 && o>=0x1100 && o<=0x1184); }

/* wide value palette (thorough): counts, powers of two up to 1024, K-derived, byte-fills, budget candidates. */
static const uint32_t WIDE[]={0,1,2,4,8,16,32,64,128,256,512,1024,0x1b,0x40,0x80,0xb1,0xff,0x100,0x200,0x400};
/* build the candidate list for the mode. cs is caller-allocated (cap entries).
 *   sched : 0x1040 value sweep 0..0x1ff.
 *   regs  : non-essential present regs x WIDE, skipping the hard-wedge 0x11xx scratch (board-safe).
 *   full  : EVERY present reg (incl essential+dangerous) x WIDE — thorough, risk authorized 2026-07-07.
 *   combo : 0x1040 in {base/candidates} x every other non-essential present reg x a small palette (2-reg). */
static int build_cands(struct cand*cs,int cap,const char*mode,int K,int mc_phys){
    int n=0; (void)mc_phys;
    if(strcmp(mode,"full")==0){
        for(int k=0;k+1<REGCMD_I4_N && n<cap;k+=2){
            uint32_t o=REGCMD_I4[k]&0xffff, b=REGCMD_I4[k+1]>>16;
            if(dangerous(b,o)) continue;   /* confirmed pure bus-lock hazard, no row-lever — skip (see Exp log) */
            for(unsigned v=0;v<sizeof WIDE/sizeof*WIDE && n<cap;v++) cs[n++]=(struct cand){b,o,WIDE[v],0,0,0};
        }
    } else if(strcmp(mode,"combo")==0){
        uint32_t sched[]={0x1b,0x40,0x80,0xb1,0xff,(uint32_t)(0xb1+K/32)};   /* co-varied 0x1040 values */
        uint32_t v2[]={0,mc_phys,64,128,256,512};                             /* companion budget values */
        for(unsigned s=0;s<sizeof sched/sizeof*sched;s++)
          for(int k=0;k+1<REGCMD_I4_N && n<cap;k+=2){
            uint32_t o=REGCMD_I4[k]&0xffff, b=REGCMD_I4[k+1]>>16;
            if(essential(b,o)||dangerous(b,o)||(b==0x201&&o==0x1040)) continue;
            for(unsigned v=0;v<sizeof v2/sizeof*v2 && n<cap;v++)
                cs[n++]=(struct cand){0x201,0x1040,sched[s], b,o,v2[v]};
          }
    } else if(strcmp(mode,"regs")==0){
        for(int k=0;k+1<REGCMD_I4_N && n<cap;k+=2){
            uint32_t o=REGCMD_I4[k]&0xffff, b=REGCMD_I4[k+1]>>16;
            if(essential(b,o)||dangerous(b,o)) continue;
            for(unsigned v=0;v<sizeof WIDE/sizeof*WIDE && n<cap;v++) cs[n++]=(struct cand){b,o,WIDE[v],0,0,0};
        }
    } else { /* "sched": exhaustive value sweep of the K-reduction schedule reg */
        for(uint32_t v=0; v<=0x1ff && n<cap; v++) cs[n++]=(struct cand){0x201,0x1040,v,0,0,0};
    }
    return n;
}

static int score_rows(const int16_t*raw,const int32_t*ref,int M,int N){
    int hit=0; for(int m=0;m<M;m++){ int ok=1;
        for(int n=0;n<N&&ok;n++) if(raw[(size_t)(2*m)*N+n]!=ref[(size_t)m*N+n]) ok=0;
        if(ok) hit++; } return hit;
}
/* is a 64-wide vector present contiguously anywhere in raw? (collision-proof block-presence test) */
static int vec_in_raw(const int16_t*raw,size_t rawlen,const int32_t*want){
    for(size_t off=0; off+64<=rawlen; off++){ int ok=1;
        for(int c=0;c<64&&ok;c++) if(raw[off+c]!=want[c]) ok=0; if(ok) return 1; } return 0;
}
static int blacklisted(uint32_t b,uint32_t r){ FILE*f=fopen("i4_fuzz_blacklist.txt","r"); if(!f)return 0;
    unsigned fb,fr; int bad=0; while(fscanf(f,"%x %x",&fb,&fr)==2) if(fb==b&&fr==r){bad=1;break;} fclose(f); return bad; }
static void blacklist(uint32_t b,uint32_t r){ if(blacklisted(b,r))return; FILE*f=fopen("i4_fuzz_blacklist.txt","a");
    if(f){fprintf(f,"%x %x\n",b,r); fclose(f);} }
static void read_state(int*done,int*inflight){ *done=0;*inflight=-1; FILE*f=fopen("i4_fuzz_state.txt","r");
    if(f){ char c; if(fscanf(f," %c",&c)==1 && c=='D'){*done=-2;} else { rewind(f); if(fscanf(f,"%d %d",done,inflight)!=2){*done=0;*inflight=-1;} } fclose(f);} }
static void write_state(int done,int inflight){ FILE*f=fopen("i4_fuzz_state.txt","w"); if(f){fprintf(f,"%d %d\n",done,inflight); fclose(f);} }
static void write_done(void){ FILE*f=fopen("i4_fuzz_state.txt","w"); if(f){fprintf(f,"DONE\n"); fclose(f);} }

int main(int argc,char**argv){
    setbuf(stdout, NULL);   /* unbuffered: output survives a timeout -s INT kill mid-sweep (piped stdout is block-buffered) */
    int K=argc>1?atoi(argv[1]):512, M=argc>2?atoi(argv[2]):8; const char*mode=argc>3?argv[3]:"sched";
    int N=NN, mc_phys=2*M;
    ork_npu*ctx=ork_npu_init(); if(!ctx){printf("init failed (NPU?)\n");return 1;}

    int8_t*A=malloc((size_t)M*K),*B=malloc((size_t)K*N); int32_t*ref=malloc((size_t)M*N*4);
    int16_t*raw=malloc((size_t)2*M*N*2);
    for(size_t i=0;i<(size_t)M*K;i++)A[i]=r4();
    for(size_t i=0;i<(size_t)K*N;i++)B[i]=r4();
    for(int m=0;m<M;m++)for(int n=0;n<N;n++){int s=0;for(int k=0;k<K;k++)s+=A[(size_t)m*K+k]*B[(size_t)k*N+n];ref[(size_t)m*N+n]=s;}

    setenv("ORK_I4_ALAY","1",1);   /* per-row contiguous A (Exp-2026-06-19); ork_npu_probe_i4_mm honors it */
    /* short submit timeout so a wedging candidate fails FAST in-process (kernel times out the job, ~1.5s)
     * and the next probe's ACT_RESET recovers the NPU — the whole sweep runs in one process, no external
     * kill needed. Overridable; the external `timeout -s INT` is only a backstop for a true hard hang. */
    if(!getenv("ORK_I4_PROBE_TO_MS")) setenv("ORK_I4_PROBE_TO_MS","1500",1);

    ork_i4_fuzz_clear();
    int base=(ork_npu_probe_i4_mm(ctx,M,K,N,A,B,raw)==0)?score_rows(raw,ref,M,N):-1;
    if(!strcmp(mode,"base")){ printf("K=%d M=%d N=%d -> rows %d\n",K,M,N,base); ork_npu_free(ctx); return 0; }
    /* stream: attempt int4 STREAMING by applying int8-stream's config to int4 (from the synth_i8 vs synth_i4
     * diff): contiguous output-stride (0x405c=(M-1)<<16), M-count=M (not 2M), 0x4038 out-width, and sweep the
     * K-reduction schedule 0x1040. Score CONTIGUOUS rows (raw[m*N+n]); >16384/K (the batch CBUF cap, =8 @K2048)
     * means we escaped residency = STREAMING. The batch mismatch (schedule w/ batch output) is why 0x1040
     * poisoned before; here we pair it with the streaming output. */
    if(!strcmp(mode,"stream")){
        int cap = 16384 / K; if(cap<1)cap=1;
        printf("[stream] K=%d M=%d N=%d: batch stride-2 baseline=%d rows, CBUF cap=%d; sweeping 0x1040 for CONTIGUOUS rows>%d (=streaming)\n",K,M,N,base,cap,cap);
        /* the int8-stream schedule value for this K (synth_i8 mg formula): base = 177 - 15*(K/512 - 1) */
        int i8base = 177 - 15*((K/512) - 1); if(i8base<0x1b) i8base=0x1b;
        uint32_t cand[]={0x1b,0x40,0x60,(uint32_t)i8base-4,(uint32_t)i8base,(uint32_t)i8base+4,0x90,0xa0,0xb1,0xc0,0xd0,0xff,0x100,0x120};
        printf("[stream] int8 mg value for K=%d ~= 0x%x; testing candidates x A-layout {per-row,interleaved}:\n",K,i8base);
        int best=0;
      for(int alay=1; alay>=0; alay--){    /* test BOTH int4 A layouts (per-row=1, interleaved cube=0) */
        char av[2]={(char)('0'+alay),0}; setenv("ORK_I4_ALAY",av,1);
        printf("  --- ORK_I4_ALAY=%d ---\n",alay);
        for(unsigned ci=0; ci<sizeof cand/sizeof*cand; ci++){ uint32_t sch=cand[ci];
            ork_i4_fuzz_clear();
            ork_i4_fuzz_add(0x201,0x1020,0x10000u|M); ork_i4_fuzz_add(0x201,0x1084,0x10000u|M); ork_i4_fuzz_add(0x201,0x102c,(uint32_t)M);
            ork_i4_fuzz_add(0x1001,0x4034,(uint32_t)(M-1)); ork_i4_fuzz_add(0x801,0x3014,(uint32_t)(M-1)<<16);
            ork_i4_fuzz_add(0x1001,0x405c,(uint32_t)(M-1)<<16);                          /* contiguous M-stride (stream) */
            ork_i4_fuzz_add(0x1001,0x4038,(uint32_t)((((N/4)-1)<<16)|((N/4)-1)));
            ork_i4_fuzz_add(0x201,0x1040,sch);                                           /* K-reduction schedule */
            int rc=ork_npu_probe_i4_mm(ctx,M,K,N,A,B,raw);
            int hit=0; if(rc==0) for(int m=0;m<M;m++){ int ok=1; for(int n=0;n<N&&ok;n++) if(raw[(size_t)m*N+n]!=ref[(size_t)m*N+n]) ok=0; if(ok) hit++; }
            printf("  0x1040=0x%03x -> %d contiguous rows (rc=%d)%s\n",sch,hit,rc, hit>cap?"  <<< STREAMING!":"");
            if(hit>best) best=hit;
        }
      }   /* end A-layout loop */
        printf("[stream] best %d contiguous rows (cap %d) — %s\n", best, cap, best>cap?"STREAMING FOUND":"no streaming (capped)");
        ork_i4_fuzz_clear(); ork_npu_free(ctx); return 0;
    }
    /* i8batch: the controlled A/B. Run int8 STREAM (baseline, contiguous) then apply the int4 BATCH trigger
     * (0x405c=0 + mc_phys=2M encoding) to int8 and map where each row's output lands — does int8 flip from
     * contiguous stream to a batch layout? If so, 0x405c (± the mc encoding) IS the stream/batch selector. */
    if(!strcmp(mode,"i8batch")){
        int8_t*A8=malloc((size_t)M*K),*B8=malloc((size_t)K*N); int32_t*ref8=malloc((size_t)M*N*4);
        int32_t*raw8=malloc((size_t)2*M*N*4);
        for(size_t i=0;i<(size_t)M*K;i++)A8[i]=r4();
        for(size_t i=0;i<(size_t)K*N;i++)B8[i]=r4();
        for(int m=0;m<M;m++)for(int n=0;n<N;n++){int s=0;for(int k=0;k<K;k++)s+=A8[(size_t)m*K+k]*B8[(size_t)k*N+n];ref8[(size_t)m*N+n]=s;}
        printf("[i8batch] K=%d M=%d N=%d\n",K,M,N);
        /* helper: for each row, find its physical location (row-major offset/N) in raw8, contiguous match */
        #define ROWMAP(tag) do{ printf("  %s row->physrow:",tag); for(int m=0;m<M;m++){ long at=-1; \
            for(size_t off=0; off+N<=(size_t)2*M*N; off++){ int ok=1; for(int n=0;n<N&&ok;n++) if(raw8[off+n]!=ref8[(size_t)m*N+n]) ok=0; if(ok){at=(long)off;break;} } \
            printf(" %d:%s%ld", m, at<0?"X":"", at<0?0:at/N); } printf("\n"); }while(0)
        ork_i8_fuzz_clear();
        int rc1=ork_npu_probe_i8_mm(ctx,M,K,N,A8,B8,raw8);
        printf("  STREAM (baseline, rc=%d):\n",rc1); if(rc1==0) ROWMAP("stream");
        ork_i8_fuzz_clear();
        ork_i8_fuzz_add(0x1001,0x405c,0);                                         /* the batch trigger */
        ork_i8_fuzz_add(0x201,0x1020,0x10000u|(2*M)); ork_i8_fuzz_add(0x201,0x1084,0x10000u|(2*M)); ork_i8_fuzz_add(0x201,0x102c,(uint32_t)(2*M));
        ork_i8_fuzz_add(0x1001,0x4034,(uint32_t)(2*M-1)); ork_i8_fuzz_add(0x801,0x3014,(uint32_t)(2*M-1)<<16);
        setenv("ORK_I8_PROBE_SCHED","0",1);                                       /* no 0x1040 streaming schedule (batch) */
        int rc2=ork_npu_probe_i8_mm(ctx,M,K,N,A8,B8,raw8);
        printf("  BATCH (0x405c=0 + mc_phys=2M + NO sched, rc=%d):\n",rc2); if(rc2==0) ROWMAP("batch");
        unsetenv("ORK_I8_PROBE_SCHED");
        printf("[i8batch] done — physrow=m means contiguous(stream); physrow=2m means stride-2(batch); X=not found\n");
        free(A8);free(B8);free(ref8);free(raw8); ork_i8_fuzz_clear(); ork_npu_free(ctx); return 0;
    }
    /* wtest <blk> <reg>: hunt a WEIGHT-bank register that lifts the 131072 weight budget. At K (large) + N=128
     * (2 blocks), the batch computes blk0 but drops blk1 (weight Ncore*K > 131072). If a reg value makes blk1
     * appear, it enlarged the weight budget → wide-N would batch in ONE submit (no N-subslice). */
    if(!strcmp(mode,"wtest") && argc>5){
        uint32_t vb=(uint32_t)strtoul(argv[4],0,0), vr=(uint32_t)strtoul(argv[5],0,0);
        int WN=128;
        int8_t*B2=malloc((size_t)K*WN); int32_t*ref2=malloc((size_t)M*WN*4); int16_t*raw2=malloc((size_t)2*M*WN*2);
        for(size_t i=0;i<(size_t)K*WN;i++)B2[i]=r4();
        for(int m=0;m<M;m++)for(int n=0;n<WN;n++){int s=0;for(int k=0;k<K;k++)s+=A[(size_t)m*K+k]*B2[(size_t)k*WN+n];ref2[(size_t)m*WN+n]=s;}
        int32_t blk1[64]; for(int c=0;c<64;c++) blk1[c]=ref2[64+c];   /* row0, block1 (n=64..127) */
        setenv("ORK_I4_ALAY","1",1);
        ork_i4_fuzz_clear(); ork_npu_probe_i4_mm(ctx,M,K,WN,A,B2,raw2);
        int base_b1=vec_in_raw(raw2,(size_t)2*M*WN,blk1);
        printf("[wtest] reg %x/%x @ K=%d M=%d WN=128: baseline blk1 present=%d (hunting lift to 1)\n",vb,vr,K,M,base_b1);
        uint32_t vals[]={0,1,2,4,8,16,32,64,128,256,512,(uint32_t)(K/16),(uint32_t)(K/8),(uint32_t)(K/4),
                         (uint32_t)(K*WN/2),(uint32_t)(K*WN),0xff,0x100,0x200,0x400};
        for(unsigned i=0;i<sizeof vals/sizeof*vals;i++){
            ork_i4_fuzz_clear(); ork_i4_fuzz_add(vb,vr,vals[i]);
            ork_npu_probe_i4_mm(ctx,M,K,WN,A,B2,raw2);
            int b1=vec_in_raw(raw2,(size_t)2*M*WN,blk1);
            if(b1>base_b1) printf("  0x%-6x -> blk1=%d  <<< WEIGHT-BANK LEVER\n",vals[i],b1);
        }
        printf("[wtest] done\n");
        free(B2);free(ref2);free(raw2); ork_npu_free(ctx); return 0;
    }
    /* vsweep <blk> <reg>: characterize one register — sweep its value 0..0x200, print rows-matched for
     * each (not just HITs). Reveals the value->rows law of a candidate budget reg (e.g. 0x201/0x107c). */
    if(!strcmp(mode,"vsweep") && argc>5){
        uint32_t vb=(uint32_t)strtoul(argv[4],0,0), vr=(uint32_t)strtoul(argv[5],0,0);
        printf("[vsweep] reg %x/%x @ K=%d M=%d (base rows %d):\n",vb,vr,K,M,base);
        for(uint32_t v=0; v<=0x200; v++){
            ork_i4_fuzz_clear(); ork_i4_fuzz_add(vb,vr,v);
            int rc=ork_npu_probe_i4_mm(ctx,M,K,N,A,B,raw);
            int rows=(rc==0)?score_rows(raw,ref,M,N):-1;
            if(rows>base||rows<0||(v%64)==0) printf("  0x%03x -> rows %d%s\n",v,rows,rows>base?"  <<":"");
        }
        ork_i4_fuzz_clear(); ork_npu_free(ctx); return 0;
    }
    /* bankprobe <blk> <reg>: value-level, wedge-RESUMABLE sweep of a suspected CBUF bank-partition register
     * (one that BLACKLISTED wholesale after a single wedge — a bank reg hard-wedges on bad partitions but a
     * specific value may enlarge the activation bank -> rows>base). State = i4_fuzz_state.txt (value index);
     * on resume a still-inflight value is SKIPPED (not blacklisted — we're probing this reg). 0x107c=K/16
     * stays active (synth default), so this co-varies with it. Run under the autonomous power-cycle loop. */
    if(!strcmp(mode,"bankprobe") && argc>5){
        uint32_t bb=(uint32_t)strtoul(argv[4],0,0), br=(uint32_t)strtoul(argv[5],0,0);
        uint32_t bvals[]={1,2,3,4,6,8,12,16,24,32,48,64,0x80,0x100,0x200,0x400};
        int nb=sizeof bvals/sizeof*bvals;
        int done,inflight; read_state(&done,&inflight);
        if(done==-2){ printf("[bankprobe] DONE (rm i4_fuzz_state.txt to re-run)\n"); ork_npu_free(ctx); return 0; }
        if(inflight>=0 && inflight<nb){ printf("[bankprobe] value idx %d (0x%x) WEDGED -> skip\n",inflight,bvals[inflight]); if(done<=inflight)done=inflight+1; }
        printf("[bankprobe] reg %x/%x @ K=%d M=%d, base rows %d, from idx %d/%d\n",bb,br,K,M,base,done,nb);
        for(int i=done;i<nb;i++){
            write_state(i,i); fflush(stdout);
            ork_i4_fuzz_clear(); ork_i4_fuzz_add(bb,br,bvals[i]);
            int rc=ork_npu_probe_i4_mm(ctx,M,K,N,A,B,raw);
            int rows=(rc==0)?score_rows(raw,ref,M,N):-1;
            write_state(i+1,-1);
            printf("  0x%-4x -> rows %d%s\n",bvals[i],rows,rows>base?"  <<< BANK LEVER":"");
            if(rows>base){ FILE*hf=fopen("i4_fuzz_hits.txt","a"); if(hf){fprintf(hf,"BANK reg %x/%x=0x%x -> rows %d (base %d) K=%d M=%d\n",bb,br,bvals[i],rows,base,K,M); fclose(hf);} }
        }
        write_done(); printf("[bankprobe] complete\n"); ork_i4_fuzz_clear(); ork_npu_free(ctx); return 0;
    }
    printf("[fuzz] K=%d M=%d (mc_phys=%d) N=%d mode=%s  baseline rows-matched = %d / %d\n",K,M,mc_phys,N,mode,base,M);
    if(base<0){printf("[fuzz] baseline submit failed (NPU wedged?) — aborting\n");return 1;}

    static struct cand cs[8192]; int ncand=build_cands(cs,8192,mode,K,mc_phys);
    { const char*ls=argc>4?argv[4]:0;   /* 4th arg "list" (or an index): print candidate mapping, no submits */
      if(ls){ if(!strcmp(ls,"list")){ for(int i=0;i<ncand;i++) printf("cand %d: reg %x/%x = 0x%x\n",i,cs[i].blk,cs[i].reg,cs[i].val); }
              else { int i=atoi(ls); if(i>=0&&i<ncand) printf("cand %d: reg %x/%x = 0x%x\n",i,cs[i].blk,cs[i].reg,cs[i].val); }
              ork_npu_free(ctx); return 0; } }
    int done,inflight; read_state(&done,&inflight);
    if(done==-2){ printf("[fuzz] state = DONE (rm i4_fuzz_state.txt to re-run)\n"); return 0; }
    if(inflight>=0 && inflight<ncand){
        printf("[fuzz] resume: cand %d was in-flight (reg %x/%x=0x%x) -> WEDGE, blacklisting reg\n",
               inflight,cs[inflight].blk,cs[inflight].reg,cs[inflight].val);
        blacklist(cs[inflight].blk,cs[inflight].reg); if(done<=inflight)done=inflight+1;
    }
    printf("[fuzz] %d candidates, starting at %d\n",ncand,done);

    int best=base;
    for(int i=done;i<ncand;i++){
        if(blacklisted(cs[i].blk,cs[i].reg)){ write_state(i+1,-1); continue; }
        write_state(i,i); fflush(stdout);                       /* in-flight BEFORE submit (wedge trace) */
        ork_i4_fuzz_clear(); ork_i4_fuzz_add(cs[i].blk,cs[i].reg,cs[i].val);
        if(cs[i].blk2) ork_i4_fuzz_add(cs[i].blk2,cs[i].reg2,cs[i].val2);
        int rc=ork_npu_probe_i4_mm(ctx,M,K,N,A,B,raw);
        int rows=(rc==0)?score_rows(raw,ref,M,N):-1;
        write_state(i+1,-1);
        if(rows>base){ printf("  HIT  reg %x/%x=0x%x %s -> rows %d (base %d) %s\n",
                              cs[i].blk,cs[i].reg,cs[i].val, cs[i].blk2?"(+combo)":"",rows,base,rows==M?"*** FULL ***":"");
            FILE*hf=fopen("i4_fuzz_hits.txt","a"); if(hf){fprintf(hf,"reg %x/%x=0x%x",cs[i].blk,cs[i].reg,cs[i].val);
                              if(cs[i].blk2)fprintf(hf," + %x/%x=0x%x",cs[i].blk2,cs[i].reg2,cs[i].val2);
                              fprintf(hf," -> rows %d (base %d) K=%d M=%d\n",rows,base,K,M); fclose(hf);} }
        else if(rc!=0) printf("  .    reg %x/%x=0x%x submit rc=%d\n",cs[i].blk,cs[i].reg,cs[i].val,rc);
        if(rows>best)best=rows;
        if((i%64)==0){ printf("  [..%d/%d best=%d]\n",i,ncand,best); fflush(stdout); }
    }
    ork_i4_fuzz_clear(); write_done();
    printf("[fuzz] DONE. best rows-matched = %d / %d (baseline %d)%s\n",best,M,base,best>base?"  <-- IMPROVED":"");
    free(A);free(B);free(ref);free(raw); ork_npu_free(ctx); return 0;
}
