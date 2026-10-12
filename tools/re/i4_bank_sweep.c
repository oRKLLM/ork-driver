/* tools/re/i4_bank_sweep.c — a MODEL for the int4 BCHAIN bank budgets, timed.
 *
 * THE MODEL UNDER TEST. int4 never recomputes 0x1040 (the "poison pill" — the int8 formula corrupts
 * it), so the CBUF bank split is frozen at the captured 177 = 0xb1 = DATA_BANK=1 / WEIGHT_BANK=11.
 * If 0x1040 means for int4 what the fp16 work proved it means, then BOTH BCHAIN budgets fall out of
 * that one frozen register:
 *
 *   weight per N-chunk : Wb*K/2 bytes <= WEIGHT_BANK * 32768   => Wb <= 720896/K   (WBNK=11)
 *   activation per grp : H *K   bytes <= DATA_BANK   * 32768/2 => H  <= 16384/K    (DBNK=1, /2 = ping-pong)
 *
 * Both match measurement (720896 exactly; H=CEIL(16384/K) at six K), which is why this is worth
 * testing properly rather than sweeping blind.
 *
 * WHAT WOULD FALSIFY IT — and what it predicts:
 *   - Rebalancing 0x1040 must MOVE both ceilings in opposite directions. 0x84 (DBNK=4/WBNK=8)
 *     should give 4x the H budget and 8/11 of the Wb budget. If the ceilings do NOT move, the
 *     budgets are not coming from this register and the model is wrong.
 *   - That is also the LEVER: H is rows per weight stream, so 4x H = 4x less weight traffic on the
 *     int4 prefill path. Wb is only submit-count, and the default sits far below its ceiling.
 *
 * METHOD — and the trap this tool exists to avoid. A checksum ALONE cannot validate a config: the
 * driver self-heals a timed-out submit and returns the RIGHT answer ~800x slower. So every point is
 * TIMED and classified against a known-good baseline:
 *     OK       correct + fast          (< SLOW_X * baseline)
 *     SELFHEAL correct + slow          -> a FAILED config the retry rescued. NOT usable.
 *     WRONG    incorrect checksum
 * ORK_I4_WB / ORK_I4_H / ORK_I4_1040 are re-read per call, so one process sweeps without the
 * per-process getenv-caching dance.
 *
 *   make i4_bank_sweep
 *   sudo tools/util/npu_guard.sh -- env ORK_MM_TIMEOUT=2000 ./i4_bank_sweep <K> <N> <M> [mode]
 *      mode: wb (default) | split | joint
 */
#define _GNU_SOURCE
#include "ork_npu.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define SLOW_X 8.0     /* > 8x baseline == the self-heal path, not a valid config */

static unsigned long long fnv64(const void *p,size_t n){
    const unsigned char *b=p; unsigned long long h=1469598103934665603ULL;
    for(size_t i=0;i<n;i++){ h^=b[i]; h*=1099511628211ULL; } return h;
}
static double now_us(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec*1e6+t.tv_nsec/1e3; }

static ork_npu *C_; static ork_w *W_; static int M_,N_,K_;
static int8_t *A_; static int32_t *Cbuf_;

/* ONE run only (no warmup): cheap probe used by the joint sweep so a bad point costs one
 * self-heal (~1-2 s) instead of five. */
static unsigned long long run_once(double *us,int *rcout){
    memset(Cbuf_,0,(size_t)M_*N_*4);
    double t0=now_us();
    int rc=ork_i4_mm_run(C_,W_,M_,A_,Cbuf_); *rcout=rc;
    *us=now_us()-t0;
    if(rc) return 0;
    return fnv64(Cbuf_,(size_t)M_*N_*4);
}

/* one timed run at the current env knobs; returns cksum (0 on rc!=0) and writes us */
static unsigned long long run1(int iters,double *us,int *rcout){
    memset(Cbuf_,0,(size_t)M_*N_*4);
    int rc=ork_i4_mm_run(C_,W_,M_,A_,Cbuf_); *rcout=rc;
    if(rc){ *us=0; return 0; }
    for(int i=0;i<2;i++) ork_i4_mm_run(C_,W_,M_,A_,Cbuf_);
    double t0=now_us(); for(int i=0;i<iters;i++) ork_i4_mm_run(C_,W_,M_,A_,Cbuf_);
    *us=(now_us()-t0)/iters;
    return fnv64(Cbuf_,(size_t)M_*N_*4);
}
static const char *classify(unsigned long long ck,unsigned long long ref,double us,double base){
    if(!ck)            return "rc-fail";
    if(ck!=ref)        return "WRONG";
    if(us > SLOW_X*base) return "SELFHEAL";   /* correct only because the retry rescued it */
    return "ok";
}

int main(int argc,char**argv){
    K_=argc>1?atoi(argv[1]):2048; N_=argc>2?atoi(argv[2]):1024; M_=argc>3?atoi(argv[3]):128;
    const char *mode=argc>4?argv[4]:"wb";
    int iters=getenv("ORK_ITERS")?atoi(getenv("ORK_ITERS")):5;
    if(K_%32||N_%64){ printf("need K%%32, N%%64\n"); return 2; }

    C_=ork_npu_init(); if(!C_){ printf("no board\n"); return 0; }
    int8_t *B=malloc((size_t)K_*N_); A_=malloc((size_t)M_*K_); Cbuf_=malloc((size_t)M_*N_*4);
    if(!A_||!B||!Cbuf_){ printf("OOM\n"); return 2; }
    for(size_t i=0;i<(size_t)K_*N_;i++) B[i]=(int8_t)((int)((i*7)%15)-7);
    for(size_t i=0;i<(size_t)M_*K_;i++) A_[i]=(int8_t)((int)((i*13)%31)-15);
    W_=ork_i4_mm_pack(C_,K_,N_,B); if(!W_){ printf("pack fail\n"); return 2; }

    /* baseline = shipped defaults (no overrides) */
    unsetenv("ORK_I4_WB"); unsetenv("ORK_I4_H"); unsetenv("ORK_I4_1040"); unsetenv("ORK_I4_DBNK");
    double base; int rc; unsigned long long ref=run1(iters,&base,&rc);
    if(rc){ printf("baseline rc=%d\n",rc); return 2; }
    printf("K=%d N=%d M=%d  baseline %.1f us  ref=%016llx (model: Wb<=%d, H<=%d)\n",
           K_,N_,M_,base, ref, 720896/K_, (16384+K_-1)/K_);

    char buf[32];
    if(!strcmp(mode,"wb")){
        printf("\n%-7s %-9s %-10s %s\n","Wb","Wb*K/2 B","us","verdict   (model: fits WBNK=11 => <=360448 B)");
        for(int wb=64; wb<=N_ && wb<=1024; wb*=2){
            snprintf(buf,sizeof buf,"%d",wb); setenv("ORK_I4_WB",buf,1);
            double us; unsigned long long ck=run1(iters,&us,&rc);
            long bytes=(long)wb*K_/2;
            printf("%-7d %-9ld %-10.1f %-9s %s\n",wb,bytes,us,classify(ck,ref,us,base),
                   bytes<=360448?"(model says OK)":"(model says OVER)");
        }
        unsetenv("ORK_I4_WB");
    } else if(!strcmp(mode,"split")){
        /* 0x1040 rebalance: does the split MOVE the H ceiling? v = (WBNK<<4)|DBNK */
        int splits[]={0xb1,0xa2,0x93,0x84,0x75,0x66};
        printf("\n%-8s %-5s %-5s %-6s %-10s %s\n","0x1040","DBNK","WBNK","H","us","verdict");
        for(size_t si=0; si<sizeof splits/sizeof*splits; si++){
            int v=splits[si], dbnk=v&0xf, wbnk=(v>>4)&0xf;
            snprintf(buf,sizeof buf,"%d",v); setenv("ORK_I4_1040",buf,1);
            int hbase=(16384+K_-1)/K_;
            int hs[4]={hbase, hbase*2, hbase*4, hbase*8};
            for(int hi=0; hi<4; hi++){
                if(hs[hi]<2||hs[hi]>512) continue;
                snprintf(buf,sizeof buf,"%d",hs[hi]); setenv("ORK_I4_H",buf,1);
                double us; unsigned long long ck=run1(iters,&us,&rc);
                printf("0x%-6x %-5d %-5d %-6d %-10.1f %s\n",v,dbnk,wbnk,hs[hi],us,classify(ck,ref,us,base));
            }
            unsetenv("ORK_I4_H");
        }
        unsetenv("ORK_I4_1040");
    } else {
        /* JOINT mode: sweep the 0x1040 DATA/WEIGHT split TOGETHER with a matched Wb.
         * ORK_I4_DBNK=n sets 0x1040=((12-n)<<4)|n AND H=n*hcap(K); we supply the paired Wb.
         * Two independent passes; a point is VALID only if both passes are correct AND fast. */
        int ns[16]={1,2,3,4,6,8,11}, nn=7;
        int wbs[16]={64,128,256}, nw=3;
        int hforce=getenv("ORK_SWEEP_H")?atoi(getenv("ORK_SWEEP_H")):0;   /* force H independent of n */
        int reps=getenv("ORK_SWEEP_REPS")?atoi(getenv("ORK_SWEEP_REPS")):2;
        if(reps<2)reps=2; if(reps>8)reps=8;
        { const char*e=getenv("ORK_SWEEP_N"); if(e){ nn=0; for(const char*q=e;*q&&nn<16;){ ns[nn++]=atoi(q); while(*q&&*q!=',')q++; if(*q)q++; } } }
        { const char*e=getenv("ORK_SWEEP_WB"); if(e){ nw=0; for(const char*q=e;*q&&nw<16;){ wbs[nw++]=atoi(q); while(*q&&*q!=',')q++; if(*q)q++; } } }
        int hbase=(16384+K_-1)/K_;
        printf("\n%-4s %-7s %-5s %-5s %-6s %-6s %-9s %-10s %-10s %s\n",
               "n","0x1040","Hmax","Wb","NG","NC","programs","pass1 us","pass2 us","verdict");
        for(int ni=0; ni<nn; ni++){
            int n=ns[ni], h=n*hbase; if(h>64)h=64;
            snprintf(buf,sizeof buf,"%d",n); setenv("ORK_I4_DBNK",buf,1);
            if(hforce){ h=hforce; snprintf(buf,sizeof buf,"%d",h); setenv("ORK_I4_H",buf,1); }
            for(int wi=0; wi<nw; wi++){
                int wb=wbs[wi]; if(wb>N_) continue;
                snprintf(buf,sizeof buf,"%d",wb); setenv("ORK_I4_WB",buf,1);
                double u[8]={0}; int rcx=0, bad=0; unsigned long long ck=0;
                double best=0;
                for(int r=0;r<reps;r++){
                    int rr; unsigned long long cc=run_once(&u[r],&rr);
                    if(rr){ rcx=1; break; }
                    if(cc!=ref){ bad=1; break; }
                    if(u[r]>SLOW_X*base){ bad=2; break; }
                    ck=cc; if(!best||u[r]<best) best=u[r];
                }
                int NG=(M_+h-1)/h, NC=(N_+wb-1)/wb;
                const char *v = rcx?"rc-fail" : (bad==1)?"WRONG" : (bad==2)?"SELFHEAL" : "ok";
                printf("%-4d 0x%-5x %-5d %-5d %-6d %-6d %-9d %-10.1f",
                       n,((12-n)<<4)|n,h,wb,NG,NC,NG*NC,u[0]);
                for(int r=1;r<reps;r++) printf(" %-10.1f",u[r]);
                printf("  %s  (best %.1f, %.2fx)\n", v, best, best?base/best:0.0);
                fflush(stdout);
                (void)ck;
            }
            unsetenv("ORK_I4_WB");
            if(hforce) unsetenv("ORK_I4_H");
        }
        unsetenv("ORK_I4_DBNK"); unsetenv("ORK_I4_WB");
        /* re-time the best-looking valid points properly (iters-averaged) */
    }
    ork_mm_free(C_,W_); ork_npu_free(C_);
    return 0;
}
