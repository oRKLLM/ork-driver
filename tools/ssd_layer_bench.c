/* tools/ssd_layer_bench.c — measure ONE Mamba-2/SSD layer's SCAN matmul workload on NPU vs CPU at
 * REAL shapes, to replace the assumed GMAC/s rates in tools/ssd_opcount.c with measured ones.
 *
 * The chunked-SSD scan's matmul stages (group-batched where B/C are shared; HG=H/G heads per group
 * stacked along head_dim). Both operands are per-chunk ACTIVATIONS, so the NPU primitive is
 * ork_bmm_i8 (dynamic·dynamic, per-batch repack+run) — NOT run_stream (needs a resident weight).
 *   scores  C·B^T      nbatch=G·NC   [CS, Nst]·[Nst, CS]     (K=Nst, N=CS)
 *   cstate  Xt·B        nbatch=G·NC   [HG*P, CS]·[CS, Nst]    (K=CS,  N=Nst)   <- dominant
 *   Y_off   SI·C^T      nbatch=G·NC   [HG*P, Nst]·[Nst, CS]   (K=Nst, N=CS)    <- dominant
 *   Y_diag  M·xbar      nbatch=H·NC   [CS, CS]·[CS, P]        (K=CS,  N=P)     <- many tiny
 *
 * THROUGHPUT bench (dummy int8 data, no correctness — like mc_prof/rknn_vs_ork). Reports per-stage and
 * per-layer wall + effective GMAC/s for NPU (ork_bmm_i8) and a pthread int8 GEMM on the 4 big cores,
 * then extrapolates a full-model prefill SSD-scan time (× n_layers) and tok/s for both.
 *
 *   make ssd_layer_bench && sudo ./ssd_layer_bench [preset 0=130m 1=7b] [L]      (board only)
 */
#define _GNU_SOURCE
#include "ork_npu.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <pthread.h>
#include <sched.h>

static double now_us(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec*1e6+t.tv_nsec*1e-3; }

/* ---- CPU int8 GEMM, M-rows split across NT threads pinned to the big cores (mirror -t4).
 * Inner loop is a CONTIGUOUS int8 dot product (B pre-transposed to Bt[N][K] per batch, timed like
 * ork_bmm's repack) so -O3 -march=native vectorizes it (NEON). ---- */
typedef struct { const int8_t*A; const int8_t*Bt; int32_t*C; int M,K,N,nbatch,r0,r1,core; } cpuw;
static void*cpu_gemm_worker(void*vp){
    cpuw*w=vp; cpu_set_t set; CPU_ZERO(&set); CPU_SET(w->core,&set);
    pthread_setaffinity_np(pthread_self(),sizeof set,&set);
    int K=w->K,N=w->N,M=w->M;
    for(int b=0;b<w->nbatch;b++){
        const int8_t*A=w->A+(size_t)b*M*K, *Bt=w->Bt+(size_t)b*K*N;
        int32_t*C=w->C+(size_t)b*M*N;
        for(int i=w->r0;i<w->r1;i++){ const int8_t*a=A+(size_t)i*K;
            for(int j=0;j<N;j++){ const int8_t*bt=Bt+(size_t)j*K; int32_t acc=0;
                for(int k=0;k<K;k++) acc+=(int32_t)a[k]*(int32_t)bt[k];
                C[(size_t)i*N+j]=acc; }
        }
    }
    return NULL;
}
#define NT 4
static double cpu_gemm(const int8_t*A,const int8_t*B,int32_t*C,int32_t*scratch_unused,
                       int8_t*Bt,int nbatch,int M,int K,int N){
    pthread_t th[NT]; cpuw w[NT]; int per=(M+NT-1)/NT;
    double t0=now_us();
    /* transpose B[b] [K,N] -> Bt[b] [N,K] (timed, matches ork_bmm repack cost) */
    for(int b=0;b<nbatch;b++){ const int8_t*Bb=B+(size_t)b*K*N; int8_t*Tb=Bt+(size_t)b*K*N;
        for(int k=0;k<K;k++) for(int j=0;j<N;j++) Tb[(size_t)j*K+k]=Bb[(size_t)k*N+j]; }
    for(int t=0;t<NT;t++){ int r0=t*per,r1=r0+per; if(r1>M)r1=M; if(r0>M)r0=M;
        w[t]=(cpuw){A,Bt,C,M,K,N,nbatch,r0,r1,4+t};
        pthread_create(&th[t],NULL,cpu_gemm_worker,&w[t]); }
    for(int t=0;t<NT;t++) pthread_join(th[t],NULL);
    (void)scratch_unused;
    return now_us()-t0;
}

typedef struct { const char*name; int H,P,Nst,G,CS,layers; } model;

/* one stage: nbatch matmuls of [M,K]·[K,N]; time NPU (ork_bmm_i8) + CPU; return macs, fill wall/gmac */
static double bench_stage(ork_npu*c,const char*tag,int nbatch,int M,int K,int N,
                          double*npu_ms,double*npu_g,double*cpu_ms,double*cpu_g){
    size_t szA=(size_t)nbatch*M*K, szB=(size_t)nbatch*K*N, szC=(size_t)nbatch*M*N;
    int8_t*A=malloc(szA),*B=malloc(szB),*Bt=malloc(szB); int32_t*C=malloc(szC*4);
    for(size_t i=0;i<szA;i++)A[i]=(int8_t)(i&3)-1; for(size_t i=0;i<szB;i++)B[i]=(int8_t)(i&3)-1;
    double macs=(double)nbatch*M*K*N;
    /* NPU via ork_bmm_i8 (real dynamic·dynamic primitive: per-batch repack of B + gather of A + run) */
    ork_bmm_i8(c,nbatch,M,K,N,A,B,C);                      /* warm */
    double n0=now_us(); ork_bmm_i8(c,nbatch,M,K,N,A,B,C); double nus=now_us()-n0;
    /* NPU-IDEAL: pack B[0] as a RESIDENT weight ONCE, reuse across all batches (no repack/gather).
     * Not correct math — isolates the NPU's raw matmul ceiling at this shape from ork_bmm overhead. */
    double rus=0; ork_w*w=ork_mm_pack_i8(c,K,N,B);
    if(w){ ork_mm_run_i8(c,w,M,A,C);                        /* warm */
        double r0=now_us(); for(int b=0;b<nbatch;b++) ork_mm_run_i8(c,w,M,A+(size_t)b*M*K,C); rus=now_us()-r0;
        ork_mm_free(c,w); }
    /* CPU */
    double c0=cpu_gemm(A,B,C,NULL,Bt,nbatch,M,K,N);        /* warm */
    double cus=cpu_gemm(A,B,C,NULL,Bt,nbatch,M,K,N);
    *npu_ms=nus/1000; *npu_g=macs/(nus*1e-6)/1e9;
    *cpu_ms=cus/1000; *cpu_g=macs/(cus*1e-6)/1e9;
    double rms=rus/1000, rg=rus>0?macs/(rus*1e-6)/1e9:0;
    printf("  %-8s nb=%-4d [%d,%d]x[%d,%d]  bmm %6.1fG | NPUideal %6.1fG (%.2fx cpu) | CPU %6.1fG %6.2fms  bmm %.2fx cpu\n",
           tag,nbatch,M,K,K,N,*npu_g,rg, rus>0?cus/rus:0, *cpu_g,*cpu_ms, cus/nus);
    free(A);free(B);free(Bt);free(C);
    return macs;
}

int main(int argc,char**argv){
    setvbuf(stdout,NULL,_IONBF,0);
    int preset=argc>1?atoi(argv[1]):0; int L=argc>2?atoi(argv[2]):256;
    model M130={"mamba2-130m",24,64,128,1,64,24}, M7B={"Codestral-7B",128,64,128,8,64,64};
    model m=preset?M7B:M130;
    int NC=(L+m.CS-1)/m.CS, HG=m.H/m.G; if(HG<1)HG=1;
    ork_npu*c=ork_npu_init(); if(!c){printf("no NPU\n");return 0;}
    printf("=== SSD LAYER SCAN BENCH — %s, one layer, prefill L=%d (NC=%d, HG=%d) ===\n",m.name,L,NC,NC>0?HG:HG);
    printf("(dummy int8, throughput only. NPU=ork_bmm_i8 real primitive; CPU=int8 GEMM on 4 big cores.)\n");

    double npu_tot=0,cpu_tot=0,macs_tot=0,nms,ng,cms,cg;
    double cpu_sc,cpu_cs,cpu_yo;   /* grouped-stage CPU ms, to compare against the fused chain */
    macs_tot+=bench_stage(c,"scores", m.G*NC,  m.CS,     m.Nst, m.CS,  &nms,&ng,&cms,&cg); npu_tot+=nms; cpu_tot+=cms; cpu_sc=cms;
    macs_tot+=bench_stage(c,"cstate", m.G*NC,  HG*m.P,   m.CS,  m.Nst, &nms,&ng,&cms,&cg); npu_tot+=nms; cpu_tot+=cms; cpu_cs=cms;
    macs_tot+=bench_stage(c,"Y_off",  m.G*NC,  HG*m.P,   m.Nst, m.CS,  &nms,&ng,&cms,&cg); npu_tot+=nms; cpu_tot+=cms; cpu_yo=cms;
    macs_tot+=bench_stage(c,"Y_diag", m.H*NC,  m.CS,     m.CS,  m.P,   &nms,&ng,&cms,&cg); npu_tot+=nms; cpu_tot+=cms;
    /* FULLY-GROUPED Y_diag (L-factorization: exp(Acs[l]) out, exp(-Acs[s]) into xbar; masked G stays
     * per-GROUP) => ONE [CS,CS]x[CS, HG*P] matmul per (group,chunk), NOT H tiny ones. Measure the shape. */
    double ydg_np,ydg_ng,ydg_cp,ydg_cg;
    double ydg_macs=bench_stage(c,"Ydiag_g",m.G*NC, m.CS, m.CS, HG*m.P, &ydg_np,&ydg_ng,&ydg_cp,&ydg_cg);
    printf("  (Ydiag_g REPLACES per-head Y_diag under full grouping: %d matmuls vs %d)\n", m.G*NC, m.H*NC);

    double layer_npu_ms=npu_tot, layer_cpu_ms=cpu_tot;
    double full_npu_ms=layer_npu_ms*m.layers, full_cpu_ms=layer_cpu_ms*m.layers;
    printf("\n  per-layer SCAN matmul: NPU %.2f ms | CPU %.2f ms  (%.2f G MACs)  -> CPU %s NPU by %.2fx\n",
           layer_npu_ms,layer_cpu_ms,macs_tot/1e9, layer_cpu_ms<layer_npu_ms?"BEATS":"loses to",
           layer_cpu_ms<layer_npu_ms?layer_npu_ms/layer_cpu_ms:layer_cpu_ms/layer_npu_ms);
    printf("  full-model (%d layers) SCAN-only prefill of L=%d:  NPU %.0f ms (%.0f tok/s) | CPU %.0f ms (%.0f tok/s)\n",
           m.layers,L, full_npu_ms, L/(full_npu_ms*1e-3), full_cpu_ms, L/(full_cpu_ms*1e-3));
    printf("  (SCAN ONLY — in/out projections are separate big GEMMs on the NPU either way.)\n");

    /* ================= FUSED ON-NPU SCAN: all grouped matmuls in ONE chained submit ================
     * The user's hypothesis: keep the whole scan on the NPU as ONE PC-chained submit -> the ~48us
     * per-submit floor is paid ONCE, not per matmul, and operands stay resident (no repack). Compare to
     * the SAME matmuls as N separate submits, and to the CPU doing the grouped scan (SDOT, fused kernel). */
    double cpu_grouped = cpu_sc+cpu_cs+cpu_yo+ydg_cp;   /* CPU: scores+cstate+Y_off+Ydiag_g */
    printf("\n  === FUSED ON-NPU SCAN (grouped matmuls, one chained submit, resident operands) ===\n");
    double ff=0,pf=0,fp=0,pp=0; int of=0,op=0;
    int rf =ork_ssd_fused_scan_bench(c,m.H,m.P,m.Nst,m.G,m.CS,NC,20,0,0,&ff,&pf,&of); /* fp16, factored Ydiag_g (fast, fp16-fragile) */
    int rp =ork_ssd_fused_scan_bench(c,m.H,m.P,m.Nst,m.G,m.CS,NC,20,0,1,&fp,&pp,&op); /* fp16, per-head Y_diag (fp16-STABLE) */
    if(!rf) printf("  fp16 FACTORED (fast, Ydiag_g): %.2f ms [bit-exact %s]  -- OVERFLOWS fp16 at large chunk decay (ssd_coherence)\n", ff/1000, of?"YES":"NO");
    else    printf("  fp16 factored rc=%d\n",rf);
    if(!rp) printf("  fp16 PER-HEAD (fp16-STABLE)  : %.2f ms [bit-exact %s]  -- COHERENT at all dt (rel-L2 1e-3..6e-3)\n", fp/1000, op?"YES":"NO");
    else    printf("  fp16 per-head rc=%d\n",rp);
    printf("  CPU grouped scan (SDOT, 4 big cores): %.2f ms\n", cpu_grouped);
    if(!rp){ double s=fp/1000;
      printf("  --> COHERENT fp16 (per-head) vs CPU: %s by %.2fx | full-model (%d layers) L=%d: NPU %.0f ms (%.0f tok/s) vs CPU %.0f ms (%.0f tok/s)\n",
             s<cpu_grouped?"NPU WINS":"CPU wins", s<cpu_grouped?cpu_grouped/s:s/cpu_grouped,
             m.layers,L, s*m.layers, L/(s*m.layers*1e-3), cpu_grouped*m.layers, L/(cpu_grouped*m.layers*1e-3)); }

    /* ================= END-TO-END LAYER: does the NPU<->CPU handoff erode the CPU-scan win? =========
     * Real layer: in_proj (NPU big GEMM) -> SSD scan -> out_proj (NPU big GEMM). The HYBRID path bounces
     * NPU->CPU->NPU (2 handoffs/layer); the ALL-NPU path stays on the NPU (0 handoffs) but pays the 3-4x
     * slower scan. Time both back-to-back so every real transition/sync/thread-wake cost is included. */
    int dmodel = preset?4096:768, dinner = m.H*m.P;
    int dinproj = 2*dinner + 2*m.G*m.Nst + m.H; dinproj = (dinproj+31)&~31;  /* %32 */
    int8_t *win_w=calloc((size_t)dmodel*dinproj,1), *wout_w=calloc((size_t)dinner*dmodel,1);
    for(size_t i=0;i<(size_t)dmodel*dinproj;i++)win_w[i]=(int8_t)(i&3)-1;
    for(size_t i=0;i<(size_t)dinner*dmodel;i++)wout_w[i]=(int8_t)(i&3)-1;
    ork_w *w_in = ork_mm_pack_i8(c, dmodel, dinproj, win_w);
    ork_w *w_out= ork_mm_pack_i8(c, dinner, dmodel, wout_w);
    if(w_in && w_out){
        int8_t *act_in=malloc((size_t)L*dmodel), *act_mid=malloc((size_t)L*dinner);
        int32_t *proj_in_out=malloc((size_t)L*dinproj*4), *proj_out_out=malloc((size_t)L*dmodel*4);
        for(size_t i=0;i<(size_t)L*dmodel;i++)act_in[i]=1; for(size_t i=0;i<(size_t)L*dinner;i++)act_mid[i]=1;
        /* scan buffers (reused across stages; sized to the largest) */
        int nbG=m.G*NC, nbH=m.H*NC, Mbig=HG*m.P;
        size_t maxA=(size_t)nbH*m.CS*m.CS; if((size_t)nbG*Mbig*m.Nst>maxA)maxA=(size_t)nbG*Mbig*m.Nst;
        size_t maxB=(size_t)nbG*m.Nst*m.CS; if((size_t)nbH*m.CS*m.P>maxB)maxB=(size_t)nbH*m.CS*m.P;
        size_t maxC=(size_t)nbG*Mbig*m.Nst; if((size_t)nbH*m.CS*m.P>maxC)maxC=(size_t)nbH*m.CS*m.P;
        int8_t *sA=malloc(maxA),*sB=malloc(maxB),*sBt=malloc(maxB); int32_t*sC=malloc(maxC*4);
        for(size_t i=0;i<maxA;i++)sA[i]=1; for(size_t i=0;i<maxB;i++)sB[i]=1;
        #define SCAN_NPU() do{ \
            ork_bmm_i8(c,nbG,m.CS,m.Nst,m.CS,sA,sB,sC); ork_bmm_i8(c,nbG,Mbig,m.CS,m.Nst,sA,sB,sC); \
            ork_bmm_i8(c,nbG,Mbig,m.Nst,m.CS,sA,sB,sC); ork_bmm_i8(c,nbH,m.CS,m.CS,m.P,sA,sB,sC); }while(0)
        #define SCAN_CPU() do{ \
            cpu_gemm(sA,sB,sC,NULL,sBt,nbG,m.CS,m.Nst,m.CS); cpu_gemm(sA,sB,sC,NULL,sBt,nbG,Mbig,m.CS,m.Nst); \
            cpu_gemm(sA,sB,sC,NULL,sBt,nbG,Mbig,m.Nst,m.CS); cpu_gemm(sA,sB,sC,NULL,sBt,nbH,m.CS,m.CS,m.P); }while(0)
        int IT=20;
        /* warm */ ork_mm_run_i8(c,w_in,L,act_in,proj_in_out); SCAN_CPU(); ork_mm_run_i8(c,w_out,L,act_mid,proj_out_out); SCAN_NPU();
        double h0=now_us();
        for(int it=0;it<IT;it++){ ork_mm_run_i8(c,w_in,L,act_in,proj_in_out); SCAN_CPU(); ork_mm_run_i8(c,w_out,L,act_mid,proj_out_out); }
        double thyb=(now_us()-h0)/IT/1000;
        double a0=now_us();
        for(int it=0;it<IT;it++){ ork_mm_run_i8(c,w_in,L,act_in,proj_in_out); SCAN_NPU(); ork_mm_run_i8(c,w_out,L,act_mid,proj_out_out); }
        double tall=(now_us()-a0)/IT/1000;
        /* isolate the projections alone to derive the transition overhead in the hybrid */
        double p0=now_us(); for(int it=0;it<IT;it++){ ork_mm_run_i8(c,w_in,L,act_in,proj_in_out); ork_mm_run_i8(c,w_out,L,act_mid,proj_out_out); } double tproj=(now_us()-p0)/IT/1000;
        printf("\n  === END-TO-END LAYER (in_proj NPU + scan + out_proj NPU), L=%d ===\n",L);
        printf("  projections alone (NPU, K=%d/%d): %.2f ms | scan_cpu(isolated) %.2f | scan_npu(isolated) %.2f\n",
               dmodel,dinner,tproj,layer_cpu_ms,layer_npu_ms);
        printf("  HYBRID (NPU->CPU->NPU, 2 handoffs): %.2f ms  | ALL-NPU (0 handoffs): %.2f ms  -> %s by %.2fx\n",
               thyb,tall, thyb<tall?"HYBRID wins":"ALL-NPU wins", thyb<tall?tall/thyb:thyb/tall);
        printf("  handoff overhead = hybrid - (proj + scan_cpu) = %.2f - (%.2f + %.2f) = %.2f ms/layer (2 transitions)\n",
               thyb,tproj,layer_cpu_ms, thyb-tproj-layer_cpu_ms);
        free(act_in);free(act_mid);free(proj_in_out);free(proj_out_out);free(sA);free(sB);free(sBt);free(sC);
    }
    if(w_in)ork_mm_free(c,w_in); if(w_out)ork_mm_free(c,w_out);

    /* ================= fp16 REALITY CHECK: the scan needs fp16 (int8 is numerically incoherent,
     * ssd_coherence.c: maxrel 1e2-1e4). fp16 is the 2-byte datapath (~3.3x int8). Measure the grouped
     * scan matmuls in fp16 (ork_bmm_fp16) vs int8 (ork_bmm_i8) vs CPU to get the REAL fp16-scan cost. */
    printf("\n  === fp16 SCAN (REQUIRED for coherence) vs int8 vs CPU, grouped shapes ===\n");
    { struct { const char*t; int nb,M,K,N; } gs[4] = {
        {"scores", m.G*NC, m.CS, m.Nst, m.CS}, {"Ydiag_g",m.G*NC, m.CS, m.CS, HG*m.P},
        {"cstate", m.G*NC, HG*m.P, m.CS, m.Nst}, {"Y_off", m.G*NC, HG*m.P, m.Nst, m.CS} };
      double i8_tot=0, f16_tot=0, cpu_tot2=0;
      for(int s=0;s<4;s++){ int nb=gs[s].nb,M=gs[s].M,K=gs[s].K,N=gs[s].N;
        size_t szA=(size_t)nb*M*K, szB=(size_t)K*N, szC=(size_t)M*N;
        int8_t*A8=malloc(szA),*B8=malloc(szB),*Bt=malloc((size_t)nb*K*N); int32_t*C8=malloc(szC*4);
        ork_f16*Af=malloc(szA*sizeof(ork_f16)),*Bf=malloc(szB*sizeof(ork_f16)); float*Cf=malloc(szC*4);
        for(size_t i=0;i<szA;i++){A8[i]=1; Af[i]=(ork_f16)1.0f;} for(size_t i=0;i<szB;i++){B8[i]=1; Bf[i]=(ork_f16)1.0f;}
        /* RESIDENT weight (no repack) — pack once, run per batch — matching the NPU-ideal methodology,
         * isolating the pure fp16-vs-int8 NPU COMPUTE ratio at this shape (the fused-chain regime). */
        ork_w*wi=ork_mm_pack_i8(c,K,N,B8); ork_w*wf=ork_mm_pack(c,K,N,Bf); double i8=1e9,f16=1e9;
        if(wi){ ork_mm_run_i8(c,wi,M,A8,C8); double a0=now_us(); for(int b=0;b<nb;b++) ork_mm_run_i8(c,wi,M,A8,C8); i8=(now_us()-a0)/1000; ork_mm_free(c,wi); }
        if(wf){ ork_mm_run(c,wf,M,Af,Cf); double b0=now_us(); for(int b=0;b<nb;b++) ork_mm_run(c,wf,M,Af,Cf); f16=(now_us()-b0)/1000; ork_mm_free(c,wf); }
        double *B8t=(double*)0; (void)B8t;
        double cpu=cpu_gemm(A8,B8,C8,NULL,Bt,1,M,K,N)*nb; cpu=cpu_gemm(A8,B8,C8,NULL,Bt,1,M,K,N)/1000*nb;
        printf("  %-8s int8 %6.2fms | fp16 %6.2fms (%.2fx int8) | CPU %6.2fms\n",gs[s].t,i8,f16,f16/i8,cpu);
        i8_tot+=i8; f16_tot+=f16; cpu_tot2+=cpu;
        free(A8);free(B8);free(Bt);free(C8);free(Af);free(Bf);free(Cf);
      }
      double fused_f16_est = 2.69 * (f16_tot/i8_tot);  /* 130m fused int8 = 2.69ms; scale by resident fp16/int8 */
      printf("  GROUPED SCAN TOTAL (resident): int8 %.2fms | fp16 %.2fms (%.2fx int8) | CPU %.2fms\n",
             i8_tot,f16_tot,f16_tot/i8_tot,cpu_tot2);
      printf("  --> FUSED fp16 scan est (int8-fused 2.69ms x fp16/int8 %.2f) = %.2f ms vs CPU %.2f ms: %s\n",
             f16_tot/i8_tot, fused_f16_est, cpu_tot2,
             fused_f16_est<cpu_tot2? "NPU still wins (fp16 coherent + fused)" : "CPU wins (fp16 penalty erases it)");
      printf("  (int8 fused = 2.66x but INCOHERENT (maxrel 1e2-1e4); fp16 = coherent, this is the real contender.)\n");
    }
    ork_npu_free(c);
    return 0;
}
