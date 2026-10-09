/* tiny_e2e_gdn_check — the matmul-form chunkwise GDN (tiny_e2e_gdn.h) against the validated scalar
 * one (gdn_chunk_bwd.c), before the trainer is allowed to use it.
 *
 * gdn_chunk_bwd.c is #included at fp64 as the ORACLE (its own self-checks: identical to the sequential
 * recurrence at every chunk length, finite differences, positive controls). tiny_e2e_gdn.h is the same
 * derivation with its [MM] lines as real matmul calls, run here at fp32 with the matmuls on
 *   cpu   fp32 CPU            -> must agree to fp32 rounding (~1e-6)
 *   emu   fp16 operands, CPU  -> what fp16 operands cost
 *   npu   fp16 on the NPU     -> only in a -DWITH_NPU build; must sit next to emu
 * on two input regimes: the moderate decays of gdn_chunk_bwd.c's own test (g in 0.15..0.95), and
 * STRONG decays drawn the way the tiny model's init produces them (log g down to -20 per token, i.e.
 * gamma_t/gamma_s spanning hundreds of orders of magnitude inside one 64-token chunk) -- the fp16
 * range risk the scope report left open.
 *
 * Positive controls: gm_BRK=1 (gate coupling through M dropped) and gm_BRK=2 (Gram dK^T term dropped)
 * must FAIL the cpu comparison, or the comparison proves nothing.
 *
 *   cc -O2 -Itools -o /tmp/gchk tools/tiny_e2e_gdn_check.c -lm && /tmp/gchk
 *   (board)  cc -O2 -DWITH_NPU -Iinclude -Isrc -Itools tools/tiny_e2e_gdn_check.c libork_npu.a -lm -pthread
 */
#define GDN_CHUNK_NO_MAIN 1
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
#include "gdn_chunk_bwd.c"
#include "tiny_e2e_gdn.h"
#pragma GCC diagnostic pop

static float *fq,*fk,*fv,*flg,*fbe,*fY,*fdY,*fdq,*fdk,*fdv,*fdlg,*fdbe,*fent;

static double relf(const float*a,const double*b,size_t n){
    double w=0,m=0;
    for(size_t i=0;i<n;i++){ double d=fabs((double)a[i]-b[i]); if(d>w) w=d; if(fabs(b[i])>m) m=fabs(b[i]); }
    return m>0? w/m : w;
}

static double run_mine(int CH,gm_scr*w,double *out6,int quiet){
    gm_fwd(w,T,CH,fq,fk,fv,flg,fbe,fY,fent);
    gm_bwd(w,T,CH,fq,fk,fv,flg,fbe,fent,fdY,fdq,fdk,fdv,fdlg,fdbe);
    double r[6]={ relf(fY,oY,(size_t)T*DV), relf(fdq,odq,(size_t)T*DK), relf(fdk,odk,(size_t)T*DK),
                  relf(fdv,odv,(size_t)T*DV), relf(fdlg,odlg,T), relf(fdbe,odbe,T) };
    double worst=0; for(int i=0;i<6;i++){ if(r[i]>worst) worst=r[i]; if(out6) out6[i]=r[i]; }
    if(!quiet) printf("Y %.2e  dq %.2e  dk %.2e  dv %.2e  dlogg %.2e  dbeta %.2e   worst %.2e",
                      r[0],r[1],r[2],r[3],r[4],r[5],worst);
    return worst;
}

int main(int argc,char**argv){
    T=argc>1?atoi(argv[1]):256; DK=64; DV=64; int CH=64;
    double DYS=argc>2?atof(argv[2]):1e-3;   /* cotangent scale: 1e-6 puts dY in fp16's SUBNORMAL range */
    size_t tk=(size_t)T*DK, tv=(size_t)T*DV, sz=(size_t)DV*DK;
    q=malloc(tk*RSZ); k=malloc(tk*RSZ); v=malloc(tv*RSZ); g=malloc((size_t)T*RSZ); bet=malloc((size_t)T*RSZ);
    S0=calloc(sz,RSZ); dY=malloc(tv*RSZ); dSo=calloc(sz,RSZ); Y=malloc(tv*RSZ); Sf=malloc(sz*RSZ);
    dq=malloc(tk*RSZ); dk=malloc(tk*RSZ); dv=malloc(tv*RSZ); dlg=malloc((size_t)T*RSZ);
    dbe=malloc((size_t)T*RSZ); dSi=malloc(sz*RSZ);
    oY=malloc(tv*RSZ); oSf=malloc(sz*RSZ); odq=malloc(tk*RSZ); odk=malloc(tk*RSZ); odv=malloc(tv*RSZ);
    odlg=malloc((size_t)T*RSZ); odbe=malloc((size_t)T*RSZ); odSi=malloc(sz*RSZ);
    fq=malloc(tk*4); fk=malloc(tk*4); fv=malloc(tv*4); flg=malloc((size_t)T*4); fbe=malloc((size_t)T*4);
    fY=malloc(tv*4); fdY=malloc(tv*4); fdq=malloc(tk*4); fdk=malloc(tk*4); fdv=malloc(tv*4);
    fdlg=malloc((size_t)T*4); fdbe=malloc((size_t)T*4); fent=malloc((size_t)((T+CH-1)/CH)*sz*4);
    gm_scr *w=gm_new(CH,DK,DV);
#ifdef WITH_NPU
    g_npu=ork_npu_init(); if(!g_npu){ printf("ork_npu_init failed\n"); return 2; }
#endif
    int fails=0;
    const char *regime[2]={"moderate decay  g in (0.15,0.95)","STRONG decay    log g in (-20,-0.02)"};
    for(int reg=0;reg<2;reg++){
        /* q,k scaled as the model scales them (rms_norm then Dk^-1 / Dk^-0.5), v ~ O(1) */
        for(int t=0;t<T;t++){
            double sq=0,sk=0; double qq[256],kk[256];
            for(int j=0;j<DK;j++){ qq[j]=rnd(); kk[j]=rnd(); sq+=qq[j]*qq[j]; sk+=kk[j]*kk[j]; }
            sq=sqrt(sq/DK+1e-6); sk=sqrt(sk/DK+1e-6);
            for(int j=0;j<DK;j++){ QI(t,j)=qq[j]/sq/DK; KI(t,j)=kk[j]/sk/sqrt((double)DK); }
            for(int i=0;i<DV;i++){ VI(t,i)=rnd(); dY[(size_t)t*DV+i]=DYS*rnd(); }
            double lgv = reg==0 ? log(0.55+0.4*rnd()) : -exp(3.0*rnd()) ;   /* -0.05..-20 */
            g[t]=exp(lgv); bet[t]=0.5+0.4*rnd();
            flg[t]=(float)lgv; fbe[t]=(float)bet[t];
        }
        for(size_t i=0;i<tk;i++){ fq[i]=(float)q[i]; fk[i]=(float)k[i]; }
        for(size_t i=0;i<tv;i++){ fv[i]=(float)v[i]; fdY[i]=(float)dY[i]; }
        /* oracle on the fp32-rounded inputs, so the comparison sees only the implementation */
        for(size_t i=0;i<tk;i++){ q[i]=fq[i]; k[i]=fk[i]; }
        for(size_t i=0;i<tv;i++){ v[i]=fv[i]; dY[i]=fdY[i]; }
        for(int t=0;t<T;t++){ g[t]=exp((double)flg[t]); bet[t]=fbe[t]; }
        seq_fwd_bwd();                       /* oracle: the sequential recurrence (oY, odq, ...) */
        bwd_chunked(CH); { double d=0,m=0; for(size_t i=0;i<tk;i++){ d=fmax(d,fabs(dk[i]-odk[i])); m=fmax(m,fabs(odk[i])); }
            printf("\n%s   T=%d dk=dv=%d chunk=%d  |dY|<=%g\n  oracle self-check (gdn_chunk_bwd chunkwise vs sequential, fp64): dk %.1e\n",
                   regime[reg],T,DK,CH,DYS,d/m); }

        const char *mn[3]={"cpu fp32","emu fp16 ops","npu fp16"};
        int nmode=2;
#ifdef WITH_NPU
        nmode=3;
#endif
        for(int sc=0;sc<2;sc++){
            g_mm_scale=sc;
            for(int md=0;md<nmode;md++){
                if(md==0 && sc==1) continue;
                g_mm_mode[MMC_GDN]=md;
                printf("  %-12s scale=%d  ",mn[md],sc);
                mm_stats_reset();
                double wst=run_mine(CH,w,NULL,0);
                /* keep the emu outputs so the npu run is compared to them DIRECTLY: equal printed
                 * errors vs the oracle are what a silent CPU fallback would also print */
                static float *e_dk,*e_dlg,*e_Y;
                if(md==1){ free(e_dk); free(e_dlg); free(e_Y);
                    e_dk=malloc(tk*4); e_dlg=malloc((size_t)T*4); e_Y=malloc(tv*4);
                    memcpy(e_dk,fdk,tk*4); memcpy(e_dlg,fdlg,(size_t)T*4); memcpy(e_Y,fY,tv*4); }
                if(md==2){
                    double a=0,b=0,c2=0,ma=0,mb=0,mc=0;
                    for(size_t i=0;i<tk;i++){ a=fmax(a,fabs(fdk[i]-e_dk[i])); ma=fmax(ma,fabs(e_dk[i])); }
                    for(int i=0;i<T;i++){ b=fmax(b,fabs(fdlg[i]-e_dlg[i])); mb=fmax(mb,fabs(e_dlg[i])); }
                    for(size_t i=0;i<tv;i++){ c2=fmax(c2,fabs(fY[i]-e_Y[i])); mc=fmax(mc,fabs(e_Y[i])); }
                    printf("\n      npu vs emu directly: Y %.2e dk %.2e dlogg %.2e   [%ld matmuls, %ld cpu-fallback, npu %.3fs of %.3fs]",
                           c2/mc,a/ma,b/mb,g_mm_calls[MMC_GDN],g_mm_fallback[MMC_GDN],g_mm_npu_sec[MMC_GDN],g_mm_sec[MMC_GDN]);
                }
                double bar = md==0 ? 1e-5 : 1e-1;
                printf("   %s\n", wst<=bar?"ok":"FAIL");
                if(wst>bar) fails++;
            }
        }
        g_mm_mode[MMC_GDN]=MM_CPU; g_mm_scale=0;
        for(int b=1;b<=2;b++){
            gm_BRK=b; double wst=run_mine(CH,w,NULL,1); gm_BRK=0;
            printf("  positive control %d (%s): worst %.2e  %s\n",b,b==1?"gate coupling via M dropped":"Gram dK^T dropped",
                   wst, wst>1e-5?"FAIL (as required)":"PASSED -- check is vacuous");
            if(!(wst>1e-5)) fails++;
        }
    }
#ifdef WITH_NPU
    mm_npu_free_all(); ork_npu_free(g_npu);
#endif
    printf("\n%s\n",fails?"SOME FAILED":"ALL PASS");
    return fails!=0;
}
