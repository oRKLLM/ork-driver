/* kv_append_probe — validate Tier 12f resident-KV APPEND. Append K/V key-by-key into resident packed weights
 * (ork_kv_resident_alloc/ork_kv_append), run decode attention, and check it EQUALS the full-repack path
 * (ork_i8_mm_pack of the whole K^T/V — same int8 bytes, so bit-for-bit) AND a CPU fp reference. Proves the tiled
 * write in ork_kv_append matches ork_i8_mm_pack's layout (incl. multi-tile V for L>1024).
 *   sudo env ORK_MM_TIMEOUT=3000 ./kv_append_probe [L] [HD]
 */
#define _GNU_SOURCE
#include "ork_npu.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
static int run_attn(ork_npu*c,ork_w*wkt,ork_w*wv,int8_t*Q8,int L,int HD,float scale,float qs,float ks,float vs,float*att){
    int32_t *scores=malloc((size_t)L*4),*attv=malloc((size_t)HD*4); double*sc=malloc((size_t)L*sizeof(double)); int8_t*w8=malloc((size_t)L);
    int rc=-1;
    ork_mm_task_i8 t1={wkt,1,Q8,scores}; if(ork_i8_mm_run_chain(c,1,&t1)) goto out;
    { double mx=-1e300; for(int j=0;j<L;j++){ sc[j]=(double)scores[j]/((double)qs*ks)*scale; if(sc[j]>mx)mx=sc[j]; }
      double Z=0; for(int j=0;j<L;j++){ sc[j]=exp(sc[j]-mx); Z+=sc[j]; } if(Z<=0)Z=1;
      double wmax=0; for(int j=0;j<L;j++){ sc[j]/=Z; if(sc[j]>wmax)wmax=sc[j]; } double ws=127.0/(wmax>1e-9?wmax:1.0);
      for(int j=0;j<L;j++){ int wi=(int)lrint(sc[j]*ws); w8[j]=(int8_t)(wi>127?127:(wi<0?0:wi)); }
      ork_mm_task_i8 t2={wv,1,w8,attv}; if(ork_i8_mm_run_chain(c,1,&t2)) goto out;
      for(int e=0;e<HD;e++) att[e]=(float)((double)attv[e]/(ws*vs)); rc=0; }
out: free(scores);free(attv);free(sc);free(w8); return rc;
}
int main(int argc,char**argv){
    int L=argc>1?atoi(argv[1]):512, HD=argc>2?atoi(argv[2]):128, Kp=512;
    setvbuf(stdout,0,_IONBF,0);
    if(L%32){ printf("L must be %%32\n"); return 2; }
    ork_npu*c=ork_npu_init(); if(!c){printf("init failed\n");return 2;}
    printf("kv_append_probe: L=%d HD=%d (V K-tiles=%d) — resident append vs full-repack vs CPU\n",L,HD,(L+1023)/1024);
    float scale=1.0f/sqrtf((float)HD);
    float *K=malloc((size_t)L*HD*4),*V=malloc((size_t)L*HD*4),*Q=malloc((size_t)HD*4);
    uint32_t g=0x1234; for(int e=0;e<HD;e++){ g=g*1664525u+1013904223u; Q[e]=((int)((g>>26)%17)-8)*0.1f; }
    for(size_t i=0;i<(size_t)L*HD;i++){ g=g*1664525u+1013904223u; K[i]=((int)((g>>26)%17)-8)*0.1f; }
    for(size_t i=0;i<(size_t)L*HD;i++){ g=g*1664525u+1013904223u; V[i]=((int)((g>>26)%17)-8)*0.1f; }
    float qmax=1e-6f,kmax=1e-6f,vmax=1e-6f;
    for(int e=0;e<HD;e++){ float a=fabsf(Q[e]); if(a>qmax)qmax=a; }
    for(size_t i=0;i<(size_t)L*HD;i++){ float ka=fabsf(K[i]),va=fabsf(V[i]); if(ka>kmax)kmax=ka; if(va>vmax)vmax=va; }
    float qs=127.0f/qmax, ks=127.0f/kmax, vs=127.0f/vmax;
    int8_t *Q8=calloc((size_t)Kp,1); for(int e=0;e<HD;e++) Q8[e]=(int8_t)lrintf(Q[e]*qs);

    /* reference A: FULL REPACK (pack the whole K^T/V at once) */
    int8_t *KTp=calloc((size_t)Kp*L,1),*Vp=malloc((size_t)L*HD);
    for(int e=0;e<HD;e++)for(int j=0;j<L;j++) KTp[(size_t)e*L+j]=(int8_t)lrintf(K[(size_t)j*HD+e]*ks);
    for(size_t i=0;i<(size_t)L*HD;i++) Vp[i]=(int8_t)lrintf(V[i]*vs);
    ork_w *rkt=ork_i8_mm_pack(c,Kp,L,KTp), *rv=ork_i8_mm_pack(c,L,HD,Vp);
    if(!rkt||!rv){ printf("repack pack fail\n"); return 2; }
    float *att_ref=malloc((size_t)HD*4); if(run_attn(c,rkt,rv,Q8,L,HD,scale,qs,ks,vs,att_ref)){ printf("repack run fail\n"); return 1; }

    /* path B: RESIDENT APPEND, key by key */
    ork_kv_resident *kv=ork_kv_resident_alloc(c,HD,L); if(!kv){ printf("kv alloc fail (L>nmax?)\n"); return 2; }
    int8_t *kcol=malloc((size_t)HD),*vrow=malloc((size_t)HD);
    for(int j=0;j<L;j++){ for(int e=0;e<HD;e++){ kcol[e]=(int8_t)lrintf(K[(size_t)j*HD+e]*ks); vrow[e]=(int8_t)lrintf(V[(size_t)j*HD+e]*vs); }
        if(ork_kv_append(c,kv,j,kcol,vrow)){ printf("append fail @%d\n",j); return 1; } }
    float *att_res=malloc((size_t)HD*4); if(run_attn(c,kv->wkt,kv->wv,Q8,L,HD,scale,qs,ks,vs,att_res)){ printf("resident run fail\n"); return 1; }

    /* CPU fp reference */
    float *att_cpu=malloc((size_t)HD*4);
    { double mx=-1e300,*sc=malloc((size_t)L*sizeof(double));
      for(int j=0;j<L;j++){ double s=0; for(int e=0;e<HD;e++)s+=Q[e]*K[(size_t)j*HD+e]; sc[j]=s*scale; if(sc[j]>mx)mx=sc[j]; }
      double Z=0; for(int j=0;j<L;j++){ sc[j]=exp(sc[j]-mx); Z+=sc[j]; }
      for(int e=0;e<HD;e++){ double a=0; for(int j=0;j<L;j++)a+=sc[j]*V[(size_t)j*HD+e]; att_cpu[e]=(float)(a/Z); } free(sc); }

    double d_repack=0, d_cpu=0, rm=0;
    for(int e=0;e<HD;e++){ if(fabsf(att_ref[e])>rm)rm=fabsf(att_ref[e]);
        double a=fabs(att_res[e]-att_ref[e]); if(a>d_repack)d_repack=a;
        double b=fabs(att_res[e]-att_cpu[e]); if(b>d_cpu)d_cpu=b; }
    double rel_cpu=d_cpu/(rm>1e-6?rm:1);
    printf("  append vs full-repack: max|diff|=%.6f  %s\n", d_repack, d_repack<1e-4?"IDENTICAL":"MISMATCH");
    printf("  append vs CPU fp     : rel-err=%.4f  %s\n", rel_cpu, rel_cpu<0.08?"COHERENT":"CHECK");
    int ok = d_repack<1e-4 && rel_cpu<0.08;
    printf("%s\n", ok?"PASS — resident append == full-repack, coherent vs CPU (Tier 12f append validated)":"FAIL");
    ork_kv_resident_free(c,kv); ork_w_free(rkt); ork_w_free(rv); ork_npu_free(c);
    return ok?0:1;
}
