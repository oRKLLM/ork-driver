/* test_gdn_chunk_npu — real-data numerics for one Gated-DeltaNet (GDA) chunk-scan layer on the NPU
 * (ork_gdn_scan_f32), validated vs the definitional fp64 delta-rule recurrence (same ground truth as
 * examples/test_gdn_chunk.c). The delta-rule twin of test_ssd_chunk_npu.c.
 *
 * ork_gdn_scan_f32 runs the 6 matmul stages (Sk,Sq,KK,KQ,O_intra,Sdelta) on the fused-multicore fp16
 * stream and the UT-transform (triangular solve) on the CPU by forward substitution. This asserts the
 * on-NPU output matches the sequential recurrence within fp16 tolerance. Skips (exit 0) with no NPU
 * (board only for a real result). Part of `make test`.
 */
#define _GNU_SOURCE
#include "ork_npu.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

static float frs(void){ return ((float)rand()/RAND_MAX)*2.0f-1.0f; }   /* [-1,1) */

/* definitional gated-delta-rule recurrence (fp64), layout q,k,v[ns,nt,nh,d] g,beta[ns,nt,nh]
 * s0,s_new[ns,nh,d,d] (key,val) o[ns,nt,nh,d]. */
static void gdn_seq_ref(int d,int nh,int nt,int ns,const float*s0,const float*q,const float*k,
                        const float*v,const float*g,const float*beta,float*o,float*s_new){
    const double qs=1.0/sqrt((double)d);
    double *S=malloc((size_t)d*d*sizeof(double)),*pred=malloc((size_t)d*sizeof(double));
    for(int seq=0;seq<ns;seq++)for(int h=0;h<nh;h++){
        for(int i=0;i<d*d;i++) S[i]=s0[((size_t)seq*nh+h)*d*d+i];
        for(int t=0;t<nt;t++){
            double gate=exp(g[(size_t)(seq*nt+t)*nh+h]);
            for(int i=0;i<d*d;i++) S[i]*=gate;
            for(int val=0;val<d;val++){ double a=0; for(int key=0;key<d;key++) a+=S[(size_t)key*d+val]*k[((size_t)(seq*nt+t)*nh+h)*d+key]; pred[val]=a; }
            double bt=beta[(size_t)(seq*nt+t)*nh+h];
            for(int key=0;key<d;key++){ double kk=k[((size_t)(seq*nt+t)*nh+h)*d+key];
                for(int val=0;val<d;val++) S[(size_t)key*d+val]+=kk*bt*(v[((size_t)(seq*nt+t)*nh+h)*d+val]-pred[val]); }
            for(int val=0;val<d;val++){ double a=0; for(int key=0;key<d;key++) a+=S[(size_t)key*d+val]*q[((size_t)(seq*nt+t)*nh+h)*d+key]*qs; o[((size_t)(seq*nt+t)*nh+h)*d+val]=(float)a; }
        }
        for(int i=0;i<d*d;i++) s_new[((size_t)seq*nh+h)*d*d+i]=(float)S[i];
    }
    free(S);free(pred);
}

static double rel_l2(const float*a,const float*b,size_t n){ double sd=0,sr=0; for(size_t i=0;i<n;i++){ double e=(double)a[i]-b[i]; sd+=e*e; sr+=(double)b[i]*b[i]; } return sqrt(sd/(sr+1e-12)); }

static int run_case(ork_npu*c,const char*tag,int d,int nh,int nt,int ns,double tol){
    size_t nqkv=(size_t)ns*nt*nh*d, ngb=(size_t)ns*nt*nh, nst=(size_t)ns*nh*d*d;
    float *q=malloc(nqkv*4),*k=malloc(nqkv*4),*v=malloc(nqkv*4),*g=malloc(ngb*4),*bt=malloc(ngb*4),*s0=malloc(nst*4);
    float *o=malloc(nqkv*4),*sn=malloc(nst*4),*oref=malloc(nqkv*4),*snref=malloc(nst*4);
    for(size_t i=0;i<nqkv;i++){ q[i]=frs(); k[i]=frs(); v[i]=frs(); }
    /* real GDN L2-normalizes q,k per (token,head) before the scan (fla l2norm) so k_l·k_s∈[-1,1] and
     * the UT-transform (I+A)^{-1} stays well-conditioned; unnormalized k blows it up as d grows. */
    for(size_t j=0;j<(size_t)ns*nt*nh;j++){ double nk=0,nq=0;
        for(int e=0;e<d;e++){ nk+=(double)k[j*d+e]*k[j*d+e]; nq+=(double)q[j*d+e]*q[j*d+e]; }
        float ik=(float)(1.0/sqrt(nk+1e-12)), iq=(float)(1.0/sqrt(nq+1e-12));
        for(int e=0;e<d;e++){ k[j*d+e]*=ik; q[j*d+e]*=iq; } }
    for(size_t i=0;i<ngb;i++){ g[i]=-(((float)rand()/RAND_MAX)*0.6f+0.02f); bt[i]=((float)rand()/RAND_MAX)*0.9f+0.05f; }
    for(size_t i=0;i<nst;i++) s0[i]=frs()*0.1f;

    int rc=ork_gdn_scan_f32(c,d,nh,nt,ns,s0,q,k,v,g,bt,o,sn);
    if(rc){ fprintf(stderr,"[%s] ork_gdn_scan_f32 rc=%d\n",tag,rc); free(q);free(k);free(v);free(g);free(bt);free(s0);free(o);free(sn);free(oref);free(snref); return 1; }
    gdn_seq_ref(d,nh,nt,ns,s0,q,k,v,g,bt,oref,snref);
    double ro=rel_l2(o,oref,nqkv), rss=rel_l2(sn,snref,nst);
    int fail=!(ro<=tol && rss<=tol);
    fprintf(stderr,"[%s] d=%d nh=%d nt=%d ns=%d  o rel-L2=%.3e  s_new rel-L2=%.3e  %s\n",tag,d,nh,nt,ns,ro,rss,fail?"FAIL":"OK");
    free(q);free(k);free(v);free(g);free(bt);free(s0);free(o);free(sn);free(oref);free(snref);
    return fail;
}

int main(void){
    srand(20260713);
    ork_npu *c=ork_npu_init();
    if(!c){ fprintf(stderr,"[gdn-npu] no NPU (init failed) — skipping (exit 0)\n"); return 0; }
    int fail=0;
    /* fp16 stream + fp64 forward-subst → rel-L2 ~1e-3; the delta rule can amplify, tol 3e-2 (coherence). */
    fail |= run_case(c,"gdn-npu-1chunk", 32, 2, 64,  1, 3e-2);
    fail |= run_case(c,"gdn-npu-multi",  32, 4, 192, 1, 3e-2);
    fail |= run_case(c,"gdn-npu-qwen",  128, 2, 128, 1, 3e-2);   /* Ornith/Qwen3-Next head_dim=128, CS=64 */
    fail |= run_case(c,"gdn-npu-2seq",   64, 2, 128, 2, 3e-2);
    ork_npu_free(c);
    fprintf(stderr, fail ? "\nTEST_GDN_CHUNK_NPU: FAIL\n" : "\nTEST_GDN_CHUNK_NPU: PASS\n");
    return fail?1:0;
}
