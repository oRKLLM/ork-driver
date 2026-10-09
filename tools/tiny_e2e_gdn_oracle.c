/* tiny_e2e_gdn_oracle — gdn_chunk_bwd.c (fp64) behind a float interface, as its own translation unit.
 *
 * Debug oracle for train_tiny_e2e.c (-DGDN_ORACLE): replaces the fp32 matmul-form recurrence with the
 * validated fp64 scalar one, so a model-level gradient mismatch can be split into "the recurrence"
 * versus "everything around it". Separate TU because gdn_chunk_bwd.c's globals (T, q, k, ...) collide
 * with the trainer's. Zero entry state, zero final-state cotangent, g = exp(log g) (the oracle takes g).
 */
#define GDN_CHUNK_NO_MAIN 1
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
#pragma GCC diagnostic ignored "-Wunused-variable"
#include "gdn_chunk_bwd.c"
#pragma GCC diagnostic pop

static int cap_T=-1, cap_K=-1, cap_V=-1;
static void ensure(int t,int nk,int nv){
    if(t==cap_T && nk==cap_K && nv==cap_V) return;
    T=t; DK=nk; DV=nv; cap_T=t; cap_K=nk; cap_V=nv;
    size_t tk=(size_t)T*DK, tv=(size_t)T*DV, sz=(size_t)DV*DK;
    q=realloc(q,tk*RSZ); k=realloc(k,tk*RSZ); v=realloc(v,tv*RSZ); g=realloc(g,T*RSZ); bet=realloc(bet,T*RSZ);
    S0=realloc(S0,sz*RSZ); dY=realloc(dY,tv*RSZ); dSo=realloc(dSo,sz*RSZ); Y=realloc(Y,tv*RSZ); Sf=realloc(Sf,sz*RSZ);
    dq=realloc(dq,tk*RSZ); dk=realloc(dk,tk*RSZ); dv=realloc(dv,tv*RSZ); dlg=realloc(dlg,T*RSZ);
    dbe=realloc(dbe,T*RSZ); dSi=realloc(dSi,sz*RSZ);
    memset(S0,0,sz*RSZ); memset(dSo,0,sz*RSZ);
}
static void load(int t,int nk,int nv,const float*fq,const float*fk,const float*fv,const float*flg,const float*fb){
    ensure(t,nk,nv);
    for(size_t i=0;i<(size_t)T*DK;i++){ q[i]=fq[i]; k[i]=fk[i]; }
    for(size_t i=0;i<(size_t)T*DV;i++) v[i]=fv[i];
    for(int i=0;i<T;i++){ g[i]=exp((double)flg[i]); bet[i]=fb[i]; }
}
void gdn_oracle_fwd(int t,int nk,int nv,int ch,const float*fq,const float*fk,const float*fv,const float*flg,
                    const float*fb,float*fy){
    load(t,nk,nv,fq,fk,fv,flg,fb);
    fwd_chunked(ch);
    for(size_t i=0;i<(size_t)T*DV;i++) fy[i]=(float)Y[i];
}
void gdn_oracle_bwd(int t,int nk,int nv,int ch,const float*fq,const float*fk,const float*fv,const float*flg,
                    const float*fb,const float*fdy,float*gq,float*gk,float*gv,float*glg,float*gb){
    load(t,nk,nv,fq,fk,fv,flg,fb);
    for(size_t i=0;i<(size_t)T*DV;i++) dY[i]=fdy[i];
    bwd_chunked(ch);
    for(size_t i=0;i<(size_t)T*DK;i++){ gq[i]=(float)dq[i]; gk[i]=(float)dk[i]; }
    for(size_t i=0;i<(size_t)T*DV;i++) gv[i]=(float)dv[i];
    for(int i=0;i<T;i++){ glg[i]=(float)dlg[i]; gb[i]=(float)dbe[i]; }
}
