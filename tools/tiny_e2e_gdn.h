/* tiny_e2e_gdn.h — chunkwise Gated DeltaNet, forward + backward, with every [MM] step a real matmul.
 *
 * gdn_chunk_bwd.c is the validated derivation (fp64 vs the sequential recurrence to ~1e-14 at every
 * chunk length, vs finite differences, two positive controls, vs MLX fp32 fixtures at T=1024 d=128),
 * but it computes its [MM] lines as scalar loops, so it cannot put anything on the NPU. This file is
 * the SAME sequence of steps -- same variable names, same order, same adjoint, line for line -- with
 * each [MM] line replaced by an mm_nn/mm_nt/mm_tn call (tiny_e2e_mm.h), so the chunk matmuls can be
 * routed to the NPU. It is checked against gdn_chunk_bwd.c itself by tools/tiny_e2e_gdn_check.c
 * (which #includes both), at fp64 algebra level and at fp32, before the trainer uses it.
 *
 * One deliberate difference from gdn_chunk_bwd.c: the input is log g, not g. The model computes
 * log g = -exp(A_log) softplus(a + dt_bias) directly; going through g = exp(.) and back through log()
 * would turn every head whose decay underflows fp32 (A softplus > ~87, reachable at A ~ 16) into
 * log(0) = -inf. The gradient returned is d/dlog g, which is also what the decay backward wants.
 *
 * Layout: per (batch, head), q,k [T,DK], v [T,DV], logg, beta [T], state S [DV,DK] (MLX's layout).
 * Entry state is zero and the final-state cotangent is zero (the model does not carry state across
 * sequences). The forward keeps one state per chunk boundary; the backward replays each chunk from
 * its checkpoint and walks it back (the same memory trade gdn_chunk_bwd.c and the MLX VJP make).
 */
#ifndef TINY_E2E_GDN_H
#define TINY_E2E_GDN_H
#include "tiny_e2e_mm.h"

static int gm_BRK = 0;      /* 1 = drop gate coupling through M (drat), 2 = drop the Gram dK^T term */

typedef struct {
    int C, DK, DV;
    double *L;                                             /* C, cumsum of log g in DOUBLE: see gm_chunk_fwd */
    float *gam,*tail,*dL;                                  /* C   */
    float *rat,*KK,*QK,*M,*P,*drat,*dM,*dP,*dQK,*dKK;      /* C*C */
    float *S0K,*Bm,*U,*QS,*Ut,*dUt,*dU,*Z,*dS0K,*dQS,*tCV; /* C*DV */
    float *tCK;                                            /* C*DK */
    float *tVK;                                            /* DV*DK */
} gm_scr;

static gm_scr *gm_new(int C,int DK,int DV){
    gm_scr *w=calloc(1,sizeof *w); w->C=C; w->DK=DK; w->DV=DV;
    size_t c=C, cc=(size_t)C*C, cv=(size_t)C*DV, ck=(size_t)C*DK, vk=(size_t)DV*DK;
    w->L=calloc(c,8);
    float **p1[]={&w->gam,&w->tail,&w->dL};
    float **p2[]={&w->rat,&w->KK,&w->QK,&w->M,&w->P,&w->drat,&w->dM,&w->dP,&w->dQK,&w->dKK};
    float **p3[]={&w->S0K,&w->Bm,&w->U,&w->QS,&w->Ut,&w->dUt,&w->dU,&w->Z,&w->dS0K,&w->dQS,&w->tCV};
    for(unsigned i=0;i<sizeof p1/sizeof*p1;i++) *p1[i]=calloc(c,4);
    for(unsigned i=0;i<sizeof p2/sizeof*p2;i++) *p2[i]=calloc(cc,4);
    for(unsigned i=0;i<sizeof p3/sizeof*p3;i++) *p3[i]=calloc(cv,4);
    w->tCK=calloc(ck,4); w->tVK=calloc(vk,4);
    return w;
}

/* One chunk forward. q,k,v,logg,beta point at the chunk's first token. Fills w for the backward. */
static void gm_chunk_fwd(gm_scr*w,int C,const float*q,const float*k,const float*v,const float*logg,
                         const float*beta,const float*Sent,float*Sout,float*Y){
    const int DK=w->DK, DV=w->DV;
    /* L in double: with strong decays L reaches -400 inside a chunk, where an fp32 ulp is 3e-5, and
     * exp(L_t - L_s) inherits that as RELATIVE error on every ratio (measured 1.0e-05 worst on the
     * strong-decay check with a float cumsum). Same values, no cancellation, for C extra doubles. */
    double a=0;
    for(int t=0;t<C;t++){ a+=logg[t]; w->L[t]=a; w->gam[t]=(float)exp(a); }
    g_mm_class=MMC_GDN;
    mm_nt(C,DK,C, k,k, w->KK);                                     /* KK  = K Kt   [MM] */
    mm_nt(C,DK,C, q,k, w->QK);                                     /* QK  = Q Kt   [MM] */
    mm_nt(C,DK,DV, k,Sent, w->S0K);                                /* S0K = K St   [MM] */
    mm_nt(C,DK,DV, q,Sent, w->QS);                                 /* QS  = Q St   [MM] */
    for(int t=0;t<C;t++) for(int s=0;s<C;s++){
        size_t ix=(size_t)t*C+s;
        float r = s<=t ? (float)exp(w->L[t]-w->L[s]) : 0.f;
        w->rat[ix]=r;
        w->M[ix] = s<t  ? beta[t]*r*w->KK[ix] : 0.f;
        w->P[ix] = s<=t ? r*w->QK[ix] : 0.f;
    }
    for(int t=0;t<C;t++) for(int i=0;i<DV;i++)
        w->Bm[t*DV+i]=beta[t]*(v[(size_t)t*DV+i]-w->gam[t]*w->S0K[t*DV+i]);
    for(int t=0;t<C;t++){                                         /* (I+M) U = Bm */
        for(int i=0;i<DV;i++) w->U[t*DV+i]=w->Bm[t*DV+i];
        for(int s=0;s<t;s++){ float m=w->M[(size_t)t*C+s]; if(m==0.f) continue;
            for(int i=0;i<DV;i++) w->U[t*DV+i]-=m*w->U[s*DV+i]; }
    }
    g_mm_class=MMC_GDN;
    mm_nn(C,C,DV, w->P,w->U, w->tCV);                              /* P U          [MM] */
    for(int t=0;t<C;t++) for(int i=0;i<DV;i++)
        Y[(size_t)t*DV+i]=w->gam[t]*w->QS[t*DV+i]+w->tCV[t*DV+i];
    for(int s=0;s<C;s++) w->tail[s]=(float)exp(w->L[C-1]-w->L[s]);
    for(int s=0;s<C;s++) for(int i=0;i<DV;i++) w->Ut[s*DV+i]=w->U[s*DV+i]*w->tail[s];
    mm_tn(DV,C,DK, w->Ut,k, w->tVK);                               /* (U.tail)t K  [MM] */
    float gC=w->gam[C-1];
    for(size_t i=0;i<(size_t)DV*DK;i++) Sout[i]=gC*Sent[i]+w->tVK[i];
}

/* One chunk backward: dSout in, dSin out; writes the chunk's slices of dq,dk,dv,dlogg,dbeta. */
static void gm_chunk_bwd(gm_scr*w,int C,const float*q,const float*k,const float*v,const float*beta,
                         const float*Sent,const float*dY,const float*dSout,float*dSin,
                         float*dq,float*dk,float*dv,float*dlogg,float*dbeta){
    const int DK=w->DK, DV=w->DV;
    memset(w->dL,0,(size_t)C*4); memset(w->dU,0,(size_t)C*DV*4); memset(w->drat,0,(size_t)C*C*4);
    memset(dq,0,(size_t)C*DK*4); memset(dk,0,(size_t)C*DK*4); memset(dbeta,0,(size_t)C*4);

    float gC=w->gam[C-1]; double dgC=0;                                         /* S_out */
    for(size_t i=0;i<(size_t)DV*DK;i++){ dSin[i]=gC*dSout[i]; dgC+=(double)dSout[i]*Sent[i]; }
    w->dL[C-1]+=gC*(float)dgC;
    g_mm_class=MMC_GDN;
    mm_nt(C,DK,DV, k,dSout, w->dUt);                               /* dW = K dSot  [MM] */
    mm_nn(C,DV,DK, w->Ut,dSout, w->tCK);                           /* dK += (U.tail) dSo [MM] */
    for(size_t i=0;i<(size_t)C*DK;i++) dk[i]+=w->tCK[i];
    for(int s=0;s<C;s++){
        float dt=0;
        for(int i=0;i<DV;i++){ w->dU[s*DV+i]+=w->dUt[s*DV+i]*w->tail[s]; dt+=w->dUt[s*DV+i]*w->U[s*DV+i]; }
        if(s<C-1){ w->dL[C-1]+=dt*w->tail[s]; w->dL[s]-=dt*w->tail[s]; }   /* s=C-1 is +x-x: see below */
    }

    for(int t=0;t<C;t++){                                                        /* Y */
        float dgam=0;
        for(int i=0;i<DV;i++){ dgam+=dY[(size_t)t*DV+i]*w->QS[t*DV+i];
                               w->dQS[t*DV+i]=w->gam[t]*dY[(size_t)t*DV+i]; }
        w->dL[t]+=w->gam[t]*dgam;
    }
    mm_nn(C,DV,DK, w->dQS,Sent, w->tCK);                           /* dQ += (gam.dY) S  [MM] */
    for(size_t i=0;i<(size_t)C*DK;i++) dq[i]+=w->tCK[i];
    mm_tn(DV,C,DK, w->dQS,q, w->tVK);                              /* dS += (gam.dY)t Q [MM] */
    for(size_t i=0;i<(size_t)DV*DK;i++) dSin[i]+=w->tVK[i];
    mm_nt(C,DV,C, dY,w->U, w->dP);                                 /* dP = dY Ut        [MM] */
    mm_tn(C,C,DV, w->P,dY, w->tCV);                                /* dU += Pt dY       [MM] */
    for(size_t i=0;i<(size_t)C*DV;i++) w->dU[i]+=w->tCV[i];
    for(int t=0;t<C;t++) for(int s=0;s<C;s++){                                    /* P */
        size_t ix=(size_t)t*C+s;
        if(s<=t){ w->dQK[ix]=w->dP[ix]*w->rat[ix]; w->drat[ix]+=w->dP[ix]*w->QK[ix]; }
        else      w->dQK[ix]=0.f;
    }
    mm_nn(C,C,DK, w->dQK,k, w->tCK);                               /* dQ += dQK K       [MM] */
    for(size_t i=0;i<(size_t)C*DK;i++) dq[i]+=w->tCK[i];
    mm_tn(C,C,DK, w->dQK,q, w->tCK);                               /* dK += dQKt Q      [MM] */
    for(size_t i=0;i<(size_t)C*DK;i++) dk[i]+=w->tCK[i];

    for(int t=C-1;t>=0;t--){                                                     /* (I+Mt) Z = dU */
        for(int i=0;i<DV;i++) w->Z[t*DV+i]=w->dU[t*DV+i];
        for(int s=t+1;s<C;s++){ float m=w->M[(size_t)s*C+t]; if(m==0.f) continue;
            for(int i=0;i<DV;i++) w->Z[t*DV+i]-=m*w->Z[s*DV+i]; }
    }
    mm_nt(C,DV,C, w->Z,w->U, w->dM);                               /* dM = -(Z Ut).strict [MM] */
    for(int t=0;t<C;t++) for(int s=0;s<C;s++){ size_t ix=(size_t)t*C+s; w->dM[ix] = s<t ? -w->dM[ix] : 0.f; }

    for(int t=0;t<C;t++){                                                        /* Bm */
        float b=beta[t], dgam=0, db=0;
        for(int i=0;i<DV;i++){
            float z=w->Z[t*DV+i];
            db += z*(v[(size_t)t*DV+i]-w->gam[t]*w->S0K[t*DV+i]);
            dv[(size_t)t*DV+i] = z*b;
            w->dS0K[t*DV+i]    = -z*b*w->gam[t];
            dgam += -b*z*w->S0K[t*DV+i];
        }
        dbeta[t]+=db;
        w->dL[t]+=w->gam[t]*dgam;
    }
    mm_nn(C,DV,DK, w->dS0K,Sent, w->tCK);                          /* dK += dS0K S      [MM] */
    for(size_t i=0;i<(size_t)C*DK;i++) dk[i]+=w->tCK[i];
    mm_tn(DV,C,DK, w->dS0K,k, w->tVK);                             /* dS += dS0Kt K     [MM] */
    for(size_t i=0;i<(size_t)DV*DK;i++) dSin[i]+=w->tVK[i];

    memset(w->dKK,0,(size_t)C*C*4);                                              /* M, then KK */
    for(int t=0;t<C;t++) for(int s=0;s<t;s++){
        size_t ix=(size_t)t*C+s; float dm=w->dM[ix]; if(dm==0.f) continue;
        dbeta[t]+=dm*w->rat[ix]*w->KK[ix];
        if(gm_BRK!=1) w->drat[ix]+=dm*beta[t]*w->KK[ix];
        w->dKK[ix]=dm*beta[t]*w->rat[ix];
    }
    /* KK is a Gram matrix: dK_t += dKK[t,s] K_s AND dK_s += dKK[t,s] K_t, i.e. dK += (dKK + dKKt) K.
     * The symmetrised dKK goes in the dM buffer (dM is consumed above). */
    for(int t=0;t<C;t++) for(int s=0;s<C;s++)
        w->dM[(size_t)t*C+s] = w->dKK[(size_t)t*C+s] + (gm_BRK==2 ? 0.f : w->dKK[(size_t)s*C+t]);
    mm_nn(C,C,DK, w->dM,k, w->tCK);                                /* dK += (dKK+dKKt) K [MM] */
    for(size_t i=0;i<(size_t)C*DK;i++) dk[i]+=w->tCK[i];

    /* ratio. The diagonal s=t is skipped: it adds r to dL_t and subtracts it again, exactly zero in
     * real arithmetic, but rat[t,t]=1 makes r the LARGEST term, and in fp32 (dL_t + r) - r rounds away
     * the low bits of a dL_t that can be 1e4x smaller. Measured: with the diagonal kept, the model's
     * d/dA_log was 1.6e-03 off MLX while the fp64 oracle recurrence in the same model gave 1.4e-06.
     * gdn_chunk_bwd.c keeps the diagonal and is right to: in fp64 the loss is invisible. */
    for(int t=0;t<C;t++) for(int s=0;s<t;s++){
        float r=w->drat[(size_t)t*C+s]*w->rat[(size_t)t*C+s];
        w->dL[t]+=r; w->dL[s]-=r;
    }
    float acc=0;                                                                 /* L = cumsum(log g) */
    for(int t=C-1;t>=0;t--){ acc+=w->dL[t]; dlogg[t]=acc; }
}

/* Whole sequence for one (batch, head): zero entry state; ent receives nc checkpoints [DV,DK]. */
static void gm_fwd(gm_scr*w,int T,int CH,const float*q,const float*k,const float*v,const float*logg,
                   const float*beta,float*Y,float*ent){
    const int DK=w->DK, DV=w->DV; size_t sz=(size_t)DV*DK;
    float *S=calloc(sz,4), *Sn=malloc(sz*4);
    int nc=(T+CH-1)/CH;
    for(int c=0;c<nc;c++){
        int t0=c*CH, C=T-t0<CH?T-t0:CH;
        memcpy(ent+(size_t)c*sz,S,sz*4);
        gm_chunk_fwd(w,C,q+(size_t)t0*DK,k+(size_t)t0*DK,v+(size_t)t0*DV,logg+t0,beta+t0,S,Sn,Y+(size_t)t0*DV);
        memcpy(S,Sn,sz*4);
    }
    free(S); free(Sn);
}
static void gm_bwd(gm_scr*w,int T,int CH,const float*q,const float*k,const float*v,const float*logg,
                   const float*beta,const float*ent,const float*dY,
                   float*dq,float*dk,float*dv,float*dlogg,float*dbeta){
    const int DK=w->DK, DV=w->DV; size_t sz=(size_t)DV*DK;
    float *dS=calloc(sz,4), *dSn=malloc(sz*4), *Sn=malloc(sz*4), *Yt=malloc((size_t)CH*DV*4);
    int nc=(T+CH-1)/CH;
    for(int c=nc-1;c>=0;c--){
        int t0=c*CH, C=T-t0<CH?T-t0:CH;
        const float *Sent=ent+(size_t)c*sz;
        gm_chunk_fwd(w,C,q+(size_t)t0*DK,k+(size_t)t0*DK,v+(size_t)t0*DV,logg+t0,beta+t0,Sent,Sn,Yt);
        gm_chunk_bwd(w,C,q+(size_t)t0*DK,k+(size_t)t0*DK,v+(size_t)t0*DV,beta+t0,Sent,dY+(size_t)t0*DV,
                     dS,dSn,dq+(size_t)t0*DK,dk+(size_t)t0*DK,dv+(size_t)t0*DV,dlogg+t0,dbeta+t0);
        memcpy(dS,dSn,sz*4);
    }
    free(dS); free(dSn); free(Sn); free(Yt);
}
#endif /* TINY_E2E_GDN_H */
