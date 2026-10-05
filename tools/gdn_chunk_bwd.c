/* gdn_chunk_bwd — chunkwise-parallel Gated DeltaNet, forward AND hand-derived backward, in C.
 *
 * WHY THIS AND NOT THE RECURRENT FORM. tools/gdn_bwd_fp64_ref.c proves the recurrent backward, but
 * the recurrent form advances a [Dv,Dk] state once per token: serial depth T, and all of its
 * arithmetic is outer products and mat-vecs — work shapes that leave a matrix unit idle. The NPU is
 * a matrix unit. The chunkwise form cuts serial depth to T/C and turns the intra-chunk work into
 * dense matmuls, which is what makes any of this reach the hardware at all. Every matmul marked
 * [MM] below is an ork_f16_mm_pack/run call in a real implementation.
 *
 * THERE IS NOTHING TO PORT. The MLX reference (qwen35-lora-moe/kernels/) implements the chunkwise
 * FORWARD by hand but obtains its backward from mx.vjp through that same forward — deliberately,
 * "correct by construction and immune to the usual failure of adjoint code". C has no autograd, so
 * the adjoint IS the work here, and the failure mode that protects against is exactly what this
 * file risks. Hence two independent oracles below, not one.
 *
 * FORWARD, per (batch, head), chunk length C, entry state S [Dv,Dk]:
 *      L     = cumsum(log g)                       gamma_t = exp(L_t)
 *      ratio[t,s] = exp(L_t - L_s)                 s <= t only; the exponent is <= 0 there
 *      KK    = K Kt                          [MM]  M[t,s] = b_t ratio[t,s] KK[t,s], s < t
 *      S0K   = K St                          [MM]  Bm[t]  = b_t (v_t - gamma_t S0K[t])
 *      (I+M) U = Bm                                unit lower triangular solve
 *      P[t,s] = ratio[t,s] (Q Kt)[t,s]       [MM]  s <= t
 *      Y     = gamma . (Q St) + P U          [MM]
 *      tail_s = exp(L_{C-1} - L_s)
 *      S_out = gamma_{C-1} S + (U . tail)t K [MM]
 *
 * BACKWARD. Each forward line differentiated, in reverse. Z is the adjoint of the solve:
 *      S_out  dS += g_C dSo       dL_{C-1} += g_C <dSo,S>      dW = K dSot     dK += (U.tail) dSo
 *             dU += dW . tail     dtail = <dW,U>               dL_{C-1} += <dtail,tail>, dL_s -= "
 *      Y      dL_t += gamma_t <dY_t, QS_t>         dQ += (gamma.dY) S    dS += (gamma.dY)t Q
 *             dP = dY Ut                           dU += Pt dY
 *      P      dQK = dP . ratio                     dratio += dP . QK     dQ += dQK K, dK += dQKt Q
 *      solve  (I+Mt) Z = dU                        dB = Z                dM = -(Z Ut) . strict
 *      Bm     db_t += <Z_t, v_t - gamma_t S0K_t>   dV = Z . b            dS0K = -Z . b . gamma
 *             dL_t += -gamma_t b_t <Z_t, S0K_t>
 *      S0K    dK += dS0K S                         dS += dS0Kt K
 *      M      db_t += <dM_t, ratio_t . KK_t>       dratio += dM . b . KK  dKK = dM . b . ratio
 *      KK     dK_t += dKK[t,s] K_s   AND   dK_s += dKK[t,s] K_t          (both: KK is a Gram matrix)
 *      ratio  dL_t += dratio[t,s] ratio[t,s],  dL_s -= same              (s <= t)
 *      L      dlogg_t = sum_{r>=t} dL_r                                   reverse cumsum
 *
 * The adjoint of the solve is the one step that is not a transcription of a forward line:
 * U = A^-1 B with A = I+M gives dB = A^-t dU (a transposed, i.e. upper, triangular solve) and
 * dA = -Z Ut. Forming A^-1 is never needed in either direction.
 *
 * Memory: the forward keeps ONE state per chunk boundary; the backward replays a chunk from its
 * checkpoint and walks it back, so only one chunk of intermediates is ever live. That is the same
 * trade the MLX custom VJP makes, and the reason a long sequence fits at all.
 *
 * fp64 throughout and CPU-only: this is the reference the fp16/NPU implementation gets checked
 * against, so it must not itself be the thing in question.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* The arithmetic precision is a compile-time choice so the SAME implementation can be run as the
 * fp64 algebra oracle and as the fp32 thing a real trainer would run. Two sources would drift. */
#ifndef GDN_REAL
#define GDN_REAL double
#endif
typedef GDN_REAL real;
#define RSZ (sizeof(real))

static int T,DK,DV;
static real *q,*k,*v,*g,*bet,*S0;          /* inputs:  q,k [T,DK]  v [T,DV]  g,bet [T]  S0 [DV,DK] */
static real *dY,*dSo;                      /* cotangents: dY [T,DV]  dSo [DV,DK] */
static real *Y,*Sf;                        /* outputs:    Y  [T,DV]  Sf [DV,DK]  */
static real *dq,*dk,*dv,*dlg,*dbe,*dSi;    /* grads; dlg is d/dlog g, dg = dlg/g */

#define QI(t,j) q[(size_t)(t)*DK+(j)]
#define KI(t,j) k[(size_t)(t)*DK+(j)]
#define VI(t,i) v[(size_t)(t)*DV+(i)]
#define SI(p,i,j) p[(size_t)(i)*DK+(j)]

/* 1 = drop the gate coupling through M; 2 = drop the second dK term from the Gram matrix.
 * Both are plausible transcription slips, and both must make the checks below FAIL. */
static int BRK=0;

typedef struct {
    real *L,*gam,*tail,*dL;              /* C   */
    real *rat,*KK,*QK,*M,*P,*drat,*dM;   /* C*C */
    real *S0K,*Bm,*U,*QS,*dU,*Z,*dS0K;   /* C*DV*/
} scr;

static scr *scr_new(int C){
    scr*w=calloc(1,sizeof *w); size_t c=C, cc=(size_t)C*C, cv=(size_t)C*DV;
    real**p1[]={&w->L,&w->gam,&w->tail,&w->dL};
    real**p2[]={&w->rat,&w->KK,&w->QK,&w->M,&w->P,&w->drat,&w->dM};
    real**p3[]={&w->S0K,&w->Bm,&w->U,&w->QS,&w->dU,&w->Z,&w->dS0K};
    for(unsigned i=0;i<sizeof p1/sizeof*p1;i++) *p1[i]=calloc(c,RSZ);
    for(unsigned i=0;i<sizeof p2/sizeof*p2;i++) *p2[i]=calloc(cc,RSZ);
    for(unsigned i=0;i<sizeof p3/sizeof*p3;i++) *p3[i]=calloc(cv,RSZ);
    return w;
}

/* One chunk forward: entry state Sent -> Y[t0..t0+C) and Sout. Fills w for the backward. */
static void chunk_fwd(scr*w,int t0,int C,const real*Sent,real*Sout){
    real a=0;
    for(int t=0;t<C;t++){ a+=(real)log((double)g[t0+t]); w->L[t]=a; w->gam[t]=exp(a); }
    for(int t=0;t<C;t++) for(int s=0;s<C;s++){
        w->rat[t*C+s] = s<=t ? exp(w->L[t]-w->L[s]) : 0.0;
        real qk=0,kk=0;
        for(int j=0;j<DK;j++){ qk+=QI(t0+t,j)*KI(t0+s,j); kk+=KI(t0+t,j)*KI(t0+s,j); }
        w->QK[t*C+s]=qk; w->KK[t*C+s]=kk;
        w->M[t*C+s] = s<t  ? bet[t0+t]*w->rat[t*C+s]*kk : 0.0;
        w->P[t*C+s] = s<=t ? w->rat[t*C+s]*qk : 0.0;
    }
    for(int t=0;t<C;t++) for(int i=0;i<DV;i++){
        real sk=0; for(int j=0;j<DK;j++) sk+=KI(t0+t,j)*SI(Sent,i,j);
        w->S0K[t*DV+i]=sk;
        w->Bm[t*DV+i]=bet[t0+t]*(VI(t0+t,i)-w->gam[t]*sk);
    }
    for(int t=0;t<C;t++){                                   /* (I+M) U = Bm, forward substitution */
        for(int i=0;i<DV;i++) w->U[t*DV+i]=w->Bm[t*DV+i];
        for(int s=0;s<t;s++){ real m=w->M[t*C+s]; if(m==0) continue;
            for(int i=0;i<DV;i++) w->U[t*DV+i]-=m*w->U[s*DV+i]; }
    }
    for(int t=0;t<C;t++) for(int i=0;i<DV;i++){
        real qs=0; for(int j=0;j<DK;j++) qs+=QI(t0+t,j)*SI(Sent,i,j);
        w->QS[t*DV+i]=qs;
        real y=w->gam[t]*qs;
        for(int s=0;s<=t;s++) y+=w->P[t*C+s]*w->U[s*DV+i];
        Y[(size_t)(t0+t)*DV+i]=y;
    }
    for(int s=0;s<C;s++) w->tail[s]=exp(w->L[C-1]-w->L[s]);
    for(int i=0;i<DV;i++) for(int j=0;j<DK;j++){
        real o=w->gam[C-1]*SI(Sent,i,j);
        for(int s=0;s<C;s++) o+=w->U[s*DV+i]*w->tail[s]*KI(t0+s,j);
        SI(Sout,i,j)=o;
    }
}

/* One chunk backward: dSout in, dSin out; accumulates into the global dq/dk/dv/dlg/dbe slices. */
static void chunk_bwd(scr*w,int t0,int C,const real*Sent,const real*dSout,real*dSin){
    memset(w->dL,0,(size_t)C*RSZ);      memset(w->dU,0,(size_t)C*DV*RSZ);
    memset(w->drat,0,(size_t)C*C*RSZ);  memset(w->dM,0,(size_t)C*C*RSZ);
    memset(dSin,0,(size_t)DV*DK*RSZ);

    real gC=w->gam[C-1], dgC=0;                                            /* S_out */
    for(int i=0;i<DV;i++) for(int j=0;j<DK;j++){
        SI(dSin,i,j)+=gC*SI(dSout,i,j);
        dgC+=SI(dSout,i,j)*SI(Sent,i,j);
    }
    w->dL[C-1]+=gC*dgC;
    for(int s=0;s<C;s++){
        real dt=0;
        for(int i=0;i<DV;i++){
            real dw=0; for(int j=0;j<DK;j++) dw+=SI(dSout,i,j)*KI(t0+s,j);
            w->dU[s*DV+i]+=dw*w->tail[s];
            dt+=dw*w->U[s*DV+i];
        }
        for(int j=0;j<DK;j++){ real acc=0;
            for(int i=0;i<DV;i++) acc+=SI(dSout,i,j)*w->U[s*DV+i];
            dk[(size_t)(t0+s)*DK+j]+=acc*w->tail[s]; }
        w->dL[C-1]+=dt*w->tail[s];
        w->dL[s]  -=dt*w->tail[s];
    }

    for(int t=0;t<C;t++){                                                    /* Y, then P */
        real dgam=0;
        for(int i=0;i<DV;i++) dgam+=dY[(size_t)(t0+t)*DV+i]*w->QS[t*DV+i];
        w->dL[t]+=w->gam[t]*dgam;
        for(int i=0;i<DV;i++){
            real dqs=w->gam[t]*dY[(size_t)(t0+t)*DV+i];
            for(int j=0;j<DK;j++){
                dq[(size_t)(t0+t)*DK+j]+=dqs*SI(Sent,i,j);
                SI(dSin,i,j)           +=dqs*QI(t0+t,j);
            }
        }
        for(int s=0;s<=t;s++){
            real dP=0;
            for(int i=0;i<DV;i++) dP+=dY[(size_t)(t0+t)*DV+i]*w->U[s*DV+i];
            real dQK=dP*w->rat[t*C+s];
            w->drat[t*C+s]+=dP*w->QK[t*C+s];
            for(int j=0;j<DK;j++){
                dq[(size_t)(t0+t)*DK+j]+=dQK*KI(t0+s,j);
                dk[(size_t)(t0+s)*DK+j]+=dQK*QI(t0+t,j);
            }
            real p=w->P[t*C+s];
            for(int i=0;i<DV;i++) w->dU[s*DV+i]+=p*dY[(size_t)(t0+t)*DV+i];
        }
    }

    for(int t=C-1;t>=0;t--){                                                 /* (I+Mt) Z = dU */
        for(int i=0;i<DV;i++) w->Z[t*DV+i]=w->dU[t*DV+i];
        for(int s=t+1;s<C;s++){ real m=w->M[s*C+t]; if(m==0) continue;
            for(int i=0;i<DV;i++) w->Z[t*DV+i]-=m*w->Z[s*DV+i]; }
    }
    for(int t=0;t<C;t++) for(int s=0;s<t;s++){
        real acc=0; for(int i=0;i<DV;i++) acc+=w->Z[t*DV+i]*w->U[s*DV+i];
        w->dM[t*C+s]=-acc;
    }

    for(int t=0;t<C;t++){                                                    /* Bm */
        real b=bet[t0+t], dgam=0;
        for(int i=0;i<DV;i++){
            real z=w->Z[t*DV+i];
            dbe[t0+t]               += z*(VI(t0+t,i)-w->gam[t]*w->S0K[t*DV+i]);
            dv[(size_t)(t0+t)*DV+i]  = z*b;
            w->dS0K[t*DV+i]          = -z*b*w->gam[t];
            dgam                    += -b*z*w->S0K[t*DV+i];
        }
        w->dL[t]+=w->gam[t]*dgam;
    }
    for(int t=0;t<C;t++) for(int i=0;i<DV;i++){                              /* S0K */
        real d=w->dS0K[t*DV+i];
        for(int j=0;j<DK;j++){
            dk[(size_t)(t0+t)*DK+j]+=d*SI(Sent,i,j);
            SI(dSin,i,j)           +=d*KI(t0+t,j);
        }
    }
    for(int t=0;t<C;t++) for(int s=0;s<t;s++){                               /* M, then KK */
        real dm=w->dM[t*C+s]; if(dm==0) continue;
        dbe[t0+t]+=dm*w->rat[t*C+s]*w->KK[t*C+s];
        if(BRK!=1) w->drat[t*C+s]+=dm*bet[t0+t]*w->KK[t*C+s];
        real dkk=dm*bet[t0+t]*w->rat[t*C+s];
        for(int j=0;j<DK;j++){
            dk[(size_t)(t0+t)*DK+j]+=dkk*KI(t0+s,j);
            if(BRK!=2) dk[(size_t)(t0+s)*DK+j]+=dkk*KI(t0+t,j);
        }
    }
    for(int t=0;t<C;t++) for(int s=0;s<=t;s++){                              /* ratio */
        real r=w->drat[t*C+s]*w->rat[t*C+s];
        w->dL[t]+=r; w->dL[s]-=r;
    }
    real acc=0;                                                            /* L = cumsum(log g) */
    for(int t=C-1;t>=0;t--){ acc+=w->dL[t]; dlg[t0+t]=acc; }
}

/* Whole sequence, chunk length CH. */
static void fwd_chunked(int CH){
    scr*w=scr_new(CH);
    real *S=malloc((size_t)DV*DK*RSZ), *Sn=malloc((size_t)DV*DK*RSZ);
    memcpy(S,S0,(size_t)DV*DK*RSZ);
    for(int t0=0;t0<T;t0+=CH){
        int C=T-t0<CH?T-t0:CH;
        chunk_fwd(w,t0,C,S,Sn);
        memcpy(S,Sn,(size_t)DV*DK*RSZ);
    }
    memcpy(Sf,S,(size_t)DV*DK*RSZ);
    free(S); free(Sn);
}

static void bwd_chunked(int CH){
    int nc=(T+CH-1)/CH;
    scr*w=scr_new(CH);
    real *ent=malloc((size_t)nc*DV*DK*RSZ);                 /* one checkpoint per chunk boundary */
    real *S=malloc((size_t)DV*DK*RSZ), *Sn=malloc((size_t)DV*DK*RSZ);
    memcpy(S,S0,(size_t)DV*DK*RSZ);
    for(int c=0;c<nc;c++){                                  /* forward, keeping only the entries */
        memcpy(ent+(size_t)c*DV*DK,S,(size_t)DV*DK*RSZ);
        int t0=c*CH, C=T-t0<CH?T-t0:CH;
        chunk_fwd(w,t0,C,S,Sn);
        memcpy(S,Sn,(size_t)DV*DK*RSZ);
    }
    memset(dq,0,(size_t)T*DK*RSZ); memset(dk,0,(size_t)T*DK*RSZ);
    memset(dv,0,(size_t)T*DV*RSZ); memset(dbe,0,(size_t)T*RSZ); memset(dlg,0,(size_t)T*RSZ);
    real *dS=malloc((size_t)DV*DK*RSZ), *dSn=malloc((size_t)DV*DK*RSZ);
    memcpy(dS,dSo,(size_t)DV*DK*RSZ);
    for(int c=nc-1;c>=0;c--){                               /* replay each chunk, then walk it back */
        int t0=c*CH, C=T-t0<CH?T-t0:CH;
        const real*Sent=ent+(size_t)c*DV*DK;
        chunk_fwd(w,t0,C,Sent,Sn);
        chunk_bwd(w,t0,C,Sent,dS,dSn);
        memcpy(dS,dSn,(size_t)DV*DK*RSZ);
    }
    memcpy(dSi,dS,(size_t)DV*DK*RSZ);
    free(ent); free(S); free(Sn); free(dS); free(dSn);
}

/* ---- ORACLE 1: the sequential recurrence, forward and backward, with the same cotangents. ---- */
static real *oY,*oSf,*odq,*odk,*odv,*odlg,*odbe,*odSi;

static void seq_fwd_bwd(void){
    size_t sz=(size_t)DV*DK;
    real *St=malloc((size_t)T*sz*RSZ), *Sd=malloc((size_t)T*sz*RSZ), *e=malloc((size_t)DV*RSZ);
    real *kv=malloc((size_t)DV*RSZ);
    real *S=malloc(sz*RSZ); memcpy(S,S0,sz*RSZ);
    for(int t=0;t<T;t++){
        real *sd=Sd+(size_t)t*sz, *st=St+(size_t)t*sz;
        for(size_t i=0;i<sz;i++) sd[i]=g[t]*S[i];
        for(int i=0;i<DV;i++){ real a=0;
            for(int j=0;j<DK;j++) a+=sd[(size_t)i*DK+j]*KI(t,j);
            e[i]=(VI(t,i)-a)*bet[t]; }
        for(int i=0;i<DV;i++) for(int j=0;j<DK;j++)
            st[(size_t)i*DK+j]=sd[(size_t)i*DK+j]+e[i]*KI(t,j);
        for(int i=0;i<DV;i++){ real a=0;
            for(int j=0;j<DK;j++) a+=st[(size_t)i*DK+j]*QI(t,j);
            oY[(size_t)t*DV+i]=a; }
        memcpy(S,st,sz*RSZ);
    }
    memcpy(oSf,S,sz*RSZ);

    real *A=malloc(sz*RSZ), *dSd=malloc(sz*RSZ), *dd=malloc((size_t)DV*RSZ);
    memcpy(A,dSo,sz*RSZ);
    for(int t=T-1;t>=0;t--){
        real *sd=Sd+(size_t)t*sz, *st=St+(size_t)t*sz;
        const real *Sprev = t? St+(size_t)(t-1)*sz : S0;
        for(int i=0;i<DV;i++){ real a=0;
            for(int j=0;j<DK;j++) a+=sd[(size_t)i*DK+j]*KI(t,j);
            kv[i]=a; e[i]=(VI(t,i)-a)*bet[t]; }
        for(int i=0;i<DV;i++) for(int j=0;j<DK;j++)              /* A += dy (x) q */
            A[(size_t)i*DK+j]+=dY[(size_t)t*DV+i]*QI(t,j);
        for(int j=0;j<DK;j++){ real a=0;
            for(int i=0;i<DV;i++) a+=st[(size_t)i*DK+j]*dY[(size_t)t*DV+i];
            odq[(size_t)t*DK+j]=a; }
        for(int i=0;i<DV;i++){ real a=0;
            for(int j=0;j<DK;j++) a+=A[(size_t)i*DK+j]*KI(t,j);
            dd[i]=a; }
        for(int j=0;j<DK;j++){ real a=0;
            for(int i=0;i<DV;i++) a+=A[(size_t)i*DK+j]*e[i];
            odk[(size_t)t*DK+j]=a; }
        double dbet=0;
        for(int i=0;i<DV;i++){
            odv[(size_t)t*DV+i]=dd[i]*bet[t];
            dbet+=dd[i]*(VI(t,i)-kv[i]);
        }
        odbe[t]=dbet;
        for(int j=0;j<DK;j++){ real a=0;
            for(int i=0;i<DV;i++) a+=sd[(size_t)i*DK+j]*(-dd[i]*bet[t]);
            odk[(size_t)t*DK+j]+=a; }
        for(int i=0;i<DV;i++) for(int j=0;j<DK;j++)
            dSd[(size_t)i*DK+j]=A[(size_t)i*DK+j]-dd[i]*bet[t]*KI(t,j);
        double dg=0;
        for(size_t i=0;i<sz;i++) dg+=dSd[i]*Sprev[i];
        odlg[t]=dg*g[t];                                         /* d/dlog g = g d/dg */
        for(size_t i=0;i<sz;i++) A[i]=dSd[i]*g[t];
    }
    memcpy(odSi,A,sz*RSZ);
    free(St);free(Sd);free(e);free(kv);free(S);free(A);free(dSd);free(dd);
}

/* ---- ORACLE 2: central finite differences of  Lsc = <dY,Y> + <dSo,S_final>. ---- */
static int CHFD;
static double lsc(void){
    fwd_chunked(CHFD);
    double L=0;
    for(size_t i=0;i<(size_t)T*DV;i++) L+=dY[i]*Y[i];
    for(size_t i=0;i<(size_t)DV*DK;i++) L+=dSo[i]*Sf[i];
    return L;
}

static unsigned s_=314159;
static double rnd(void){ s_=s_*1103515245u+12345u; return (double)((int)((s_>>13)&0x7fff)-16384)/16384.0; }

static int fails=0;
static int cmp(const char*nm,const real*a,const real*b,size_t n,double tol,int quiet){
    double worst=0,mx=0;
    for(size_t i=0;i<n;i++){ double d=fabs(a[i]-b[i]); if(d>worst) worst=d;
                             if(fabs(b[i])>mx) mx=fabs(b[i]); }
    double rel=mx>0?worst/mx:worst;
    if(!quiet) printf("  %-22s  max|d|/max|ref| %.3e   %s\n", nm, rel, rel<=tol?"PASS":"FAIL");
    return rel>tol;
}

/* The finite-difference FLOOR has to be computed, not guessed. (Lp-Lm)/(2 eps) loses
 * |L| * eps_machine / eps to cancellation, and at T=128 d=128 the loss is a sum of ~33k terms, so
 * that floor is ~1e-06 -- thousands of times the 1e-9 an earlier version of this function assumed.
 * With the wrong floor this reported dk as FAIL at 5.4e-05 while dk agreed with the sequential
 * backward to 8.5e-15. Same mistake as in bwd_ops_fp64_check.c; same fix. */
static int fdcheck(const char*nm,real*p,const real*grd,int n,double eps){
    int bad=0; double worst=0, wabs=0;
    double L0=lsc();
    double atol=16.0*fabs(L0)*2.220446049250313e-16/eps; if(atol<1e-12) atol=1e-12;
    for(int s=0;s<8;s++){
        int i=(s*n)/8+s; if(i>=n) i=n-1;
        real sv=p[i];
        p[i]=sv+eps; double Lp=lsc();
        p[i]=sv-eps; double Lm=lsc();
        p[i]=sv;
        double fd=(Lp-Lm)/(2*eps), an=grd[i], ad=fabs(an-fd);
        double rel=fabs(fd)>0?ad/fabs(fd):(fabs(an)>0?1e9:0);
        if(!(ad<=atol || rel<=2e-6)) bad++;
        if(rel>worst){ worst=rel; wabs=ad; }
    }
    printf("  %-22s  worst rel %.3e  |d| %.2e  (floor %.1e)  %s\n",
           nm, worst, wabs, atol, bad?"FAIL":(worst>2e-6?"PASS (within fd floor)":"PASS"));
    return bad!=0;
}

#ifndef GDN_CHUNK_NO_MAIN
int main(int argc,char**argv){
    T  = argc>1?atoi(argv[1]):24;
    DK = argc>2?atoi(argv[2]):8;
    DV = argc>3?atoi(argv[3]):8;
    int CH = argc>4?atoi(argv[4]):8;
    /* FD step. 1e-4, not the reflexive 1e-6: Lsc sums ~33k terms at T=128 d=128, so the
     * cancellation floor |L| eps_machine / eps is ~1.3e-06 at 1e-6 and the check flags noise.
     * Sweeping eps is also how to tell the two apart -- measured on dk's worst component,
     *   eps    1e-7     1e-6     1e-5     1e-4     1e-3
     *   |d|    2.7e-06  3.2e-06  4.2e-08  1.4e-08  7.0e-10
     * which falls with eps, so it is cancellation. A wrong gradient gives a FLAT row. */
    double FDEPS = argc>5?atof(argv[5]):1e-4;
    size_t tk=(size_t)T*DK, tv=(size_t)T*DV, sz=(size_t)DV*DK;

    q=malloc(tk*RSZ); k=malloc(tk*RSZ); v=malloc(tv*RSZ); g=malloc((size_t)T*RSZ); bet=malloc((size_t)T*RSZ);
    S0=malloc(sz*RSZ); dY=malloc(tv*RSZ); dSo=malloc(sz*RSZ); Y=malloc(tv*RSZ); Sf=malloc(sz*RSZ);
    dq=malloc(tk*RSZ); dk=malloc(tk*RSZ); dv=malloc(tv*RSZ); dlg=malloc((size_t)T*RSZ);
    dbe=malloc((size_t)T*RSZ); dSi=malloc(sz*RSZ);
    oY=malloc(tv*RSZ); oSf=malloc(sz*RSZ); odq=malloc(tk*RSZ); odk=malloc(tk*RSZ); odv=malloc(tv*RSZ);
    odlg=malloc((size_t)T*RSZ); odbe=malloc((size_t)T*RSZ); odSi=malloc(sz*RSZ);
    if(!odSi){ printf("alloc failed\n"); return 1; }

    for(int t=0;t<T;t++){
        for(int j=0;j<DK;j++){ QI(t,j)=rnd(); KI(t,j)=rnd(); }
        for(int i=0;i<DV;i++){ VI(t,i)=rnd(); dY[(size_t)t*DV+i]=rnd(); }
        g[t]=0.55+0.4*rnd();                 /* decay in (0.15, 0.95) */
        bet[t]=0.5+0.4*rnd();
    }
    for(size_t i=0;i<sz;i++){ S0[i]=0.3*rnd(); dSo[i]=rnd(); }

    printf("Chunkwise-parallel GDN backward   T=%d dk=%d dv=%d chunk=%d   fp64, CPU only\n\n",
           T,DK,DV,CH);

    /* 1. the chunkwise FORWARD must be the sequential recurrence, to machine precision. */
    seq_fwd_bwd();
    fwd_chunked(CH);
    printf("forward, chunkwise vs sequential recurrence:\n");
    fails += cmp("Y",       Y,  oY,  tv, 1e-11, 0);
    fails += cmp("S_final", Sf, oSf, sz, 1e-11, 0);

    /* 2. the chunkwise BACKWARD must be the sequential backward. Algebraically the same function,
     *    so at fp64 this is an identity check, not a tolerance check. */
    bwd_chunked(CH);
    printf("\nbackward, chunkwise vs sequential recurrence (same cotangents):\n");
    fails += cmp("dq",     dq,  odq,  tk, 1e-10, 0);
    fails += cmp("dk",     dk,  odk,  tk, 1e-10, 0);
    fails += cmp("dv",     dv,  odv,  tv, 1e-10, 0);
    fails += cmp("dlog g", dlg, odlg, T,  1e-10, 0);
    fails += cmp("dbeta",  dbe, odbe, T,  1e-10, 0);
    fails += cmp("dS_in",  dSi, odSi, sz, 1e-10, 0);

    /* 3. and it must match finite differences of its OWN forward — catching any error the two
     *    hand derivations could share. */
    CHFD=CH;
    printf("\nbackward vs central finite differences of the chunkwise forward:\n");
    bwd_chunked(CH);
    real *sdq=malloc(tk*RSZ),*sdk=malloc(tk*RSZ),*sdv=malloc(tv*RSZ),
           *sdb=malloc((size_t)T*RSZ),*sds=malloc(sz*RSZ),*sdg=malloc((size_t)T*RSZ);
    memcpy(sdq,dq,tk*RSZ); memcpy(sdk,dk,tk*RSZ); memcpy(sdv,dv,tv*RSZ);
    memcpy(sdb,dbe,(size_t)T*RSZ); memcpy(sds,dSi,sz*RSZ);
    for(int t=0;t<T;t++) sdg[t]=dlg[t]/g[t];      /* dL/dg from dL/dlog g */
    fails += fdcheck("dq",    q,   sdq, (int)tk, FDEPS);
    fails += fdcheck("dk",    k,   sdk, (int)tk, FDEPS);
    fails += fdcheck("dv",    v,   sdv, (int)tv, FDEPS);
    fails += fdcheck("dg",    g,   sdg, T,       FDEPS);
    fails += fdcheck("dbeta", bet, sdb, T,       FDEPS);
    fails += fdcheck("dS_in", S0,  sds, (int)sz, FDEPS);

    /* 4. chunk length is an implementation choice: every length must give the same gradients,
     *    and chunk=1 must degenerate to the sequential form. */
    printf("\nchunk-length invariance (every length vs the sequential backward):\n");
    int lens[]={1,2,3,5,7,8,16,32,T}, nbad=0;
    for(unsigned i=0;i<sizeof lens/sizeof*lens;i++){
        int c=lens[i]; if(c<1||c>T) continue;
        bwd_chunked(c);
        int b = cmp("",dq,odq,tk,1e-10,1) + cmp("",dk,odk,tk,1e-10,1)
              + cmp("",dv,odv,tv,1e-10,1) + cmp("",dlg,odlg,T,1e-10,1)
              + cmp("",dbe,odbe,T,1e-10,1) + cmp("",dSi,odSi,sz,1e-10,1);
        printf("  chunk=%-3d  %s\n", c, b?"FAIL":"ok");
        nbad+=b!=0;
    }
    fails+=nbad;

    /* 5. positive controls: with one adjoint term removed the checks must FAIL. Without this the
     *    PASSes above could mean the tolerances are loose rather than the derivation right. */
    printf("\npositive controls (each MUST fail):\n");
    const char*names[]={"gate coupling via M dropped","Gram-matrix dK term dropped"};
    int caught=0;
    for(int b=1;b<=2;b++){
        BRK=b; bwd_chunked(CH); BRK=0;
        int bad = cmp("",dk,odk,tk,1e-10,1) + cmp("",dlg,odlg,T,1e-10,1);
        printf("  %-30s  %s\n", names[b-1], bad?"FAIL (as required)":"PASSED -- check is vacuous");
        caught += bad!=0;
    }
    if(caught<2){ printf("  a control did not fail; the checks prove nothing.\n"); fails++; }

    printf("\n%s\n", fails==0
      ? "ALL PASS -- the chunkwise backward is the sequential one, at every chunk length."
      : "SOME FAILED -- see above.");
    return fails!=0;
}
#endif /* GDN_CHUNK_NO_MAIN */
