/* tools/ssd_coherence.c — the DECISIVE numerics gate for the fused on-NPU SSD scan.
 *
 * The NPU's int8 matmul is EXACT (int8·int8->int32, validated bit-exact), so the numerical coherence of
 * the fused scan is fully determined by the int8 QUANTIZATION of the matmul operands — computable exactly
 * on CPU here, no board needed. This decides the int8-vs-fp16 gate that the fused bench (2.66x, all-ones)
 * couldn't: does the fast group-batched scan stay CORRECT with int8 matmuls on REAL data?
 *
 * Two Y_diag formulations (the risky stage — its decay L is per-head):
 *   (a) NON-FACTORED per-head:  M[l,s]=G[l,s]*L[l,s] (L in (0,1], BOUNDED), Y_diag = M·xbar   (slow: H matmuls)
 *   (b) FACTORED group-batched: Xtil[s,·]=exp(-Acs[s])·xbar[s,·], Yd=maskedG·Xtil, *exp(Acs[l])  (fast: G matmuls)
 * (b) is what makes the NPU win (2.66x) — but Xtil grows like exp(-Acs[s]) ALONG THE CONTRACTION dim s,
 * which int8 per-channel scaling (per-row-A/per-col-B) CANNOT absorb (scales must factor out of the sum).
 * This tool measures whether that kills accuracy.
 *
 *   cc -O2 -o ssd_coherence tools/ssd_coherence.c -lm && ./ssd_coherence      (CPU only, no NPU)
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

typedef struct { int H,P,Nst,G,CS,NC; } dims;
#define IXHP(t,h,p) (((size_t)(t)*d->H+(h))*d->P+(p))
#define IXDT(t,h)   ((size_t)(t)*d->H+(h))
#define IXBC(t,g,n) (((size_t)(t)*d->G+(g))*d->Nst+(n))

static double frand(void){return (double)rand()/RAND_MAX;}
static double frs(void){return frand()*2-1;}

/* ground truth: definitional sequential recurrence */
static void seq(const dims*d,const double*x,const double*dt,const double*A,const double*B,const double*C,const double*D,double*y){
    int L=d->NC*d->CS; double*st=calloc((size_t)d->P*d->Nst,sizeof(double));
    for(int h=0;h<d->H;h++){int g=h%d->G; memset(st,0,(size_t)d->P*d->Nst*sizeof(double));
        for(int t=0;t<L;t++){double dtv=dt[IXDT(t,h)],dA=exp(dtv*A[h]);
            for(int p=0;p<d->P;p++){double xv=x[IXHP(t,h,p)],dtx=dtv*xv;
                for(int n=0;n<d->Nst;n++) st[(size_t)p*d->Nst+n]=st[(size_t)p*d->Nst+n]*dA+dtx*B[IXBC(t,g,n)];
                double acc=0; for(int n=0;n<d->Nst;n++) acc+=C[IXBC(t,g,n)]*st[(size_t)p*d->Nst+n];
                y[IXHP(t,h,p)]=acc+D[h]*xv;}}}
    free(st);
}

/* round a double to the nearest IEEE-754 half (fp16) value and back — mirrors NPU fp16 storage. */
static double hf(double x){ return (double)(_Float16)x; }

/* exp via a PWL LUT — mirrors the NPU SDP output-stage exp (ork_f16_mm_build_lut, ~1024 fp16 segments over
 * a band). All scan exp args are <=0 (decay), so band [-30,0]; nodes + interp result rounded to fp16. */
static double pwl_exp(double x){
    const double lo=-30.0, hi=0.0; const int NS=1024;
    if(x>=hi) return 1.0; if(x<=lo) return 0.0;
    double t=(x-lo)/(hi-lo)*NS; int i=(int)t; double f=t-i;
    double y0=(double)(_Float16)exp(lo+(hi-lo)*i/NS), y1=(double)(_Float16)exp(lo+(hi-lo)*(i+1)/NS);
    return (double)(_Float16)(y0+f*(y1-y0));
}

/* qmode: 0=fp32, 1=int8 per-tensor, 2=int8 per-channel, 3=fp16 (A,B rounded to fp16, fp32 accumulate —
 * exactly the RK3588 fp16 matmul: fp16 operands, fp32 acc, fp32 out). out[M,N] = A[M,K]·B[K,N]. */
static void mmq(double*out,const double*A,const double*B,int M,int K,int N,int qmode){
    if(qmode==0){ for(int m=0;m<M;m++)for(int n=0;n<N;n++){double a=0;for(int k=0;k<K;k++)a+=A[(size_t)m*K+k]*B[(size_t)k*N+n];out[(size_t)m*N+n]=a;} return; }
    if(qmode==3){ for(int m=0;m<M;m++)for(int n=0;n<N;n++){double a=0;for(int k=0;k<K;k++)a+=hf(A[(size_t)m*K+k])*hf(B[(size_t)k*N+n]);out[(size_t)m*N+n]=a;} return; }
    signed char*qA=malloc((size_t)M*K),*qB=malloc((size_t)K*N);
    double*sA=malloc((size_t)M*sizeof(double)),*sB=malloc((size_t)N*sizeof(double));
    if(qmode==1){ double ma=0,mb=0; for(size_t i=0;i<(size_t)M*K;i++)if(fabs(A[i])>ma)ma=fabs(A[i]); for(size_t i=0;i<(size_t)K*N;i++)if(fabs(B[i])>mb)mb=fabs(B[i]);
        double s1=ma>0?ma/127:1, s2=mb>0?mb/127:1; for(int m=0;m<M;m++)sA[m]=s1; for(int n=0;n<N;n++)sB[n]=s2;
        for(size_t i=0;i<(size_t)M*K;i++){int v=(int)lround(A[i]/s1); qA[i]=v>127?127:v<-127?-127:v;}
        for(size_t i=0;i<(size_t)K*N;i++){int v=(int)lround(B[i]/s2); qB[i]=v>127?127:v<-127?-127:v;}
    } else { /* per-channel */
        for(int m=0;m<M;m++){double mx=0;for(int k=0;k<K;k++)if(fabs(A[(size_t)m*K+k])>mx)mx=fabs(A[(size_t)m*K+k]); sA[m]=mx>0?mx/127:1;
            for(int k=0;k<K;k++){int v=(int)lround(A[(size_t)m*K+k]/sA[m]); qA[(size_t)m*K+k]=v>127?127:v<-127?-127:v;}}
        for(int n=0;n<N;n++){double mx=0;for(int k=0;k<K;k++)if(fabs(B[(size_t)k*N+n])>mx)mx=fabs(B[(size_t)k*N+n]); sB[n]=mx>0?mx/127:1;
            for(int k=0;k<K;k++){int v=(int)lround(B[(size_t)k*N+n]/sB[n]); qB[(size_t)k*N+n]=v>127?127:v<-127?-127:v;}}
    }
    for(int m=0;m<M;m++)for(int n=0;n<N;n++){long a=0;for(int k=0;k<K;k++)a+=(long)qA[(size_t)m*K+k]*qB[(size_t)k*N+n];out[(size_t)m*N+n]=a*sA[m]*sB[n];}
    free(qA);free(qB);free(sA);free(sB);
}

/* group-batched chunked SSD scan. factored=0: Y_diag per-head non-factored (bounded M). factored=1: L-factored
 * group matmul (fast, but Xtil grows along contraction). qmode passed to all matmuls. elementwise in fp32. */
static void scan_gb(const dims*d,const double*x,const double*dt,const double*A,const double*B,const double*C,const double*D,
                    double*y,int qmode,int factored,int ewf,int lutexp){
#define H(v) (ewf?hf(v):(v))
#define EXPF(v) (lutexp?pwl_exp(v):exp(v))
    const int CS=d->CS,NC=d->NC,P=d->P,Nst=d->Nst;
    double*Abar=malloc(CS*sizeof(double)),*Acs=malloc(CS*sizeof(double)),*xbar=malloc((size_t)CS*P*sizeof(double));
    double*state_in=calloc((size_t)P*Nst,sizeof(double)),*cstate=malloc((size_t)P*Nst*sizeof(double)),*acl=malloc(NC*sizeof(double));
    double*Cc=malloc((size_t)CS*Nst*sizeof(double)),*Bc=malloc((size_t)CS*Nst*sizeof(double));
    double*G=malloc((size_t)CS*CS*sizeof(double)),*Xtil=malloc((size_t)CS*P*sizeof(double)),*Yd=malloc((size_t)CS*P*sizeof(double));
    double*Bt=malloc((size_t)Nst*CS*sizeof(double)),*tmp=malloc((size_t)CS*P*sizeof(double));
    for(int h=0;h<d->H;h++){int g=h%d->G; memset(state_in,0,(size_t)P*Nst*sizeof(double));
        for(int c=0;c<NC;c++){int base=c*CS; double run=0;
            for(int l=0;l<CS;l++){double dtv=dt[IXDT(base+l,h)]; Abar[l]=H(dtv*A[h]); run+=Abar[l]; Acs[l]=H(run);
                for(int p=0;p<P;p++) xbar[(size_t)l*P+p]=H(dtv*x[IXHP(base+l,h,p)]);
                for(int n=0;n<Nst;n++){Cc[(size_t)l*Nst+n]=H(C[IXBC(base+l,g,n)]); Bc[(size_t)l*Nst+n]=H(B[IXBC(base+l,g,n)]);}}
            acl[c]=Acs[CS-1];
            if(c>0){double dp=H(EXPF(acl[c-1])); for(size_t i=0;i<(size_t)P*Nst;i++) state_in[i]=H(dp*state_in[i]+cstate[i]);}
            /* scores G = C·Bᵀ  ([CS,Nst]·[Nst,CS]) */
            for(int s=0;s<CS;s++)for(int n=0;n<Nst;n++) Bt[(size_t)n*CS+s]=Bc[(size_t)s*Nst+n];
            mmq(G,Cc,Bt,CS,Nst,CS,qmode);
            /* Y_diag */
            if(factored){
                for(int s=0;s<CS;s++){double es=H(EXPF(-Acs[s])); for(int p=0;p<P;p++) Xtil[(size_t)s*P+p]=H(es*xbar[(size_t)s*P+p]);}
                double*Gm=malloc((size_t)CS*CS*sizeof(double));
                for(int l=0;l<CS;l++)for(int s=0;s<CS;s++) Gm[(size_t)l*CS+s]=(s<=l)?H(G[(size_t)l*CS+s]):0;   /* causal mask */
                mmq(Yd,Gm,Xtil,CS,CS,P,qmode);                                   /* [CS,CS]·[CS,P] */
                for(int l=0;l<CS;l++){double el=H(EXPF(Acs[l])); for(int p=0;p<P;p++) Yd[(size_t)l*P+p]=H(Yd[(size_t)l*P+p]*el);}
                free(Gm);
            } else {
                double*M=malloc((size_t)CS*CS*sizeof(double));
                for(int l=0;l<CS;l++)for(int s=0;s<CS;s++) M[(size_t)l*CS+s]=(s<=l)?H(G[(size_t)l*CS+s]*EXPF(Acs[l]-Acs[s])):0;
                mmq(Yd,M,xbar,CS,CS,P,qmode);                                    /* bounded M */
                free(M);
            }
            /* Y_off = (C·state_in)·exp(Acs[l]) ; state_in is [P,Nst] -> need [Nst,P]; do C·SIt */
            double*SIt=malloc((size_t)Nst*P*sizeof(double));
            for(int n=0;n<Nst;n++)for(int p=0;p<P;p++) SIt[(size_t)n*P+p]=state_in[(size_t)p*Nst+n];
            mmq(tmp,Cc,SIt,CS,Nst,P,qmode);                                      /* [CS,Nst]·[Nst,P]=[CS,P] */
            for(int l=0;l<CS;l++){int t=base+l; double el=H(EXPF(Acs[l])); for(int p=0;p<P;p++) y[IXHP(t,h,p)]=Yd[(size_t)l*P+p]+H(tmp[(size_t)l*P+p]*el)+D[h]*x[IXHP(t,h,p)];}
            free(SIt);
            /* cstate[p,n] = Σ_s exp(Acs[CS-1]-Acs[s])·B[s,n]·xbar[s,p] = (decay·xbar)ᵀ·B */
            double*Xd=malloc((size_t)CS*P*sizeof(double));
            for(int s=0;s<CS;s++){double ds=H(EXPF(Acs[CS-1]-Acs[s])); for(int p=0;p<P;p++) Xd[(size_t)s*P+p]=H(ds*xbar[(size_t)s*P+p]);}
            double*Xdt=malloc((size_t)P*CS*sizeof(double));
            for(int s=0;s<CS;s++)for(int p=0;p<P;p++) Xdt[(size_t)p*CS+s]=Xd[(size_t)s*P+p];
            mmq(cstate,Xdt,Bc,P,CS,Nst,qmode);                                   /* [P,CS]·[CS,Nst]=[P,Nst] */
            free(Xd);free(Xdt);
        }
    }
    free(Abar);free(Acs);free(xbar);free(state_in);free(cstate);free(acl);free(Cc);free(Bc);free(G);free(Xtil);free(Yd);free(Bt);free(tmp);
}

static double maxrel(const dims*d,const double*a,const double*b){int L=d->NC*d->CS;double mx=0;
    for(size_t i=0;i<(size_t)L*d->H*d->P;i++){double e=fabs(a[i]-b[i])/(fabs(b[i])+1e-6);if(e>mx)mx=e;}return mx;}
/* relative L2 (the standard metric; robust to near-zero outputs, unlike maxrel): ||a-b||/||b|| */
static double rel_l2(const dims*d,const double*a,const double*b){int L=d->NC*d->CS;double num=0,den=0;
    for(size_t i=0;i<(size_t)L*d->H*d->P;i++){double e=a[i]-b[i];num+=e*e;den+=b[i]*b[i];}return sqrt(num/(den+1e-30));}

static void run(dims d,double dtscale){
    int L=d.NC*d.CS;
    double*x=malloc((size_t)L*d.H*d.P*sizeof(double)),*dt=malloc((size_t)L*d.H*sizeof(double));
    double*A=malloc(d.H*sizeof(double)),*B=malloc((size_t)L*d.G*d.Nst*sizeof(double)),*C=malloc((size_t)L*d.G*d.Nst*sizeof(double)),*D=malloc(d.H*sizeof(double));
    double*yref=malloc((size_t)L*d.H*d.P*sizeof(double)),*y=malloc((size_t)L*d.H*d.P*sizeof(double));
    for(size_t i=0;i<(size_t)L*d.H*d.P;i++)x[i]=frs();
    for(size_t i=0;i<(size_t)L*d.H;i++){double v=frand()*dtscale; dt[i]=v>1e-4?v:1e-4;}
    for(int h=0;h<d.H;h++){A[h]=-exp(frand()*1.5); D[h]=frand();}
    for(size_t i=0;i<(size_t)L*d.G*d.Nst;i++){B[i]=frs();C[i]=frs();}
    seq(&d,x,dt,A,B,C,D,yref);
    struct{const char*t;int q,f,ewf,lut;}cases[]={
        {"fp32 grouped (sanity)",0,1,0,0},
        {"int8 per-channel, per-head",2,0,0,0},
        {"ALL-fp16 (mm+ew), per-head",3,0,1,0},
        {"ALL-fp16 + LUT-exp, per-head  <- full on-NPU",3,0,1,1}};
    printf("== dt~U(0,%.2f)  H=%d P=%d N=%d G=%d CS=%d NC=%d ==\n",dtscale,d.H,d.P,d.Nst,d.G,d.CS,d.NC);
    for(unsigned i=0;i<sizeof(cases)/sizeof(cases[0]);i++){ scan_gb(&d,x,dt,A,B,C,D,y,cases[i].q,cases[i].f,cases[i].ewf,cases[i].lut);
        printf("  %-44s rel-L2 = %.3e   (maxrel %.1e)\n",cases[i].t,rel_l2(&d,y,yref),maxrel(&d,y,yref)); }
    printf("\n");
    free(x);free(dt);free(A);free(B);free(C);free(D);free(yref);free(y);
}

int main(void){
    srand(20260712);
    printf("SSD int8 COHERENCE vs fp32 reference (NPU int8 matmul is exact => this IS the NPU's numerics)\n");
    printf("FACTORED = fast group-batched (2.66x NPU win). per-head = slow but bounded operands.\n\n");
    /* realistic dt (softplus output is typically small); larger dt stresses the exp dynamic range */
    run((dims){.H=4,.P=64,.Nst=128,.G=1,.CS=64,.NC=2}, 0.10);
    run((dims){.H=4,.P=64,.Nst=128,.G=1,.CS=64,.NC=2}, 0.30);
    run((dims){.H=4,.P=64,.Nst=128,.G=1,.CS=64,.NC=2}, 1.00);
    return 0;
}
