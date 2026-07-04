/* tools/ewmul_probe.c — board diagnostic for the fused EW-mul (SwiGLU dual-input) output stage.
 *
 * Runs ork_npu_probe_i8_ewmul at the captured shape (M=8,N=32) and prints, per output column,
 *   acc = sum_k A*B, r = requant(acc), G = 2nd input, C = NPU output
 * so the multiply/scale semantics can be read off (is C ~ r*G, r*G>>7, clamp(r*G), ...?).
 * First goal: does the 126-reg spliced program EXECUTE (not wedge)? Then read the numeric relationship.
 *
 *   make ewmul_probe && sudo ./ewmul_probe            (board only)
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "ork_npu.h"

#define M 8
#define K 512
#define N 32

static signed char A[M*K], B[K*N], G[M*N]; static int8_t C[M*N];

static int clampi8(long v){ if(v>127)v=127; if(v<-128)v=-128; return (int)v; }

int main(int argc,char**argv){
    ork_npu*c=ork_npu_init(); if(!c){printf("no board / init failed\n");return 0;}
    if(!ork_ppu_fuse_enabled(c)){printf("PPU fuse not enabled on this SoC — SKIP\n");ork_npu_free(c);return 0;}

    if(argc>1 && !strcmp(argv[1],"sg")){
        /* PER-ELEMENT validation with the NVDLA cube layout (rkt_ml.c): cube(m,n)=(n/16)*128+m*16+(n%16),
         * m=token(W=8), n=channel(C=64), ATOM=16. Marshal up/silu in, run mul, de-marshal out, check vs CPU. */
        const int Mt=8,Nc=64;
        static int8_t up[512], si[512], ac[512], bc[512], oc2[512];
        for(int m=0;m<Mt;m++)for(int n=0;n<Nc;n++){ up[m*Nc+n]=(int8_t)(((m*3+n)%15)-7); si[m*Nc+n]=(int8_t)(((n*2+m)%11)-5); }
        int ia=argc>4?atoi(argv[4]):16, oa=argc>5?atoi(argv[5]):16;   /* input-atom, output-atom (sweep) */
        for(int m=0;m<Mt;m++)for(int n=0;n<Nc;n++){ int p=(n/ia)*(8*ia)+m*ia+(n%ia); ac[p]=up[m*Nc+n]; bc[p]=si[m*Nc+n]; }
        int mult=argc>2?atoi(argv[2]):16384, shift=argc>3?atoi(argv[3]):14;   /* gain=mult/2^shift, default 1 */
        setenv("ORK_EW_AOFF","0",1);setenv("ORK_EW_COFF","0",1);setenv("ORK_EW_BIAS","0",1);
        char ms[16],ss[16]; sprintf(ms,"%d",mult);sprintf(ss,"%d",shift); setenv("ORK_EW_MULT",ms,1);setenv("ORK_EW_SHIFT",ss,1);
        double us=0; int r=ork_npu_probe_i8_mul(c,ac,bc,512,oc2,&us);
        if(r){ printf("sg returned %d\n",r); ork_npu_free(c); return 1; }
        int mism=0,mx=0,shown=0;
        for(int m=0;m<Mt;m++)for(int n=0;n<Nc;n++){ int p=(n/oa)*(8*oa)+m*oa+(n%oa); int o=oc2[p];
            long prod=(long)up[m*Nc+n]*si[m*Nc+n]; long e=(prod*mult)>>shift; if(e>127)e=127; if(e<-128)e=-128;
            int d=abs(o-(int)e); if(d){ mism++; if(d>mx)mx=d; if(shown<8){printf("  MISM m=%d n=%d up=%d si=%d exp=%ld got=%d\n",m,n,up[m*Nc+n],si[m*Nc+n],e,o);shown++;} } }
        printf("SG per-element cube mul: %.1fus  mism=%d/512  max|err|=%d  (%s)\n", us, mism, mx, mism==0?"BIT-EXACT":"off");
        ork_npu_free(c); return 0;
    }

    if(argc>1 && !strcmp(argv[1],"mul")){
        /* standalone SDP element-wise MUL: a*b -> out. uniform a,b (layout-agnostic) validates it executes+multiplies. */
        int av=argc>2?atoi(argv[2]):4, bv=argc>3?atoi(argv[3]):8; int Nn=512;
        static int8_t a[512],b[512],o[512];
        int ramp=(av==999), perm=(av==998);  /* 999=ramp; 998=position-index probe (map the layout permutation) */
        int bperm=(bv==998);                  /* b=998 => position-index on the EW operand b */
        for(int i=0;i<Nn;i++){ a[i]=perm?(int8_t)(i&0x3f):(ramp?(int8_t)((i%64)-32):(int8_t)av);
                               b[i]=bperm?(int8_t)(i&0x3f):(int8_t)bv; }
        double us=0; int r=ork_npu_probe_i8_mul(c,a,b,Nn,o,&us);
        if(r){ printf("mul returned %d (%s)\n",r,r==-1?"WEDGED":"bad dims"); ork_npu_free(c); return 1; }
        printf("STANDALONE MUL EXECUTED OK, %.1f us. a=%s b=%d\n", us, perm?"posidx":(ramp?"ramp":"const"), bv);
        int rows=(perm||ramp||bperm)?Nn:16;
        printf(" i:  a   b   out\n"); for(int i=0;i<rows;i++) printf(" %2d: %3d %3d  %4d\n", i, a[i], b[i], (int)o[i]);
        ork_npu_free(c); return 0;
    }

    if(argc>1 && !strcmp(argv[1],"ew64")){
        /* path-a EW-mul at N=64 (matches the ORK_EW_RK override values) — wedge binary-search vehicle */
        static signed char A[8*512], B[512*64], G[8*64]; static int8_t C[8*64];
        for(int i=0;i<8*512;i++)A[i]=1; for(int i=0;i<512*64;i++)B[i]=1;
        for(int m=0;m<8;m++)for(int n=0;n<64;n++)G[m*64+n]=(signed char)(n-32);
        double us=0; int r=ork_npu_probe_i8_ewmul(c,8,512,64,A,B,G,0x4000,14,C,&us);
        if(r){ printf("ew64 WEDGED (rc=%d) RK=%s\n",r,getenv("ORK_EW_RK")?getenv("ORK_EW_RK"):"0"); ork_npu_free(c); return 1; }
        printf("ew64 EXECUTED %.1fus RK=%s. out[0..15]: ",us,getenv("ORK_EW_RK")?getenv("ORK_EW_RK"):"0");
        for(int i=0;i<16;i++)printf("%d ",C[i]); printf("\n");
        ork_npu_free(c); return 0;
    }

    if(argc>1 && !strcmp(argv[1],"dump64")){
        /* dump ork's synth_i8(8,512,64)+set_i8_ewmul program (ORK_EW_NOMUL for plain int8-out) so it can
         * be diffed against RKNN's REGCMD_EWMUL_LIN to find why the EW graft wedges. */
        setenv("ORK_EW_DUMP","1",1);
        static signed char A[8*512], B[512*64], G[8*64]; static int8_t C[8*64];
        for(int i=0;i<8*512;i++)A[i]=1; for(int i=0;i<512*64;i++)B[i]=1; for(int i=0;i<8*64;i++)G[i]=1;
        ork_npu_probe_i8_ewmul(c,8,512,64,A,B,G,0x100,14,C,0);
        ork_npu_free(c); return 0;
    }

    if(argc>1 && !strcmp(argv[1],"tmpl")){
        /* Path (b): submit RKNN's captured op VERBATIM with arbitrary in-bounds data — does it EXECUTE? */
        /* uniform data so the conv layout is irrelevant: every output position = same acc, same 2nd-input */
        int iv=argc>2?atoi(argv[2]):1, wv=argc>3?atoi(argv[3]):1, gv=argc>4?atoi(argv[4]):64;
        static signed char in[4096], wt[0x6000], gl[0x2000]; static int8_t out[4096];
        memset(in,iv,4096); memset(wt,wv,0x6000); memset(gl,gv,0x2000);
        double us=0; int r=ork_npu_probe_i8_ewmul_tmpl(c,in,4096,wt,0x6000-0x2300,gl,0x2000-0x400,out,4096,&us);
        if(r){ printf("verbatim EW-mul returned %d (%s)\n",r,r==-1?"WEDGED":"bad dims"); ork_npu_free(c); return 1; }
        printf("VERBATIM EW-mul EXECUTED OK, %.1f us. in=%d wt=%d gl(silu)=%d\n", us,iv,wv,gv);
        printf("output bytes[0..31]: "); for(int i=0;i<32;i++)printf("%d ",out[i]); printf("\n");
        ork_npu_free(c); return 0;
    }

    if(argc>1 && !strcmp(argv[1],"lin")){
        /* Path (b) matmul replay at K=512,N=64,M=8. Uniform A,B so up_acc is constant per column;
         * G ramps so we can read out = f(requant(up_acc), G). */
        const int Ml=8,Kl=512,Nl=64;
        static signed char A[8*512], B[512*64], G[8*64]; static int8_t C[8*64];
        int av=argc>2?atoi(argv[2]):1, bv=argc>3?atoi(argv[3]):1;
        int gc=argc>4?atoi(argv[4]):0x7fff;   /* 4th arg: constant G (layout-independent); else ramp */
        for(int i=0;i<Ml*Kl;i++)A[i]=av;
        for(int i=0;i<Kl*Nl;i++)B[i]=bv;
        if(gc!=0x7fff) for(int i=0;i<Ml*Nl;i++)G[i]=(signed char)gc;               /* constant G */
        else for(int m=0;m<Ml;m++)for(int n=0;n<Nl;n++)G[m*Nl+n]=(signed char)(n-32);
        double us=0; int r=ork_npu_probe_i8_ewmul_lin(c,A,B,G,C,&us);
        if(r){ printf("ewmul_lin returned %d (%s)\n",r,r==-1?"WEDGED":"bad dims"); ork_npu_free(c); return 1; }
        long acc=(long)av*bv*Kl;   /* every column: sum_k A*B */
        printf("MATMUL EW-mul EXECUTED OK, %.1f us. A=%d B=%d up_acc(all cols)=%ld\n",us,av,bv,acc);
        printf(" n:  G   C(npu)\n");
        for(int n=0;n<Nl;n+=4) printf(" %2d: %4d  %4d\n", n, G[n], (int)C[n]);
        ork_npu_free(c); return 0;
    }

    int mult = argc>1?atoi(argv[1]):0x0100;   /* small requant so acc*mult>>shift stays in int8 range */
    int shift= argc>2?atoi(argv[2]):14;

    /* small structured inputs: A rows constant so acc is easy to predict; B columns ramp; G a ramp */
    for(int m=0;m<M;m++)for(int k=0;k<K;k++)A[m*K+k]=1;                 /* acc[m,n] = sum_k B[k,n] */
    for(int k=0;k<K;k++)for(int n=0;n<N;n++)B[k*N+n]=(signed char)((n%3)-1);  /* -1/0/1 -> acc = K*(n%3-1) */
    for(int m=0;m<M;m++)for(int n=0;n<N;n++)G[m*N+n]=(signed char)(n-16);      /* -16..15 */

    double us=0; int r=ork_npu_probe_i8_ewmul(c,M,K,N,A,B,G,mult,shift,C,&us);
    if(r){ printf("probe returned %d (%s)\n", r, r==-1?"WEDGED":"bad dims"); ork_npu_free(c); return 1; }
    printf("EW-mul probe OK, %.1f us. mult=0x%x shift=%d\n", us, mult, shift);
    printf(" n : acc      req      G     C(npu)   r*G   (r*G)>>7  clamp(r*G>>?)\n");
    for(int n=0;n<N;n++){
        long acc=0; for(int k=0;k<K;k++) acc += (long)A[0*K+k]*B[k*N+n];
        long req = (acc*mult) >> shift;               /* candidate requant (floor; HW is round-half-even) */
        int rq = clampi8(req);
        int g  = G[0*N+n];
        printf(" %2d: %6ld  %6d  %4d   %4d   %6d  %6d\n",
               n, acc, rq, g, (int)C[0*N+n], rq*g, (rq*g)>>7);
    }
    ork_npu_free(c); return 0;
}
