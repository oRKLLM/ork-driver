/* inject_map — DETERMINISTICALLY map rkllm's fold layout by CONTROLLED INJECTION through its captured regcmd.
 * Runs rkllm's exact regcmd (/tmp/mm_regcmd.txt) via ork_npu_replay_i8 but with OUR chosen A/W bytes:
 *   - weight buffer = all zero except ONE byte =1 at offset WPOS  -> only the (k,n) that byte encodes contributes
 *   - A buffer = a recoverable ramp: A[i] = (i % 251) - 125       -> each contributing product reveals its A byte
 * Then every NONZERO output element C[idx] tells us: this output slot (idx) receives weight-byte WPOS, with a
 * value = A[the A byte the fold paired with WPOS]. Sweeping WPOS maps weight-offset -> (output slot, A offset),
 * i.e. the full input/weight/output layout — no guessing, no layout priors. Execution is rkllm's proven regcmd
 * (runs on ork w/o wedge). Feeds first-K weight only (full 4x span wedges).
 *   sudo env WPOS=0 ORK_MM_TIMEOUT=6 ./inject_map [M=8] [K=3584] [N=1216]   (dumps nonzero C to stdout)
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include "ork_npu.h"
extern int ork_npu_replay_i8(ork_npu*,const unsigned*,int,int,int,int,const signed char*,int,const signed char*,int,int*,int,double*);

int main(int argc,char**argv){
    setvbuf(stdout,NULL,_IONBF,0);
    int M=argc>1?atoi(argv[1]):8, K=argc>2?atoi(argv[2]):3584, N=argc>3?atoi(argv[3]):1216;
    /* WPOS list: remaining argv are weight byte offsets to probe in ONE process (one model load) */
    long wposv[64]; int nwp=0;
    for(int i=4;i<argc && nwp<64;i++) wposv[nwp++]=atol(argv[i]);
    if(nwp==0){ wposv[0]=getenv("WPOS")?atol(getenv("WPOS")):0; nwp=1; }
    uint32_t rc[512]; FILE*f=fopen("/tmp/mm_regcmd.txt","r"); int rn=0;
    if(!f){perror("mm_regcmd.txt");return 2;} while(rn<512 && fscanf(f,"%x",&rc[rn])==1) rn++; fclose(f);
    printf("inject_map M=%d K=%d N=%d rn=%d nwp=%d\n",M,K,N,rn,nwp);

    size_t asz=(size_t)M*K, wsz=(size_t)K*N;
    int nmap = getenv("NMAP")!=NULL;   /* NMAP: A=all 1s, weight[wposv[i]]=1<<i (i<7); each output column's value */
                                        /* becomes a BITMASK of which injected offsets map to it -> maps many */
                                        /* offsets -> n in ONE submit (k within column not distinguished). */
    signed char*A=malloc(asz);
    if(nmap) memset(A,1,asz); else for(size_t i=0;i<asz;i++) A[i]=(signed char)((int)(i%251)-125);
    signed char*Wt=calloc(wsz,1);

    ork_npu*c=ork_npu_init(); if(!c){printf("init (board only)\n");return 77;}
    ork_npu_set_core_budget(c,1);
    { int8_t*wd=calloc(wsz,1); int32_t*td=calloc((size_t)8*N,4); int8_t*ad=calloc((size_t)8*K,1);   /* warm */
      ork_w*w=ork_mm_pack_i8(c,K,N,wd); if(w){ ork_mm_run_i8(c,w,8,ad,td); ork_mm_free(c,w);} free(wd);free(td);free(ad); }
    int32_t*Craw=calloc((size_t)M*N,4);

    if(nmap){   /* ONE submit: weight[wposv[i]]=1<<i, A=1 -> C[m][n] = sum of bits for offsets mapping to (any k, n) */
        memset(Wt,0,wsz);
        for(int i=0;i<nwp && i<7;i++) if(wposv[i]>=0&&(size_t)wposv[i]<wsz) Wt[wposv[i]]=(signed char)(1<<i);
        double us=0; int r=ork_npu_replay_i8(c,rc,rn,M,K,N,A,(int)asz,Wt,(int)wsz,Craw,1,&us);
        if(r){ printf("NMAP replay rc=%d\n",r); ork_npu_free(c); return 1; }
        printf("NMAP offsets:"); for(int i=0;i<nwp&&i<7;i++) printf(" [%d]=%ld",i,wposv[i]); printf("  (bit i set in a column's value => offset i maps to that n)\n");
        for(int i=0;i<M*N;i++) if(Craw[i]){ int n=(i/(M*4))*4+(i%4); printf("  C[%d] (n=%d) = %d  bits=0x%x\n",i,n,Craw[i],Craw[i]); }
        ork_npu_free(c); return 0;
    }
    /* decode helpers: A offset = nc16(m,k,M) = (k/16)*(M*16)+m*16+(k%16); output idx = c4(m,n,M) = (n/4)*(M*4)+m*4+(n%4) */
    for(int wi=0; wi<nwp; wi++){
        long wpos=wposv[wi];
        memset(Wt,0,wsz); if(wpos>=0&&(size_t)wpos<wsz) Wt[wpos]=1;
        double us=0; int r=ork_npu_replay_i8(c,rc,rn,M,K,N,A,(int)asz,Wt,(int)wsz,Craw,1,&us);
        if(r){ printf("WPOS=%ld replay rc=%d\n",wpos,r); continue; }
        int nz=0, idx0=-1; long val0=0;
        for(int i=0;i<M*N;i++) if(Craw[i]){ nz++; if(idx0<0){idx0=i; val0=Craw[i];} }
        /* n from idx0 (m=0 slot has smallest idx for the column): idx0=(n/4)*(M*4)+(n%4) => n=(idx0/(M*4))*4+(idx0%4) */
        int nn = (idx0>=0)? (idx0/(M*4))*4 + (idx0%4) : -1;
        /* k from val0 (=A[nc16(0,k,M)]=ramp): Aoff=(val0+125); Aoff=(k/16)*(M*16)+(k%16) => k=(Aoff/(M*16))*16+(Aoff%16) */
        long aoff = (val0+125); int kk = (aoff>=0)? (int)((aoff/(M*16))*16 + (aoff%16)) : -1;
        printf("WPOS=%-8ld nz=%d  idx0=%d val0=%ld  => n=%d k=%d\n",wpos,nz,idx0,val0,nn,kk);
    }
    ork_npu_free(c); return 0;
}
