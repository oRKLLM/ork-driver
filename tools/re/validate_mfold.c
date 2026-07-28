/* validate_mfold.c — #39 step 3: bit-exact validation of synth_i8_mfold as a TRUE single-task program.
 * mc=M (one tile => input & output both width M, symmetric, no chain, no asymmetry). Known A,W -> CPU ref;
 * pack A NC1HWC2 (C2=16, width M) + weight ork-fmt; submit ork's own synth_i8_mfold regcmd; de-tile C
 * NC1HWC2 (width M); compare bit-exact. Run under nohup so a stall can't be ssh-orphaned.
 *   sudo env ORK_MFOLD=1 ORK_MM_TIMEOUT=6 ./validate_mfold [M] [K] [N]     (default 16 3584 1216)
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include "ork_npu.h"
extern void ork_i8_fuzz_add(uint32_t blk, uint32_t reg, uint32_t val);   /* RE fuzz override hook in npu.c */

#define FA 16
static size_t nc_off(int m,int c,int M){ return (size_t)(c/FA)*((size_t)M*FA)+(size_t)m*FA+(c%FA); }
static size_t ork_woff(int n,int k,int K){ int KT=(K+31)/32; return ((size_t)(n/32)*KT+(k/32))*1024+(size_t)(n%32)*32+(k%32); }

int main(int argc,char**argv){
    setvbuf(stdout,NULL,_IONBF,0);
    int M=argc>1?atoi(argv[1]):16, K=argc>2?atoi(argv[2]):3584, N=argc>3?atoi(argv[3]):1216;
    printf("validate_mfold(synth,single-tile) M=%d K=%d N=%d\n",M,K,N);

    int probe=getenv("ORK_MF_PROBE")?atoi(getenv("ORK_MF_PROBE")):0;   /* 0=random 1=all-ones 2=K-window mask */
    int klo=getenv("ORK_MF_KLO")?atoi(getenv("ORK_MF_KLO")):0, khi=getenv("ORK_MF_KHI")?atoi(getenv("ORK_MF_KHI")):K;
    int8_t*A=malloc((size_t)M*K),*W=malloc((size_t)K*N); int32_t*Cref=calloc((size_t)M*N,4);
    for(size_t i=0;i<(size_t)M*K;i++) A[i]=probe?1:(int8_t)((i*2654435761u>>27)%7 - 3);
    /* probe 2: A=1 everywhere, W[k][n]=1 only for k in [klo,khi) -> uniform C = |read_set ∩ [klo,khi)| */
    for(int k=0;k<K;k++)for(int n=0;n<N;n++) W[(size_t)k*N+n]= probe==2 ? ((k>=klo&&k<khi)?1:0) : (probe?1:(int8_t)(((size_t)k*N+n)*40503u>>13)%7 - 3);
    printf("probe=%d (0=random,1=ones,2=Kwin[%d,%d))\n",probe,klo,khi);
    for(int m=0;m<M;m++)for(int n=0;n<N;n++){ long s=0; for(int k=0;k<K;k++) s+=(long)A[(size_t)m*K+k]*W[(size_t)k*N+n]; Cref[(size_t)m*N+n]=(int32_t)s; }

    size_t Aelems=(size_t)((K+FA-1)/FA)*M*FA;
    int8_t*Anc=calloc(Aelems,1); for(int m=0;m<M;m++)for(int k=0;k<K;k++) Anc[nc_off(m,k,M)]=A[(size_t)m*K+k];
    int8_t*Wok=calloc((size_t)K*N,1); for(int k=0;k<K;k++)for(int n=0;n<N;n++) Wok[ork_woff(n,k,K)]=W[(size_t)k*N+n];

    ork_npu*c=ork_npu_init(); if(!c){printf("init (board only)\n");return 77;}
    ork_npu_set_core_budget(c,1);
    /* GROUND TRUTH: ork's own (known-correct) normal int8 matmul on the SAME logical A,W. Also serves as warmup. */
    int32_t*Cnorm=calloc((size_t)M*N,4);
    { ork_w*w=ork_mm_pack_i8(c,K,N,W); if(w){ork_mm_run_i8(c,w,M,A,Cnorm);ork_mm_free(c,w);} }
    { int cb=0; for(size_t i=0;i<(size_t)M*N;i++) if(Cnorm[i]!=Cref[i])cb++;
      printf("ork-normal vs CPU ref: %s (mism=%d)  [Cref[0..3]=%d %d %d %d]\n", cb?"DIFFER":"match", cb, Cref[0],Cref[1],Cref[2],Cref[3]); }

    /* RE sweep hook: ORK_MF_FUZZ="0x201:0x1044:0x560,0x201:0x107c:0x40" overrides regs in synth_i8_mfold
     * (applied last, wins) so schedule regs can be swept from env with no recompile. */
    { const char*fz=getenv("ORK_MF_FUZZ");
      if(fz){ char buf[512]; strncpy(buf,fz,sizeof buf-1); buf[sizeof buf-1]=0;
        for(char*t=strtok(buf,",");t;t=strtok(NULL,",")){ unsigned b,r,v;
          if(sscanf(t,"%x:%x:%x",&b,&r,&v)==3){ ork_i8_fuzz_add(b,r,v); printf("fuzz 0x%x:0x%x=0x%x\n",b,r,v); } } } }
    unsigned rc[512]; int rn=0;
    const char*rfile=getenv("ORK_MF_REGCMD");
    if(rfile){ FILE*f=fopen(rfile,"r"); if(!f){printf("no regcmd %s\n",rfile);return 1;}
        while(rn<512&&fscanf(f,"%x",&rc[rn])==1)rn++; fclose(f);
        /* zero any chain descriptor so it's a clean single task */
        for(int k=0;k+1<rn;k+=2){ unsigned o=rc[k]&0xffff,b=(rc[k+1]>>16)&0xffff; if(b==0x101&&(o==0x0010||o==0x0014)){rc[k]&=0xffff;rc[k+1]&=0xffff0000u;} }
        printf("REPLAY exact capture %s rn=%d, submitting...\n",rfile,rn);
    } else {
        setenv("ORK_MFOLD","1",1);
        rn=ork_npu_synth_i8_dump(c,M,K,N,rc,512);
        if(rn<0){printf("synth rc=%d\n",rn);return 1;}
        printf("synth mfold rn=%d, submitting...\n",rn);
    }

    int32_t*Cout=calloc((size_t)M*N,4); double us=0;
    int r=ork_npu_replay_i8(c,rc,rn,M,K,N,Anc,Wok,(int)((size_t)K*N),Cout,2,&us);
    if(r){ printf("submit rc=%d (STALL/err)\n",r); ork_npu_free(c); return 1; }
    /* INT32 output. de-tile NC1HWC2 (M as width): off(m,n)=(n/16)*(M*16)+m*16+(n%16) int32 elements. */
    int bad=0,first=-1; long mx=0;
    for(int m=0;m<M;m++)for(int n=0;n<N;n++){ int32_t got=Cout[nc_off(m,n,M)], ref=Cref[(size_t)m*N+n];
        long e=labs((long)got-ref); if(e){bad++; if(first<0)first=m*N+n;} if(e>mx)mx=e; }
    printf("RESULT int32 M-fold vs CPU ref: %s  mism=%d/%d maxerr=%ld  %.1f us/submit\n",
           bad?"MISMATCH":"BIT-EXACT ***",bad,M*N,mx,us);
    if(bad){ printf("  Cref[0..3]=%d %d %d %d  Cout(nc_off 0,0)=%d (1,0)=%d (0,1)=%d\n",
             Cref[0],Cref[1],Cref[2],Cref[3],Cout[nc_off(0,0,M)],Cout[nc_off(1,0,M)],Cout[nc_off(0,1,M)]);
      printf("  raw Cout[0..7]=%d %d %d %d %d %d %d %d\n",Cout[0],Cout[1],Cout[2],Cout[3],Cout[4],Cout[5],Cout[6],Cout[7]); }
    ork_npu_free(c); return bad?1:0;
}
