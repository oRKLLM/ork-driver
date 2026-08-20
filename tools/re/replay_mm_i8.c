/* replay_mm_i8.c — #38: replay rkllm's CAPTURED int8 matmul (regcmd + real tiled weight + real A) on ork's
 * submit path, VERIFY the output bit-exactly against rkllm's captured C, and TIME it vs ork's own kernel.
 * Consumes the RKDUMP_MM dump: /tmp/mm_{meta.txt,regcmd.txt,weight.bin,A.bin,C.bin}.
 *   ./replay_mm_i8 [iters]
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include "ork_npu.h"

static double now_us(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec*1e6+t.tv_nsec/1e3; }
static void* slurp(const char*p, size_t*n){ FILE*f=fopen(p,"rb"); if(!f)return NULL; fseek(f,0,SEEK_END); long s=ftell(f); fseek(f,0,SEEK_SET);
    void*b=malloc(s); if(fread(b,1,s,f)!=(size_t)s){fclose(f);free(b);return NULL;} fclose(f); if(n)*n=s; return b; }

int main(int argc,char**argv){
    setvbuf(stdout,NULL,_IONBF,0);   /* unbuffered: see progress even if a wedge kills us mid-run */
    int iters=argc>1?atoi(argv[1]):100;
    int M=0,K=0,N=0; { FILE*f=fopen("/tmp/mm_meta.txt","r"); if(!f){fprintf(stderr,"no mm_meta.txt\n");return 1;}
        char k[32]; int v; while(fscanf(f,"%31s %d",k,&v)==2){ if(!strcmp(k,"M"))M=v; else if(!strcmp(k,"K"))K=v; else if(!strcmp(k,"N"))N=v; } fclose(f); }
    /* regcmd words */
    unsigned *rc=malloc(4096*sizeof(unsigned)); int rn=0;
    { FILE*f=fopen("/tmp/mm_regcmd.txt","r"); if(!f){fprintf(stderr,"no mm_regcmd.txt\n");return 1;} while(rn<4096&&fscanf(f,"%x",&rc[rn])==1)rn++; fclose(f); }
    size_t wn=0,an=0,cn=0;
    int8_t *B=slurp("/tmp/mm_weight.bin",&wn), *A=slurp("/tmp/mm_A.bin",&an);
    int32_t *Cref=slurp("/tmp/mm_C.bin",&cn);
    printf("captured: M=%d K=%d N=%d rn=%d  weight=%zuB(K*N=%d) A=%zuB C=%zuB\n",M,K,N,rn,wn,K*N,an,cn);
    if(!B||!A||!Cref){ fprintf(stderr,"missing dump files\n"); return 1; }

    ork_npu*c=ork_npu_init(); if(!c){ printf("init failed (board only)\n"); return 77; }
    printf("ork %s soc=%s cores=%d\n",ork_npu_version(),ork_npu_soc(c),ork_npu_cores(c));
    double gmac_num=(double)M*K*N/1e3;

    /* (1) ork's own kernel FIRST — warms the NPU into int8 mode (a raw foreign submit into a cold/wrong-mode
     * context wedges) AND gives the speed reference. Single-core, same (M,K,N). */
    ork_npu_set_core_budget(c,1);
    double us_o=0;
    int8_t *Ao=malloc((size_t)M*K),*Bo=malloc((size_t)K*N); int32_t*Co=malloc((size_t)M*N*4);
    memset(Ao,1,(size_t)M*K); memset(Bo,1,(size_t)K*N);
    ork_w*w=ork_i8_mm_pack(c,K,N,Bo);
    if(w && ork_i8_mm_run(c,w,M,Ao,Co)==0){
        double t0=now_us(); for(int i=0;i<iters;i++) ork_i8_mm_run(c,w,M,Ao,Co); us_o=(now_us()-t0)/iters;
        printf("ork 1-core kernel:   %.1f us/matmul  %.1f GMAC/s  (warmed int8)\n",us_o, us_o>0?gmac_num/us_o:0);
        ork_mm_free(c,w);
    } else printf("ork kernel pack/run failed\n");

    /* (2) replay rkllm's regcmd with its REAL weight + A; verify vs captured C */
    printf("submitting rkllm regcmd...\n");
    int32_t *Cout=calloc((size_t)M*N,4); double us_r=0;
    int rr=ork_i8_npu_replay(c,rc,rn,M,K,N,A,(int)an,B,(int)wn,Cout,iters,&us_r);
    if(rr){ printf("REPLAY FAILED rc=%d\n",rr); }
    else {
        int bad=0,first=-1; long maxe=0;
        for(int i=0;i<M*N;i++){ long e=labs((long)Cout[i]-(long)Cref[i]); if(e){bad++; if(first<0)first=i;} if(e>maxe)maxe=e; }
        printf("REPLAY vs captured C: %s  mism=%d/%d maxerr=%ld", bad?"MISMATCH":"BIT-EXACT ***", bad, M*N, maxe);
        if(bad&&first>=0) printf("  first@%d got=%d ref=%d",first,Cout[first],Cref[first]);
        printf("\n  rkllm regcmd replay: %.1f us/matmul  %.1f GMAC/s\n",us_r, us_r>0?gmac_num/us_r:0);
        if(us_o>0) printf("  => rkllm/ork = %.2fx\n", us_r/us_o);
    }
    ork_npu_free(c); return 0;
}
