/* max_reduce_probe — validate the BATCHED on-NPU per-row max-reduce (ork_npu_row_max_i8): all M rows
 * reduced together, one EW-max submit per tree level via base-offset addressing. Compares every row's
 * NPU max to a CPU reference; times it. BOARD: sudo ./max_reduce_probe [M] [N] */
#include "ork_npu.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
int main(int argc,char**argv){
    ork_npu*c=ork_npu_init(); if(!c){printf("init failed\n");return 2;}
    if(!ork_ppu_fuse_enabled(c)){printf("PPU fuse not enabled — SKIP\n");ork_npu_free(c);return 0;}
    int M=argc>1?atoi(argv[1]):64, N=argc>2?atoi(argv[2]):256;
    int8_t*a=malloc((size_t)M*N),*out=malloc(M); int*cpu=malloc(M*sizeof(int));
    unsigned s=98765;
    for(int m=0;m<M;m++){ int mx=-128; for(int n=0;n<N;n++){ s=s*1103515245+12345; int8_t v=(int8_t)((int)((s>>16)&0xff)-128); a[(size_t)m*N+n]=v; if(v>mx)mx=v; } cpu[m]=mx; }
    double us=0; int rc=ork_npu_row_max_i8(c,a,M,N,out,&us);
    int bad=0; for(int m=0;m<M;m++) if(out[m]!=cpu[m]){ if(bad<4)printf("  row %d: NPU=%d CPU=%d\n",m,out[m],cpu[m]); bad++; }
    printf("M=%d N=%d rc=%d : %d/%d rows exact, %.0f us total (%.1f us/row)  %s\n",
           M,N,rc,M-bad,M,us,us/M,(rc==0&&!bad)?"OK":"FAIL");
    free(a);free(out);free(cpu); ork_npu_free(c);
    return (rc==0&&!bad)?0:1;
}
