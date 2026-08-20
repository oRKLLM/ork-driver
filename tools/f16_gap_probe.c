/* f16_gap_probe — (B') does an inter-slice drain-gap let two WIDE fp16 matmuls (distinct weight buffers) ride ONE
 * PC-chain submit without the cross-buffer CDMA wild? Runs both the NO-GAP baseline and the perchan-GAP variant
 * on a wide Kp, prints rc (nonzero = wedge) + nonzero-output counts. BOARD: sudo env ORK_EW_TIMEOUT=3000 ./f16_gap_probe [M Kp N] */
#include "ork_npu.h"
#include <stdio.h>
#include <stdlib.h>
int main(int argc,char**argv){
    int M=argc>1?atoi(argv[1]):16, Kp=argc>2?atoi(argv[2]):2048, N=argc>3?atoi(argv[3]):512;
    ork_npu*c=ork_npu_init(); if(!c){printf("init failed\n");return 1;}
    printf("== f16_gap_probe M=%d Kp=%d N=%d (2 wide fp16 mm, distinct weights, one PC-chain) ==\n",M,Kp,N);
    long z0,z1; double us;
    int rc0=ork_f16_npu_gap_probe(c,M,Kp,N,0,&z0,&z1,&us);
    printf("  NO-GAP : rc=%d  nz0=%ld nz1=%ld (/%d)  %.1f us  %s\n",rc0,z0,z1,M*N,us, rc0?"WEDGE/err":"submitted");
    long y0,y1; double us2;
    int rc1=ork_f16_npu_gap_probe(c,M,Kp,N,1,&y0,&y1,&us2);
    printf("  GAP    : rc=%d  nz0=%ld nz1=%ld (/%d)  %.1f us  %s\n",rc1,y0,y1,M*N,us2, rc1?"WEDGE/err":"submitted");
    ork_npu_free(c);
    return 0;
}
