/* sdp_max_fuzz — realize an on-NPU element-wise ALU op by retargeting the SDP EW_ALU_ALGO field
 * (rocket_registers.h: 0x4070 EW_CFG, bits[19:16] = EW_ALU_ALGO; NVDLA firmware map_alu_op[] order
 * = MAX=0,MIN=1,SUM=2,EQL=3; our ADD template has algo=2=SUM, confirming the enum). Drives
 * ork_npu_probe_i8_mul with the ALU-active (ADD-routing) config + algo overridden. With operand a=0,
 * the three ops have DISTINCT signatures: MAX->max(0,b)=ReLU(b); MIN->min(0,b)=-ReLU(-b); SUM->b.
 * Confirms MAX=0 works on-NPU (no fuzzing of unknowns — codes are from the driver). BOARD: sudo ./sdp_max_fuzz */
#include "ork_npu.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
int main(void){
    ork_npu*c=ork_npu_init(); if(!c){printf("init failed\n");return 2;}
    if(!ork_ppu_fuse_enabled(c)){printf("PPU fuse not enabled — SKIP\n");ork_npu_free(c);return 0;}
    int n=64; int8_t a[64],b[64],o[64];
    for(int i=0;i<n;i++){ a[i]=0; b[i]=(int8_t)(i-32); }   /* a=0, b ramps -32..+31 */
    setenv("ORK_EW_AOFF","0",1); setenv("ORK_EW_COFF","0",1); setenv("ORK_EW_BIAS","0",1);
    setenv("ORK_EW_MULT","16384",1); setenv("ORK_EW_SHIFT","14",1);   /* gain = 1 */
    setenv("ORK_EW_R40","0x00020040",1); setenv("ORK_EW_R48","0x40000000",1);   /* ALU-active (ADD routing) */
    struct{int algo;const char*nm;}cs[]={{0,"MAX"},{1,"MIN"},{2,"SUM"}};
    for(unsigned k=0;k<3;k++){
        char r70[16]; snprintf(r70,sizeof r70,"0x%08x",0x904002c0u|((unsigned)cs[k].algo<<16));
        setenv("ORK_EW_R70",r70,1);
        memset(o,0,sizeof o); double us=0;
        int r=ork_npu_probe_i8_mul(c,a,b,n,o,&us);
        /* sample at b=-20 (idx12), b=-4 (idx28), b=+4 (idx36), b=+20 (idx52) */
        printf("algo=%d %-3s R70=%s rc=%d : out[b=-20]=%d out[b=-4]=%d out[b=+4]=%d out[b=+20]=%d\n",
               cs[k].algo,cs[k].nm,r70,r, o[12],o[28],o[36],o[52]);
    }
    printf("\nExpect (a=0): MAX -> 0,0,~4,~20 (ReLU) ; MIN -> ~-20,~-4,0,0 ; SUM -> ~-20,~-4,~4,~20\n");
    ork_npu_free(c); return 0;
}
