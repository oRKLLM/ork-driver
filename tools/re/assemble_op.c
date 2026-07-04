/* assemble_op.c — assemble a bit-exact on-NPU op from a captured RKNN dump, WITHOUT decoding the op's index
 * math. RKNN's activation LUT and its index/scale registers are a self-consistent matched pair: LUT[idx(in)]
 * is correct silu for RKNN's OWN idx(in), whatever its shape. So we transplant BOTH verbatim — the LUT (from
 * the 1097-reg LUT-load task's 0x4104 writes) and the compute regcmd (the 69-reg task) — and ork replays them,
 * patching only buffer addresses + M/N geometry. Result: bit-exact to RKNN in every precision (int8/int16/fp16),
 * with the LUT/params baked into a header => NO runtime RKNN dependency.
 *
 * Feed it a full capture dump (from tools/re/regcmd_capture.c) on stdin:
 *   sudo env LD_PRELOAD=.../regcmd_capture.so ... ./run_rknn model.rknn 2>dump
 *   ./assemble_op NAME < dump > src/regcmd_NAME.h
 * It finds the LUT-load regcmd (~2210 words) and the compute regcmd (~154 words) by size, and emits
 *   static const int16_t LUT_NAME[<n>]  = {...};   // RKNN's curve, in 0x4104 write order
 *   static const uint32_t REGCMD_NAME[<m>] = {...}; // the compute op regcmd (matched index/scale params baked in)
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#define MAXW 8192
#define MAXREGION 32

struct region { int nwords; uint32_t w[4096]; };
static struct region reg[MAXREGION];
static int nreg = 0;

int main(int argc, char **argv){
    const char *name = argc>1?argv[1]:"OP";
    char line[1024];
    int cur = -1, want = 0, got = 0;
    while (fgets(line, sizeof line, stdin)) {
        char *h = strstr(line, "regcmd (");
        if (h) { /* start a new region: "--- regcmd (N u32 words) ---" */
            if (nreg < MAXREGION) { cur = nreg++; want = atoi(h+8); got = 0; reg[cur].nwords = 0; }
            continue;
        }
        if (cur < 0) continue;
        char *p = strchr(line, ']'); p = p ? p+1 : line;   /* skip [nnn] index */
        char *tok = strtok(p, " \t\n");
        while (tok) {
            int len = strlen(tok), ok = (len==8);
            for (int i=0; ok && i<8; i++){ char ch=tok[i]; if(!((ch>='0'&&ch<='9')||(ch>='a'&&ch<='f')||(ch>='A'&&ch<='F'))) ok=0; }
            if (ok && got < want && reg[cur].nwords < 4096) { reg[cur].w[reg[cur].nwords++] = (uint32_t)strtoul(tok,NULL,16); got++; }
            tok = strtok(NULL, " \t\n");
        }
        if (got >= want) cur = -1;
    }
    /* find the LUT-load region (largest, ~2210 words) and the compute region (~146-154 words) */
    int lutr = -1, cmpr = -1;
    for (int i=0;i<nreg;i++){ if (reg[i].nwords > 1500) { if(lutr<0||reg[i].nwords>reg[lutr].nwords) lutr=i; }
                              else if (reg[i].nwords >= 140 && reg[i].nwords <= 160) { if(cmpr<0) cmpr=i; } }
    if (lutr<0 || cmpr<0) { fprintf(stderr,"assemble_op: could not find LUT-load (%d) and compute (%d) regions among %d\n", lutr, cmpr, nreg); return 1; }

    /* extract LUT: 0x4104 writes, value = (w0>>16) | ((w1&0xffff)<<16), taken as signed 16-bit */
    static int16_t lut[2048]; int nl=0;
    for (int k=0; k+1 < reg[lutr].nwords; k+=2){
        uint32_t w0=reg[lutr].w[k], w1=reg[lutr].w[k+1];
        if ((w0 & 0xffff) == 0x4104) { int32_t v = (int32_t)((w0>>16) | ((w1&0xffff)<<16)); lut[nl++] = (int16_t)v; }
    }
    /* trim compute regcmd trailing zero pairs */
    int m = reg[cmpr].nwords; while (m>=2 && reg[cmpr].w[m-1]==0 && reg[cmpr].w[m-2]==0) m-=2;

    printf("/* Assembled by tools/re/assemble_op.c from an RKNN capture — LUT + compute regcmd (matched pair).\n");
    printf(" * Bit-exact to RKNN: replay REGCMD_%s (patch addresses + M/N via set_mul_geom) after streaming\n", name);
    printf(" * LUT_%s through the REGCMD_SILU_LUT loader. No runtime RKNN dependency. */\n", name);
    printf("#define LUT_%s_N %d\n", name, nl);
    printf("static const int16_t LUT_%s[LUT_%s_N]={\n", name, name);
    for (int i=0;i<nl;i++){ printf("%d,", lut[i]); if((i%16)==15) printf("\n"); }
    printf("\n};\n");
    printf("#define REGCMD_%s_N %d\n", name, m);
    printf("static const uint32_t REGCMD_%s[REGCMD_%s_N]={\n", name, name);
    for (int i=0;i<m;i++){ printf("0x%08x,", reg[cmpr].w[i]); if((i%6)==5) printf("\n"); }
    printf("\n};\n");
    fprintf(stderr,"assemble_op: LUT_%s=%d entries, REGCMD_%s=%d words (from regions lut=%dw compute=%dw)\n",
            name, nl, name, m, reg[lutr].nwords, reg[cmpr].nwords);
    return 0;
}
