/**
 * @file regcmd_decode.c
 * @brief Decode a captured/synth regcmd word-dump into NAMED registers + NAMED values (RE tool, no NPU).
 *
 * Reads hex words from @c argv[1] (or stdin) in the RKDUMP dump format (space/newline-separated %08x words,
 * as written by @c tools/re/regcmd_capture.c → /tmp/mm_regcmd.txt, mm_chain_N.txt, and validate_mfold's
 * @c ORK_MF_REGCMD files). A POSIX regex pulls the 8-hex-digit words out of arbitrary text, so a raw capture
 * log can be piped in directly and stray lines are ignored.
 *
 * For each (block,offset,value) triplet [word pair: @c off=w[k]&0xffff, @c blk=w[k+1]>>16,
 * @c val=(w[k]>>16)|((w[k+1]&0xffff)<<16)] it reports, driven ENTIRELY by @c ork_regs.h so it can never drift:
 *   - the REGISTER name (#ORK_REGS) + its description, or `?? UNKNOWN REGISTER` if (blk,off) is not in the table;
 *   - the VALUE's named setting (@c OKV_* via #OKV_TBL) for enum-like registers, or `!! unknown value`
 *     (with the list of known settings) if it matches none — how a foreign/vendor value ork has no name for is surfaced;
 *   - composer field decode for @c OKC_* registers (CBUF_CON0 banks, DATA_SIZE0 width/height);
 *   - `[IOVA]` for address registers;
 *   - `!! bits outside field mask` if the value sets bits the register's documented mask forbids.
 *
 * @par Build
 *   @code cc -Isrc -Iinclude -o regcmd_decode tools/re/regcmd_decode.c @endcode  (or `make regcmd_decode`)
 * @par Use
 *   @code ./regcmd_decode /tmp/mm_regcmd.txt @endcode  or  @code cat dump | ./regcmd_decode @endcode
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <ctype.h>
#include <regex.h>
#include "ork_regs.h"

/**
 * @brief VALUE-name table: the @c OKV_* named settings, keyed by the register they belong to.
 *
 * A register that appears here is "enum-like" — a value matching none of its entries is flagged. Kept in
 * sync with the @c OKV_* block in ork_regs.h (single source: the constants themselves are @c \#included).
 */
static const struct { enum ork_reg_id id; uint32_t val; const char *name; } OKV_TBL[] = {
    {RK_DPU_OUT_PRECISION, OKV_OUT_PREC_INT8,   "OKV_OUT_PREC_INT8"},
    {RK_DPU_OUT_PRECISION, OKV_OUT_PREC_INT32,  "OKV_OUT_PREC_INT32"},
    {RK_CNA_CONV_CON1,     OKV_CONV1_STD,       "OKV_CONV1_STD"},
    {RK_CNA_CONV_CON1,     OKV_CONV1_MFOLD,     "OKV_CONV1_MFOLD"},
    {RK_CNA_DMA_CON2,      OKV_DMA2_CONTIGUOUS, "OKV_DMA2_CONTIGUOUS"},
    {RK_DPU_SURFACE_ADD,   OKV_ELEMSZ_INT8,     "OKV_ELEMSZ_INT8"},
    {RK_DPU_SURFACE_ADD,   OKV_ELEMSZ_INT32,    "OKV_ELEMSZ_INT32"},
    {RK_DPU_SURFACE_ADD,   OKV_SURFADD_MFOLD,   "OKV_SURFADD_MFOLD"},
};
enum { OKV_N = (int)(sizeof OKV_TBL / sizeof OKV_TBL[0]) };

/** @brief True if @p id is a packed-field register the decoder should field-decode (OKC_* composed). */
static int is_composer(enum ork_reg_id id){ return id==RK_CNA_CBUF_CON0||id==RK_CNA_DATA_SIZE0||id==RK_CNA_DATA_SIZE0_MIR; }
/** @brief True if @p id holds an IOVA (annotated `[IOVA]`, never flagged as an unknown value). */
static int is_addr(enum ork_reg_id id){ return id==RK_CNA_FEATURE_DATA_ADDR||id==RK_CNA_WEIGHT_DATA_ADDR||id==RK_DPU_DST_BASE_ADDR||id==RK_PC_NEXT_ADDR; }

/**
 * @brief Find the #ork_reg_id for a (block,offset) pair.
 * @param blk regcmd block id.
 * @param off register offset within the block.
 * @return the index into #ORK_REGS, or -1 if the pair is not named.
 */
static int reg_id(uint32_t blk, uint32_t off){
    for(int i=0;i<RK_REG__COUNT;i++) if(ORK_REGS[i].blk==blk && ORK_REGS[i].off==off) return i;
    return -1;
}

/**
 * @brief Decode one regcmd word array into an annotated listing (register name, value name, description, flags).
 * @param w   regcmd words (value/target pairs, as captured).
 * @param n   number of words in @p w.
 * @param out stream to write the annotated listing to.
 * @return the count of flagged items (unknown-register + unknown-value + out-of-mask); 0 if all recognized.
 */
static int ork_regcmd_decode(const uint32_t *w, int n, FILE *out){
    int unk_reg=0, unk_val=0, oob_mask=0;
    for(int k=0;k+1<n;k+=2){
        uint32_t off=w[k]&0xffffu, blk=w[k+1]>>16, val=(w[k]>>16)|((w[k+1]&0xffffu)<<16);
        int id=reg_id(blk,off);
        fprintf(out,"[%3d] blk=0x%04x off=0x%04x ", k/2, blk, off);
        if(id<0){ fprintf(out,"%-24s = 0x%08x  ?? UNKNOWN REGISTER (not in ork_regs.h)\n","(unnamed)",val); unk_reg++; continue; }
        const ork_reg_desc *d=&ORK_REGS[id];
        fprintf(out,"%-24s = 0x%08x  ", d->name, val);
        /* enum-like? (has OKV_ entries) */
        int has_okv=0, matched=0;
        for(int i=0;i<OKV_N;i++) if(OKV_TBL[i].id==id){ has_okv=1; if(OKV_TBL[i].val==val){ fprintf(out,"-> %s", OKV_TBL[i].name); matched=1; break; } }
        if(has_okv && !matched){
            fprintf(out,"!! unknown value (known:");
            for(int i=0;i<OKV_N;i++) if(OKV_TBL[i].id==id) fprintf(out," %s=0x%x",OKV_TBL[i].name,OKV_TBL[i].val);
            fprintf(out,")"); unk_val++;
        } else if(!has_okv){
            if(is_composer(id)){
                if(id==RK_CNA_CBUF_CON0) fprintf(out,"-> {data_bank=%u weight_bank=%u}", val&0xf, (val>>4)&0xf);
                else                     fprintf(out,"-> {width=%u height=%u}", (val>>16)&0x7ff, val&0x7ff);
            } else if(is_addr(id)) fprintf(out,"[IOVA]");
            else fprintf(out,"(computed/raw)");
        }
        if(d->mask!=OKR_ANY && (val & ~d->mask)){ fprintf(out,"  !! bits 0x%x outside field mask 0x%x", val & ~d->mask, d->mask); oob_mask++; }
        if(d->desc) fprintf(out,"\n        └─ %s", d->desc);   /* overlay the ork_regs.h Doxygen description */
        fprintf(out,"\n");
    }
    fprintf(out,"---- %d reg-writes | flags: %d unknown-register, %d unknown-value, %d out-of-mask ----\n",
            n/2, unk_reg, unk_val, oob_mask);
    return unk_reg+unk_val+oob_mask;
}

/**
 * @brief Entry point: slurp input (file arg or stdin), regex-extract the hex words, decode, print.
 * @return 0 if every register+value was recognized; 1 if anything was flagged; 2 on input error.
 */
int main(int argc, char **argv){
    /* slurp input (file arg or stdin) */
    FILE *f = (argc>1) ? fopen(argv[1],"r") : stdin;
    if(!f){ fprintf(stderr,"cannot open %s\n", argv[1]); return 2; }
    char *buf=NULL; size_t cap=0, len=0; int c;
    while((c=fgetc(f))!=EOF){ if(len+1>=cap){ cap=cap?cap*2:4096; buf=realloc(buf,cap); } buf[len++]=(char)c; }
    if(len) buf[len]=0; else { fprintf(stderr,"empty input\n"); return 2; }
    if(f!=stdin) fclose(f);

    /* regex-extract 8-hex-digit words (ignore addresses/prose in a mixed log) */
    regex_t re; regcomp(&re,"[0-9a-fA-F]{8}",REG_EXTENDED);
    uint32_t *w=NULL; int n=0, wc=0;
    const char *p=buf; regmatch_t m;
    while(regexec(&re,p,1,&m,0)==0){
        /* require a WHOLE token of exactly 8 hex (bounded by non-hex on both sides) so we don't slice longer hex */
        int so=(int)m.rm_so, eo=(int)m.rm_eo;
        int lok = (p+so==buf) || !isxdigit((unsigned char)p[so-1]);
        int rok = !isxdigit((unsigned char)p[eo]);
        if(lok && rok){ if(n>=wc){ wc=wc?wc*2:256; w=realloc(w,(size_t)wc*4); } char t[9]; memcpy(t,p+so,8); t[8]=0; w[n++]=(uint32_t)strtoul(t,NULL,16); }
        p += eo>so?eo:so+1;
    }
    regfree(&re);
    fprintf(stdout,"regcmd_decode: %d hex words (%d reg-writes)\n", n, n/2);
    int flagged = ork_regcmd_decode(w, n, stdout);
    free(buf); free(w);
    return flagged?1:0;
}
