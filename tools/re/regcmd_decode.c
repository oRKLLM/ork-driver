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
    {RK_CNA_CONV_CON1,     OKV_CONV1_GROUP_LINE,       "OKV_CONV1_GROUP_LINE"},
    {RK_CNA_CONV_CON1,     OKV_CONV1_PLAIN,     "OKV_CONV1_PLAIN"},
    /* 0x1080 CNA_DMA_CON2 is a COMPUTED 16-bit stride (not an enum) — the old OKV_DMA2_CONTIGUOUS=0x0fffffe8
     * was a 32-bit-misread phantom, so it is intentionally NOT listed here (decodes as computed/raw). */
    {RK_DPU_SURFACE_ADD,   OKV_ELEMSZ_INT8,     "OKV_ELEMSZ_INT8"},
    {RK_DPU_SURFACE_ADD,   OKV_ELEMSZ_INT32,    "OKV_ELEMSZ_INT32"},
    {RK_DPU_SURFACE_ADD,   OKV_SURFADD_MFOLD,   "OKV_SURFADD_MFOLD"},
};
enum { OKV_N = (int)(sizeof OKV_TBL / sizeof OKV_TBL[0]) };

/** @brief True if @p id is a packed-field register the decoder should field-decode (OKC_* composed). */
static int is_composer(enum ork_reg_id id){ return id==RK_CNA_CBUF_CON0||id==RK_CNA_DATA_SIZE0||id==RK_CNA_DATA_SIZE0_MIR; }
/** @brief True if @p id holds an IOVA (annotated `[IOVA]`, never flagged as an unknown value). */
static int is_addr(enum ork_reg_id id){ return id==RK_CNA_FEATURE_DATA_ADDR||id==RK_CNA_WEIGHT_DATA_ADDR||id==RK_DPU_DST_BASE_ADDR||id==RK_PC_NEXT_ADDR; }
/** @brief True if @p id's VALUE is a genuine 32-bit quantity (value = (w0>>16) | ((w1&0xffff)<<16)).
 *  EVERY OTHER register is 16-bit: value = w0>>16, and w1&0xffff is a SEPARATE mode/secondary field — folding it
 *  into the value is the misread that made 0x1080 look like the 0x0fffffe8 wedge sentinel. So default is 16-bit;
 *  only addresses + the provably-packed regs are wide. (Migrating this to a per-row flag in ork_regs.h is #40.) */
static int is_wide(enum ork_reg_id id){
    return is_addr(id)
        || id==RK_CNA_DATA_SIZE0 || id==RK_CNA_DATA_SIZE0_MIR   /* (width<<16)|height */
        || id==RK_CNA_DATA_SIZE1                                /* ((K-1)<<16)|K */
        || id==RK_CNA_WEIGHT_SIZE0                              /* K*N bytes (exceeds 16 bits) */
        || id==RK_CNA_WEIGHT_SIZE2                              /* 0x1010000|N (bit 24 set) */
        || id==RK_CNA_DMA_CON2                                  /* signed 28-bit surface stride (w1-low = sign ext) */
        || id==RK_CNA_CVT_CON1 || id==RK_CNA_CVT_CON2 || id==RK_CNA_CVT_CON3 || id==RK_CNA_CVT_CON4; /* scale<<16|offset */
}
/** @brief True if @p id's (wide) value is TWO'S-COMPLEMENT signed over its mask width. 0x1080 CNA_DMA_CON2 is a
 *  signed surface stride: negatives (e.g. -3*M) sign-extend into w1-low as 0xfff — verified 0/4114 violations in
 *  the rkllm capture. The old "0x0fffffe8 sentinel" was just -24 (= -3*8) read as unsigned. NOT a mode field. */
static int is_signed(enum ork_reg_id id){ return id==RK_CNA_DMA_CON2; }

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
    /* Pre-scan the fold tile's M (0x102c) and primary feature-DMA burst (0x107c) so the CDMA schedule regs can be
     * checked against the derived rules: burst 0x107c = min(4*M,128) for matched single-pass tiles (else it is a
     * per-output-position prefetch on the large accumulate tiles), and stride 0x1080 = +/-(0x107c - M). */
    int Mtile=0, curburst=0;   /* curburst tracks the MOST-RECENT 0x107c so 0x1080 pairs with ITS burst, not the first */
    for(int k=0;k+1<n && k<216;k+=2){ uint32_t b=w[k+1]>>16, o=w[k]&0xffffu, v=w[k]>>16;
        if(b==0x201 && o==0x102c && !Mtile) Mtile=(int)v; }
    for(int k=0;k+1<n;k+=2){
        uint32_t off=w[k]&0xffffu, blk=w[k+1]>>16;
        uint32_t val16=w[k]>>16, extra=w[k+1]&0xffffu;   /* w0 high = 16-bit value; w1 low = a SEPARATE mode/high field */
        int id=reg_id(blk,off);
        fprintf(out,"[%3d] blk=0x%04x off=0x%04x ", k/2, blk, off);
        if(id<0){ /* width unknown -> show BOTH interpretations rather than guess (16-bit val+extra, and the 32-bit combine) */
            fprintf(out,"%-24s val16=0x%04x extra=0x%04x (32b=0x%08x)  ?? UNKNOWN REGISTER (not in ork_regs.h)\n",
                    "(unnamed)", val16, extra, val16|(extra<<16)); unk_reg++; continue; }
        const ork_reg_desc *d=&ORK_REGS[id];
        int wide=is_wide(id);
        uint32_t val = wide ? (val16|(extra<<16)) : val16;   /* 16-bit regs: value=w0>>16; extra shown separately, NOT folded in */
        if(id==RK_CNA_DMA_CON1) curburst=(int)val;           /* running burst: 0x1080 below pairs with the nearest preceding 0x107c */
        if(wide && is_signed(id)){
            uint32_t m=(d->mask==OKR_ANY)?0xffffffffu:d->mask, sb=(m>>1)+1;   /* sign bit = top set bit of the mask */
            long sv=(long)(val&m); if((val&m)&sb) sv-=(long)(sb<<1);
            fprintf(out,"%-24s = 0x%08x (signed %ld)  ", d->name, val, sv);
        } else if(wide) fprintf(out,"%-24s = 0x%08x  ", d->name, val);
        else            fprintf(out,"%-24s = 0x%04x (extra=0x%04x)  ", d->name, val, extra);
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
            else if(id==RK_CNA_DMA_CON1){          /* 0x107c feature-DMA burst: rule = min(4*M,128) for matched tiles */
                if(Mtile>0){ int exp=4*Mtile>128?128:4*Mtile;
                    if((int)val==exp) fprintf(out,"(feature DMA burst = 4*M = %d, MATCHED)", exp);
                    else fprintf(out,"(feature DMA burst = %u; ANOMALY != 4*M(%d): large-tile per-output-position prefetch, cap 128)", val, exp); }
                else fprintf(out,"(feature DMA burst)"); }
            else if(id==RK_CNA_DMA_CON2){          /* 0x1080 signed surface stride: rule = +/-(0x107c - M) */
                uint32_t m=(d->mask==OKR_ANY)?0xffffffffu:d->mask, sb=(m>>1)+1; long sv=(long)(val&m); if((val&m)&sb) sv-=(long)(sb<<1);
                if(Mtile>0 && curburst>0 && (sv==(long)curburst-Mtile || sv==(long)Mtile-curburst)){
                    if(sv==-3L*Mtile) fprintf(out,"(surface stride %ld = -3*M = -(0x107c-M) [burst=4*M])", sv);
                    else              fprintf(out,"(surface stride %ld = %c(0x107c - M))", sv, sv>=0?'+':'-'); }
                else fprintf(out,"(surface stride %ld; NOTE: != +/-(0x107c=%d - M=%d))", sv, curburst, Mtile); }
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
