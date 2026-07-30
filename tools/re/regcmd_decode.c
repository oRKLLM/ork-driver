/**
 * @file regcmd_decode.c
 * @brief Decode a captured/synth regcmd dump into NAMED registers + NAMED values, across the FULL task sequence
 *        with NVDLA register-file PERSISTENCE tracking (RE tool, no NPU).
 *
 * Reads hex words from @c argv[1] (or stdin) in the RKDUMP dump format (space/newline-separated %08x words, as
 * written by @c tools/re/regcmd_capture.c → /tmp/mm_regcmd.txt, mm_chain_N.txt, and validate_mfold's
 * @c ORK_MF_REGCMD files). A per-line 8-hex-word scan pulls the words out of arbitrary text; prose is ignored.
 *
 * NEVER PRESUMES A SELF-CONTAINED TILE. RK3588/NVDLA tasks are DELTA-encoded: the register file PERSISTS across
 * chained tasks, so a task rewrites only the registers it changes and INHERITS the rest from its predecessors
 * (proven across the prefill: block 0x1001 output-stage registers are progressively omitted down the chain; see
 * full_sdp.py). Decoding a single task in isolation therefore MISREPRESENTS the state actually in force. So this
 * tool decodes EVERY task in the input and maintains a persistent register file:
 *   - each task is segmented from the input (by `--- regcmd` markers, or an explicit task word-count argv[2],
 *     or — for a lone tile file — the whole input as task 0);
 *   - every register write is annotated {NEW} / {CHANGED 0x..->0x.. @taskN} vs the inherited value;
 *   - each task reports how many registers it INHERITS (set by a prior task, not rewritten here) — the explicit
 *     "this task is not self-contained" signal;
 *   - a final EFFECTIVE STATE dump lists every register ever set, its final value, and the task that last wrote
 *     it — the complete config in force, which no single task carries.
 *
 * Per-register decode (driven ENTIRELY by @c ork_regs.h so it can never drift): register name + description (or
 * `?? UNKNOWN REGISTER`), the value's named setting (@c OKV_*) or `!! unknown value`, composer field decode
 * (@c OKC_*), `[IOVA]` for addresses, signed-stride / burst-rule annotations, and `!! bits outside field mask`.
 *
 * @par Build  @code make regcmd_decode @endcode
 * @par Use
 *   @code ./regcmd_decode /tmp/mm_regcmd_m8.txt            # a lone tile (task 0)
 *         ./regcmd_decode chain_dump.txt                    # a multi-task dump (`--- regcmd` markers segment it)
 *         ./regcmd_decode flat_PxN_stream.txt 232           # a flat P*232-word stream, chunked into 232-word tasks
 *         cat dump | ./regcmd_decode @endcode
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
 */
static const struct { enum ork_reg_id id; uint32_t val; const char *name; } OKV_TBL[] = {
    {RK_DPU_OUT_PRECISION, OKV_OUT_PREC_INT8,   "OKV_OUT_PREC_INT8"},
    {RK_DPU_OUT_PRECISION, OKV_OUT_PREC_INT32,  "OKV_OUT_PREC_INT32"},
    {RK_CNA_CONV_CON1,     OKV_CONV1_GROUP_LINE,       "OKV_CONV1_GROUP_LINE"},
    {RK_CNA_CONV_CON1,     OKV_CONV1_PLAIN,     "OKV_CONV1_PLAIN"},
    {RK_DPU_SURFACE_ADD,   OKV_ELEMSZ_INT8,     "OKV_ELEMSZ_INT8"},
    {RK_DPU_SURFACE_ADD,   OKV_ELEMSZ_INT32,    "OKV_ELEMSZ_INT32"},
    {RK_DPU_SURFACE_ADD,   OKV_SURFADD_MFOLD,   "OKV_SURFADD_MFOLD"},
};
enum { OKV_N = (int)(sizeof OKV_TBL / sizeof OKV_TBL[0]) };

/** @brief Persistent register file across tasks (NVDLA register-file persistence). Indexed by #ork_reg_id. */
struct rstate { uint32_t val[RK_REG__COUNT]; int set[RK_REG__COUNT]; int task[RK_REG__COUNT]; };

static int is_composer(enum ork_reg_id id){ return id==RK_CNA_CBUF_CON0||id==RK_CNA_DATA_SIZE0||id==RK_CNA_DATA_SIZE0_MIR; }
static int is_addr(enum ork_reg_id id){ return id==RK_CNA_FEATURE_DATA_ADDR||id==RK_CNA_WEIGHT_DATA_ADDR||id==RK_DPU_DST_BASE_ADDR||id==RK_PC_NEXT_ADDR; }
static int is_wide(enum ork_reg_id id){
    return is_addr(id)
        || id==RK_CNA_CONV_CON1
        || id==RK_CNA_DATA_SIZE0 || id==RK_CNA_DATA_SIZE0_MIR
        || id==RK_CNA_DATA_SIZE1
        || id==RK_CNA_WEIGHT_SIZE0
        || id==RK_CNA_WEIGHT_SIZE2
        || id==RK_CNA_DMA_CON2
        || id==RK_CNA_CVT_CON1 || id==RK_CNA_CVT_CON2 || id==RK_CNA_CVT_CON3 || id==RK_CNA_CVT_CON4;
}
static int is_signed(enum ork_reg_id id){ return id==RK_CNA_DMA_CON2; }

static int reg_id(uint32_t blk, uint32_t off){
    for(int i=0;i<RK_REG__COUNT;i++) if(ORK_REGS[i].blk==blk && ORK_REGS[i].off==off) return i;
    return -1;
}

/**
 * @brief Decode ONE task's regcmd words, tracking the persistent register file @p st.
 * @param w,n   this task's words.
 * @param taskno task index (0-based).
 * @param st    persistent register file (updated); registers not rewritten here stay INHERITED.
 * @param out   output stream.
 * @return count of flagged items (unknown-register + unknown-value + out-of-mask).
 */
static int decode_task(const uint32_t *w, int n, int taskno, struct rstate *st, FILE *out){
    int unk_reg=0, unk_val=0, oob_mask=0, nwrite=0;
    /* M for this task: its own 0x102c, else INHERIT the persistent one (the burst/stride rules key on M, and a
     * delta-encoded task legitimately omits 0x102c and inherits it — do not presume it is present). */
    int id102c=reg_id(0x201,0x102c), Mtile=0, curburst=0;
    for(int k=0;k+1<n && k<216;k+=2){ if((w[k+1]>>16)==0x201 && (w[k]&0xffffu)==0x102c && !Mtile) Mtile=(int)(w[k]>>16); }
    if(!Mtile && id102c>=0 && st->set[id102c]) Mtile=(int)st->val[id102c];   /* inherited M */
    if(id102c>=0 && st->set[id102c] && !curburst){ int idb=reg_id(0x201,0x107c); if(idb>=0 && st->set[idb]) curburst=(int)st->val[idb]; }

    for(int k=0;k+1<n;k+=2){
        uint32_t off=w[k]&0xffffu, blk=w[k+1]>>16;
        uint32_t val16=w[k]>>16, extra=w[k+1]&0xffffu;
        if(blk==0 && off==0 && val16==0 && extra==0) continue;   /* padding word-pair */
        int id=reg_id(blk,off); nwrite++;
        fprintf(out,"  [%3d] blk=0x%04x off=0x%04x ", k/2, blk, off);
        if(id<0){
            fprintf(out,"%-24s val16=0x%04x extra=0x%04x (32b=0x%08x)  ?? UNKNOWN REGISTER (not in ork_regs.h)\n",
                    "(unnamed)", val16, extra, val16|(extra<<16)); unk_reg++; continue; }
        const ork_reg_desc *d=&ORK_REGS[id];
        int wide=is_wide(id);
        uint32_t val = wide ? (val16|(extra<<16)) : val16;
        if(id==RK_CNA_DMA_CON1) curburst=(int)val;
        if(wide && is_signed(id)){
            uint32_t m=(d->mask==OKR_ANY)?0xffffffffu:d->mask, sb=(m>>1)+1;
            long sv=(long)(val&m); if((val&m)&sb) sv-=(long)(sb<<1);
            fprintf(out,"%-24s = 0x%08x (signed %ld)  ", d->name, val, sv);
        } else if(wide) fprintf(out,"%-24s = 0x%08x  ", d->name, val);
        else            fprintf(out,"%-24s = 0x%04x (extra=0x%04x)  ", d->name, val, extra);
        int has_okv=0, matched=0;
        for(int i=0;i<OKV_N;i++) if(OKV_TBL[i].id==id){ has_okv=1; if(OKV_TBL[i].val==val){ fprintf(out,"-> %s", OKV_TBL[i].name); matched=1; break; } }
        if(has_okv && !matched){
            fprintf(out,"!! unknown value (known:");
            for(int i=0;i<OKV_N;i++) if(OKV_TBL[i].id==id) fprintf(out," %s=0x%x",OKV_TBL[i].name,OKV_TBL[i].val);
            fprintf(out,")"); unk_val++;
        } else if(!has_okv){
            if(is_composer(id)){
                if(id==RK_CNA_CBUF_CON0)
                    fprintf(out,"-> {data_bank=%u weight_bank=%u fc_data_bank=%u DATA_REUSE=%u WEIGHT_REUSE=%u}",
                            val&0xf, (val>>4)&0xf, (val>>8)&0x7, (val>>12)&1, (val>>13)&1);
                else                     fprintf(out,"-> {width=%u height=%u}", (val>>16)&0x7ff, val&0x7ff);
            } else if(is_addr(id)) fprintf(out,"[IOVA]");
            else if(id==RK_CNA_FC_CON0)
                fprintf(out,"-> {FC_SKIP_EN=%u FC_SKIP_DATA=%u}", val&1, (val>>16)&0xffff);
            else if(id==RK_CNA_DMA_CON1){
                if(Mtile>0){ int exp=4*Mtile>128?128:4*Mtile;
                    if((int)val==exp) fprintf(out,"(feature DMA burst = 4*M = %d, MATCHED)", exp);
                    else fprintf(out,"(feature DMA burst = %u; ANOMALY != 4*M(%d): large-tile per-output-position prefetch, cap 128)", val, exp); }
                else fprintf(out,"(feature DMA burst)"); }
            else if(id==RK_CNA_DMA_CON2){
                uint32_t m=(d->mask==OKR_ANY)?0xffffffffu:d->mask, sb=(m>>1)+1; long sv=(long)(val&m); if((val&m)&sb) sv-=(long)(sb<<1);
                if(Mtile>0 && curburst>0 && (sv==(long)curburst-Mtile || sv==(long)Mtile-curburst)){
                    if(sv==-3L*Mtile) fprintf(out,"(surface stride %ld = -3*M = -(0x107c-M) [burst=4*M])", sv);
                    else              fprintf(out,"(surface stride %ld = %c(0x107c - M))", sv, sv>=0?'+':'-'); }
                else fprintf(out,"(surface stride %ld; NOTE: != +/-(0x107c=%d - M=%d))", sv, curburst, Mtile); }
            else fprintf(out,"(computed/raw)");
        }
        if(d->mask!=OKR_ANY && (val & ~d->mask)){ fprintf(out,"  !! bits 0x%x outside field mask 0x%x", val & ~d->mask, d->mask); oob_mask++; }
        /* PERSISTENCE delta vs the inherited value (never presume self-contained) */
        if(!st->set[id])                 fprintf(out,"  {NEW}");
        else if(st->val[id]!=val)        fprintf(out,"  {CHANGED 0x%x->0x%x @task%d}", st->val[id], val, st->task[id]);
        st->val[id]=val; st->set[id]=1; st->task[id]=taskno;
        if(d->desc) fprintf(out,"\n         └─ %s", d->desc);
        fprintf(out,"\n");
    }
    /* inheritance summary: registers set by a PRIOR task and NOT rewritten here => this task is not self-contained */
    int inherited=0, everset=0;
    for(int i=0;i<RK_REG__COUNT;i++) if(st->set[i]){ everset++; if(st->task[i]<taskno) inherited++; }
    fprintf(out,"  -- task %d: %d reg-writes | inherits %d/%d registers (set by a prior task, not rewritten here)"
                " | flags: %d unk-reg %d unk-val %d oob-mask\n", taskno, nwrite, inherited, everset, unk_reg, unk_val, oob_mask);
    return unk_reg+unk_val+oob_mask;
}

/** @brief Extract whole 8-hex-digit tokens from @p line, appending to @p *w (grown as needed). */
static void extract_words(const char *line, uint32_t **w, int *n, int *cap){
    static regex_t re; static int inited=0;
    if(!inited){ regcomp(&re,"[0-9a-fA-F]{8}",REG_EXTENDED); inited=1; }
    const char *p=line; regmatch_t m;
    while(regexec(&re,p,1,&m,0)==0){
        int so=(int)m.rm_so, eo=(int)m.rm_eo;
        int lok=(p+so==line)||!isxdigit((unsigned char)p[so-1]);
        int rok=!isxdigit((unsigned char)p[eo]);
        if(lok && rok){ if(*n>=*cap){ *cap=*cap?*cap*2:256; *w=realloc(*w,(size_t)*cap*4); }
            char t[9]; memcpy(t,p+so,8); t[8]=0; (*w)[(*n)++]=(uint32_t)strtoul(t,NULL,16); }
        p += eo>so?eo:so+1;
    }
}

/**
 * @brief Entry point: slurp input, segment into tasks, decode each with persistence, dump effective state.
 * @return 0 if every register+value in every task was recognized; 1 if anything flagged; 2 on input error.
 */
int main(int argc, char **argv){
    FILE *f = (argc>1 && strcmp(argv[1],"-")) ? fopen(argv[1],"r") : stdin;
    if(!f){ fprintf(stderr,"cannot open %s\n", argv[1]); return 2; }
    int task_words = (argc>2) ? atoi(argv[2]) : 0;   /* explicit flat-stream chunk size (else marker/whole-file) */

    /* read all lines; segment tasks on `--- regcmd` markers, skipping metadata/data lines */
    uint32_t **tw=NULL; int *tn=NULL; int ntask=0, tcap=0;    /* per-task word arrays */
    uint32_t *cur=NULL; int curn=0, curc=0; int have_marker=0;
    char *line=NULL; size_t lc=0; ssize_t ll;
    #define _FLUSH() do{ if(curn){ if(ntask>=tcap){ tcap=tcap?tcap*2:16; tw=realloc(tw,(size_t)tcap*sizeof*tw); tn=realloc(tn,(size_t)tcap*sizeof*tn);} \
                         tw[ntask]=cur; tn[ntask]=curn; ntask++; cur=NULL; curn=curc=0; } }while(0)
    while((ll=getline(&line,&lc,f))!=-1){
        if(strstr(line,"--- regcmd")){ have_marker=1; _FLUSH(); continue; }          /* task boundary */
        if(strstr(line,"=== SUBMIT")||strstr(line,"[dump")||strstr(line,"task[")) continue; /* metadata / capture data */
        extract_words(line,&cur,&curn,&curc);
    }
    _FLUSH();
    if(f!=stdin) fclose(f);
    free(line);

    /* flat-stream re-chunk: if an explicit task size was given (or a single blob with no markers and the caller
     * wants chunking), concatenate and split into task_words-sized tasks */
    if(task_words>0){
        uint32_t *all=NULL; int an=0, ac=0;
        for(int t=0;t<ntask;t++){ if(an+tn[t]>ac){ ac=an+tn[t]; all=realloc(all,(size_t)ac*4);} memcpy(all+an,tw[t],(size_t)tn[t]*4); an+=tn[t]; free(tw[t]); }
        free(tw); free(tn); tw=NULL; tn=NULL; ntask=tcap=0;
        for(int off=0;off<an;off+=task_words){ int len=an-off<task_words?an-off:task_words;
            if(ntask>=tcap){ tcap=tcap?tcap*2:16; tw=realloc(tw,(size_t)tcap*sizeof*tw); tn=realloc(tn,(size_t)tcap*sizeof*tn);}
            uint32_t*seg=malloc((size_t)len*4); memcpy(seg,all+off,(size_t)len*4); tw[ntask]=seg; tn[ntask]=len; ntask++; }
        free(all);
    }
    if(ntask==0){ fprintf(stderr,"empty input\n"); return 2; }

    int total=0; for(int t=0;t<ntask;t++) total+=tn[t];
    fprintf(stdout,"regcmd_decode: %d task(s), %d hex words total%s\n", ntask, total,
            have_marker?" (segmented by `--- regcmd` markers)":(task_words>0?" (chunked by task-size)":" (single task — no markers)"));

    struct rstate st; memset(&st,0,sizeof st);
    int flagged=0;
    for(int t=0;t<ntask;t++){
        fprintf(stdout,"\n===== TASK %d (%d words, %d reg-writes) =====\n", t, tn[t], tn[t]/2);
        flagged += decode_task(tw[t], tn[t], t, &st, stdout);
    }

    /* EFFECTIVE STATE — the complete config in force after all tasks (what no single task carries). Sorted by
     * (block, offset). This is the "full decode of everything available", never presuming one tile is complete. */
    fprintf(stdout,"\n===== EFFECTIVE STATE after %d task(s) — every register in force, last writer =====\n", ntask);
    int shown=0;
    for(uint32_t blk=0; blk<=0xffff; blk++){
        for(int i=0;i<RK_REG__COUNT;i++){
            if(!st.set[i] || ORK_REGS[i].blk!=blk) continue;
            const ork_reg_desc *d=&ORK_REGS[i];
            fprintf(stdout,"  blk=0x%04x off=0x%04x %-24s = 0x%08x   (last set @task%d)\n",
                    d->blk, d->off, d->name, st.val[i], st.task[i]); shown++;
        }
    }
    fprintf(stdout,"  (%d registers in force)\n", shown);

    for(int t=0;t<ntask;t++) free(tw[t]);
    free(tw); free(tn);
    return flagged?1:0;
}
