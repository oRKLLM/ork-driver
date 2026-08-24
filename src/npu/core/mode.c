/* npu/core/mode.c — the mode-transition layer: ork_npu_enter, driven by the XSPEC policy table.
 * Part of the dtype-agnostic substrate; interface in npu/core.h. Lifted verbatim from npu.c by the
 * precision split (MODULARIZE_PLAN.md round 1). */
#define _GNU_SOURCE   /* CPU_SET/pthread_setaffinity_np, as npu.c does */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <fcntl.h>
#include <errno.h>
#include <pthread.h>
#include <sched.h>
#include <signal.h>
#include <sys/mman.h>
#include <sys/ioctl.h>
#include <stdarg.h>
#include <sys/prctl.h>
#include "ork_regs.h"
#include "regcmd_i8.h"
#include "npu/internal.h"
#include "npu/core.h"
#include "npu/core/core.h"

static const struct ork_xspec XSPEC[XP_NPROFILE] = {
  /* XP_MC_MM      3596  run_multicore   */ { KWP_MC,  RC_I8ENTRY,       TG_PERCORE, WC_NOTKW,         TG_PERCORE, WC_NT_NOTKW, 1 },
  /* XP_SC_MM      4211  run single-core */ { KWP_SC,  RC_I8ENTRY,       TG_SCALAR,  WC_NOTKW,         TG_SCALAR,  WC_NT_NOTKW, 1 },
  /* XP_CHAIN_NT   7194/7477 chain       */ { KWP_MC,  RC_NOTLIVE_NOTKW, TG_SCALAR,  WC_NOTKW,         TG_SCALAR,  WC_NOTKW,    1 },
  /* XP_STREAM_I8  7821  run_stream_i8   */ { KWP_MC,  RC_NOTLIVE_NOTKW, TG_BOTH,    WC_NOTLIVE_NOTKW, TG_NONE,    WC_NONE,     1 },
  /* XP_STREAM_F16 7892/8026 stream f16  */ { KWP_F16, RC_NOTKW,         TG_BOTH,    WC_NOTKW,         TG_NONE,    WC_NONE,     1 },
  /* XP_I4_MC      4000/4161 int4 mc     */ { KWP_NTI, RC_NOTKW,         TG_PERCORE, WC_NOTKW,         TG_PERCORE, WC_NT,       1 },
  /* XP_I4_MWARM   8692/8783/8816 int4   */ { KWP_NTI, RC_NOTKW,         TG_PERCORE, WC_NOTKW,         TG_NONE,    WC_NONE,     1 },
  /* XP_I4_INCR    8552  int4 incr       */ { KWP_NTI, RC_NOTKW,         TG_NONE,    WC_NONE,          TG_NONE,    WC_NONE,     1 },
  /* XP_I4CHAIN    8497  run_chain_i4 (4)*/ { KWP_NONE,RC_ALWAYS,        TG_SCALAR,  WC_ALWAYS,        TG_SCALAR,  WC_ALWAYS,   1 },
  /* XP_I4_STREAM  9014  stream i4 (5)   */ { KWP_NONE,RC_ALWAYS,        TG_BOTH,    WC_ALWAYS,        TG_NONE,    WC_NONE,     1 },
  /* XP_SDP        activation/ewmul (4)  */ { KWP_NONE,RC_SDPKW,         TG_NONE,    WC_NONE,          TG_NONE,    WC_NONE,     1 },  /* setdt=1: record SDP mode so a following matmul (int4-grouped/fp16/int8) re-transitions -> resets, fixing the mm-after-SDP wedge */
};


/* MODE-TRANSITION RE hooks (mode_probe.c). The standalone SDP ops (ork_npu_ewmul_*, orki_i16_act_lut →
 * exp/silu/…) reprogram the pipeline (their own ACT_RESET + SDP regcmd) but leave c->last_dt / c->warmed
 * untouched, so a following SAME-dtype matmul sees dt==last_dt and SKIPS its reset/re-warm — running a
 * matmul regcmd on an SDP-configured pipeline. These expose the two candidate fixes so the probe can
 * measure which is sufficient:
 *   _invalidate: clear the cached mode state ONLY (last_dt=-1, warmed=0, per-core mwarm=0) — the next
 *                matmul then takes its own reset/re-warm path (fp16 entry = warmed=0 re-warm, int8 entry =
 *                ACT_RESET). No explicit HW reset here. Tests whether re-warm alone clears the wedge.
 *   _reset:      an explicit HW ACT_RESET AND invalidate — the heavyweight, always-safe reinit. */
/* Probe hooks, settable by call OR by env so the ggml consumer can exercise them on a real model:
 * -1 = not yet resolved, read the env once; 0/1 = explicitly set. */
int orki_xspec_noreset = -1;
int orki_xspec_nores(void){ if(orki_xspec_noreset<0){ const char*e=getenv("ORK_XSPEC_NORESET"); orki_xspec_noreset=(e&&atoi(e))?1:0; } return orki_xspec_noreset; }
int orki_xspec_nocle(void){ if(orki_xspec_noclear<0){ const char*e=getenv("ORK_XSPEC_NOCLEAR"); orki_xspec_noclear=(e&&atoi(e))?1:0; } return orki_xspec_noclear; }
void ork_npu_set_xspec_noreset(int on){ orki_xspec_noreset = on ? 1 : 0; }
int orki_xspec_noclear = -1;   /* probe hook: skip the warm/size CLEARS (the OTHER half of what a keep-warm
                               * predicate suppresses). Separating the two matters: the 27B corruption was
                               * produced by a kw=1 that turned off reset AND clears together, so which one
                               * was load-bearing was never established. */
void ork_npu_set_xspec_noclear(int on){ orki_xspec_noclear = on ? 1 : 0; }

/* ============================ MODE-TRANSITION LAYER (ork_npu_enter) ============================
 * SINGLE owner of "what does moving the NPU's stateful regcmd datapath from mode X to mode Y
 * require" — the ACT_RESET / re-warm (warmed, mwarm[]) / buffer-realloc (ccsz, mccsz[]) policy that
 * was previously copy-pasted (and quietly drifted) inline into every run/stream/chain/int4 entry.
 *
 * Each run path calls ork_npu_enter(c, target_marker, profile, chain) FIRST; the per-profile row of XSPEC
 * below IS the policy for that path. A profile is a faithful, byte-for-byte transcription of the site
 * it replaced (verified `make test` byte-identical across all dtypes and both keep-warm knobs), so
 * Phase-1 behavior was UNCHANGED — the consolidation was behavior-preserving. The drift is visible AS
 * DATA, and a policy change is a one-row edit — e.g. PHASE 2 (2026-07-14) converged the →I8_CHAIN
 * profiles: XP_CHAIN_NT used to ignore ORK_SSM_KEEPWARM (KWP_NTL + RC_NOTLIVE), so a chain entered
 * from an fp16 op ate a full ~105ms ACT_RESET where the stream profiles kept warm; switching it to
 * KWP_MC + RC_NOTLIVE_NOTKW eliminated that (chain_xition_probe: reset-cost 53538us→~0, coherent), and
 * the two stream-int8 profiles collapsed into one (XP_STREAM_I8). See the wiki "Exp-2026-07-14 Mode-
 * Transition Layer" for the full Phase-2 record and AGENTS.md §"Mode-transition layer" for how to add/change.
 *
 * EXHAUSTIVE (from -> to) permutation space — modes = { COLD(-1), F16(0), I8(1), I4(2), I8_CHAIN(3),
 * I4_CHAIN(4), I4_STREAM(5) }, plus SDP = a TRANSIENT activation/ewmul reset with NO stored marker.
 * `from` (= c->last_dt) enters ONLY through the
 * ORK_I8_LIVE / ORK_INT_DT / ORK_KW_DT predicates, so a row is keyed by (target, caller-scope), not by
 * an enumerated `from` — that collapses the NxN matrix to one row per historical site:
 *   ->F16/I8 matmul : reset only ENTERING int8 from a non-int8-live mode (first-int8-submit wedge);
 *                     fp16 never resets. Keep-warm across int8<->fp16 (ORK_SSM_KEEPWARM, default on).
 *   ->I4           : reset entering int4 from a non-int mode; keep-warm int<->int (ORK_MIXED_NOTHRASH).
 *   ->I8_CHAIN(3)  : DT_I8<->DT_I8_CHAIN is NOT a hw mode change (ORK_I8_LIVE) -> no reset.
 *   ->I4_CHAIN(4)  : unconditional reset on entry (single-core int4 M=1 chain).
 *   ->I4_STREAM(5) : unconditional reset on entry.
 *   ->SDP          : activation/ewmul reprogram the pipeline but correctly LEAVE last_dt untouched
 *                    (setdt=0), so the NEXT matmul keeps warm (no ~105us re-warm). The historical
 *                    "SDP->matmul wedge" was NOT a last_dt issue — it was the c->task LUT-descriptor
 *                    poisoning (nuance #1), fixed independently in 98c00b1 (Exp-2026-07-12). Board
 *                    mode_probe (2026-07-14) confirms EVERY SDP->matmul is SAFE with NO reset, and
 *                    that FORCING one costs ~105us/transition for zero correctness gain. XP_SDP is
 *                    therefore KEEP-WARM-AWARE: rst=RC_SDPKW (reset iff !ork_sdp_noreset(), i.e. only
 *                    when the ORK_SDP_NORESET skip is OFF), setdt=0 (no marker, leaves last_dt). This is
 *                    the op-local SDP reset expressed AS DATA — byte-identical to the historical inline
 *                    `if(!ork_sdp_noreset()) orki_act(RESET)`, default-SKIP so it does NOT re-introduce the
 *                    churn ORK_SSM_KEEPWARM removes. NEVER set XP_SDP to RC_ALWAYS (that forces the reset).
 *                    Wired via ork_npu_enter(c, c->last_dt, XP_SDP, OCK_NONE); SDP ops still not yet
 *                    converted keep the inline form (identical behavior) pending a Phase-2 sweep.
 * NUANCE #1 (kept SEPARATE, per Exp-2026-07-12): the c->task LUT-descriptor poisoning is a DISTINCT
 * axis from precision-mode and ACT_RESET does NOT fix it — the layer owns only the precision reset;
 * the c->task save/restore stays an op responsibility (no clr_task cell is wired in Phase 1). */
/* CHAINING MECHANISM in effect for a transition — passed as explicit state to ork_npu_enter so the
 * policy can branch on it for the few handoffs where the mechanism genuinely matters, and ignore it
 * (the common case) otherwise. OCK_NONE = plain per-matmul run / run_multicore / int4 batch;
 * OCK_SW = run_stream_* round-robin (multi-submit, per-core); OCK_HW = run_chain_i8 / chain_progs
 * PC-chain (one submit, task_number>1); OCK_FUSED = run_chain_i8_ffn static regcmd graph (carries
 * in-chain SDP/LUT ops — ping-pong/LUT-commit rules differ; that specialness lives in the chain body). */
int ork_npu_enter(ork_npu *c, int to, int profile, int chain){
    const struct ork_xspec *x=&XSPEC[profile];
    int from=c->last_dt, fd=c->fd;
    /* Record the chaining mechanism as transition state. For the common case it has NO bearing on the
     * precision-mode reset, so the XSPEC row below is mechanism-agnostic. When a transition is found to
     * need mechanism-specific handling (e.g. OCK_FUSED's in-chain LUT), branch on `chain` here — the
     * hook is intentionally explicit even though no entry-transition currently requires it (validated:
     * mode_probe + test_ssd_chunk_npu). See AGENTS.md §"Mode-transition layer". */
    c->last_chain = chain;
    if(orki_xprof<0){ const char*e=getenv("ORK_XPROF"); orki_xprof=(e&&atoi(e))?1:0; }
    if(orki_xprof){ int fi=from+1; if(fi<0)fi=0; if(fi>7)fi=7; orki_xcount[profile][fi]++; }
    if(x->setdt && from==to) return 0;                 /* mirrors the old `if(dt!=c->last_dt)` guard */
    int kw=0;
    switch(x->kwp){
      case KWP_MC:  kw=(ork_nothrash()&&ORK_INT_DT(from)&&ORK_INT_DT(to)) || (ork_f16warm()&&ORK_KW_DT(from)&&ORK_KW_DT(to)); break;
      case KWP_SC:  kw= ork_f16warm()&&ORK_KW_DT(from)&&ORK_KW_DT(to); break;
      case KWP_NTI: kw= ork_nothrash()&&ORK_INT_DT(from); break;
      case KWP_NTL: kw= ork_nothrash()&&ORK_I8_LIVE(from); break;
      case KWP_F16: kw= ork_f16warm()&&ORK_KW_DT(from); break;
      default:      kw=0;
    }
    int rst=0;
    switch(x->rst){
      case RC_NOTKW:         rst=!kw; break;
      case RC_I8ENTRY:       rst=(to==DT_I8 && !ORK_I8_LIVE(from) && !kw); break;
      case RC_NOTLIVE:       rst=!ORK_I8_LIVE(from); break;
      case RC_NOTLIVE_NOTKW: rst=(!ORK_I8_LIVE(from) && !kw); break;
      case RC_ALWAYS:        rst=1; break;
      case RC_SDPKW:         rst=!ork_sdp_noreset(); break;   /* transient SDP: reset only if the keep-warm skip is OFF (ORK_SDP_NORESET=0) — byte-identical to the old inline `if(!ork_sdp_noreset())` */
      default:               rst=0;
    }
    /* ORK_DEBUG_RESET attribution: the generic ACT_RESET log names only a return address, which is always
     * ork_npu_enter — useless for deciding WHICH XSPEC row is firing. Name the profile and the transition. */
    if(rst && getenv("ORK_DEBUG_RESET"))
        fprintf(stderr,"[ork XSPEC] reset profile=%d from=%d to=%d chain=%d\n", profile, from, to, chain);
    /* PROBE HOOK (percore_mode_probe): suppress the ACT_RESET only, leaving every other XSPEC effect
     * (warm/size clears, last_dt) intact — so the probe isolates "is the reset itself load-bearing?" */
    if(rst && orki_xspec_nores()) rst=0;
    if(rst){
        /* ORK_DEBUG_RESET prices the transition. MEASURED 2026-08-24 on the 27B mixed-tier run: mean
         * 105 ms per ACT_RESET, 160 of them = 16.8 s of a 161.6 s scored run (10.4%). This is the ioctl
         * alone -- the induced cold re-warm and per-core LUT reload land later in the run path -- so it is
         * a LOWER bound. 105 ms is also why a standalone SDP op can never live on the NPU: at a few hundred
         * activations per forward the transitions alone would dwarf the compute, which is what pushed
         * activations onto CPU/NEON and left in-chain fusion as the only viable on-NPU form. */
        if(getenv("ORK_DEBUG_RESET")){
            static double acc=0; static long nn=0;
            double t0=ork_now_us(); orki_act(fd,RKNPU_ACT_RESET,0); acc+=ork_now_us()-t0; nn++;
            fprintf(stderr,"[ork XSPEC-COST] n=%ld total=%.1fms mean=%.0fus\n", nn, acc/1000.0, acc/nn);
        } else orki_act(fd,RKNPU_ACT_RESET,0);
        for(int i=0;i<ORK_MAXCORE;i++){ c->chain_lut_devloaded[i]=0; c->chain_task_built[i]=0; }   /* a reset clears the SDP LUT SRAM (all cores) + the mode pipeline -> force a per-core reload and a task rebuild */
    }
    int wclr=0;
    switch(x->wc){
      case WC_NOTKW:         wclr=!kw; break;
      case WC_NOTLIVE_NOTKW: wclr=(!ORK_I8_LIVE(from) && !kw); break;
      case WC_ALWAYS:        wclr=1; break;
      default:               wclr=0;
    }
    if(orki_xspec_nocle()) wclr=0;
    if(wclr){ if(x->wtg&TG_SCALAR)c->warmed=0; if(x->wtg&TG_PERCORE)for(int i=0;i<ORK_MAXCORE;i++)c->mwarm[i]=0; }
    int sclr=0;
    switch(x->sc){
      case WC_NOTKW:    sclr=!kw; break;
      case WC_NT:       sclr=!ork_nothrash(); break;
      case WC_NT_NOTKW: sclr=(!ork_nothrash() && !kw); break;
      case WC_ALWAYS:   sclr=1; break;
      default:          sclr=0;
    }
    if(sclr){ if(x->stg&TG_SCALAR)c->ccsz=0; if(x->stg&TG_PERCORE)for(int i=0;i<ORK_MAXCORE;i++)c->mccsz[i]=0; }
    if(x->setdt) c->last_dt=to;
    return 1;
}
