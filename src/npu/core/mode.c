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
    if(rst){ orki_act(fd,RKNPU_ACT_RESET,0); for(int i=0;i<ORK_MAXCORE;i++){ c->chain_lut_devloaded[i]=0; c->chain_task_built[i]=0; } }   /* a reset clears the SDP LUT SRAM (all cores) + the mode pipeline -> force a per-core reload and a task rebuild */
    int wclr=0;
    switch(x->wc){
      case WC_NOTKW:         wclr=!kw; break;
      case WC_NOTLIVE_NOTKW: wclr=(!ORK_I8_LIVE(from) && !kw); break;
      case WC_ALWAYS:        wclr=1; break;
      default:               wclr=0;
    }
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
