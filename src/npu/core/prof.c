/* npu/core/prof.c — profiling and timing accessors: the per-run / per-core / submit-floor counters and
 * the dump helpers. No dtype in any contract; see npu/core.h. Lifted verbatim from npu.c. */
#define _GNU_SOURCE   /* CPU_SET/pthread_setaffinity_np, as npu.c does */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>
#include <sys/prctl.h>
#include "ork_regs.h"
#include "regcmd_i8.h"
#include "npu/internal.h"
#include "npu/core.h"

void ork_load_prof_dump(void){
    if(!orki_load_prof) return;
    double tot=orki_lp_alloc+orki_lp_mmap+orki_lp_prime+orki_lp_create+orki_lp_memcpy+orki_lp_bf; if(tot<=0) tot=1;
    fprintf(stderr,"[ork LOAD_PROF] %ld chunks, %.2f GiB imported, %.2f s in import path:\n",
            orki_lp_nchunk, orki_lp_bytes/(1024.0*1024.0*1024.0), tot/1e6);
    fprintf(stderr,"  dma_heap_alloc %.2fs (%.0f%%) | mmap %.2fs (%.0f%%) | prime_fd %.2fs (%.0f%%) | mem_create %.2fs (%.0f%%) | memcpy(Bb) %.2fs (%.0f%%) | retile(Bf) %.2fs (%.0f%%)\n",
            orki_lp_alloc/1e6,100*orki_lp_alloc/tot, orki_lp_mmap/1e6,100*orki_lp_mmap/tot, orki_lp_prime/1e6,100*orki_lp_prime/tot,
            orki_lp_create/1e6,100*orki_lp_create/tot, orki_lp_memcpy/1e6,100*orki_lp_memcpy/tot, orki_lp_bf/1e6,100*orki_lp_bf/tot);
    orki_lp_alloc=orki_lp_mmap=orki_lp_prime=orki_lp_create=orki_lp_memcpy=orki_lp_bf=0; orki_lp_nchunk=0; orki_lp_bytes=0;
}

void ork_npu_mc_reset(void){ for(int i=0;i<MCPROF_MAX;i++){orki_mc_copy[i]=orki_mc_sub[i]=orki_mc_acc[i]=orki_mc_synth[i]=0;orki_mc_n[i]=0;} }

void ork_npu_mc_timing(int core,double*copy,double*sub,double*acc,long*n){
    if(copy)*copy=orki_mc_copy[core]; if(sub)*sub=orki_mc_sub[core]; if(acc)*acc=orki_mc_acc[core]; if(n)*n=orki_mc_n[core]; }

double ork_npu_mc_synth(int core){ return (core>=0&&core<MCPROF_MAX)?orki_mc_synth[core]:0; }

void ork_npu_run_timing(double*setup,double*submit,double*copy,long*n){ if(setup)*setup=orki_rt_setup; if(submit)*submit=orki_rt_submit; if(copy)*copy=orki_rt_copy; if(n)*n=orki_rt_n; }

void ork_npu_floor_timing(double*ioctl_us,double*hw_us,long long*hw_raw_last,long*n){
    if(ioctl_us)*ioctl_us=orki_fd_ioctl_us; if(hw_us)*hw_us=orki_fd_hw_us; if(hw_raw_last)*hw_raw_last=orki_fd_hw_raw_last; if(n)*n=orki_fd_n; }

void ork_npu_floor_reset(void){ orki_fd_ioctl_us=0; orki_fd_hw_us=0; orki_fd_hw_raw_last=0; orki_fd_n=0; }

void ork_npu_xprof_dump(void){
    if(orki_xprof<=0) return;
    fprintf(stderr,"[ork XPROF] transition counts (profile <- from):\n");
    for(int p=0;p<XP_NPROFILE;p++){ long tot=0; for(int f=0;f<8;f++) tot+=orki_xcount[p][f]; if(!tot) continue;
        fprintf(stderr,"  %-11s:",orki_XPNAME[p]); for(int f=0;f<8;f++) if(orki_xcount[p][f]) fprintf(stderr," %s=%ld",orki_XFROM[f],orki_xcount[p][f]); fprintf(stderr,"\n"); }
}
