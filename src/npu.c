/* npu.c — core regcmd matmul engine for the Rockchip NPU (see include/ork_npu.h).
 *
 * Raw DRM submission (no librknnrt): synthesizes a register-command program per matmul tile
 * and submits it to the `rknpu` kernel driver. fp16 A x fp16 B -> fp32 C. Tiling:
 *   - K split into <= soc.ks slices, partials accumulated (host-side, fp32);
 *   - N split into <= soc.nmax output-column slices (the NPU caps output width);
 *   - each slice M-tiled: clean power-of-2 Kp uses the single-submit internal M-scheduler,
 *     odd remainder Kp falls back to one internal M-tile per submit (correct for any Kp).
 * One reused feature buffer (the NPU caches feature state by address) + a one-time
 * cold-start warmup per fresh output buffer. SoC-specific numbers come from soc.h.
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include "rknpu_ioctl.h"
#include "regcmd_array_4x32x16.h"
#include "ork_npu.h"
#include "soc.h"
typedef ork_f16 f16;

struct buf { uint32_t handle; uint64_t dma, obj; void *cpu; size_t size; };
struct ork_npu { int fd; const struct ork_soc *soc; struct buf regcmd, task, Af, Cc; size_t ccsz; float *cres; size_t cressz; int warmed; };
struct ork_w   { int K, N, Sk, Sn; struct buf *Bb; };   /* Bb[ns*Sk + ks], K-split x N-split */

static size_t pgup(size_t s){return (s+4095)&~((size_t)4095);}
static struct buf bcreate(int fd,size_t size,uint32_t flags){
    struct rknpu_mem_create c; memset(&c,0,sizeof c); c.size=pgup(size); c.flags=flags; c.core_mask=RKNPU_CORE0_MASK;
    if(ioctl(fd,DRM_IOCTL_RKNPU_MEM_CREATE,&c)){perror("CREATE");return (struct buf){0};}
    struct rknpu_mem_map m; memset(&m,0,sizeof m); m.handle=c.handle;
    if(ioctl(fd,DRM_IOCTL_RKNPU_MEM_MAP,&m)){perror("MAP");return (struct buf){0};}
    void*p=mmap(NULL,c.size,PROT_READ|PROT_WRITE,MAP_SHARED,fd,m.offset);
    if(p==MAP_FAILED){perror("mmap");return (struct buf){0};}
    return (struct buf){c.handle,c.dma_addr,c.obj_addr,p,c.size};
}
static void bdestroy(int fd,struct buf*b){ if(!b->cpu)return; munmap(b->cpu,b->size);
    struct rknpu_mem_destroy d; memset(&d,0,sizeof d); d.handle=b->handle; d.obj_addr=b->obj; ioctl(fd,DRM_IOCTL_RKNPU_MEM_DESTROY,&d); b->cpu=0; }
static void bsync(int fd,struct buf*b,uint32_t f){struct rknpu_mem_sync s;memset(&s,0,sizeof s);s.obj_addr=b->obj;s.size=b->size;s.flags=f;ioctl(fd,DRM_IOCTL_RKNPU_MEM_SYNC,&s);}
static void act(int fd,uint32_t f,uint32_t v){struct rknpu_action a={.flags=f,.value=v};ioctl(fd,DRM_IOCTL_RKNPU_ACTION,&a);}
/* replace ALL matching regcmd entries — the template repeats some regs (e.g. 0x1040) and
 * the NPU uses a later copy, so a first-match-only patch leaves stale values. */
static void setr(uint32_t*rc,int n,uint32_t b,uint32_t o,uint32_t v){for(int k=0;k+1<n;k+=2)if((rc[k]&0xffff)==o&&(rc[k+1]>>16)==b){rc[k]=(o)|((v&0xffff)<<16);rc[k+1]=(b<<16)|((v>>16)&0xffff);}}
/* sched=1: single-submit internal M-scheduler (clean power-of-2 Kp); sched=0: one M-tile. */
static void synth(uint32_t*rc,int mc,int K,int N,uint32_t aA,uint32_t aB,uint32_t aC,int sched,int cbuf){
    memcpy(rc,REGCMD,REGCMD_N*4);
    setr(rc,REGCMD_N,0x201,0x1024,((K-1)<<16)|K);setr(rc,REGCMD_N,0x201,0x1030,K*N*2);setr(rc,REGCMD_N,0x201,0x1034,K*2);
    setr(rc,REGCMD_N,0x201,0x1044,K/32);setr(rc,REGCMD_N,0x201,0x1088,K);setr(rc,REGCMD_N,0x201,0x107c,K/8);
    setr(rc,REGCMD_N,0x201,0x1020,0x10000|mc);setr(rc,REGCMD_N,0x201,0x1084,0x10000|mc);setr(rc,REGCMD_N,0x201,0x102c,mc);
    setr(rc,REGCMD_N,0x1001,0x4034,mc-1);setr(rc,REGCMD_N,0x1001,0x405c,(mc-1)<<16);setr(rc,REGCMD_N,0x801,0x3014,(mc-1)<<16);
    setr(rc,REGCMD_N,0x1001,0x403c,((N-1)<<16)|(N-1));setr(rc,REGCMD_N,0x1001,0x4058,N-1);setr(rc,REGCMD_N,0x1001,0x4038,(((N/4)-1)<<16)|((N/4)-1));
    setr(rc,REGCMD_N,0x201,0x1038,0x1010000|N);setr(rc,REGCMD_N,0x801,0x3018,N-1);
    if(sched){
        int R=cbuf/K; if(R<1)R=1; int rows=(mc+1<R)?(mc+1):R; setr(rc,REGCMD_N,0x201,0x1010,16*rows);
        int kk=K/256,lg=0; while(kk>1){kk>>=1;lg++;} int base=0xb1-15*((1<<lg)-1),slope=15*(1<<lg),mg=mc/64; if(mg<1)mg=1;
        int v=base-slope*(mg-1); if(v<0x1b)v=0x1b; setr(rc,REGCMD_N,0x201,0x1040,v);
    } else { setr(rc,REGCMD_N,0x201,0x1010,16*(mc+1)); }
    setr(rc,REGCMD_N,0x201,0x1070,aA);setr(rc,REGCMD_N,0x201,0x1110,aB);setr(rc,REGCMD_N,0x1001,0x4020,aC);
}

ork_npu *ork_npu_init(void){
    const struct ork_soc *soc=ork_soc_detect();
    if(!soc){fprintf(stderr,"[ork] unknown SoC (no device-tree match) — cannot select NPU params\n");return NULL;}
    if(!soc->validated) fprintf(stderr,"[ork] WARNING: %s params are inherited/untested — validate with the regression suite\n",soc->id);
    const char*card=getenv("ORK_NPU_CARD"); if(!card)card=soc->card;
    int fd=open(card,O_RDWR); if(fd<0){perror("open NPU card");return NULL;}
    act(fd,RKNPU_GET_DRV_VERSION,0);act(fd,RKNPU_POWER_ON,0);act(fd,RKNPU_SET_PROC_NICE,(uint32_t)-19);
    ork_npu *c=calloc(1,sizeof *c); c->fd=fd; c->soc=soc;
    c->regcmd=bcreate(fd,4096,0x403); c->task=bcreate(fd,4096,0x40b); c->Af=bcreate(fd,(size_t)4*32768*2,0x403);
    struct rknpu_task t; memset(&t,0,sizeof t); t.enable_mask=0xd;t.int_mask=0x300;t.int_clear=0x1ffff;t.regcfg_amount=108;t.regcmd_addr=c->regcmd.dma;
    memcpy(c->task.cpu,&t,sizeof t); bsync(fd,&c->task,RKNPU_MEM_SYNC_TO_DEVICE|RKNPU_MEM_SYNC_FROM_DEVICE);
    if(!c->regcmd.cpu||!c->task.cpu||!c->Af.cpu){ork_npu_free(c);return NULL;}
    return c;
}
void ork_npu_free(ork_npu *c){ if(!c)return; int fd=c->fd;
    bdestroy(fd,&c->regcmd);bdestroy(fd,&c->task);bdestroy(fd,&c->Af);bdestroy(fd,&c->Cc);
    free(c->cres); if(fd>=0)close(fd); free(c); }
const char *ork_npu_soc(const ork_npu *c){return c->soc->id;}
int ork_npu_cores(const ork_npu *c){return c->soc->cores;}
int ork_npu_validated(const ork_npu *c){return c->soc->validated;}

ork_w *ork_mm_pack(ork_npu *c,int K,int N,const f16 *B){
    if(K%32||N%16) return NULL;
    int KS=c->soc->ks, NMAX=c->soc->nmax;
    int Sk=(K+KS-1)/KS, Sn=(N+NMAX-1)/NMAX;
    ork_w *w=calloc(1,sizeof *w); w->K=K;w->N=N;w->Sk=Sk;w->Sn=Sn; w->Bb=calloc((size_t)Sk*Sn,sizeof(struct buf));
    for(int ns=0;ns<Sn;ns++){int n0=ns*NMAX,Nc=(N-n0<NMAX)?(N-n0):NMAX,NN=Nc/16;
      for(int ks=0;ks<Sk;ks++){int k0=ks*KS,Kp=(K-k0<KS)?(K-k0):KS,KT=Kp/32;
        struct buf*b=&w->Bb[(size_t)ns*Sk+ks]; *b=bcreate(c->fd,(size_t)Kp*Nc*2,0x403); f16*bb=b->cpu;
        for(int nt=0;nt<NN;nt++)for(int kt=0;kt<KT;kt++)for(int nl=0;nl<16;nl++)for(int kk=0;kk<32;kk++)
            bb[nt*KT*16*32+kt*16*32+nl*32+kk]=B[(size_t)(k0+kt*32+kk)*N+(n0+nt*16+nl)];
        bsync(c->fd,b,RKNPU_MEM_SYNC_TO_DEVICE|RKNPU_MEM_SYNC_FROM_DEVICE);bsync(c->fd,b,RKNPU_MEM_SYNC_TO_DEVICE);}}
    return w;
}
void ork_w_free(ork_w *w){ if(!w)return; free(w->Bb); free(w); }   /* device buffers freed at ctx teardown */

int ork_mm_run(ork_npu *c,ork_w *w,int M,const f16 *A,float *C){
    int fd=c->fd,K=w->K,N=w->N, KS=c->soc->ks, NMAX=c->soc->nmax, CBUF=c->soc->cbuf_elems;
    size_t need=(size_t)M*N*4;
    if(c->cressz<need){c->cres=realloc(c->cres,need);c->cressz=need;}
    memset(c->cres,0,need);
    size_t maxout=0; for(int k0=0;k0<K;k0+=KS){int Kp=(K-k0<KS)?(K-k0):KS;int sd=((Kp&(Kp-1))==0);int R=CBUF/Kp;if(R<1)R=1;
        int chunk=sd?4*R:((CBUF/2)/Kp); if(chunk<1)chunk=1; int rows=chunk<M?chunk:M; int nc=N<NMAX?N:NMAX; size_t o=(size_t)rows*nc*4; if(o>maxout)maxout=o;}
    if(c->ccsz<maxout){bdestroy(fd,&c->Cc);c->Cc=bcreate(fd,maxout,0x403);c->ccsz=maxout;c->warmed=0; if(!c->Cc.cpu)return -1;}
    for(int ns=0;ns<w->Sn;ns++){int n0=ns*NMAX,Nc=(N-n0<NMAX)?(N-n0):NMAX;
      for(int ks=0;ks<w->Sk;ks++){int k0=ks*KS,Kp=(K-k0<KS)?(K-k0):KS;
        int sched=((Kp&(Kp-1))==0), R=CBUF/Kp; if(R<1)R=1; int chunk=sched?4*R:((CBUF/2)/Kp); if(chunk<1)chunk=1;
        struct buf*Bb=&w->Bb[(size_t)ns*w->Sk+ks];
        for(int m0=0;m0<M;m0+=chunk){int mc=(M-m0<chunk)?(M-m0):chunk; if(mc<=0)continue;
            f16*ad=c->Af.cpu; for(int r=0;r<mc;r++)for(int j=0;j<Kp;j++) ad[(size_t)r*Kp+j]=A[(size_t)(m0+r)*K+k0+j];
            bsync(fd,&c->Af,RKNPU_MEM_SYNC_TO_DEVICE);
            uint32_t rc[REGCMD_N]; synth(rc,mc,Kp,Nc,(uint32_t)c->Af.dma,(uint32_t)Bb->dma,(uint32_t)c->Cc.dma,sched,CBUF);
            memcpy(c->regcmd.cpu,rc,sizeof rc); bsync(fd,&c->regcmd,RKNPU_MEM_SYNC_TO_DEVICE);
            struct rknpu_submit sub;memset(&sub,0,sizeof sub);sub.flags=0x5;sub.timeout=6000;sub.task_number=1;sub.task_obj_addr=c->task.obj;sub.core_mask=RKNPU_CORE0_MASK;sub.fence_fd=-1;sub.subcore_task[0]=(struct rknpu_subcore_task){0,1};
            int reps=c->warmed?1:2;   /* cold-start warmup: first submit to a fresh output buffer is stale */
            for(int rep=0;rep<reps;rep++){ if(ioctl(fd,DRM_IOCTL_RKNPU_SUBMIT,&sub)){perror("SUBMIT");return -1;} bsync(fd,&c->Cc,RKNPU_MEM_SYNC_FROM_DEVICE); }
            c->warmed=1;
            float*cc=c->Cc.cpu; for(int r=0;r<mc;r++)for(int n=0;n<Nc;n++) c->cres[(size_t)(m0+r)*N+(n0+n)]+=cc[(size_t)r*Nc+n];
        }
      }
    }
    memcpy(C,c->cres,need); return 0;
}
