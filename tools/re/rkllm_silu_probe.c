/* tools/re/rkllm_silu_probe.c — LD_PRELOAD shim that decodes EVERY NPU submit rkllm issues into ONE compact
 * line per op (dims + output-format registers + activation/LUT markers), so we can scan a full-model forward
 * (~1600 submits) and pinpoint how rkllm runs the FFN / SiLU — specifically whether it emits NON-int8 output
 * (0x40c0!=0x20 / 0x4010 PREC!=0) or fuses an activation LUT (0x4104 writes / silu regs), which would show how
 * it keeps activation precision (lower PPL) that our int8-only fused stage can't.
 *
 * Clean-room interoperability RE: observes the OPEN hardware register-command ISA the runtime programs (same
 * regcmd encoding as ork's regcmd_i8.h), NOT any proprietary code. Same interposition as regcmd_capture.c.
 *
 *   gcc -shared -fPIC -O2 -I<ork-driver/src> -o rkllm_silu_probe.so rkllm_silu_probe.c -ldl
 *   sudo env LD_PRELOAD=$PWD/rkllm_silu_probe.so LD_LIBRARY_PATH=<rkllm libdir> ./rkllm_bench <model> ...
 * Env: RKP_FILTER=act  -> print only ops with an activation LUT (0x4104) or non-int8 output (recommended)
 *      RKP_MAX=N        -> stop decoding after N submits
 *      RKP_DUMP_LUT=1   -> also hexdump the LUT-load regcmd (the 0x4104 curve) of the first activation op
 */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <stdint.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include "rknpu_ioctl.h"

#define MAXB 4096
struct ent { uint32_t handle; uint64_t dma, obj, size, off; void *cpu; };
static struct ent tab[MAXB]; static int nent = 0;
static int (*real_ioctl)(int, unsigned long, ...) = NULL;
static void *(*real_mmap)(void *, size_t, int, int, int, off_t) = NULL;
static void *(*real_mmap64)(void *, size_t, int, int, int, off_t) = NULL;
static int g_filter_act = 0, g_max = 0, g_dump_lut = 0, g_submits = 0, g_acts = 0;

__attribute__((constructor)) static void init(void){
    real_ioctl = (int(*)(int,unsigned long,...))dlsym(RTLD_NEXT,"ioctl");
    real_mmap  = (void*(*)(void*,size_t,int,int,int,off_t))dlsym(RTLD_NEXT,"mmap");
    real_mmap64= (void*(*)(void*,size_t,int,int,int,off_t))dlsym(RTLD_NEXT,"mmap64");
    g_filter_act = getenv("RKP_FILTER") && !strcmp(getenv("RKP_FILTER"),"act");
    g_max = getenv("RKP_MAX") ? atoi(getenv("RKP_MAX")) : 0;
    g_dump_lut = getenv("RKP_DUMP_LUT") != NULL;
    fprintf(stderr,"[rkp] loaded (filter=%s max=%d)\n", g_filter_act?"act":"all", g_max);
}
static struct ent *by_handle(uint32_t h){ for(int i=0;i<nent;i++) if(tab[i].handle==h) return &tab[i]; return NULL; }
static struct ent *by_off(uint64_t o){ for(int i=0;i<nent;i++) if(tab[i].off==o) return &tab[i]; return NULL; }
static struct ent *by_dma_range(uint64_t d, uint64_t *off){ uint32_t d32=(uint32_t)d;
    for(int i=0;i<nent;i++){ uint32_t b=(uint32_t)tab[i].dma; if(d32>=b && d32<b+tab[i].size){ *off=d32-b; return &tab[i]; } } return NULL; }
static void record_map(off_t off, void*p){ struct ent*e=by_off((uint64_t)off); if(e && p!=MAP_FAILED) e->cpu=p; }
void *mmap(void*a,size_t l,int pr,int fl,int fd,off_t o){ if(!real_mmap)real_mmap=(void*(*)(void*,size_t,int,int,int,off_t))dlsym(RTLD_NEXT,"mmap"); void*p=real_mmap(a,l,pr,fl,fd,o); record_map(o,p); return p; }
void *mmap64(void*a,size_t l,int pr,int fl,int fd,off_t o){ if(!real_mmap64)real_mmap64=(void*(*)(void*,size_t,int,int,int,off_t))dlsym(RTLD_NEXT,"mmap64"); void*p=real_mmap64(a,l,pr,fl,fd,o); record_map(o,p); return p; }

/* regcmd is (target,value) pairs: word[k]=reg16|(val_lo16<<16), word[k+1]=block16<<16|val_hi16 */
static uint32_t rc_get(const uint32_t*w,int n,uint16_t reg,int*found){
    for(int k=0;k+1<n;k+=2) if((w[k]&0xffff)==reg){ if(found)*found=1; return ((w[k]>>16)&0xffff)|((w[k+1]&0xffff)<<16); }
    if(found)*found=0; return 0;
}
static int rc_count(const uint32_t*w,int n,uint16_t reg){ int c=0; for(int k=0;k+1<n;k+=2) if((w[k]&0xffff)==reg)c++; return c; }

int ioctl(int fd, unsigned long req, ...){
    va_list ap; va_start(ap,req); void*arg=va_arg(ap,void*); va_end(ap);
    if(!real_ioctl) real_ioctl=(int(*)(int,unsigned long,...))dlsym(RTLD_NEXT,"ioctl");
    int ret=real_ioctl(fd,req,arg);
    if(req==DRM_IOCTL_RKNPU_MEM_CREATE){ struct rknpu_mem_create*m=arg;
        if(nent<MAXB){ tab[nent]=(struct ent){m->handle,m->dma_addr,m->obj_addr,m->size,0,NULL}; nent++; } }
    else if(req==DRM_IOCTL_RKNPU_MEM_MAP){ struct rknpu_mem_map*m=arg; struct ent*e=by_handle(m->handle); if(e)e->off=m->offset; }
    else if(req==DRM_IOCTL_RKNPU_SUBMIT){
        struct rknpu_submit*s=arg; int sid=g_submits++;
        if(g_max && sid>=g_max) return ret;
        struct ent*te=NULL; for(int i=0;i<nent;i++) if(tab[i].obj==s->task_obj_addr){ te=&tab[i]; break; }
        if(!te || !te->cpu) return ret;
        struct rknpu_task*t=(struct rknpu_task*)te->cpu;
        for(uint32_t i=0;i<s->task_number;i++){
            uint64_t roff=0; struct ent*re=by_dma_range(t[i].regcmd_addr,&roff); if(!re||!re->cpu) continue;
            const uint32_t*w=(const uint32_t*)((char*)re->cpu+roff); int n=(int)t[i].regcfg_amount*2;
            int f4104=rc_count(w,n,0x4104), f4108=0,f4010f=0,f40c0f=0,fK=0,fN=0;
            uint32_t r4010=rc_get(w,n,0x4010,&f4010f), r40c0=rc_get(w,n,0x40c0,&f40c0f);
            uint32_t r4050=rc_get(w,n,0x4050,NULL), r4038=rc_get(w,n,0x4038,NULL);
            uint32_t r4084=rc_get(w,n,0x4084,NULL), r4088=rc_get(w,n,0x4088,NULL);
            uint32_t r4108=rc_get(w,n,0x4108,&f4108);
            uint32_t Kr=rc_get(w,n,0x1024,&fK), Nr=rc_get(w,n,0x1038,&fN);
            int is_act = f4104>0 || f4108 || (f4010f && (r4010&3)!=0) || (f40c0f && r40c0!=0x20 && r40c0!=0);
            if(g_filter_act && !is_act) continue;
            if(is_act) g_acts++;
            fprintf(stderr,"[rkp S%04d T%u] en=0x%x rc=%u K=%u N=%u | out 4010=%08x 40c0=%02x 4050=%03x 4038=%08x | R %x>>%u | LUT(4104)=%d 4108=%s%s\n",
                sid,i,t[i].enable_mask,t[i].regcfg_amount, fK?(Kr&0xffff):0, fN?(Nr&0xffff):0,
                r4010, r40c0&0xff, r4050&0xfff, r4038, r4084, r4088, f4104, f4108?"y":"n",
                is_act?"  <== ACTIVATION":"");
            if(g_dump_lut && f4104>0 && g_acts<=1){
                fprintf(stderr,"  [rkp LUT-load regcmd, %d words]:\n",n);
                for(int k=0;k<n && k<80;k+=4) fprintf(stderr,"    %08x %08x %08x %08x\n",w[k],w[k+1<n?k+1:0],w[k+2<n?k+2:0],w[k+3<n?k+3:0]);
            }
        }
    }
    return ret;
}
