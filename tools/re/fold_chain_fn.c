
/* #39 WEIGHT-RESIDENT FOLD CHAIN: replay a captured fold regcmd for P width-`w` row-tiles in ONE
 * task_number=P submit, sharing ONE weight buffer (loaded once) so the chain amortizes the K*N weight DMA
 * across all P tiles (rkllm's weight-resident row-tile idea). Chain descriptor mirrors run_chain_i8
 * (words 216-219: 0x0010=next-regcmd-addr, 0x0014=0x37; terminal program clears them). Apacked = P tiles,
 * each `tileAbytes` (nc16 width-w); Bpacked = woff weight (shared, Bbytes); Craw = P*(w*N) int32 (c4).
 * us = avg total-submit over iters. Single-core (core0). 0/ok, -2 bad shape, <0 submit error. */
int ork_npu_replay_i8_chain(ork_npu *c, const uint32_t *regcmd, int rn, int w, int K, int N,
                            const signed char *Apacked, int tileAbytes, const signed char *Bpacked, int Bbytes,
                            int P, int *Craw, int iters, double *us){
    int fd=c->fd; if(fd<0) return -3; if(rn<8||rn>2048||P<1||P>128||w<1||(K%32)||(N%16)) return -2;
    int dom=c->dom_active;
    size_t tileC=(size_t)w*N;                                   /* per-tile C int32 elems (c4) */
    size_t bsz=(Bbytes>(int)((size_t)K*N))?(size_t)Bbytes:(size_t)K*N;
    size_t aszg=(size_t)P*tileAbytes+(1u<<20), bszg=bsz*8+(1u<<20), cszg=(size_t)P*tileC*4*2+65536;
    struct buf A =bcreate(fd,aszg,0x403,dom);                       if(!A.cpu)  return -2;
    struct buf B =bcreate(fd,bszg,0x403,dom);                       if(!B.cpu) {bdestroy(fd,&A);return -2;}
    struct buf Cc=bcreate(fd,cszg,0x403,dom);                       if(!Cc.cpu){bdestroy(fd,&A);bdestroy(fd,&B);return -2;}
    struct buf RCb=bcreate(fd,(size_t)P*rn*4,0x403,dom);           if(!RCb.cpu){bdestroy(fd,&A);bdestroy(fd,&B);bdestroy(fd,&Cc);return -2;}
    struct buf TKb=bcreate(fd,(size_t)P*sizeof(struct rknpu_task),0x403,dom); if(!TKb.cpu){bdestroy(fd,&A);bdestroy(fd,&B);bdestroy(fd,&Cc);bdestroy(fd,&RCb);return -2;}
    memset(A.cpu,0,aszg); memset(B.cpu,0,bszg); memset(Cc.cpu,0,cszg);
    if(Apacked) memcpy(A.cpu,Apacked,(size_t)P*tileAbytes);
    if(Bpacked) memcpy(B.cpu,Bpacked,(Bbytes>0)?(size_t)Bbytes:(size_t)K*N);
    bsync(fd,&A,RKNPU_MEM_SYNC_TO_DEVICE); bsync(fd,&B,RKNPU_MEM_SYNC_TO_DEVICE); bsync(fd,&Cc,RKNPU_MEM_SYNC_TO_DEVICE);
    uint32_t *rc0=(uint32_t*)RCb.cpu; struct rknpu_task *tk=(struct rknpu_task*)TKb.cpu;
    for(int t=0;t<P;t++){
        uint32_t *rc=rc0+(size_t)t*rn; memcpy(rc,regcmd,(size_t)rn*4);
        setrn(rc,rn,RK_CNA_FEATURE_DATA_ADDR,(uint32_t)(A.dma+(size_t)t*tileAbytes));   /* per-tile A */
        setrn(rc,rn,RK_CNA_WEIGHT_DATA_ADDR,(uint32_t)B.dma);                            /* SHARED weight (resident) */
        setrn(rc,rn,RK_DPU_DST_BASE_ADDR,(uint32_t)(Cc.dma+(size_t)t*tileC*4));          /* per-tile C */
        /* the captured regcmd's words 216.. are the RKDUMP +16 margin = the NEXT task's regcmd bleeding in
         * (0x1040/0x100c/...). regcfg_amount=108 means only words 0-215 are the tile's regs; zero the rest so
         * the chain walk doesn't execute the bleed, then write ONLY our chain descriptor at 216-219. */
        for(int z=216;z<rn;z++) rc[z]=0;
        if(t+1<P){ uint64_t nx=RCb.dma+(size_t)(t+1)*rn*4;
            rc[216]=0x0010|((nx&0xffff)<<16); rc[217]=(0x0101u<<16)|((nx>>16)&0xffff);
            rc[218]=0x0014|(0x0037u<<16);     rc[219]=(0x0101u<<16)|0; }
        /* else terminal: 216.. already zeroed */
        memset(&tk[t],0,sizeof tk[t]);
        tk[t].enable_mask=0xd; tk[t].int_mask=0x300; tk[t].int_clear=0x1ffff; tk[t].regcfg_amount=108;
        tk[t].regcmd_addr=RCb.dma+(size_t)t*rn*4;
    }
    bsync(fd,&RCb,RKNPU_MEM_SYNC_TO_DEVICE); bsync(fd,&TKb,RKNPU_MEM_SYNC_TO_DEVICE|RKNPU_MEM_SYNC_FROM_DEVICE);
    int ret=-1; struct rknpu_submit sub;
    #define _FCSUB() do{ memset(&sub,0,sizeof sub); sub.flags=ork_ppflags(); sub.task_number=(uint32_t)P; \
        sub.task_obj_addr=TKb.obj; sub.core_mask=RKNPU_CORE0_MASK; sub.fence_fd=-1; sub.timeout=mm_timeout_ms(); \
        sub.subcore_task[0]=(struct rknpu_subcore_task){0,(uint32_t)P}; }while(0)
    _FCSUB(); if(rknpu_submit_ioctl(fd,&sub,dom)){ goto done; }        /* warm */
    bsync(fd,&Cc,RKNPU_MEM_SYNC_FROM_DEVICE);
    if(Craw) memcpy(Craw,Cc.cpu,(size_t)P*tileC*4);
    { double t0=ork_now_us(); for(int i=0;i<iters;i++){ _FCSUB(); if(rknpu_submit_ioctl(fd,&sub,dom)){ goto done; } }
      if(us) *us=(ork_now_us()-t0)/(iters>0?iters:1); ret=0; }
    #undef _FCSUB
done:
    bdestroy(fd,&A); bdestroy(fd,&B); bdestroy(fd,&Cc); bdestroy(fd,&RCb); bdestroy(fd,&TKb); return ret;
}
