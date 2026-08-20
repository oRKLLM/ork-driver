/* tools/stream_prefetch_probe.c — DECISIVE measurement for the int4 prefetch-inflate streaming design.
 *
 * THE QUESTION: when an int4-stored weight is streamed through the 4 GiB NPU window, the per-swap work is
 *   FILL  (int4->int8 inflate + tile into a NON-mapped "staging" dma-buf, cacheable ~10GB/s, + dma-buf clean)
 *   MAP   (bare MEM_CREATE import of the staged buffer -> IOVA; zero-copy, no copy)
 *   SUBMIT(the NPU int8 matmul)
 * A double-buffered loop prefetches FILL(slot N+1) + MAP(slot N+1) on a background thread while the main
 * thread SUBMITs slot N. Does the prefetch FULLY HIDE the fill+map behind the submit?
 *   overlapped wall/swap  ~= t_submit  => fully hidden (streaming ~= resident speed)
 *   overlapped wall/swap  >  t_submit  => prep-bound; residual = overlapped - t_submit
 * speedup = serial / overlapped.  Crossover M = smallest M where overlapped ~= t_submit.
 *
 * Also: the int8 "fill-once-resident -> bare MAP-only" swap latency (MEM_CREATE map of an ALREADY-FILLED
 * dma-buf, NO fill) — isolating the map from the fill (the earlier fixed-slot bench bundled them).
 *
 * CORRECTNESS: the overlapped streaming output is checked bit-exact vs a CPU int8 reference (the int8
 * codes the int4 weight inflates to). No timing is reported without the correctness gate.
 *
 *   make stream_prefetch_probe
 *   sudo systemctl stop orkllm
 *   sudo taskset -c 4-7 ./stream_prefetch_probe
 *   sudo systemctl start orkllm
 *
 * Governors MUST be at max (DDR dmc=performance@2112MHz, A76 cpu4..7=performance@2.4GHz) — the bench warns.
 */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <pthread.h>
#include "ork_npu.h"

/* ---- staging-ring diagnostic helpers exported from npu.c (not in the public header) ---- */
struct ork_stage;
struct ork_stage *ork_stage_create(ork_npu *c, int K, int N);
void  ork_stage_fill (ork_npu *c, struct ork_stage *s, const ork_w *src);
int   ork_stage_map  (ork_npu *c, struct ork_stage *s);
int   ork_stage_run  (ork_npu *c, struct ork_stage *s, int M, const int8_t *A, int32_t *C);
void  ork_stage_unmap(ork_npu *c, struct ork_stage *s);
void  ork_stage_free (ork_npu *c, struct ork_stage *s);
void  ork_i8_slice_direct_inflate(const ork_w *w, int8_t *i8, int kind); /* nibble -> linear int8[N*K] */

static double now_us(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec*1e6+t.tv_nsec/1e3; }
static int cmp_d(const void*a,const void*b){ double x=*(const double*)a,y=*(const double*)b; return x<y?-1:x>y?1:0; }
static double median(double*v,int n){ qsort(v,n,sizeof(double),cmp_d); return v[n/2]; }

#define WARM 3
#define REPS 15

/* governor check (warn loudly — a parked DDR governor invalidates bandwidth-sensitive numbers) */
static int rl(const char*p,char*b,int c){ FILE*f=fopen(p,"r"); if(!f)return -1; if(!fgets(b,c,f)){fclose(f);return -1;} fclose(f); char*nl=strchr(b,'\n'); if(nl)*nl=0; return 0; }
static void check_gov(void){
    char g[64]; int warned=0; printf("=== governors ===\n");
    if(!rl("/sys/class/devfreq/dmc/governor",g,sizeof g)){ printf("  DDR dmc = %s\n",g); if(strcmp(g,"performance")){ printf("  *** WARN: DDR not performance — bandwidth numbers SUSPECT\n"); warned=1; } }
    for(int cpu=4;cpu<=7;cpu++){ char p[128]; snprintf(p,sizeof p,"/sys/devices/system/cpu/cpu%d/cpufreq/scaling_governor",cpu);
        if(!rl(p,g,sizeof g)){ if(strcmp(g,"performance")){ printf("  *** WARN: cpu%d not performance\n",cpu); warned=1; } } }
    if(!warned) printf("  OK: DDR + A76 at performance.\n"); printf("\n");
}

/* CPU int8 reference C[M,N] = A[M,K] x Bi8[K,N] (row-major) */
static void ref_i8(int M,int K,int N,const int8_t*A,const int8_t*Bi8,int32_t*C){
    for(int m=0;m<M;m++)for(int n=0;n<N;n++){ int32_t s=0; const int8_t*a=A+(size_t)m*K;
        for(int k=0;k<K;k++) s+=(int)a[k]*(int)Bi8[(size_t)k*N+n]; C[(size_t)m*N+n]=s; }
}

/* background FILL thread arg. Only the inflate+tile (pure CPU + dma-buf cache-sync ioctl on the dma-buf
 * fd) runs on the thread; MAP/RUN/UNMAP (DRM-fd ioctls + the worker pool) stay on the main thread —
 * concurrent ioctls on the single DRM fd serialize/contend badly (the NPU is single-stream). The design
 * only asks the EXPENSIVE inflate to overlap the prior submit; the map is cheap and stays serialized. */
struct job { ork_npu*c; struct ork_stage*s; const ork_w*src; };
static void* fill_thread(void*arg){ struct job*j=arg; ork_stage_fill(j->c,j->s,j->src); return NULL; }

/* one (shape) x sweep over M. NSRC distinct int4 source weights cycled (so each swap is a genuinely
 * different weight, like streaming experts/layers). NSRC kept small (resident int4 sources are compact). */
static int run_shape(ork_npu*c, int K, int N, const int*Ms, int nM, const char*label){
    int rc=0, NSRC=3;
    printf("  [%s] building %d int4 sources...\n",label,NSRC);
    /* build NSRC int4-packed source weights (compact, resident) + their inflated int8 + CPU ref material */
    ork_w **src=calloc(NSRC,sizeof*src);
    int8_t **Bi8=calloc(NSRC,sizeof*Bi8);                 /* inflated int8 codes (linear N*K), for ref */
    float *f32=malloc((size_t)N*K*sizeof(float)), *bsc=malloc((size_t)N*sizeof(float));
    if(!src||!Bi8||!f32||!bsc){ printf("  [%s] OOM\n",label); rc=1; goto done0; }
    for(int e=0;e<NSRC;e++){
        unsigned s=0x1234u+(unsigned)(K*131+N+e*7919);
        for(size_t i=0;i<(size_t)N*K;i++){ s=s*1664525u+1013904223u; float u=(s>>8)*(1.0f/16777216.0f); f32[i]=(u-0.5f)*0.2f; }
        src[e]=ork_i4a8_mm_pack(c,K,N,f32,bsc);
        if(!src[e]){ printf("  [%s] pack_i4a8 failed K=%d N=%d\n",label,K,N); rc=1; goto done; }
        /* inflate nibble store -> linear int8[N*K], then transpose to row-major Bi8[K*N] for the CPU ref */
        int8_t *lin=malloc((size_t)N*K); ork_i8_slice_direct_inflate(src[e],lin,0); /* UNIFORM (matches pack default) */
        Bi8[e]=malloc((size_t)K*N);
        for(int n=0;n<N;n++)for(int k=0;k<K;k++) Bi8[e][(size_t)k*N+n]=lin[(size_t)n*K+k];
        free(lin);
    }

    /* two staging slots for the double buffer */
    struct ork_stage *S0=ork_stage_create(c,K,N), *S1=ork_stage_create(c,K,N);
    if(!S0||!S1){ printf("  [%s] stage_create NULL (dma-heap absent?)\n",label); rc=2; if(S0)ork_stage_free(c,S0); if(S1)ork_stage_free(c,S1); goto done; }

    printf("  [%s] K=%d N=%d  (int4 store=%.1f MiB  int8 staged=%.1f MiB)\n",
           label,K,N,(double)((size_t)K*N/2)/1048576.0,(double)((size_t)K*N)/1048576.0);

    /* ---- isolate t_fill (inflate+tile into bare staging) and t_map (bare MEM_CREATE import) ---- */
    double samp[REPS];
    for(int i=0;i<WARM;i++){ ork_stage_fill(c,S0,src[i%NSRC]); }
    for(int i=0;i<REPS;i++){ double t=now_us(); ork_stage_fill(c,S0,src[i%NSRC]); samp[i]=now_us()-t; }
    double t_fill=median(samp,REPS);
    /* map/unmap pair timed; report the map half (the swap-time import). unmap separately. */
    for(int i=0;i<WARM;i++){ ork_stage_map(c,S0); ork_stage_unmap(c,S0); }
    for(int i=0;i<REPS;i++){ double t=now_us(); ork_stage_map(c,S0); samp[i]=now_us()-t; ork_stage_unmap(c,S0); }
    double t_map=median(samp,REPS);
    double samp2[REPS];
    for(int i=0;i<REPS;i++){ ork_stage_map(c,S0); double t=now_us(); ork_stage_unmap(c,S0); samp2[i]=now_us()-t; }
    double t_unmap=median(samp2,REPS);

    printf("    t_fill(inflate+tile+clean) med=%.0f us   t_map(MEM_CREATE import) med=%.0f us   t_unmap med=%.0f us\n",
           t_fill,t_map,t_unmap);
    printf("    %-5s | %10s | %12s | %12s | %8s | %s\n","M","t_submit","serial/swap","overlap/swap","speedup","gate (overlap vs submit)");

    int32_t *Cstd=NULL;
    for(int mi=0;mi<nM;mi++){
        int M=Ms[mi];
        int8_t *A=malloc((size_t)M*K); int32_t *C=malloc((size_t)M*N*4);
        if(!A||!C){ printf("    M=%d OOM\n",M); free(A);free(C); continue; }
        unsigned as=0x55+M; for(size_t i=0;i<(size_t)M*K;i++){ as=as*1103515245+12345; A[i]=(int8_t)(((as>>16)%5)-2); }
        /* precompute CPU ref per source ONCE (ref_i8 is O(M*K*N) scalar — must be OUT of the timed loop),
         * + a per-source captured-output buffer compared AFTER timing stops. */
        int32_t **Cref=malloc(NSRC*sizeof*Cref), **Cap=malloc(NSRC*sizeof*Cap);
        for(int e=0;e<NSRC;e++){ Cref[e]=malloc((size_t)M*N*4); Cap[e]=malloc((size_t)M*N*4); ref_i8(M,K,N,A,Bi8[e],Cref[e]); }

        /* t_submit: map slot 0 with src[0], time ork_stage_run alone */
        ork_stage_fill(c,S0,src[0]); ork_stage_map(c,S0);
        if(ork_stage_run(c,S0,M,A,C)){ printf("    M=%d run FAILED\n",M); ork_stage_unmap(c,S0); free(A);free(C);free(Cref); rc=1; continue; }
        for(int i=0;i<WARM;i++) ork_stage_run(c,S0,M,A,C);
        for(int i=0;i<REPS;i++){ double t=now_us(); ork_stage_run(c,S0,M,A,C); samp[i]=now_us()-t; }
        double t_sub=median(samp,REPS);
        ork_stage_unmap(c,S0);

        /* ---- SERIAL loop: for each of NSRC*reps swaps do fill->map->run->unmap sequentially ---- */
        int SW = NSRC*4;            /* swaps to average */
        double t0=now_us();
        for(int i=0;i<SW;i++){ struct ork_stage*s=(i&1)?S1:S0; const ork_w*w=src[i%NSRC];
            ork_stage_fill(c,s,w); ork_stage_map(c,s); ork_stage_run(c,s,M,A,C); ork_stage_unmap(c,s); }
        double t_serial=(now_us()-t0)/SW;

        /* ---- OVERLAPPED double-buffer: prefetch FILL of slot N+1 on a thread while MAP+SUBMIT slot N.
         * Pipeline: prime fill(slot0); then loop i: launch thread to fill the OTHER slot for swap i+1;
         * on main map+run+unmap the current (already-filled) slot; join; swap. The fill (expensive
         * inflate) of i+1 overlaps the map+submit of i. */
        int badref=0;
        struct ork_stage *cur=S0, *nxt=S1;
        ork_stage_fill(c,cur,src[0]);                    /* prime: slot for swap 0 is filled */
        t0=now_us();
        for(int i=0;i<SW;i++){
            pthread_t th; struct job j={c,nxt,src[(i+1)%NSRC]}; int launched=0;
            if(i+1<SW){ pthread_create(&th,NULL,fill_thread,&j); launched=1; }   /* prefetch fill(i+1) */
            ork_stage_map(c,cur);                         /* map+submit the current (pre-filled) slot */
            ork_stage_run(c,cur,M,A,C);
            ork_stage_unmap(c,cur);
            if(launched) pthread_join(th,NULL);
            memcpy(Cap[i%NSRC],C,(size_t)M*N*4);          /* capture output; compare AFTER timing (cheap memcpy) */
            struct ork_stage*tmp=cur; cur=nxt; nxt=tmp;   /* next is now filled, becomes current */
        }
        double t_over=(now_us()-t0)/SW;
        /* correctness gate (outside timing): each source's captured output bit-exact vs CPU ref */
        for(int e=0;e<NSRC;e++) if(memcmp(Cap[e],Cref[e],(size_t)M*N*4)){ badref++; }

        double speedup=t_serial/t_over;
        /* the serialized non-fill cost (map+submit+unmap) is the floor the overlap can reach if the fill
         * is FULLY hidden behind it. fill hidden <=> overlap ~= floor (and <  serial = floor+fill). */
        double floor_ns=t_map+t_sub+t_unmap;
        const char*gate; char gbuf[120];
        double resid=t_over-floor_ns;
        if(resid <= 0.10*floor_ns){ snprintf(gbuf,sizeof gbuf,"FILL HIDDEN (overlap~=map+submit+unmap=%.0f; submit alone=%.0f)",floor_ns,t_sub); gate=gbuf; }
        else { snprintf(gbuf,sizeof gbuf,"fill residual %.0f us/swap over floor(%.0f)",resid,floor_ns); gate=gbuf; }
        printf("    %-5d | %9.0f | %11.0f | %11.0f | %7.2fx | %s%s\n",
               M,t_sub,t_serial,t_over,speedup,gate, badref?"  *** CORRECTNESS FAIL":"");
        if(badref){ printf("      !! %d/%d sources mismatched CPU ref\n",badref,NSRC); rc=1; }
        for(int e=0;e<NSRC;e++){ free(Cref[e]); free(Cap[e]); } free(Cref); free(Cap);
        free(A);free(C);
    }
    (void)Cstd;
    ork_stage_free(c,S0); ork_stage_free(c,S1);
done:
    for(int e=0;e<NSRC;e++){ if(src&&src[e]) ork_mm_free(c,src[e]); if(Bi8&&Bi8[e]) free(Bi8[e]); }
done0:
    free(src); free(Bi8); free(f32); free(bsc);
    printf("\n");
    return rc;
}

/* the int8 "fill-once-resident -> bare MAP-only" swap latency: fill ONCE, then time MAP alone (no fill),
 * over many swaps. Isolates the map from the fill that the earlier fixed-slot bench bundled. */
static int run_int8_maponly(ork_npu*c,int K,int N){
    struct ork_stage*S=ork_stage_create(c,K,N);
    if(!S){ printf("  int8 map-only: stage_create NULL\n"); return 2; }
    /* fill once with arbitrary int4 src inflated to int8 (we only care about the map cost) */
    float *f32=malloc((size_t)N*K*sizeof(float)),*bsc=malloc((size_t)N*sizeof(float));
    for(size_t i=0;i<(size_t)N*K;i++) f32[i]=0.01f;
    ork_w*src=ork_i4a8_mm_pack(c,K,N,f32,bsc);
    ork_stage_fill(c,S,src);                              /* ONE fill; stays resident in the bare dma-buf */
    double samp[REPS];
    for(int i=0;i<WARM;i++){ ork_stage_map(c,S); ork_stage_unmap(c,S); }
    for(int i=0;i<REPS;i++){ double t=now_us(); ork_stage_map(c,S); samp[i]=now_us()-t; ork_stage_unmap(c,S); }
    double t_map=median(samp,REPS);
    printf("  int8 fill-once-resident, bare MAP-only K=%d N=%d: %.0f us/swap (MEM_CREATE import, NO fill)\n",K,N,t_map);
    ork_mm_free(c,src); ork_stage_free(c,S); free(f32); free(bsc);
    return 0;
}

/* ---- Phase 2 validation: ork_i4a8_mm_load_import + ork_stream_pool_* bit-exact vs CPU ref ---- */
static int validate_pool(ork_npu*c){
    int K=2048,N=512,M=8,rc=0;
    printf("=== Phase 2 validation (load_i4a8_import + ork_stream_pool_*) K=%d N=%d M=%d ===\n",K,N,M);
    float *f32=malloc((size_t)N*K*sizeof(float)),*bsc=malloc((size_t)N*sizeof(float));
    unsigned s=0xABCD; for(size_t i=0;i<(size_t)N*K;i++){ s=s*1664525u+1013904223u; f32[i]=((s>>8)*(1.0f/16777216.0f)-0.5f)*0.2f; }
    ork_w*wp=ork_i4a8_mm_pack(c,K,N,f32,bsc); if(!wp){ printf("  pack_i4a8 failed\n"); return 1; }
    size_t i4sz=ork_i4a8_w_dump(wp,NULL,0); void*i4blob=malloc(i4sz); ork_i4a8_w_dump(wp,i4blob,i4sz);
    /* CPU ref from the inflated int8 codes (row-major) */
    int8_t*lin=malloc((size_t)N*K); ork_i8_slice_direct_inflate(wp,lin,0);
    int8_t*Bi8=malloc((size_t)K*N); for(int n=0;n<N;n++)for(int k=0;k<K;k++) Bi8[(size_t)k*N+n]=lin[(size_t)n*K+k]; free(lin);
    int8_t*A=malloc((size_t)M*K); for(size_t i=0;i<(size_t)M*K;i++){ s=s*1103515245+12345; A[i]=(int8_t)(((s>>16)%5)-2); }
    int32_t*Cref=malloc((size_t)M*N*4),*C=malloc((size_t)M*N*4); ref_i8(M,K,N,A,Bi8,Cref);
    ork_mm_free(c,wp);

    /* (1) ork_i4a8_mm_load_import vs CPU ref */
    ork_w*wi=ork_i4a8_mm_load_import(c,K,N,i4blob,i4sz);
    if(!wi){ printf("  load_i4a8_import NULL (import unavailable)\n"); rc=2; }
    else { ork_i8_mm_run(c,wi,M,A,C); int bad=memcmp(C,Cref,(size_t)M*N*4)?1:0;
        printf("  load_i4a8_import: %s vs CPU ref\n",bad?"WRONG":"ok"); rc|=bad; ork_mm_free(c,wi); }

    /* (2) ork_stream_pool: add i4a8 entry, map/run/unmap thrice (cache-hit remap), check each */
    ork_stream_pool*pool=ork_stream_pool_create(c);
    if(!pool){ printf("  pool_create NULL (import unavailable)\n"); rc=(rc==2)?2:rc; goto done; }
    ork_stream_entry*e4=ork_i4a8_stream_pool_add(pool,K,N,i4blob,i4sz);
    if(!e4){ printf("  pool_add_i4a8 NULL\n"); rc=1; }
    else { int bad=0; for(int it=0;it<3;it++){ ork_stream_pool_map(pool,e4); ork_stream_pool_run(pool,e4,M,A,C);
            if(memcmp(C,Cref,(size_t)M*N*4)) bad++; ork_stream_pool_unmap(pool,e4); }
        printf("  pool i4a8 (3 map/run/unmap cycles): %s  (entry RAM=%.2f MiB)\n",bad?"WRONG":"ok",(double)ork_stream_entry_bytes(e4)/1048576.0); rc|=bad; }

    /* (3) ork_stream_pool: add i8 entry (from an int8 dump of the SAME weights) */
    ork_w*w8=ork_i8_mm_pack(c,K,N,Bi8); size_t i8sz=ork_w_dump(w8,NULL,0); void*i8blob=malloc(i8sz); ork_w_dump(w8,i8blob,i8sz); ork_mm_free(c,w8);
    ork_stream_entry*e8=ork_i8_stream_pool_add(pool,K,N,i8blob,i8sz);
    if(!e8){ printf("  pool_add_i8 NULL\n"); rc=1; }
    else { ork_stream_pool_map(pool,e8); ork_stream_pool_run(pool,e8,M,A,C); int bad=memcmp(C,Cref,(size_t)M*N*4)?1:0; ork_stream_pool_unmap(pool,e8);
        printf("  pool i8: %s vs CPU ref\n",bad?"WRONG":"ok"); rc|=bad; }
    free(i8blob);

    /* (4) remove + re-add (evict then reload), check still correct + no crash */
    if(e4){ ork_stream_pool_remove(pool,e4);
        ork_stream_entry*e4b=ork_i4a8_stream_pool_add(pool,K,N,i4blob,i4sz);
        if(e4b){ ork_stream_pool_map(pool,e4b); ork_stream_pool_run(pool,e4b,M,A,C); int bad=memcmp(C,Cref,(size_t)M*N*4)?1:0; ork_stream_pool_unmap(pool,e4b);
            printf("  pool evict+reload: %s\n",bad?"WRONG":"ok"); rc|=bad; } }
    ork_stream_pool_free(pool);
done:
    free(f32);free(bsc);free(i4blob);free(Bi8);free(A);free(Cref);free(C);
    printf("\n");
    return rc;
}

int main(void){
    setvbuf(stdout,NULL,_IONBF,0);                        /* stream output (else SIGINT loses buffered lines) */
    check_gov();
    ork_npu*c=ork_npu_init(); if(!c){ printf("ork_npu_init failed (need /dev/dri + rknpu; run as root / stop orkllm)\n"); return 1; }
    printf("SoC=%s cores=%d validated=%d  ork-driver %s\n\n",ork_npu_soc(c),ork_npu_cores(c),ork_npu_validated(c),ork_npu_version());
    printf("int4 prefetch-inflate streaming probe. FILL=inflate+tile into a NON-mapped staging dma-buf;\n");
    printf("MAP=bare MEM_CREATE import; SUBMIT=NPU matmul. Double-buffer prefetches FILL+MAP(N+1) behind\n");
    printf("SUBMIT(N). gate: overlap/swap ~= t_submit => fully hidden. median of %d reps.\n\n",REPS);

    int rc=0;
    rc |= validate_pool(c);                               /* Phase 2 correctness gate FIRST (fast) */

    int Ms[]={1,8,64,128,256,512}; int nM=6;

    /* 35B-A3B expert dims (int4-stored) */
    rc |= run_shape(c,2048,512, Ms,nM,"A3B gate/up 2048x512");
    rc |= run_shape(c,512,2048, Ms,nM,"A3B down    512x2048");
    /* backbone-layer-sized matmul */
    rc |= run_shape(c,2048,2048,Ms,nM,"backbone   2048x2048");

    printf("=== int8 fill-once-resident bare-map-only (map isolated from fill) ===\n");
    rc |= run_int8_maponly(c,2048,2048);
    rc |= run_int8_maponly(c,4096,4096);

    printf("\n%s\n", rc?(rc==2?"SKIPPED (dma-heap unavailable)":"FAIL"):"ALL OK");
    ork_npu_free(c);
    return rc==2?0:rc;
}
