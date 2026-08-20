/* dom_switch_probe.c — measure the per-submit IOMMU-domain-SWITCH cost.
 *
 * The fused-FFN "fc.wg in its own domain" layout (Option D) makes every layer's gate submit jump to a
 * dedicated fc.wg domain and back, turning ~5 domain switches across 28 layers into ~56. dom_activate
 * itself is a cheap userspace scratch-handle swap (npu.c), but each submit carries iommu_domain_id and the
 * kernel rknpu driver re-attaches the domain (TTBR reprogram + TLB flush) whenever it changes. This probe
 * isolates THAT cost: run one FIXED tiny matmul repeatedly, (a) STAYING in one domain vs (b) ALTERNATING
 * two domains every submit. delta(median) = per-switch kernel attach cost. Small shape => compute is a
 * near-constant floor so the delta is the switch, not the math.
 *   ./dsp [reps=300] [K=512] [N=512] [M=1]
 */
#define _GNU_SOURCE
#include "ork_npu.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static long ns_now(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec*1000000000L+t.tv_nsec; }
static int cmp_long(const void*a,const void*b){ long x=*(const long*)a,y=*(const long*)b; return x<y?-1:x>y?1:0; }
static void stats(const char*tag,long*v,int n){
    qsort(v,n,sizeof(long),cmp_long);
    double sum=0; for(int i=0;i<n;i++) sum+=v[i];
    printf("  %-18s n=%d  min=%.2fus  median=%.2fus  mean=%.2fus  p90=%.2fus\n",
           tag,n, v[0]/1000.0, v[n/2]/1000.0, (sum/n)/1000.0, v[(int)(n*0.9)]/1000.0);
}

int main(int argc,char**argv){
    int reps = argc>1?atoi(argv[1]):300;
    int K    = argc>2?atoi(argv[2]):512;
    int N    = argc>3?atoi(argv[3]):512;
    int M    = argc>4?atoi(argv[4]):1;
    ork_npu *c = ork_npu_init();
    if(!c){ printf("no board\n"); return 0; }
    printf("[dom_switch_probe] reps=%d shape K=%d N=%d M=%d\n",reps,K,N,M);

    int8_t *B=calloc((size_t)K*N,1); for(size_t i=0;i<(size_t)K*N;i++) B[i]=(int8_t)((i%7)-3);
    int8_t *A=calloc((size_t)M*K,1); for(size_t i=0;i<(size_t)M*K;i++) A[i]=(int8_t)((i%5)-2);
    int32_t *C=calloc((size_t)M*N,4);

    ork_npu_set_pack_domain(c,0); ork_w *w0=ork_i8_mm_pack(c,K,N,B);
    ork_npu_set_pack_domain(c,1); ork_w *w1=ork_i8_mm_pack(c,K,N,B);
    if(!w0||!w1){ printf("pack failed (w0=%p w1=%p)\n",(void*)w0,(void*)w1); return 1; }
    printf("  w0 dom=%d  w1 dom=%d\n", ork_w_domain(w0), ork_w_domain(w1));

    /* warm both domains: first-touch scratch alloc + NPU warm (mode/regcmd) so timing excludes one-time cost */
    for(int i=0;i<40;i++){ ork_i8_mm_run(c,w0,M,A,C); ork_i8_mm_run(c,w1,M,A,C); }

    long *same=malloc(sizeof(long)*reps), *alt=malloc(sizeof(long)*reps);

    /* SAME: stay in domain 0 every submit — no dom switch */
    for(int i=0;i<reps;i++){ long t0=ns_now(); ork_i8_mm_run(c,w0,M,A,C); same[i]=ns_now()-t0; }

    /* ALT: alternate w0(dom0) / w1(dom1) — EVERY submit changes iommu_domain_id */
    for(int i=0;i<reps;i++){ ork_w*w=(i&1)?w1:w0; long t0=ns_now(); ork_i8_mm_run(c,w,M,A,C); alt[i]=ns_now()-t0; }

    printf("[result]\n");
    stats("same-domain",same,reps);
    stats("alternating",alt,reps);
    /* switch cost ~= median(alt) - median(same); alt averages one switch per submit */
    qsort(same,reps,sizeof(long),cmp_long); qsort(alt,reps,sizeof(long),cmp_long);
    double dmed=(alt[reps/2]-same[reps/2])/1000.0, dmin=(alt[0]-same[0])/1000.0;
    printf("  => per-switch cost ~ %.2fus (median delta), %.2fus (min delta)\n",dmed,dmin);
    printf("  context: fused Option D adds ~2 switches/layer x 28 layers = ~56 switches/token\n");
    printf("           => ~%.2fus/token added by fc.wg-domain isolation (median)\n", dmed*56.0);

    ork_npu_free(c);
    return 0;
}
