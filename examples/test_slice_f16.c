/* test_slice_f16 — the fp16 shape planner must agree with the runtime, exactly.
 *
 * The planner's only job is to answer "what will the run path do?" before the run path does it. A
 * planner that disagrees is worse than no planner: it licenses a shape the hardware refuses, which
 * is how r103 reached a training run in the first place. So the central assertion here is not that
 * the planner is self-consistent — it is that ork_f16_mm_mmax returns the SAME step the runtime
 * tiles to, at the shapes where the envelope actually bends.
 *
 * NO NPU WORK. This runs entirely on an ork_npu_init_offline() context: soc caps, no device. That
 * is deliberate and is the feature being tested — model design (FFN width, expert count, head dim)
 * has to be answerable on a workstation, not by submitting to a board and seeing if it wedges.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "ork_npu.h"
#include "ork_slice_f16.h"

static int fails = 0;

static void expect(const char *what, long got, long want)
{
    int ok = (got == want);
    printf("  %-46s got %-6ld want %-6ld %s\n", what, got, want, ok ? "ok" : "FAIL");
    if (!ok) fails++;
}

static int mmax_cb(int K, int N, void *ud){ return ork_f16_mm_mmax((const ork_npu *)ud, K, N); }

int main(void)
{
    ork_npu *c = ork_npu_init_offline("rk3588");
    if (!c) { printf("ork_npu_init_offline(rk3588) failed\n"); return 1; }
    printf("fp16 shape planner, OFFLINE (no device, no board)\n\n");

    /* 1. The envelope, at the shapes r103 proved. These are the numbers the runtime enforces. */
    printf("1. ork_f16_mm_mmax matches the measured envelope\n");
    expect("K=1024 N=16   (narrow: r103 hang boundary)", ork_f16_mm_mmax(c,1024,16),   128);
    expect("K=1024 N=32   (narrow: also hung)",          ork_f16_mm_mmax(c,1024,32),   128);
    expect("K=1024 N=3584 (wide: measured good)",        ork_f16_mm_mmax(c,1024,3584), 176);
    expect("K=512  N=16   (narrow)",                     ork_f16_mm_mmax(c,512,16),    320);
    expect("K=512  N=3584 (wide)",                       ork_f16_mm_mmax(c,512,3584),  352);
    expect("K=256  N=16   (narrow)",                     ork_f16_mm_mmax(c,256,16),    640);
    expect("K=256  N=3584 (wide)",                       ork_f16_mm_mmax(c,256,3584),  704);

    /* 2. Shapes the hardware cannot run must be REFUSED, not quietly rounded. The pack entry point
     *    accepts on K%32/N%16 alone, so these are precisely the cases a caller could not detect. */
    printf("\n2. unrunnable shapes are refused (<=0), not rounded\n");
    struct { const char *w; int K, N; } bad[] = {
        {"K=1000 (K%32 != 0)",      1000, 16},
        {"N=56   (N%16 != 0)",      1024, 56},
        {"N=0",                     1024,  0},
        {"K=0",                        0, 16},
        {"N beyond the Sn==1 cap",  1024, 1<<20},
    };
    for (unsigned i = 0; i < sizeof bad/sizeof bad[0]; i++) {
        int r = ork_f16_mm_mmax(c, bad[i].K, bad[i].N);
        printf("  %-46s -> %-6d %s\n", bad[i].w, r, r <= 0 ? "ok" : "FAIL");
        if (r > 0) fails++;
    }

    /* 3. A plan's program count must equal what the tiling actually produces, and the M step must
     *    vary per (K-slice, N-tile) — a planner that hoisted the M loop would get this wrong. */
    printf("\n3. plan program count == materialized tile count\n");
    ork_slice_f16_caps caps = ork_slice_f16_caps_rk3588();
    caps.mmax = mmax_cb; caps.ud = c;
    struct { int M, K, N; } sh[] = {
        {1024, 1024,   16},   /* the GDN in_proj_b shape */
        {4096, 1024, 3584},   /* a prefill FFN */
        { 256, 3584, 1024},   /* the down-projection */
        {   1, 1024,   16},   /* decode */
        {4096, 8192, 1024},   /* wide K: multiple K-slices */
    };
    ork_tile *t = malloc(sizeof *t * 65536);
    for (unsigned i = 0; i < sizeof sh/sizeof sh[0]; i++) {
        int want = ork_slice_f16_count(sh[i].K, sh[i].N, sh[i].M, &caps);
        int got  = ork_slice_f16_matmul(sh[i].K, sh[i].N, sh[i].M, &caps, t, 65536);
        int viaapi = -1; ork_f16_mm_plan(c, sh[i].M, sh[i].K, sh[i].N, &viaapi);
        char w[80]; snprintf(w,sizeof w,"M=%d K=%d N=%d",sh[i].M,sh[i].K,sh[i].N);
        int ok = (want == got && got == viaapi && got > 0);
        printf("  %-46s count %-6d tiles %-6d api %-6d %s\n", w, want, got, viaapi, ok?"ok":"FAIL");
        if (!ok) fails++;
        /* the tiles must actually cover [0,M)x[0,N) exactly once per K-slice */
        if (got > 0) {
            long cover = 0;
            for (int j = 0; j < got; j++) cover += (long)(t[j].m1-t[j].m0) * (t[j].n1-t[j].n0);
            long kparts = 0; { int ks = (caps.kmax/caps.kmul)*caps.kmul;
                kparts = (sh[i].K + ks - 1)/ks; }
            long want_cover = (long)sh[i].M * sh[i].N * kparts;
            if (cover != want_cover) {
                printf("    coverage %ld != %ld (M*N*Kparts) FAIL\n", cover, want_cover); fails++; }
        }
    }
    free(t);

    /* 4. No mmax callback must refuse everything rather than default to a constant — inventing a
     *    safe-looking number is the original bug (an int8 budget reused for fp16). */
    printf("\n4. a caps struct with no mmax rule refuses, it does not guess\n");
    ork_slice_f16_caps nocb = ork_slice_f16_caps_rk3588();
    expect("check() with mmax=NULL", ork_slice_f16_check(1024, 1024, &nocb), -3);

    /* 5. The caps identity must change when the rule changes and must be stable otherwise — it is
     *    what a pack stamp compares, so a false match is a pack running against wrong caps. */
    printf("\n5. resolved-caps identity is stable, and discriminates\n");
    unsigned id1 = ork_slice_f16_capid(&caps, "rk3588");
    unsigned id2 = ork_slice_f16_capid(&caps, "rk3588");
    unsigned id3 = ork_slice_f16_capid(&caps, "rk3576");
    ork_slice_f16_caps other = caps; other.kmax = 1024;
    unsigned id4 = ork_slice_f16_capid(&other, "rk3588");
    printf("  %-46s %s\n", "same caps, same soc -> same id",  id1==id2 ? "ok" : "FAIL");
    printf("  %-46s %s\n", "different soc       -> different", id1!=id3 ? "ok" : "FAIL");
    printf("  %-46s %s\n", "different K-slice   -> different", id1!=id4 ? "ok" : "FAIL");
    printf("  %-46s %s\n", "never zero (0 means unstamped)",   id1&&id3&&id4 ? "ok" : "FAIL");
    if (!(id1==id2 && id1!=id3 && id1!=id4 && id1 && id3 && id4)) fails++;

    ork_npu_free(c);
    printf("\n%s\n", fails==0 ? "SLICE F16 OK" : "SLICE F16 FAILED");
    return fails != 0;
}
