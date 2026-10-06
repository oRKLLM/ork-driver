/* test_transpose — ork_f16_transpose / ork_f32_transpose_to_f16 against a scalar reference.
 *
 * The examples ARE the test suite, so this self-validates and exits 0/nonzero. It is pure host
 * work: no NPU, no context, no board required to be idle — but it lives here, not in tools/,
 * because it gates a CORE source that shapes the operands the NPU consumes.
 *
 * WHAT IT ASSERTS, and why each case exists:
 *   - ELEMENTWISE equality with the naive loop, not a tolerance. The NEON path only moves and
 *     narrows; it must not perturb a single value. A tolerance would hide exactly the bug this
 *     found during development: a ZIP-instead-of-TRN stage produces a permutation that is wrong
 *     in placement while being numerically plausible everywhere.
 *   - RAGGED shapes (not multiples of 8, 4, or the 64-wide block) on both entrypoints, since the
 *     vector body handles R&~7 / C&~7 and leaves two scalar edge strips. 1xN and Nx1 included:
 *     a transpose of a vector is where an R/C mix-up shows up immediately.
 *   - ROUND-TRIP: transposing twice returns the original, which catches an index swap that is
 *     self-consistent within one direction.
 *   - BAD ARGS, including the aliasing case, must return -1 rather than corrupt.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "ork_npu.h"

static unsigned st = 7u;
static float rnd(void){ st = st*1103515245u + 12345u; return (float)((int)((st>>13)&0xffff)-32768)/32768.0f; }

static int fails = 0;

static void chk16(int R, int C)
{
    size_t n = (size_t)R * C;
    ork_f16 *src = malloc(n*sizeof *src), *got = malloc(n*sizeof *got),
            *ref = malloc(n*sizeof *ref), *back = malloc(n*sizeof *back);
    if (!src || !got || !ref || !back) { printf("  alloc failed\n"); fails++; return; }
    for (size_t i = 0; i < n; i++) src[i] = (ork_f16)rnd();
    for (int r = 0; r < R; r++) for (int c = 0; c < C; c++)      /* scalar reference */
        ref[(size_t)c*R + r] = src[(size_t)r*C + c];
    memset(got, 0, n*sizeof *got);
    int rc = ork_f16_transpose(got, src, R, C);
    long bad = 0;
    for (size_t i = 0; i < n; i++) if ((float)got[i] != (float)ref[i]) bad++;
    /* round trip: (A^T)^T == A */
    ork_f16_transpose(back, got, C, R);
    long rbad = 0;
    for (size_t i = 0; i < n; i++) if ((float)back[i] != (float)src[i]) rbad++;
    printf("  f16 [%4d,%4d]  rc=%d  mismatches %ld  round-trip %ld   %s\n",
           R, C, rc, bad, rbad, (rc==0 && !bad && !rbad) ? "ok" : "FAIL");
    if (rc || bad || rbad) fails++;
    free(src); free(got); free(ref); free(back);
}

static void chk32(int R, int C)
{
    size_t n = (size_t)R * C;
    float   *src = malloc(n*sizeof *src);
    ork_f16 *got = malloc(n*sizeof *got), *ref = malloc(n*sizeof *ref);
    if (!src || !got || !ref) { printf("  alloc failed\n"); fails++; return; }
    for (size_t i = 0; i < n; i++) src[i] = rnd();
    for (int r = 0; r < R; r++) for (int c = 0; c < C; c++)
        ref[(size_t)c*R + r] = (ork_f16)src[(size_t)r*C + c];   /* same scalar cast the NEON must match */
    memset(got, 0, n*sizeof *got);
    int rc = ork_f32_transpose_to_f16(got, src, R, C);
    long bad = 0;
    for (size_t i = 0; i < n; i++) if ((float)got[i] != (float)ref[i]) bad++;
    printf("  f32->f16 [%4d,%4d]  rc=%d  mismatches %ld   %s\n",
           R, C, rc, bad, (rc==0 && !bad) ? "ok" : "FAIL");
    if (rc || bad) fails++;
    free(src); free(got); free(ref);
}

int main(void)
{
    printf("transpose: NEON vs scalar reference, elementwise (host only, no NPU)\n");
    /* aligned, ragged in each dimension, both-ragged, vectors, and a training-sized case */
    const int sh[][2] = { {8,8},{64,64},{128,256},{7,5},{13,1},{1,13},{65,33},{100,7},{3,3},
                          {256,1024},{1024,256},{9,129} };
    for (unsigned i = 0; i < sizeof sh/sizeof sh[0]; i++) chk16(sh[i][0], sh[i][1]);
    printf("\n");
    for (unsigned i = 0; i < sizeof sh/sizeof sh[0]; i++) chk32(sh[i][0], sh[i][1]);

    printf("\nbad args must be rejected, not tolerated:\n");
    ork_f16 a[16] = {0}, b[16] = {0};
    float   f[16] = {0};
    struct { const char *nm; int rc; } neg[] = {
        { "NULL dst",        ork_f16_transpose(NULL, a, 4, 4) },
        { "NULL src",        ork_f16_transpose(b, NULL, 4, 4) },
        { "rows=0",          ork_f16_transpose(b, a, 0, 4)    },
        { "cols<0",          ork_f16_transpose(b, a, 4, -1)   },
        { "aliased dst==src",ork_f16_transpose(a, a, 4, 4)    },
        { "f32 NULL src",    ork_f32_transpose_to_f16(b, NULL, 4, 4) },
        { "f32 rows=0",      ork_f32_transpose_to_f16(b, f, 0, 4)    },
    };
    for (unsigned i = 0; i < sizeof neg/sizeof neg[0]; i++) {
        int ok = neg[i].rc == -1;
        printf("  %-18s rc=%d  %s\n", neg[i].nm, neg[i].rc, ok ? "ok" : "FAIL");
        if (!ok) fails++;
    }

    printf("\n%s\n", fails == 0 ? "TRANSPOSE OK" : "TRANSPOSE FAILED");
    return fails != 0;
}
