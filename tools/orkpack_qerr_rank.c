/* orkpack_qerr_rank — rank a pack's weights by stored quantisation error, and emit a promote list.
 *
 * WHY THIS EXISTS. Mixed-precision promotion ("mostly int4, the worst few at int8") needs to know WHICH
 * weights are worst, and that is only knowable after quantisation. The pack records it: every entry
 * carries a `qerr` written by ggml_backend_ork_gptq_finalize. The backend has a qerr-ranked policy that
 * consumes it, but it is reachable only through the C pack-config API or llama.cpp's common CLI args --
 * neither of which ork_ppl or ork_bench expose. This reads the pack's INDEX ONLY (no blobs, no IOVA, no
 * model load) and prints a list for ORK_I4_INT8_LAYERS, which is substring-matched per weight name.
 *
 * It also prints the DISTRIBUTION, which is the part worth looking at before trusting any promotion: the
 * policy only pays if error is concentrated in a few weights. If qerr is flat, promoting the top decile
 * buys a tenth of the gap and the whole approach is mis-sold.
 *
 * BUDGET IS IN BITS, NOT MEGABYTES. The comparison that matters is against a GGUF 4-bit file at some
 * bits-per-weight, so the natural question is "how many weights can go to int8 and still match it".
 * Promoting one weight costs K*N/2 extra bytes (4 bits -> 8 bits over K*N values).
 *
 * READ-ONLY: opens O_RDONLY, maps PROT_READ, never writes.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>

#define ORKPACK_MAGIC "ORKPK01"

struct entry { uint32_t K, N, dtype, bscale_n; uint64_t blob_off, blob_size, bscale_off, bf_size; float qerr; uint32_t _pad; };

struct row { char name[256]; float qerr; uint64_t promote_bytes; };

static int by_qerr_desc(const void *a, const void *b) {
    const float x = ((const struct row *)a)->qerr, y = ((const struct row *)b)->qerr;
    return (x < y) - (x > y);
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s <pack.orkpack> [budget_mib] [exclude_list_file]\n"
                        "  budget_mib: how much EXTRA file size promotion may spend (default 0 = report only)\n"
                        "  exclude_list_file: a comma-separated list of names ALREADY promoted by some other\n"
                        "    policy. They are skipped here and their cost is charged against the budget, so\n"
                        "    the emitted list is the qerr-ranked REMAINDER that fits in the headroom left over.\n",
                argv[0]);
        return 2;
    }
    const char *path = argv[1];
    const double budget_mib = argc > 2 ? atof(argv[2]) : 0.0;
    char *excl = NULL;
    if (argc > 3) {
        FILE *xf = fopen(argv[3], "r");
        if (!xf) { perror("exclude list"); return 2; }
        excl = (char*)calloc(1, 1u << 16);
        if (excl) { size_t rd = fread(excl, 1, (1u << 16) - 1, xf); excl[rd] = 0; }
        fclose(xf);
    }

    int fd = open(path, O_RDONLY);
    if (fd < 0) { perror("open"); return 2; }
    struct stat st; if (fstat(fd, &st)) { perror("fstat"); return 2; }
    const uint64_t sz = (uint64_t) st.st_size;
    const uint8_t *m = mmap(NULL, sz, PROT_READ, MAP_SHARED, fd, 0);
    if (m == MAP_FAILED) { perror("mmap"); return 2; }

    /* Find the magic in the tail: it is SUPPOSED to be the final 8 bytes but packs exist with padding
     * after it, so search rather than assume (see Exp-2026-10-07). */
    int64_t magic_at = -1;
    for (int64_t back = 8; back <= 128 && back <= (int64_t) sz; back++)
        if (!memcmp(m + sz - back, ORKPACK_MAGIC, 8)) { magic_at = (int64_t) sz - back; break; }
    if (magic_at < 0) { fprintf(stderr, "not an .orkpack\n"); return 2; }

    /* Footer size varies by version; accept the candidate whose index yields a printable first name. */
    uint64_t index_off = 0, fsz = 0; uint32_t n_entries = 0, version = 0; int meta_skip = 16;
    for (uint64_t cand = 24; cand <= 96 && !fsz; cand += 8) {
        if (cand > (uint64_t) magic_at + 8) break;
        const uint64_t fstart = (uint64_t) magic_at + 8 - cand;
        uint64_t io; uint32_t ne, ver;
        memcpy(&io, m + fstart, 8); memcpy(&ne, m + fstart + 8, 4); memcpy(&ver, m + fstart + 12, 4);
        if (!(io > 0 && io < sz && ne > 0 && ne < 1000000u && ver > 0 && ver <= 16u)) continue;
        for (int ms = 16; ms >= 0; ms -= 16) {
            const uint8_t *p = m + io + ms;
            if (p + 4 > m + sz) continue;
            uint32_t nl; memcpy(&nl, p, 4);
            if (nl == 0 || nl > 256 || p + 4 + nl > m + sz) continue;
            int ok = 1;
            for (uint32_t c = 0; c < nl; c++) { uint8_t ch = p[4 + c]; if (ch < 0x20 || ch > 0x7e) { ok = 0; break; } }
            if (!ok) continue;
            index_off = io; n_entries = ne; version = ver; fsz = cand; meta_skip = ms; break;
        }
    }
    if (!fsz) { fprintf(stderr, "could not parse a footer\n"); return 2; }

    struct row *rows = calloc(n_entries, sizeof *rows);
    const uint8_t *idx = m + index_off + meta_skip, *end = m + sz;
    uint32_t n = 0, n_qerr = 0;
    uint64_t total_int4_bytes = 0;
    for (uint32_t i = 0; i < n_entries; i++) {
        uint32_t nl; if (idx + 4 > end) break;
        memcpy(&nl, idx, 4); idx += 4;
        if (nl == 0 || nl > 4096 || idx + nl + sizeof(struct entry) > end) break;
        const uint32_t cp = nl < sizeof rows[n].name - 1 ? nl : (uint32_t)(sizeof rows[n].name - 1);
        memcpy(rows[n].name, idx, cp); rows[n].name[cp] = 0; idx += nl;
        struct entry e; memcpy(&e, idx, sizeof e); idx += sizeof e;
        rows[n].qerr = e.qerr;
        rows[n].promote_bytes = (uint64_t) e.K * e.N / 2;      /* 4 bits -> 8 bits over K*N values */
        total_int4_bytes += (uint64_t) e.K * e.N / 2;
        if (e.qerr > 0.0f) n_qerr++;
        n++;
    }

    printf("pack %s: v%u, %u entries parsed (%u carry qerr), int4 weight bytes %.1f MiB\n",
           path, version, n, n_qerr, total_int4_bytes / 1048576.0);
    if (!n_qerr) {
        printf("NO qerr recorded in this pack — nothing to rank. qerr is written only by a GPTQ\n"
               "finalize pass, so a plain build records none and promotion has no evidence to use.\n");
        return 1;
    }

    qsort(rows, n, sizeof *rows, by_qerr_desc);

    /* DISTRIBUTION FIRST: promotion is only worth it if error is concentrated. */
    double sum = 0; for (uint32_t i = 0; i < n; i++) sum += rows[i].qerr;
    double top10 = 0; const uint32_t d = n / 10 ? n / 10 : 1;
    for (uint32_t i = 0; i < d; i++) top10 += rows[i].qerr;
    printf("qerr: max %.4f  median %.4f  min %.4f  |  top decile holds %.1f%% of total error\n",
           rows[0].qerr, rows[n/2].qerr, rows[n-1].qerr, 100.0 * top10 / (sum > 0 ? sum : 1));

    printf("\nworst 12 by qerr:\n");
    for (uint32_t i = 0; i < n && i < 12; i++)
        printf("  %-40s qerr %.4f  +%.2f MiB to promote\n", rows[i].name, rows[i].qerr,
               rows[i].promote_bytes / 1048576.0);

    if (budget_mib <= 0) {
        printf("\n(no budget given — pass one in MiB to emit an ORK_I4_INT8_LAYERS list)\n");
        return 0;
    }

    /* Charge anything an outside policy already promoted against the budget first, so what follows is
     * genuinely the REMAINDER rather than a second full allocation. */
    double spent = 0; uint32_t promoted = 0, skipped = 0;
    if (excl) {
        for (uint32_t i = 0; i < n; i++) {
            if (!strstr(excl, rows[i].name)) continue;
            spent += rows[i].promote_bytes / 1048576.0; skipped++;
            rows[i].qerr = -1.0f;                        /* mark: already promoted, do not re-emit */
        }
        printf("\nexcluded %u already-promoted weights, %.2f MiB of the budget already committed\n",
               skipped, spent);
    }

    /* Worst-first until the budget is spent; take a smaller candidate when the next does not fit. */
    printf("\nORK_I4_INT8_LAYERS=");
    for (uint32_t i = 0; i < n; i++) {
        if (rows[i].qerr < 0.0f) continue;               /* excluded above */
        const double cost = rows[i].promote_bytes / 1048576.0;
        if (spent + cost > budget_mib) continue;
        printf("%s%s", promoted ? "," : "", rows[i].name);
        spent += cost; promoted++;
    }
    printf("\n\npromoted %u/%u weights, %.2f of %.2f MiB budget spent (+%.1f%% on the int4 weight bytes)\n",
           promoted, n, spent, budget_mib, 100.0 * spent / (total_int4_bytes / 1048576.0));
    return 0;
}
