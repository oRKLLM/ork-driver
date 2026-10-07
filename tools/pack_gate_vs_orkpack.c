/* pack_gate_vs_orkpack — does the new pack-header gate actually refuse an EXISTING .orkpack?
 *
 * test_pack_gate proves the gate refuses a SYNTHETIC unstamped blob. That is not the same claim as
 * "the packs on this machine are refused": a synthetic blob is one written to be refused, while a real
 * pack is the thing the gate was added for. This probe closes that gap by taking a REAL blob out of a
 * REAL .orkpack and handing it to the REAL loader.
 *
 * NO NPU. ork_npu_init_offline gives a caps-only context with fd=-1 -- no device, no ioctl, no IOMMU
 * mapping, no signal handler -- so this cannot wedge the board and needs no npu_guard.
 *
 * READ-ONLY. It opens the pack O_RDONLY and maps it PROT_READ. It never writes, truncates or deletes;
 * several packs on this board are deliberately lock-protected.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include "ork_npu.h"

#define ORKPACK_MAGIC "ORKPK01"

/* The footer grew over time; the magic is deliberately the LAST 8 bytes at every version, so read
 * back from EOF. v5/v6 = this 32-byte prefix; v7 adds calib + provenance ahead of the magic. */
struct foot_v5 { uint64_t index_off; uint32_t n_entries, version, ork_fmt, quant_sig; char magic[8]; };

/* Entry layout as of v7. Older packs may lack the trailing qerr/_pad, so a mis-stride is possible;
 * every field is sanity-checked before use rather than trusted. */
struct entry_v7 { uint32_t K, N, dtype, bscale_n; uint64_t blob_off, blob_size, bscale_off, bf_size; float qerr; uint32_t _pad; };

int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s <file.orkpack>\n", argv[0]); return 2; }
    const char *path = argv[1];

    int fd = open(path, O_RDONLY);
    if (fd < 0) { perror("open"); return 2; }
    struct stat st; if (fstat(fd, &st)) { perror("fstat"); return 2; }
    const uint64_t sz = (uint64_t) st.st_size;
    const uint8_t *m = mmap(NULL, sz, PROT_READ, MAP_SHARED, fd, 0);
    if (m == MAP_FAILED) { perror("mmap"); return 2; }

    printf("pack: %s  (%.2f GiB)\n", path, (double) sz / (1024.0*1024.0*1024.0));

    /* Find the magic. It is SUPPOSED to be the final 8 bytes at every version, but packs exist on this
     * board with 4 bytes of padding after it, so search the tail rather than assume. */
    int64_t magic_at = -1;
    for (int64_t back = 8; back <= 128 && back <= (int64_t) sz; back++) {
        if (memcmp(m + sz - back, ORKPACK_MAGIC, 8) == 0) { magic_at = (int64_t) sz - back; break; }
    }
    if (magic_at < 0) { printf("  NOT an .orkpack (no %s in the last 128 bytes)\n", ORKPACK_MAGIC); return 2; }
    if ((uint64_t) magic_at != sz - 8)
        printf("  NOTE: magic is NOT the final 8 bytes -- %llu trailing byte(s) follow it\n",
               (unsigned long long) (sz - 8 - (uint64_t) magic_at));

    /* The footer ENDS at magic_at+8. Its size varies by version, so try each plausible size and keep
     * the one whose index_off/n_entries are self-consistent. Guessing would misread the index. */
    struct foot_v5 f; memset(&f, 0, sizeof f);
    uint64_t fsz = 0; int meta_skip = 16;
    /* 24 = the earliest footer (index_off, n_entries, version, magic). ork_fmt/quant_sig arrived later,
     * so they only exist from 32 up; reading them on a v2 pack would print garbage. */
    for (uint64_t cand = 24; cand <= 96; cand += 8) {
        uint64_t fstart = (uint64_t) magic_at + 8 - cand;
        if (cand > (uint64_t) magic_at + 8) break;
        uint64_t io; uint32_t ne, ver, fmt = 0, qs = 0;
        memcpy(&io, m + fstart, 8); memcpy(&ne, m + fstart + 8, 4);
        memcpy(&ver, m + fstart + 12, 4);
        if (cand >= 32) { memcpy(&fmt, m + fstart + 16, 4); memcpy(&qs, m + fstart + 20, 4); }
        if (!(io > 0 && io < sz && ne > 0 && ne < 1000000u && ver > 0 && ver <= 16u)) continue;
        /* Plausible NUMBERS are not enough -- several candidate sizes produce them. Validate by
         * parsing the first index entry: a correct index_off yields a short, printable tensor name.
         * The 16-byte meta section arrived in v6, so older packs start the index at an entry. */
        int ok = 0;
        for (int ms = 16; ms >= 0 && !ok; ms -= 16) {
            const uint8_t *p = m + io + ms;
            if (p + 4 > m + sz) continue;
            uint32_t nl; memcpy(&nl, p, 4);
            if (nl == 0 || nl > 256 || p + 4 + nl > m + sz) continue;
            int printable = 1;
            for (uint32_t c = 0; c < nl; c++) { uint8_t ch = p[4 + c]; if (ch < 0x20 || ch > 0x7e) { printable = 0; break; } }
            if (!printable) continue;
            meta_skip = ms; ok = 1;
        }
        if (!ok) continue;
        f.index_off = io; f.n_entries = ne; f.version = ver; f.ork_fmt = fmt; f.quant_sig = qs;
        fsz = cand; break;
    }
    if (!fsz) { printf("  could not find a self-consistent footer in the tail\n"); return 2; }
    printf("  container: version=%u  ork_fmt=%u  quant_sig=%08x  n_entries=%u  index_off=%llu  (footer %llu B)\n",
           f.version, f.ork_fmt, f.quant_sig, f.n_entries, (unsigned long long) f.index_off,
           (unsigned long long) fsz);
    printf("  index starts with a %d-byte meta section\n", meta_skip);

    /* 1. LAYOUT-INDEPENDENT FACT: does the ork pack-header magic appear anywhere at all?
     *    A stamped pack has one per weight, each at a page boundary; a pre-header pack has none.
     *    This needs no index parsing, so it holds even if the entry stride is wrong below. */
    const uint32_t want = ORK_PACK_MAGIC;
    uint64_t hits = 0, scanned = 0;
    for (uint64_t off = 0; off + 4 <= f.index_off && scanned < 4096; off += 4096, scanned++) {
        uint32_t v; memcpy(&v, m + off, 4);   /* blobs are page-aligned, so a stamp lands on one */
        if (v == want) hits++;
    }
    printf("  ork pack-header magic (OAPK) at page boundaries: %llu -> %s\n",
           (unsigned long long) hits, hits ? "STAMPED" : "PRE-HEADER (unstamped)");

    /* 2. Walk the index and hand a REAL blob to the REAL loader. */
    const uint8_t *idx = m + f.index_off, *end = m + sz;
    idx += meta_skip;   /* meta_off, meta_size (v6+) */

    ork_npu *off_ctx = ork_npu_init_offline("rk3588");
    if (!off_ctx) { printf("  ork_npu_init_offline failed\n"); return 2; }

    int tried = 0, refused = 0, accepted = 0;
    for (uint32_t i = 0; i < f.n_entries && tried < 3; i++) {
        uint32_t nl; if (idx + 4 > end) break;
        memcpy(&nl, idx, 4); idx += 4;
        if (nl == 0 || nl > 4096 || idx + nl + sizeof(struct entry_v7) > end) {
            printf("  index walk stopped at entry %u (name_len=%u) -- entry layout differs at v%u\n", i, nl, f.version);
            break;
        }
        char nm[256]; uint32_t cp = nl < sizeof nm - 1 ? nl : sizeof nm - 1;
        memcpy(nm, idx, cp); nm[cp] = 0; idx += nl;
        struct entry_v7 e; memcpy(&e, idx, sizeof e); idx += sizeof e;

        if (e.blob_off + e.blob_size > sz || e.K == 0 || e.N == 0 || e.blob_size == 0) {
            printf("  entry %u '%s' implausible (K=%u N=%u off=%llu size=%llu) -- stride mismatch, stopping\n",
                   i, nm, e.K, e.N, (unsigned long long) e.blob_off, (unsigned long long) e.blob_size);
            break;
        }

        uint32_t first4; memcpy(&first4, m + e.blob_off, 4);
        printf("\n  entry %u: %s\n    K=%u N=%u dtype=%u blob=%llu+%llu  first4=%08x %s\n",
               i, nm, e.K, e.N, e.dtype, (unsigned long long) e.blob_off,
               (unsigned long long) e.blob_size, first4,
               first4 == want ? "(stamped)" : "(no OAPK magic)");

        /* The container's dtype CODE shifted meaning across versions (int8 is 0 in late packs, 1 in v2),
         * so classify by the byte signature instead: an int8 weight's blob is K*N tile bytes. */
        const uint64_t kn = (uint64_t) e.K * (uint64_t) e.N;
        if (!(e.blob_size >= kn && e.blob_size < kn + (kn / 4) + 65536)) {
            printf("    (blob is %llu B vs K*N=%llu -- not an int8 tile stream, skipping the loader)\n",
                   (unsigned long long) e.blob_size, (unsigned long long) kn);
            continue;
        }

        tried++;
        printf("    calling ork_i8_mm_load on the real blob ...\n");
        fflush(stdout);
        ork_w *w = ork_i8_mm_load(off_ctx, (int) e.K, (int) e.N, m + e.blob_off, (size_t) e.blob_size);
        if (w) { printf("    RESULT: ACCEPTED (loader returned non-NULL)\n"); accepted++; ork_mm_free(off_ctx, w); }
        else   { printf("    RESULT: REFUSED (loader returned NULL)\n"); refused++; }
    }

    printf("\nSUMMARY %s: pre-header=%s  int8 blobs tried=%d refused=%d accepted=%d\n",
           path, hits ? "no" : "yes", tried, refused, accepted);
    return 0;
}
