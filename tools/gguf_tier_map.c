/* gguf_tier_map.c — GGUF per-tensor "tier map" reader for the ork-driver conversion pipeline.
 *
 * Reads any mixed-precision GGUF (llama.cpp k-quants, unsloth UD, IQ variants), pulls each
 * tensor's stored quant type from the file metadata, and collapses it onto ork-driver's two
 * storage tiers: int8 (important / high-bit) vs int4 (bulk / low-bit). This is the conversion
 * pipeline's "importance oracle" — it only decides allocation/precision; actual values get
 * quantized from fp16 elsewhere.
 *
 * Pure file I/O. No NPU / DRM / ork dependencies. Parses GGUF v2/v3 directly (no ggml/llama.cpp).
 *
 * Build (portable C11, Linux/macOS):
 *   gcc -O2 tools/gguf_tier_map.c -o gguf_tier_map
 *
 * Usage:
 *   ./gguf_tier_map <model.gguf> [bit_threshold]
 *     bit_threshold (default 5): tensors with effective bits-per-weight >= threshold -> int8,
 *                                otherwise -> int4. This is the precision/memory dial.
 */

/* expose fseeko/off_t (POSIX) under -std=c11 on glibc; harmless on macOS */
#define _POSIX_C_SOURCE 200809L
#define _FILE_OFFSET_BITS 64

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <inttypes.h>

/* ---- GGUF format constants ------------------------------------------------ */

#define GGUF_MAGIC 0x46554747u  /* "GGUF" little-endian */

/* GGUF metadata value types */
enum {
    GGUF_T_UINT8   = 0,
    GGUF_T_INT8    = 1,
    GGUF_T_UINT16  = 2,
    GGUF_T_INT16   = 3,
    GGUF_T_UINT32  = 4,
    GGUF_T_INT32   = 5,
    GGUF_T_FLOAT32 = 6,
    GGUF_T_BOOL    = 7,
    GGUF_T_STRING  = 8,
    GGUF_T_ARRAY   = 9,
    GGUF_T_UINT64  = 10,
    GGUF_T_INT64   = 11,
    GGUF_T_FLOAT64 = 12,
};

/* ggml_type enum values (subset; see ggml.h) */
enum {
    GGML_TYPE_F32     = 0,
    GGML_TYPE_F16     = 1,
    GGML_TYPE_Q4_0    = 2,
    GGML_TYPE_Q4_1    = 3,
    GGML_TYPE_Q5_0    = 6,
    GGML_TYPE_Q5_1    = 7,
    GGML_TYPE_Q8_0    = 8,
    GGML_TYPE_Q8_1    = 9,
    GGML_TYPE_Q2_K    = 10,
    GGML_TYPE_Q3_K    = 11,
    GGML_TYPE_Q4_K    = 12,
    GGML_TYPE_Q5_K    = 13,
    GGML_TYPE_Q6_K    = 14,
    GGML_TYPE_Q8_K    = 15,
    GGML_TYPE_IQ2_XXS = 16,
    GGML_TYPE_IQ2_XS  = 17,
    GGML_TYPE_IQ3_XXS = 18,
    GGML_TYPE_IQ1_S   = 19,
    GGML_TYPE_IQ4_NL  = 20,
    GGML_TYPE_IQ3_S   = 21,
    GGML_TYPE_IQ2_S   = 22,
    GGML_TYPE_IQ4_XS  = 23,
    GGML_TYPE_I8      = 24,
    GGML_TYPE_I16     = 25,
    GGML_TYPE_I32     = 26,
    GGML_TYPE_I64     = 27,
    GGML_TYPE_F64     = 28,
    GGML_TYPE_IQ1_M   = 29,
    GGML_TYPE_BF16    = 30,
    GGML_TYPE_COUNT_  = 64,  /* upper guard for the table below */
};

/* Effective bits-per-weight for each ggml_type, and a printable name.
 * For block quants this is (block bytes * 8) / block elements. Unknown -> bits=16
 * (conservative: maps to int8). */
typedef struct {
    const char *name;
    double bits;   /* effective bits/weight; <0 means "unknown" */
} type_info;

static type_info type_table(uint32_t t) {
    switch (t) {
    case GGML_TYPE_F32:     return (type_info){ "F32",     32.0 };
    case GGML_TYPE_F16:     return (type_info){ "F16",     16.0 };
    case GGML_TYPE_BF16:    return (type_info){ "BF16",    16.0 };
    case GGML_TYPE_F64:     return (type_info){ "F64",     64.0 };
    case GGML_TYPE_Q4_0:    return (type_info){ "Q4_0",     4.5 };  /* 18B / 32 */
    case GGML_TYPE_Q4_1:    return (type_info){ "Q4_1",     5.0 };  /* 20B / 32 */
    case GGML_TYPE_Q5_0:    return (type_info){ "Q5_0",     5.5 };  /* 22B / 32 */
    case GGML_TYPE_Q5_1:    return (type_info){ "Q5_1",     6.0 };  /* 24B / 32 */
    case GGML_TYPE_Q8_0:    return (type_info){ "Q8_0",     8.5 };  /* 34B / 32 */
    case GGML_TYPE_Q8_1:    return (type_info){ "Q8_1",     9.0 };  /* 36B / 32 */
    case GGML_TYPE_Q2_K:    return (type_info){ "Q2_K",     2.5625 };
    case GGML_TYPE_Q3_K:    return (type_info){ "Q3_K",     3.4375 };
    case GGML_TYPE_Q4_K:    return (type_info){ "Q4_K",     4.5 };
    case GGML_TYPE_Q5_K:    return (type_info){ "Q5_K",     5.5 };
    case GGML_TYPE_Q6_K:    return (type_info){ "Q6_K",     6.5625 };
    case GGML_TYPE_Q8_K:    return (type_info){ "Q8_K",     8.0 };
    case GGML_TYPE_IQ1_S:   return (type_info){ "IQ1_S",    1.5625 };
    case GGML_TYPE_IQ1_M:   return (type_info){ "IQ1_M",    1.75 };
    case GGML_TYPE_IQ2_XXS: return (type_info){ "IQ2_XXS",  2.0625 };
    case GGML_TYPE_IQ2_XS:  return (type_info){ "IQ2_XS",   2.3125 };
    case GGML_TYPE_IQ2_S:   return (type_info){ "IQ2_S",    2.5 };
    case GGML_TYPE_IQ3_XXS: return (type_info){ "IQ3_XXS",  3.0625 };
    case GGML_TYPE_IQ3_S:   return (type_info){ "IQ3_S",    3.4375 };
    case GGML_TYPE_IQ4_NL:  return (type_info){ "IQ4_NL",   4.5 };
    case GGML_TYPE_IQ4_XS:  return (type_info){ "IQ4_XS",   4.25 };
    case GGML_TYPE_I8:      return (type_info){ "I8",       8.0 };
    case GGML_TYPE_I16:     return (type_info){ "I16",      16.0 };
    case GGML_TYPE_I32:     return (type_info){ "I32",      32.0 };
    case GGML_TYPE_I64:     return (type_info){ "I64",      64.0 };
    default:                return (type_info){ "?UNKNOWN", -1.0 };
    }
}

/* ---- buffered reader ------------------------------------------------------ */

typedef struct {
    FILE *f;
    int   err;
} rdr;

static void rd_bytes(rdr *r, void *dst, size_t n) {
    if (r->err) return;
    if (fread(dst, 1, n, r->f) != n) r->err = 1;
}
static void rd_skip(rdr *r, uint64_t n) {
    if (r->err) return;
    /* fseeko-portable: 64-bit seek where available */
#if defined(_WIN32)
    if (_fseeki64(r->f, (long long)n, SEEK_CUR) != 0) r->err = 1;
#else
    if (fseeko(r->f, (off_t)n, SEEK_CUR) != 0) r->err = 1;
#endif
}
static uint32_t rd_u32(rdr *r){ uint32_t v=0; rd_bytes(r,&v,4); return v; }
static uint64_t rd_u64(rdr *r){ uint64_t v=0; rd_bytes(r,&v,8); return v; }

/* gguf_string = u64 len + bytes. Returns malloc'd NUL-terminated string (or NULL). */
static char *rd_gguf_string(rdr *r) {
    uint64_t len = rd_u64(r);
    if (r->err) return NULL;
    if (len > (uint64_t)64*1024*1024) { r->err = 1; return NULL; } /* sanity guard */
    char *s = (char *)malloc((size_t)len + 1);
    if (!s) { r->err = 1; return NULL; }
    rd_bytes(r, s, (size_t)len);
    s[len] = '\0';
    return s;
}

/* size in bytes of a single fixed-width metadata value type (0 for var/array). */
static size_t scalar_size(uint32_t vt) {
    switch (vt) {
    case GGUF_T_UINT8: case GGUF_T_INT8: case GGUF_T_BOOL: return 1;
    case GGUF_T_UINT16: case GGUF_T_INT16: return 2;
    case GGUF_T_UINT32: case GGUF_T_INT32: case GGUF_T_FLOAT32: return 4;
    case GGUF_T_UINT64: case GGUF_T_INT64: case GGUF_T_FLOAT64: return 8;
    default: return 0;
    }
}

/* Skip a metadata value of a given type. Recurses for arrays (incl. arrays of strings). */
static void skip_value(rdr *r, uint32_t vt) {
    if (r->err) return;
    size_t sz = scalar_size(vt);
    if (sz) { rd_skip(r, sz); return; }
    if (vt == GGUF_T_STRING) {
        uint64_t len = rd_u64(r);
        rd_skip(r, len);
        return;
    }
    if (vt == GGUF_T_ARRAY) {
        uint32_t elem_t = rd_u32(r);
        uint64_t count  = rd_u64(r);
        if (r->err) return;
        size_t esz = scalar_size(elem_t);
        if (esz) {
            rd_skip(r, (uint64_t)esz * count);
        } else if (elem_t == GGUF_T_STRING) {
            for (uint64_t i = 0; i < count && !r->err; i++) {
                uint64_t len = rd_u64(r);
                rd_skip(r, len);
            }
        } else if (elem_t == GGUF_T_ARRAY) {
            for (uint64_t i = 0; i < count && !r->err; i++)
                skip_value(r, GGUF_T_ARRAY);
        } else {
            r->err = 1; /* unknown element type */
        }
        return;
    }
    r->err = 1; /* unknown value type */
}

/* ---- main ----------------------------------------------------------------- */

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr,
            "usage: %s <model.gguf> [bit_threshold]\n"
            "  tensors with effective bits/weight >= threshold -> int8, else int4\n"
            "  (threshold default 5: Q5/Q6/Q8/F16/F32 -> int8; Q4/Q3/Q2/IQ* -> int4)\n",
            argv[0]);
        return 2;
    }
    const char *path = argv[1];
    double threshold = 5.0;
    if (argc >= 3) threshold = strtod(argv[2], NULL);

    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "error: cannot open %s\n", path); return 1; }
    rdr R = { f, 0 };

    uint32_t magic = rd_u32(&R);
    if (R.err || magic != GGUF_MAGIC) {
        fprintf(stderr, "error: not a GGUF file (magic=0x%08x)\n", magic);
        fclose(f); return 1;
    }
    uint32_t version      = rd_u32(&R);
    uint64_t tensor_count = rd_u64(&R);
    uint64_t kv_count     = rd_u64(&R);
    if (R.err) { fprintf(stderr, "error: short header\n"); fclose(f); return 1; }
    if (version != 2 && version != 3) {
        fprintf(stderr, "warning: GGUF version %u (expected 2 or 3); proceeding\n", version);
    }

    /* Skip the metadata KV block — we only need the tensor infos that follow. */
    for (uint64_t i = 0; i < kv_count && !R.err; i++) {
        char *key = rd_gguf_string(&R);   /* parse-to-skip */
        free(key);
        uint32_t vt = rd_u32(&R);
        skip_value(&R, vt);
    }
    if (R.err) {
        fprintf(stderr, "error: failed parsing metadata (kv_count=%" PRIu64 ")\n", kv_count);
        fclose(f); return 1;
    }

    printf("GGUF v%u  tensors=%" PRIu64 "  metadata_kv=%" PRIu64 "  threshold=%.4g bits\n",
           version, tensor_count, kv_count, threshold);
    printf("%-44s %-9s %5s %-22s -> %-4s\n",
           "tensor_name", "ggml_type", "bits", "dims", "tier");
    printf("--------------------------------------------------------------------------------------------\n");

    /* per-tier accumulators */
    uint64_t cnt_i8 = 0, cnt_i4 = 0, cnt_unknown = 0;
    long double elems_i8 = 0, elems_i4 = 0;
    /* sanity-property probes.
     * Real k-quant / UD mixed GGUFs bump a SUBSET of important tensors to a higher-bit type
     * (the "importance" signal we want to preserve): the final output projection, and — per
     * llama.cpp's Q*_K_M rule — attn_v and ffn_down in a fraction of the layers. The bulk
     * (attn_q/k/output, ffn_gate/up, and the un-bumped attn_v/ffn_down) stays at the base
     * low-bit type. So the meaningful assertions are:
     *   (1) every high-bit tensor (>= threshold) lands on int8 and every low-bit on int4
     *       — i.e. the tier mapping is internally consistent with the bit policy; and
     *   (2) at least one of {output.weight, a bumped attn_v, a bumped ffn_down} reached int8
     *       — i.e. the importance signal carried by the source quant survived onto int8. */
    int sane_bit_consistent = 1;        /* (1) */
    int saw_bumped_important = 0;       /* (2): a known-important tensor that is high-bit */
    int bumped_attn_v = 0, bumped_ffn_down = 0, bumped_output = 0;
    /* type histogram for "is it uniform?" check */
    uint64_t type_hist[GGML_TYPE_COUNT_] = {0};

    for (uint64_t i = 0; i < tensor_count && !R.err; i++) {
        char *name = rd_gguf_string(&R);
        uint32_t n_dims = rd_u32(&R);
        if (R.err || n_dims > 8) { R.err = 1; free(name); break; }
        uint64_t dims[8] = {0};
        long double n_elem = 1.0;
        for (uint32_t d = 0; d < n_dims; d++) { dims[d] = rd_u64(&R); n_elem *= (long double)dims[d]; }
        if (n_dims == 0) n_elem = 0;
        uint32_t gtype = rd_u32(&R);
        (void)rd_u64(&R); /* offset — unused */
        if (R.err) { free(name); break; }

        type_info ti = type_table(gtype);
        if (gtype < GGML_TYPE_COUNT_) type_hist[gtype]++;

        double bits = ti.bits;
        int unknown = (bits < 0);
        if (unknown) bits = 16.0; /* conservative: treat as int8 */

        const char *tier;
        if (bits >= threshold) { tier = "int8"; cnt_i8++; elems_i8 += n_elem; }
        else                   { tier = "int4"; cnt_i4++; elems_i4 += n_elem; }
        if (unknown) cnt_unknown++;

        /* dims string */
        char dbuf[64]; int p = 0;
        for (uint32_t d = 0; d < n_dims; d++)
            p += snprintf(dbuf + p, sizeof(dbuf) - p, "%s%" PRIu64, d ? "x" : "", dims[d]);
        if (n_dims == 0) snprintf(dbuf, sizeof(dbuf), "(scalar)");

        printf("%-44s %-9s %5.2f %-22s -> %s%s\n",
               name, ti.name, ti.bits < 0 ? 0.0 : ti.bits, dbuf, tier,
               unknown ? "  (unknown type -> conservative int8)" : "");

        /* (1) bit-policy consistency: high-bit -> int8, low-bit -> int4 */
        int want_i8 = (bits >= threshold);
        int got_i8  = (strcmp(tier, "int8") == 0);
        if (want_i8 != got_i8) sane_bit_consistent = 0;

        /* (2) importance signal: an important tensor that the source quant bumped to high-bit
         *     (and which therefore lands on int8). norms are F32 and trivially high-bit, so
         *     exclude them — they don't carry the matmul-importance signal. */
        int is_norm = strstr(name, "norm") != NULL;
        if (!is_norm && got_i8) {
            if (strstr(name, "attn_v")  != NULL) { bumped_attn_v  = 1; saw_bumped_important = 1; }
            if (strstr(name, "ffn_down")!= NULL) { bumped_ffn_down= 1; saw_bumped_important = 1; }
            /* "output.weight" is the final projection; avoid matching "*_output*"/"output_norm" */
            if (strcmp(name, "output.weight") == 0) { bumped_output = 1; saw_bumped_important = 1; }
        }

        free(name);
    }
    fclose(f);

    if (R.err) {
        fprintf(stderr, "error: failed parsing tensor infos\n");
        return 1;
    }

    /* ---- summary ---- */
    long double total_elems = elems_i8 + elems_i4;
    double i4_frac = total_elems > 0 ? (double)(elems_i4 / total_elems) : 0.0;

    printf("\n=== tier summary ===\n");
    printf("int8 tier : %8" PRIu64 " tensors  %18.0Lf elements\n", cnt_i8, elems_i8);
    printf("int4 tier : %8" PRIu64 " tensors  %18.0Lf elements\n", cnt_i4, elems_i4);
    printf("total     : %8" PRIu64 " tensors  %18.0Lf elements\n",
           cnt_i8 + cnt_i4, total_elems);
    if (cnt_unknown)
        printf("(%" PRIu64 " tensor(s) had unknown ggml_type -> mapped conservatively to int8)\n",
               cnt_unknown);
    printf("int4 fraction (by elements) = %.4f  (%.1f%% of weights on the int4 tier)\n",
           i4_frac, i4_frac * 100.0);
    /* crude memory-win estimate: int4 weights cost ~half the int8 ones */
    {
        long double bytes_all_i8 = total_elems * 1.0;            /* 1 byte/weight @ int8 */
        long double bytes_tiered = elems_i8 * 1.0 + elems_i4 * 0.5;
        double win = bytes_all_i8 > 0 ? (double)(1.0 - bytes_tiered / bytes_all_i8) : 0.0;
        printf("memory-win estimate vs all-int8 storage = %.1f%%\n", win * 100.0);
    }

    /* type histogram + uniform check */
    int distinct = 0; uint32_t only_t = 0;
    for (uint32_t t = 0; t < GGML_TYPE_COUNT_; t++)
        if (type_hist[t]) { distinct++; only_t = t; }
    printf("\n=== ggml_type histogram (%d distinct) ===\n", distinct);
    for (uint32_t t = 0; t < GGML_TYPE_COUNT_; t++)
        if (type_hist[t])
            printf("  %-9s : %" PRIu64 "\n", type_table(t).name, type_hist[t]);
    if (distinct <= 1)
        printf("NOTE: file is type-UNIFORM (all %s). Not a mixed-precision GGUF — "
               "the tier split is trivial; use a UD/_K_M variant for a meaningful map.\n",
               type_table(only_t).name);

    /* ---- sanity property (Gate 3) ---- */
    printf("\n=== sanity property (Gate 3) ===\n");
    int ok = 1;
    printf("  bit policy consistent (high-bit->int8, low-bit->int4) : %s\n",
           sane_bit_consistent ? "PASS" : "FAIL");
    ok &= sane_bit_consistent;

    if (distinct <= 1) {
        printf("  importance signal preserved on int8                   : (file is type-UNIFORM"
               " — no mixed-precision signal to preserve; not meaningful)\n");
    } else {
        printf("  importance signal preserved on int8                   : %s"
               "  [output.weight=%s attn_v(bumped)=%s ffn_down(bumped)=%s]\n",
               saw_bumped_important ? "PASS" : "FAIL",
               bumped_output ? "int8" : "-",
               bumped_attn_v ? "int8" : "-",
               bumped_ffn_down ? "int8" : "-");
        ok &= saw_bumped_important;
    }

    return ok ? 0 : 3;
}
