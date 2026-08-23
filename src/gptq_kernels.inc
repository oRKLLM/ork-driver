/* gptq_kernels.inc — the GPTQ linear algebra, TEMPLATED ON WORKING PRECISION.
 *
 * Included TWICE by ork_gptq.c (GQT=double and GQT=float) so the fp32 experiment shares one source with
 * the fp64 production path instead of forking it. fp32 halves memory traffic and doubles NEON lane count
 * on the O(K^3) work, which is the whole pack cost -- but GPTQ inverts a Hessian that is near-singular by
 * construction (hence `damp`), so it is GATED OFF (ORK_GPTQ_FP32=1) until a PPL run proves it harmless.
 *
 * Optimizations here apply to BOTH precisions:
 *  - FOUR independent accumulators in every dot product. These loops are dependent FMA chains, so they
 *    were latency-bound, not throughput-bound: one accumulator stalls on the ~4-cycle FMA latency every
 *    iteration no matter how wide the vector unit is. Four in flight hide it.
 *  - J-TILING in the symmetric product, so a tile of rj rows is reused across several i rows instead of
 *    streaming the whole K x K matrix once per output row.
 * Both are pure reassociation of a sum -- they change fp rounding, which is why the pack hash moves. */

/* Lower Cholesky in place: A (sym PD) -> L lower, A = L.L^T; upper zeroed. 0 ok, -1 not positive-definite. */
static int GQ(chol_lower)(int K, GQT *A) {
    for (int i = 0; i < K; i++) {
        #pragma omp parallel for schedule(static) if (i > 256)
        for (int j = 0; j < i; j++) {
            const GQT *ri = A + (size_t)i*K, *rj = A + (size_t)j*K;
            GQT s0 = 0, s1 = 0, s2 = 0, s3 = 0;
            int k = 0;
            for (; k + 4 <= j; k += 4) {                       /* 4 accumulators: hide the FMA latency */
                s0 += ri[k+0]*rj[k+0]; s1 += ri[k+1]*rj[k+1];
                s2 += ri[k+2]*rj[k+2]; s3 += ri[k+3]*rj[k+3];
            }
            GQT s = ri[j] - (((s0 + s1) + (s2 + s3)));
            for (; k < j; k++) s -= ri[k] * rj[k];
            A[(size_t)i*K + j] = s / rj[j];
        }
        const GQT *ri = A + (size_t)i*K;
        GQT d0 = 0, d1 = 0, d2 = 0, d3 = 0; int k = 0;
        for (; k + 4 <= i; k += 4) {
            d0 += ri[k+0]*ri[k+0]; d1 += ri[k+1]*ri[k+1];
            d2 += ri[k+2]*ri[k+2]; d3 += ri[k+3]*ri[k+3];
        }
        GQT s = ri[i] - (((d0 + d1) + (d2 + d3)));
        for (; k < i; k++) s -= ri[k] * ri[k];
        if (s <= 0) return -1;                                  /* not positive-definite */
        A[(size_t)i*K + i] = (GQT)sqrt((double)s);
    }
    for (int i = 0; i < K; i++) for (int j = i + 1; j < K; j++) A[(size_t)i*K + j] = 0;
    return 0;
}

/* Given lower L, out = (L.L^T)^-1 = (L^-1)^T.(L^-1) (symmetric). Works in the TRANSPOSE of L^-1 so both
 * inner loops are unit-stride (a column walk here was once the dominant cost of the whole quantizer). */
static int GQ(inv_from_chol)(int K, const GQT *L, GQT *out) {
    GQT *LiT = (GQT*)calloc((size_t)K*K, sizeof(GQT));          /* LiT[c][i] = (L^-1)[i][c] */
    if (!LiT) return -2;
    #pragma omp parallel for schedule(dynamic, 8)
    for (int c = 0; c < K; c++) {                               /* forward-solve L.Li[:,c] = e_c */
        GQT *lic = LiT + (size_t)c*K;
        lic[c] = (GQT)1 / L[(size_t)c*K + c];
        for (int i = c + 1; i < K; i++) {
            const GQT *lr = L + (size_t)i*K;
            GQT s0 = 0, s1 = 0, s2 = 0, s3 = 0; int k = c;
            for (; k + 4 <= i; k += 4) {
                s0 += lr[k+0]*lic[k+0]; s1 += lr[k+1]*lic[k+1];
                s2 += lr[k+2]*lic[k+2]; s3 += lr[k+3]*lic[k+3];
            }
            GQT s = -(((s0 + s1) + (s2 + s3)));
            for (; k < i; k++) s -= lr[k] * lic[k];
            lic[i] = s / lr[i];
        }
    }
    /* out[i][j] = sum_{k>=max(i,j)} LiT[i][k].LiT[j][k], J-TILED: for each i, walk j in tiles so the tile's
     * rows stay resident across the i loop rather than re-streaming K x K per output row. */
    enum { JT = 8 };
    #pragma omp parallel for schedule(dynamic, 4)
    for (int i = 0; i < K; i++) {
        const GQT *ri = LiT + (size_t)i*K;
        GQT *orow = out + (size_t)i*K;
        for (int jb = 0; jb < K; jb += JT) {
            const int je = (jb + JT < K) ? jb + JT : K;
            for (int j = jb; j < je; j++) {
                const GQT *rj = LiT + (size_t)j*K;
                const int k0 = (i > j ? i : j);
                GQT s0 = 0, s1 = 0, s2 = 0, s3 = 0; int k = k0;
                for (; k + 4 <= K; k += 4) {
                    s0 += ri[k+0]*rj[k+0]; s1 += ri[k+1]*rj[k+1];
                    s2 += ri[k+2]*rj[k+2]; s3 += ri[k+3]*rj[k+3];
                }
                GQT s = ((s0 + s1) + (s2 + s3));
                for (; k < K; k++) s += ri[k] * rj[k];
                orow[j] = s;
            }
        }
    }
    free(LiT);
    return 0;
}

/* Upper Cholesky: A -> U upper with A = U^T.U; lower zeroed. U is exactly the transpose of the lower
 * factor (the factor with positive diagonal is unique), so reuse the lower routine and transpose. */
static int GQ(chol_upper)(int K, GQT *A) {
    if (GQ(chol_lower)(K, A) != 0) return -1;
    for (int i = 0; i < K; i++)
        for (int j = i + 1; j < K; j++) { A[(size_t)i*K + j] = A[(size_t)j*K + i]; A[(size_t)j*K + i] = 0; }
    return 0;
}
