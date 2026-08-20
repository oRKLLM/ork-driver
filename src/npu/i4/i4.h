/* npu/i4/i4.h — the int4 subtree's internal interface.
 *
 * The i4 sub-modules (quant/pack/run/chain/stream) are one datapath split for readability, not
 * independent components; they call each other freely. This declares what they share plus the types the
 * datapath owns. Same role npu/i8/i8.h plays for int8. Nothing outside src/npu/i4/ should include it —
 * that is why it lives INSIDE the folder, where npu/core.h and npu/internal.h (tree-wide) sit beside
 * their subsystems. If the scaffold needs a symbol from here, declare it in internal.h instead. */
#ifndef ORK_NPU_I4_H
#define ORK_NPU_I4_H
#include "npu/internal.h"
#include "npu/core.h"

/* module-owned types and constants */
#define ORK_I4_SENT16 ((int16_t)0x7fff)
struct bchdbw { ork_npu *c; int core, c0, c1, NT, K, N, NG, M, H, Wb, Wmax; ork_w *w; const int8_t *A; int32_t *C; unsigned dom; struct rknpu_submit sub; int rc; };
struct bchmw { ork_npu *c; int core, e0, e1; const ork_mm_task_i4 *ex; int K,N,H,Wb,Wmax,NC; unsigned dom; struct rknpu_submit sub; int rc; };
struct streamw4 { ork_npu *c; int core; int S; const ork_mm_task_i4 *tasks; int *ctr; int rc; };

int ork_i4_bmm(ork_npu *c, int nbatch, int M, int K, int N, const int8_t *A, const int8_t *B, int32_t *C);
int ork_i4_bmm_strided(ork_npu *c, int nbatch, int M, int K, int N, const int8_t *A, const int8_t *B, int32_t *C, const ork_bmm_strides *s);
int ork_i4_dyn_probe(ork_npu *c, int S, const ork_mm_task_i4 *tasks);
int ork_i4_batch(void);
int ork_i4_mm_run_chain(ork_npu *c, int S, const ork_mm_task_i4 *tasks);
int ork_i4_mm_run(ork_npu *c,ork_w *w,int M,const int8_t *A,int32_t *C);
int ork_i4_mm_run_experts(ork_npu *c, const ork_mm_task_i4 *ex, int ntask, int nc);
int ork_i4_mm_run_grouped(ork_npu *c,ork_w *w,int M,const int8_t *A,const float *aScale,const float *bScale,float *C);
int ork_i4_mm_run_stream(ork_npu *c, int S, const ork_mm_task_i4 *tasks);
int ork_i4_npu_probe(ork_npu *c,int M,int K,int N,int nibB,int nibA,int nov, const uint32_t *ovr_reg,const uint32_t *ovr_val, const int8_t *A,const int8_t *B,int16_t *C);
int ork_i4_npu_probe_mm(ork_npu *c,int M,int K,int N,const int8_t *A,const int8_t *B,int16_t *raw);
int orki_bch_db_cells(ork_npu *c,int i,int c0,int c1,int Wb,int N,int NG,int M,int H,int Wmax,int32_t *C,int mode,int only_tk);
int orki_i4_submit_tmo_ms(void);
int orki_i4_run_bchain_db(ork_npu *c, ork_w *w, int M, const int8_t *A, int32_t *C, int nc);
int orki_i4_run_experts_bchain_db(ork_npu *c, const ork_mm_task_i4 *ex, int ntask, int nc);
int orki_i4_slice_run(ork_npu *c, ork_w_sliced *w, int M, const int8_t *A, int32_t *C, int nc);
ork_w *ork_i4_mm_load(ork_npu *c,int K,int N,const void *blob,size_t n);
ork_w *ork_i4_mm_load_arena(ork_npu *c,int K,int N,const void *blob,size_t n);
ork_w *ork_i4_mm_load_import(ork_npu *c,int K,int N,const void *blob,size_t n);
ork_w *ork_i4a8_mm_load(ork_npu *c, int K, int N, const void *blob, size_t n);
ork_w *ork_i4a8_mm_load_import(ork_npu *c, int K, int N, const void *blob, size_t n);
ork_w *ork_i4_mm_pack(ork_npu *c,int K,int N,const int8_t *B);
ork_w *ork_i4_mm_pack_grouped(ork_npu *c,int K,int N,const int8_t *B,int G);
ork_w *ork_i4_mm_pack_to_i8(ork_npu *c, int K, int N, const int8_t *B);
ork_w *ork_i4a8_mm_pack(ork_npu *c, int K, int N, const float *f32, float *bscale_out);
ork_w *ork_i4a8_mm_pack_im(ork_npu *c, int K, int N, const float *f32, const float *imatrix, float *bscale_out);
ork_w_sliced *orki_i4_slice_pack(ork_npu *c, int K, int N, const int8_t *B);
size_t ork_i4a8_pack_cpu_blob(ork_npu *c, int K, int N, const float *f32, const float *imatrix, int nf4, void *out, size_t cap);
size_t ork_i4a8_w_dump(const ork_w *w, void *out, size_t cap);
struct ork_stream_entry *ork_i4a8_stream_pool_add(struct ork_stream_pool *p, int K, int N, const void *blob, size_t n);
void ork_i4_fuzz_add(uint32_t blk,uint32_t reg,uint32_t val);
void ork_i4_fuzz_clear(void);
void ork_i4a8_slice_direct_kind(ork_npu *c, ork_w *w, int8_t *i8scratch, int kind);
void ork_i4a8_slice_inflate(const ork_w *w, float *qf32);
void ork_i4a8_slice_inflate_kind(const ork_w *w, float *qf32, int kind);
void orki_i4_expand_chan_to_i8(const uint8_t *nib, int K, int8_t *i8);
void orki_i4_synth(uint32_t*rc,int mc,int K,int N,uint32_t aA,uint32_t aB,uint32_t aC);
void orki_i4_tile_Aslice(uint8_t*dst,const int8_t*Arow,int k0,int Kp);

/* cross-file within the i4 subtree */
void orki_i4_expand_chan_f32(const uint8_t *nib, int K, float *qf32);
void orki_nf4_inflate_chan_f32(const uint8_t *qidx, int K, const int8_t lut[16], float *qf32);
void orki_i4_quant_chan(const float *fr, int K, float scale, int sr, uint32_t *seed, uint8_t *nib, float *qf32);
void orki_nf4_quant_chan(const float *fr, int K, float absmax, int sr, uint32_t *seed, uint8_t *nib, uint8_t *qidx);
void orki_i4_tile_direct_to_i8(ork_npu *c, ork_w *w, int K, int N, int kind, int8_t *i8scratch);
void orki_i4_tile_A(uint8_t*dst,const int8_t*A,int M,int K,int nib);
void orki_i4_tile_B(uint8_t*dst,const int8_t*B,int K,int N,int nib);
float orki_wq_best_absmax(const float *fr, int K, float rawabsmax, int nf4, const float *im, float *dq);

extern int orki_i4_validate;

#endif /* ORK_NPU_I4_H */
