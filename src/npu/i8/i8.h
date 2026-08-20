/* npu/i8/i8.h — the int8 subtree's internal interface.
 *
 * The seven i8 sub-modules (regcmd/pack/fold/run/chain/dyn/probe) are one datapath split for
 * readability, not seven independent components -- they call each other freely. This declares what
 * they share, the same declare-then-move discipline npu/core.h applies to the substrate, one level
 * down. Nothing outside src/npu/i8/ should include it —
 * that is why it lives INSIDE the folder, where npu/core.h and npu/internal.h (tree-wide) sit beside
 * their subsystems. If the scaffold needs a symbol from here, declare it in internal.h instead. */
#ifndef ORK_NPU_I8_H
#define ORK_NPU_I8_H
#include "npu/internal.h"
#include "npu/core.h"

int ork_i8_bmm(ork_npu *c, int nbatch, int M, int K, int N, const int8_t *A, const int8_t *B, int32_t *C);
int ork_i8_bmm_strided(ork_npu *c, int nbatch, int M, int K, int N, const int8_t *A, const int8_t *B, int32_t *C, const ork_bmm_strides *s);
int ork_dyn_append(ork_dyn_chain *h, const ork_mm_task_i8 *task);
int ork_dyn_end(ork_dyn_chain *h);
int ork_dyn_halt(ork_dyn_chain *h, int at);
int ork_dyn_max_steps(void);
int ork_dyn_progress(ork_dyn_chain *h);
int ork_dyn_queue_drain(ork_dyn_queue *q);
int ork_dyn_queue_flush(ork_dyn_queue *q);
int ork_dyn_queue_idle(ork_dyn_queue *q);
int ork_dyn_queue_linger_us(ork_dyn_queue *q);
int ork_dyn_queue_pending(ork_dyn_queue *q);
int ork_dyn_queue_push(ork_dyn_queue *q, const ork_mm_task_i8 *task);
int ork_dyn_remaining(ork_dyn_chain *h);
int ork_dyn_seq_end(ork_dyn_chain *h);
int ork_dyn_spin_probe(ork_npu *c, int S, const ork_mm_task_i8 *tasks, int spin_us, int *spin_alive);
int ork_dyn_steps(ork_dyn_chain *h);
int ork_kv_append(ork_npu *c, ork_kv_resident *kv, int key, const int8_t *kcol, const int8_t *vrow);
int ork_i8_mm_layer(ork_npu *c, const struct ork_layer_dims *d, ork_w *wq, ork_w *wk, ork_w *wv, ork_w *wo, ork_w *wg, ork_w *wu, ork_w *wd, const float *attn_norm, const float *q_norm, const float *ffn_norm, const float *x, const float *Kc, const float *Vc, float *x_out);
int ork_i8_mm_repack(ork_npu *c,ork_w *w,int K,int N,const int8_t *B);
int ork_i8_mm_repack_dequant(ork_npu *c, ork_w *w, int K, int N, ork_dequant_row_fn fn, void *dctx, float *bscale_out);
int ork_i8_mm_repack_f32(ork_npu *c, ork_w *w, int K, int N, const float *f32, float *bscale_out);
int ork_i8_mm_run_chain(ork_npu *c, int S, const ork_mm_task_i8 *tasks);
int ork_i8_mm_run_chain_ffn(ork_npu *c, int S, const ork_mm_task_i8 *tasks, const ork_chain_op *ops, double in_scale, double out_scale);
int ork_i8_mm_run_chain_ffn_exp(ork_npu *c, int S, const ork_mm_task_i8 *tasks, const ork_chain_op *ops, double in_scale, double out_scale);
int ork_i8_mm_run_chain_ffn_exp_biased(ork_npu *c, int S, const ork_mm_task_i8 *tasks, const ork_chain_op *ops, double in_scale, double out_scale, double max_bias);
int ork_i8_mm_run_chain_gsilu(ork_npu *c, int S, const ork_mm_task_i8 *tasks, int gate_task, int r_mult, int r_shift, uint32_t out_bias, uint32_t idx_off, uint32_t cfg4068, const int16_t *lut, int nlut);
int ork_i8_mm_run_chain_sdpsilu(ork_npu *c, int S, const ork_mm_task_i8 *tasks, int sdp_task, int gate_mult, int gate_shift, double in_scale, double out_scale);
int ork_i8_mm_run(ork_npu *c,ork_w *w,int M,const int8_t *A,int32_t *C);
int ork_i8_mm_run_ewmul(ork_npu *c,ork_w *w,int M,const int8_t *A,const int8_t *G,int8_t *C,int mult,int shift);
int ork_i8_mm_run_out16(ork_npu *c,ork_w *w,int M,const int8_t *A,short *C,int mult,int shift);
int ork_i8_mm_run_out8(ork_npu *c,ork_w *w,int M,const int8_t *A,int8_t *C,int mult,int shift);
int ork_i8_mm_run_silu(ork_npu *c,ork_w *w,int M,const int8_t *A,int8_t *C, int r_mult,int r_shift,uint32_t out_bias,uint32_t idx_off,uint32_t cfg4068, const int16_t *lut,int nlut);
int ork_i8_mm_run_silu32(ork_npu *c,ork_w *w,int M,const int8_t *A,int32_t *C, int r_mult,int r_shift,uint32_t out_bias,uint32_t idx_off,uint32_t cfg4068, const int16_t *lut,int nlut);
int ork_i8_mm_run_stream(ork_npu *c, int S, const ork_mm_task_i8 *tasks);
int ork_i8_mm_run_stream_sk(ork_npu *c, int S, const ork_mm_task_i8 *tasks);
int ork_mm_silu_build_lut(ork_npu*c, double in_scale, double out_scale, int r_mult, int r_shift, uint32_t cfg4068, int16_t *lut);
int ork_i8_npu_add(ork_npu *c,const int8_t *a,const int8_t *b,int M,int N, double a_scale,double b_scale,double out_scale,int8_t *out,double *us);
int ork_i16_npu_chain_gatesilu(ork_npu *c,int M,int K,int N,const int8_t *A,const int8_t *B, int mult,int shift,double in_scale,double out_scale, int16_t *gate_out,int16_t *out,double *us);
int ork_i16_npu_chain_mm_silu(ork_npu *c,const int16_t *in,int M,int N,double in_scale,double out_scale, int16_t *out,int *mm_ran,double *us);
int ork_i16_npu_ewmul(ork_npu *c,const int16_t *up,const int16_t *silu,int M,int N,int mult,int shift,int16_t *out,double *us);
int ork_i8_npu_ewmul(ork_npu *c,const int8_t *up,const int8_t *silu,int M,int N,int mult,int shift,int8_t *out,double *us);
int ork_i8_npu_exp(ork_npu *c,const int8_t *in,int M,int N,double in_scale,double out_scale,int8_t *out,double *us);
int ork_i8_npu_exp_biased(ork_npu *c,const int8_t *in,int M,int N,double in_scale,double out_scale,double max,int8_t *out,double *us);
int ork_npu_fold_batch(ork_npu *c, int Mtot, int K, int N, int P, const int *row_off, const uint32_t *tiles, int rn, const int8_t *Apacked, const int8_t *Bpacked, int32_t *Craw, int ncore, int iters, double *us);
int ork_npu_fold_batch_w(ork_npu*c, int nw, ork_w**ws, int M, const int8_t*Araw, int32_t**Couts, int iters, double*us);
int ork_i8_npu_fold_op(ork_npu*c,int K,int N,const int8_t*Wraw,int M,const int8_t*Araw,int32_t*Cout,int iters,double*us);
int ork_i8_npu_fold_run(ork_npu*c,int K,int N,const int8_t*Wraw,int M,const int8_t*Araw,int32_t*Cout,int ncore,int iters,double*us);
int ork_npu_fold_run_w(ork_npu*c, ork_w*w, int M, const int8_t*Araw, int32_t*Cout, int iters, double*us);
int ork_i8_npu_gelu(ork_npu *c,const int8_t *in,int M,int N,double in_scale,double out_scale,int8_t *out,double *us);
int ork_npu_mfold_chain_v(ork_npu *c, int P, const int *ws, int K, int N, const uint32_t *tiles, int rn, const int8_t *Apacked, const int8_t *Bpacked, int32_t *Craw, int wreuse, int iters, double *us);
int ork_i8_npu_mul_perchan(ork_npu *c,const int8_t *a,const int8_t *b,int M,int N,int mult,int shift,int8_t *out,double *us);
int ork_i8_npu_probe_add(ork_npu *c,const int8_t *a,const int8_t *b,int M,int N, int mult,int shift,uint32_t bscale,int za,int zb,int zo,int8_t *out,double *us);
int ork_i8_npu_probe_ewmul(ork_npu *c,int M,int K,int N,const int8_t *A,const int8_t *B,const int8_t *G, int mult,int shift,int8_t *C,double *us);
int ork_i8_npu_probe_ewmul_lin(ork_npu *c,const int8_t *A,const int8_t *B,const int8_t *G,int8_t *C,double *us);
int ork_i8_npu_probe_ewmul_tmpl(ork_npu *c,const void*in,int Isz,const void*wt,int Wsz, const void*gl,int Gsz,void*out,int Osz,double *us);
int ork_i8_npu_probe_mm(ork_npu *c,int M,int K,int N,const int8_t *A,const int8_t *B,int32_t *raw);
int ork_i8_npu_probe_mul(ork_npu *c,const int8_t *a,const int8_t *b,int n,int8_t *out,double *us);
int ork_i8_npu_probe_out8(ork_npu *c,int M,int K,int N,const int8_t *A,const int8_t *B, int mult,int shift,int8_t *C,double *us);
int ork_i8_npu_probe_silu(ork_npu *c,int M,int K,int N,const int8_t *A,const int8_t *B,int8_t *C,double *us);
int ork_i8_npu_probe_silu_cfg(ork_npu *c,int M,int K,int N,const int8_t *A,const int8_t *B, int r_mult,int r_shift,uint32_t out_bias,uint32_t idx_off,uint32_t cfg4068, const int16_t *lut,int nlut,int8_t *C,double *us);
int ork_i8_npu_probe_mtile(ork_npu *c,int M,int K,int N,int mode, const int8_t *A,const int8_t *B,int32_t *C,double *us);
int ork_i8_npu_probe_silu_std(ork_npu *c,const int8_t *in,int M,int N, int r_mult,int r_shift,uint32_t out_bias,uint32_t idx_off, uint32_t cfg4064,uint32_t cfg4068,const int16_t *lut,int nlut, int8_t *out,double *us);
int ork_i16_npu_probe_silu_std(ork_npu *c,const int16_t *in,int M,int N, int r_mult,int r_shift,uint32_t out_bias,uint32_t idx_off, uint32_t cfg4064,uint32_t cfg4068,const int16_t *lut,int nlut, int16_t *out,double *us);
int ork_i8_npu_probe_single(ork_npu *c,int K,int N,const int8_t *A,const int8_t *B,int32_t *C);
int ork_i8_npu_replay(ork_npu *c, const uint32_t *regcmd, int rn, int M, int K, int N, const int8_t *Adata, int Abytes, const int8_t *Bdata, int Bbytes, int32_t *Cout, int iters, double *us);
int ork_i8_npu_replay_sweep(ork_npu *c, const uint32_t *regcmd, int rn, int M, int K, int N, const int8_t *Avar, int nvar, int astride, const int8_t *Bdata, int Bbytes, int32_t *Couts);
int ork_i8_npu_row_max(ork_npu *c, const int8_t *a, int M, int N, int8_t *out, double *us);
int ork_i8_npu_rsqrt(ork_npu *c,const int8_t *in,int M,int N,double in_scale,double out_scale,int8_t *out,double *us);
int ork_i16_npu_silu(ork_npu *c,const int16_t *in,int M,int N,double in_scale,double out_scale,int16_t *out,double *us);
int ork_i8_npu_silu(ork_npu *c,const int8_t *in,int M,int N,double in_scale,double out_scale,int8_t *out,double *us);
int ork_i8_npu_synth_dump(ork_npu *c, int mc, int K, int N, unsigned *out, int outn);
int ork_pc_run(ork_pc_chain *pc);
int ork_i8_w_attach_fold(ork_npu *c, ork_w *w, const void *blob, size_t n);
ork_async *ork_i8_mm_run_chain_async (ork_npu *c, int S, const ork_mm_task_i8 *tasks);
ork_async *ork_i8_mm_run_async (ork_npu *c, ork_w *w, int M, const int8_t *A, int32_t *C);
ork_async *ork_i8_mm_run_stream_async(ork_npu *c, int S, const ork_mm_task_i8 *tasks);
ork_dyn_chain *ork_dyn_begin(ork_npu *c, int S, const ork_mm_task_i8 *tasks);
ork_dyn_chain *ork_dyn_begin_mc(ork_npu *c, int S, const ork_mm_task_i8 *tasks, int nc);
ork_dyn_chain *ork_i8_dyn_begin_seq(ork_npu *c, int n, const ork_seq_op *ops);
ork_dyn_chain *ork_i8_dyn_begin_seq_mc(ork_npu *c, int n, const ork_seq_op *ops, int ngroups, const int *gstart, int nc);
ork_dyn_queue *ork_dyn_queue_create(ork_npu *c, int chunk_max, int ncore);
ork_kv_resident *ork_kv_resident_alloc(ork_npu *c, int HD, int Lmax);
ork_pc_chain *ork_pc_compile(ork_npu *c, int S, const ork_mm_task_i8 *tasks);
ork_w *ork_i8_mm_adopt_imported(ork_npu *c,int K,int N,int bb_fd,int bf_fd,size_t bb_bytes,size_t bf_bytes);
ork_w *ork_i8_mm_import(ork_npu *c,int K,int N,const void *blob,size_t n,size_t bf_off);
ork_w *ork_i8_mm_load_fold(ork_npu *c,int K,int N,const void *blob,size_t n);
ork_w *ork_i8_mm_load(ork_npu *c,int K,int N,const void *blob,size_t n);
ork_w *ork_i8_mm_load_flags(ork_npu *c,int K,int N,const void *blob,size_t n,unsigned flags);
ork_w *ork_i8_mm_load_import(ork_npu *c,int K,int N,const void *blob,size_t n);
ork_w *ork_i8_mm_pack(ork_npu *c,int K,int N,const int8_t *B);
ork_w *ork_i8_mm_pack_dequant(ork_npu *c, int K, int N, ork_dequant_row_fn fn, void *dctx, float *bscale_out);
ork_w *ork_i8_mm_pack_f32(ork_npu *c, int K, int N, const float *f32, float *bscale_out);
ork_w *ork_i8_mm_pack_import(ork_npu *c,int K,int N,const int8_t *B);
size_t ork_i8_w_dump_bf_cpu(ork_npu *c, int K, int N, const int8_t *B, void *out, size_t cap);
size_t ork_i8_w_dump_fold_cpu(ork_npu *c, int K, int N, const int8_t *B, void *out, size_t cap);
size_t ork_i8_w_dump_cpu(ork_npu *c, int K, int N, const int8_t *B, void *out, size_t cap);
size_t ork_i8_w_dump_cpu_st(ork_npu *c, int K, int N, const int8_t *B, void *out, size_t cap);
struct ork_stream_entry *ork_i8_stream_pool_add(struct ork_stream_pool *p, int K, int N, const void *blob, size_t n);
struct ork_stream_entry *ork_i8_stream_pool_add_raw(struct ork_stream_pool *p, int K, int N, const int8_t *B);
void ork_dyn_dump(ork_dyn_chain *h, const char *label);
void ork_dyn_queue_destroy(ork_dyn_queue *q);
void ork_dyn_queue_set_linger(ork_dyn_queue *q, int us);
void ork_i8_fuzz_add(uint32_t blk,uint32_t reg,uint32_t val);
void ork_i8_fuzz_clear(void);
void ork_kv_resident_free(ork_npu *c, ork_kv_resident *kv);
void ork_pc_free(ork_pc_chain *pc);
void ork_i8_slice_direct_inflate(const ork_w *w, int8_t *i8, int kind);
void ork_i8_slice_tile(ork_npu *c, ork_w *w, const float *qf32, float *inv1);
void ork_i8_stage_fill(ork_npu *c, struct ork_stage *s, const int8_t *B);
void orki_fold_scratch_free(ork_npu *c);
void orki_i8_synth(uint32_t*rc,int mc,int K,int N,uint32_t aA,uint32_t aB,uint32_t aC,int sched,int cbuf,int stride);
void orki_i8_synth_mfold(uint32_t*rc,int mc,int K,int N,uint32_t aA,uint32_t aB,uint32_t aC,int cbuf);

struct ork_csub { ork_npu *c; int i; struct rknpu_submit *subs; ork_w *w; ork_dyn_chain *h; int hardened; int active; int ksbar; };

struct streamw { ork_npu *c; int core; int S; const ork_mm_task_i8 *tasks; int *ctr; int rc; };

#endif /* ORK_NPU_I8_H */
