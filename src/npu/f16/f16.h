/* npu/f16/f16.h — the f16 subtree's internal interface.
 *
 * The f16 sub-modules are one datapath split for readability, not independent components; they call
 * each other freely. This declares what they share, plus the types and state the datapath owns.
 * Same role npu/i8/i8.h plays for int8. Nothing outside src/npu/f16/ should include it —
 * that is why it lives INSIDE the folder, where npu/core.h and npu/internal.h (tree-wide) sit beside
 * their subsystems. If the scaffold needs a symbol from here, declare it in internal.h instead. */
#ifndef ORK_NPU_F16_H
#define ORK_NPU_F16_H
#include "npu/internal.h"
#include "npu/core.h"



int ork_bmm_fp16(ork_npu *c, int nbatch, int M, int K, int N, const f16 *A, const f16 *B, float *C);
int ork_bmm_fp16_fused(ork_npu*c,int nb,int M,int K,int N,const f16*A,const f16*B,float*C);
int ork_bmm_fp16_stream(ork_npu*c,int nb,int M,int K,int N,const f16*A,const f16*B,float*C);
int ork_bmm_fp16_strided(ork_npu *c, int nbatch, int M, int K, int N, const f16 *A, const f16 *B, float *C, const ork_bmm_strides *s);
int ork_f16_colsplit(void);
int ork_mm_build_f16_lut(ork_npu *c, double (*fn)(double,void*), void *fnctx, double in_lo, double in_hi, int16_t *lut, double *S_out, double *R_out, double *out_scale_out);
int ork_mm_build_f16_rsqrt_lut(ork_npu *c, int n_feat, double eps, double ss_min, double ss_max, int16_t *lut, double *S_out, double *R_out, double *out_scale_out);
int ork_mm_build_f16_silu_lut(ork_npu *c, double Gmax, int16_t *lut, double *S_out, double *R_out, double *out_scale_out);
int ork_mm_inflate_i8_to_f16(ork_npu *c,ork_w *w,const int8_t *i8,const float *bscale,int K,int N);
int ork_mm_repack_f16(ork_npu *c,ork_w *w,int K,int N,const f16 *B);
int ork_mm_run_f16_act(ork_npu *c, int K, int N, const ork_f16 *B, int M, const ork_f16 *A, float *C, double (*fn)(double,void*), void *fnctx, double in_lo, double in_hi);
int ork_mm_run_f16_f16out(ork_npu *c, ork_w *w, int M, const ork_f16 *A, ork_f16 *out);
int ork_mm_run_f16_fused_act(ork_npu *c, ork_w *w, int M, const ork_f16 *A, float *C);
int ork_mm_run_f16_silu(ork_npu *c,ork_w *w,int M,const ork_f16 *A,float *C, uint32_t out_bias,uint32_t idx_off,uint32_t cfg4068,const int16_t *lut,int nlut);
int ork_mm_run_stream_f16(ork_npu *c, int S, const ork_mm_task_f16 *tasks);
int ork_mm_run_stream_f16_chain(ork_npu *c, int S, const ork_mm_task_f16 *tasks);
int ork_npu_add_f16(ork_npu *c,const ork_f16 *a,const ork_f16 *b,int M,int N,ork_f16 *out,double *us);
int ork_npu_chain_mm_perchan_f16(ork_npu *c,int M,int K,int N,const uint16_t *A,const uint16_t *B, const uint16_t *scale,uint16_t *out,double *us);
int ork_npu_ewmul_f16(ork_npu *c,const ork_f16 *up,const ork_f16 *silu,int M,int N,ork_f16 *out,double *us);
int ork_npu_f16_gap_probe(ork_npu *c, int M, int Kp, int N, int use_gap, long *nz0, long *nz1, double *us);
int ork_npu_f16_percore_probe(ork_npu*c,int M,int K,int N,const ork_f16*A,const ork_f16*B,float*Cout,double*us,int mode);
int ork_npu_l2norm_f16(ork_npu *c,int M,int n,const f16 *x,float eps,f16 *out);
int ork_npu_mm_perchan_f16(ork_npu *c,int M,int K,int N,const uint16_t *A,const uint16_t *B, const uint16_t *scale,uint16_t *out,double *us);
int ork_npu_mm_perchan_f16_diag(ork_npu *c,int M,int K,int N,const uint16_t *A,const uint16_t *B, const uint16_t *scale,uint16_t *out,double *us);
int ork_npu_mm_perchan_f16_fused(ork_npu *c,int M,int K,int N,const uint16_t *A,const uint16_t *B, const uint16_t *scale,uint16_t *out);
int ork_npu_mul_perchan_f16(ork_npu *c,const ork_f16 *a,const ork_f16 *b,int M,int N,ork_f16 *out,double *us);
int ork_npu_mul_perchan_f16_contig(ork_npu *c,const ork_f16 *a,const ork_f16 *b,int M,int N,ork_f16 *out,double *us);
int ork_npu_probe_f16_mm(ork_npu *c,int M,int K,int N,const uint16_t *A,const uint16_t *B,float *raw);
int ork_npu_probe_f16_mm_f16out(ork_npu *c,int M,int K,int N,const uint16_t *A,const uint16_t *B,uint16_t *out);
int ork_npu_probe_f16_stridedA(ork_npu *c,int M,int K,int N,const uint16_t *A,int apitch,const uint16_t *B,uint16_t *out);
int ork_npu_probe_silu_std_f16(ork_npu *c,const ork_f16 *in,int M,int N, uint32_t idx_off,uint32_t cfg4064,uint32_t cfg4068, const int16_t *lut,int nlut,ork_f16 *out,double *us);
int ork_npu_probe_slice_f16(ork_npu *c,int Kfull,int N,int Kp,int nov, const uint32_t *ovr_reg,const uint32_t *ovr_val, const f16 *A,const f16 *B,float *C);
int ork_npu_replay_full_f16(ork_npu *c,const uint32_t *loader,int ln,const ork_f16 *in,int M,int N,ork_f16 *out,double *us);
int ork_npu_replay_reshape_f16(ork_npu *c,uint16_t *gemm_raw,int gemm_words,uint16_t *reshape_raw,int reshape_words,double *us);
int ork_npu_replay_softmax_f16(ork_npu *c, const void *in, void *out, double *us);
int ork_npu_rmsnorm_f16(ork_npu *c,int M,int n,const f16 *x,const f16 *w,float eps,f16 *out);
int ork_npu_rope_neox_f16(ork_npu *c, const ork_f16 *x, int hd, int nrow, const int *pos, double freq_base, ork_f16 *out);
int ork_npu_softmax_f16(ork_npu *c,int M,int n,const f16 *x,f16 *out);
int ork_ssd_probe_fusedmm_f16(ork_npu*c,int M,int K,int N,const f16*A,const f16*B,float*C);
int ork_ssd_probe_rawmm_f16(ork_npu*c,int M,int K,int N,const f16*A,const f16*B,float*C);
ork_w *ork_mm_f16_scratch(ork_npu *c,int K,int N);
ork_w *ork_mm_pack_f16_fused_act(ork_npu *c, int K, int N, const ork_f16 *B, double (*fn)(double,void*), void *fnctx, double in_lo, double in_hi);
void ork_f16_fuzz_add(uint32_t blk,uint32_t reg,uint32_t val);
void ork_f16_fuzz_clear(void);
void orki_set_f16_out(uint32_t*rc,int N,int stride);
void orki_set_f16_out_fp16in(uint32_t*rc,int M,int N);
void orki_synth(uint32_t*rc,int mc,int K,int N,uint32_t aA,uint32_t aB,uint32_t aC,int sched,int cbuf);

void *ork_pcfd_thread(void *vp);

/* module-owned types and state */
struct tile_i8f16_arg { f16 *bb; const int8_t *Bi; const float *bscale; int KT, k0, n0, N; };
struct f16lut_rsqrt_ctx { int n_feat; double eps; };
struct f16act_neg { double (*fn)(double,void*); void *ctx; };
struct ork_rsh_patch { ork_npu *c; struct rknpu_task marker; uint32_t idx; uint32_t delay_us;
    int rcmode; uint32_t *rcword; uint32_t rcval0,rcval1; };  /* rcmode: patch a regcmd word in c->regcmd instead of a descriptor */
struct ork_pcfd_arg { int fd, core; struct buf *tk; int rc; };
struct streamw_f16 { ork_npu *c; int core; int S; const ork_mm_task_f16 *tasks; int *ctr; int rc; };
struct streamw_f16ch { ork_npu *c; int core; int ncore; int S; const ork_mm_task_f16 *tasks; int rc; };
extern struct ork_regovr orki_f16_fovr[16];
extern int orki_f16_fovr_n;

int orki_f16_mtile(int K,int M);

#endif /* ORK_NPU_F16_H */
