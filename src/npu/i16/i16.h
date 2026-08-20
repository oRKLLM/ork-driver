/* npu/i16/i16.h — the i16 subtree's internal interface.
 *
 * The i16 sub-modules are one datapath split for readability, not independent components; they call
 * each other freely. This declares what they share, plus the types and state the datapath owns.
 * Same role npu/i8/i8.h plays for int8. Nothing outside src/npu/i16/ should include it —
 * that is why it lives INSIDE the folder, where npu/core.h and npu/internal.h (tree-wide) sit beside
 * their subsystems. If the scaffold needs a symbol from here, declare it in internal.h instead. */
#ifndef ORK_NPU_I16_H
#define ORK_NPU_I16_H
#include "npu/internal.h"
#include "npu/core.h"



int ork_i16_npu_add(ork_npu *c,const int16_t *a,const int16_t *b,int M,int N, double a_scale,double b_scale,double out_scale,int16_t *out,double *us);
int ork_i16_npu_chain_gatesilu(ork_npu *c,int M,int K,int N,const int8_t *A,const int8_t *B, int mult,int shift,double in_scale,double out_scale, int16_t *gate_out,int16_t *out,double *us);
int ork_i16_npu_chain_mm_perchan(ork_npu *c,int M,int K,int N,const int8_t *A,const int8_t *B, const int16_t *scale,int m1,int s1,int m2,int s2,int16_t *out,double *us);
int ork_i16_npu_chain_mm_silu(ork_npu *c,const int16_t *in,int M,int N,double in_scale,double out_scale, int16_t *out,int *mm_ran,double *us);
int ork_i16_npu_ewmul(ork_npu *c,const int16_t *up,const int16_t *silu,int M,int N,int mult,int shift,int16_t *out,double *us);
int ork_i16_npu_exp(ork_npu *c,const int16_t *in,int M,int N,double in_scale,double out_scale,int16_t *out,double *us);
int ork_i16_npu_gelu(ork_npu *c,const int16_t *in,int M,int N,double in_scale,double out_scale,int16_t *out,double *us);
int ork_i16_npu_mul_perchan(ork_npu *c,const int16_t *a,const int16_t *b,int M,int N,int mult,int shift,int16_t *out,double *us);
int ork_i16_npu_probe_out(ork_npu *c,int M,int K,int N,const int8_t *A,const int8_t *B, int mult,int shift,int16_t *C,double *us);
int ork_i16_npu_probe_silu_std(ork_npu *c,const int16_t *in,int M,int N, int r_mult,int r_shift,uint32_t out_bias,uint32_t idx_off, uint32_t cfg4064,uint32_t cfg4068,const int16_t *lut,int nlut, int16_t *out,double *us);
int ork_i16_npu_replay_lut(ork_npu *c,const uint32_t *regcmd,int rn,const int16_t *lut,int nlut, const int16_t *in,int M,int N,int16_t *out,double *us);
int ork_i16_npu_rsqrt(ork_npu *c,const int16_t *in,int M,int N,double in_scale,double out_scale,int16_t *out,double *us);
int ork_i16_npu_silu(ork_npu *c,const int16_t *in,int M,int N,double in_scale,double out_scale,int16_t *out,double *us);
int orki_i16_act_lut(ork_npu *c,double(*f)(double),const int16_t *in,int M,int N,double in_scale,double out_scale,int16_t *out,double *us);
void orki_i16_set_out(uint32_t*rc,int N,int stride,int mult,int shift);

/* module-owned types and state */


#endif /* ORK_NPU_I16_H */
