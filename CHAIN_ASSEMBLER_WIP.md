# Chain assembler — WIP recovery doc

Goal (user: "build the chain assembler next"): chain a whole NPU-only run (FFN inner; attention block)
into ONE `task_number=N` submit, so per-op round-trips collapse. This is THE submit-reduction lever
(per-node on-NPU regressed: 9807 submits, 76 vs 97 tok/s). Board 10.3.0.236 (non-prod, wedge-OK).
Branch: ork-driver `feat/static-graph`. All gated; default path untouched (~97 tok/s).

## Landed
- **Core `ork_npu_chain_progs(c, n, progs[], dom)`** (commit 5755089): submits N pre-built heterogeneous
  programs as one PC-chain. Packs contiguously into c->regcmd (content-driven stride); for each non-last
  program, SCANS for its PC next-descriptor slot (word where reg==0x0010 paired with domain 0x0101) and
  links next-addr + next-regcfg-amount. One rknpu_task/program (own enable_mask/regcfg), ping-pong OFF.
  Compiles clean on board. Mechanism already validated by run_chain_i8 (homogeneous) + Phase-0 (matmul->silu).

## Key facts / mechanics
- Next-descriptor: `rc[slot]=0x0010|(next_lo<<16); rc[slot+1]=0x0101<<16|next_hi; rc[slot+2]=0x0014|(nreg<<16);
  rc[slot+3]=0x0101<<16` where `nreg=(next.regcfg_amount+3)/2`. Last program keeps its raise-interrupt tail.
- enable_mask: matmul=0xd (regcfg 108), SDP silu/ewmul=0x18 (silu regcfg 69).
- Matmul int16 output: `set_i16_out(rc,N,stride,mult,shift)` rewrites the DPU output stage (0x4010 precision,
  0x4038 group stride N/8-denser, 0x4050 row byte-stride 0x0248, 0x40c0 elem=2B, 0x4084/0x4088 requant
  mult/shift). Validated standalone via ork_npu_probe_i16_out.
- **Phase-0 (ork_npu_chain_mm_silu_i16) only proved the chain WALKS** — its matmul output (Cd) and silu
  input (A) are SEPARATE buffers, NOT data-connected. The real FFN chain needs matmul-out == silu-in.

## THE CRUX (increment 1, in progress)
Data-connected matmul-int16-out -> silu-int16-in: does the int8-matmul int16 OUTPUT layout
(set_i16_out: 0x4050=0x0248 row byte-stride, 0x4038 group stride, cube tiling) match the int16 silu's
EXPECTED INPUT layout (0x5018 input addr; Phase-0's EWCUBEH cube addressing)? If yes, point silu-in at
the matmul's int16 output buffer and the bridge works. If not -> RE the layout xform (may need a repack
or a matching stride on the silu input regs).

## FFN chain plan (SwiGLU), each transition a bridge to validate
1. gate = A[M,K] x Wg[K,N]  (int8 matmul, set_i16_out -> int16 G)      [0xd]
2. silu(G) -> S (int16)                                                 [0x18]   <- increment 1 = 1->2 bridge
3. up   = A[M,K] x Wu[K,N]  (int8 matmul, set_i16_out -> int16 U)      [0xd]
4. glu  = S (*) U -> H (ewmul int16)                                    [0x18]
5. down = H[M,N] x Wd[N,K]  (int8 matmul; N=6144 contraction -> K-split -> multiple programs) [0xd...]
- down K-split = multiple chained matmul programs w/ accumulate; the one many-program sub-chain.
- Intermediate buffers G/S/U/H resident in one domain; addresses threaded through each op's regcmd.

## Next steps
1. [in progress] Build + validate increment-1 data-connected matmul-i16-out -> silu chain (new fn +
   chain_gatesilu_probe). Coherence = silu(matmul(A,W)) within tol; rc=0 no wedge.
2. Add up + glu (ewmul) -> 4-op chain; validate.
3. Add down (K-split sub-chain) -> full FFN inner one submit; validate vs separate-op FFN.
4. Wire into ggml-ork FFN handler behind a flag; measure prefill vs 97 baseline + coherence (ork_ppl).
5. Attention-block chain (QK^T + softmax + A.V) as the second assembled unit.

## Rules
- Coherence-gate every increment (output vs separate-op reference). errno=110 = real wedge (NOT dmesg
  "soft reset"). `timeout` every board NPU cmd; SIGTERM not kill -9. Board self-heals; hard wedge -> plug.
- Commit each validated increment. Never clobber board WIP (board tree == feat/static-graph, synced by file).
