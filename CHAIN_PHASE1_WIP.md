# Chained-FFN Phase 1 — on-NPU matmul→silu data handoff (2026-07-11)

## Goal
Phase 0 proved a heterogeneous 2-task chain WALKS (matmul 0xd → int16-silu 0x18, one submit,
rc=0, silu err 0.07). Phase 1 = the matmul must FEED the silu on-NPU (no host round-trip),
with a COHERENT intermediate (avoid the per-tensor int8 requant PPL-55 floor).

## What's proven this session (probes: tools/i16out_probe.c, ork_npu_probe_i16_out + set_i16_out/set_f16_out)

1. **int16-matmul-output = CLOSED.** Grafting an int16 output stage (0x40c0=0x40 2-byte, swept
   0x4010/0x4050) onto synth_i8 WEDGES (errno=110) at int32-strides, garbage at int8-stride.
   Matches the PRIOR sweep already recorded at npu.c:4321 ("PREC=1 int16 across 3 configs — ALL
   soft-reset + garbage"). int16 output exists ONLY in the standalone silu program's 0x50xx lane
   (EWCUBEH set_mul_geom addressing), NOT transferable to the matmul's row-major output stage.

2. **fp16-matmul-output = ACCEPTED (NO WEDGE), but emits zeros.** Grafting the fp16 REGCMD OUT_CVT
   (0x4010=0xa8000002 PREC=2, 0x40c0=0x40, 0x4050=0x36e, gain 0x4084) onto synth_i8 RUNS clean
   (rc=0, ~70us) — the user's "fp16 works where int16 doesn't" reversal is CONFIRMED directionally
   (fp16 is a first-class NVDLA OUT_CVT target; int16 is the odd corner). BUT output is all-0.0:
   the fp16 PREC=2 CVT is a float→float converter (it came from the fp16-INPUT pipeline) and reads
   the int8 matmul's INT32 accumulator as fp16 bits. Gain fix (0x4084=0x00010001, bit16-set unit
   gain per regcmd_silu.h:420) did NOT help → the int32 acc never reaches the fp16 formatter. So
   **INT8_TO_FP16 needs CONV/proc-precision changes upstream (0x100c/0x1080/0x3010/0x4004), not just
   the SDP output stage.** (0x4010 PREC field: int8=00, int16=01, fp16=10, int32=bit31 — npu.c:4347.)

## The robust path (recommended next) — the "shim" the user asked for
Don't make the matmul convert. Use its BULLETPROOF int32 output (native accumulator, full precision,
zero requant loss = MOST coherent) and shim on the READ side:
- **Path A (read-side shim):** matmul → int32 output (known-good) → reconfigure the int16-silu SDP op
  to READ int32 input (its input CVT already converts; change ERDMA DATA_SIZE 0x5034 2→4-byte +
  input PREC). One op, full precision. RE surface = the silu's INPUT surface, not the matmul output.
- **Path B (interstitial CVT task):** matmul int32 → a tiny SDP/EW copy-with-convert task
  (int32→fp16/int16 in EWCUBEH) → silu. A literal shim task; basis = ork_npu_ewmul / copy ops.

Path A is cleanest (fewest ops, full precision). Path B is more general.

## Also viable (already works, gated off): fp16 FUSED silu
ork_mm_run_f16_silu (npu.c:4461) is a CALIBRATED, coherent (~1% err) fp16 matmul + fused SiLU with
fp16→fp32 output — RUNS on-NPU, no wedge. Gated OFF only because the fp16 MATMUL is ~3.3x int8. If
the chain tolerates fp16-matmul cost on the gate op only, this is a drop-in coherent gate TODAY.

## State of the tree
- ork-driver: set_i16_out (Phase-1 int16 scaffolding, committed f655684), NEW set_f16_out +
  ork_npu_probe_i16_out (+ORK_MM_F16OUT fp16 branch) + tools/i16out_probe.c + Makefile target.
  All internal RE scaffolding, off the product surface. COMMIT PENDING this turn.
- Board 10.3.0.236 clean (0 real wedges from Phase-1 probes — int16 wedged but self-healed).
- Product config API (2-option, fork d7a2fcdc0) unaffected — this is orthogonal RE.

## Next concrete step
Path A: read the int16-silu op's input surface regs (0x5018/0x5034/0x5040 + input PREC), build a
int32-input variant, and test matmul-int32 → silu-reads-int32 as a 2-task chain (extend chain_probe).
Coherence gate: PPL via ork_ppl once wired into ggml-ork.
