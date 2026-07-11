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

## Path A RESULT (2026-07-11): literal int32-handoff is HARDWARE-CLOSED
GROUNDED in NVDLA SDP docs (nvdla.org unit_description): **the SDP reads only INT8/INT16/FP16 from
memory; INT32 is an internal accumulator precision, NEVER a memory feature-map input format.** So
"silu reads the matmul's native int32 output from memory" is impossible on this hardware — matches
the CVT-bypass note (npu.c:4347). The memory-handoff intermediate MUST be int8/int16/fp16:
  - int8 : matmul writes ✓ (set_i8_out8), silu reads ✓ — but per-tensor requant floor = PPL 55.
  - int16: silu reads ✓, matmul writes ✗ (int16-matmul-output CLOSED). DEAD for handoff.
  - fp16 : silu reads ✓, matmul writes = INT8_TO_FP16 (the ONE open lead) — runs no-wedge but zeros.

INT8_TO_FP16 crack attempt: swept 0x4010 (DATA_FORMAT) out-precision field {0x2,0x8002,0x08000002,
0x28000002} keeping fp16 gain 0x00010001 — ALL HANG (61s job-timeout, soft-reset, NPU self-healed,
test_matmul clean after). This is bitfield-guessing that wedges — STOP guessing on-board (AGENTS.md).
The right way to crack INT8_TO_FP16 DATA_FORMAT = CAPTURE it from RKNN (regcmd_capture toolkit on
.239), not board sweeps.

## Where Path A leaves us — the coherent on-NPU silu needs the ACC ON-CHIP (fused), not a mem handoff
The memory-format wall means a 2-task chain can't carry a lossless intermediate the silu can read
(int16 uncomposable, int32 unreadable, fp16 needs a capture). The config that ALREADY WORKS coherent
is the FUSED path — accumulator flows conv→LUT on-chip, no memory round-trip:
  - set_f16_silu (fp16 matmul + fused SiLU, fp16 acc→LUT): WORKS, ~1% err, gated off for fp16-mm 3.3x.
  - TASK #35 PRIZE (uncracked): int8 matmul FUSED with an int16-RESOLUTION LUT index (widen the
    acc→index map beyond the int8 requant that causes PPL 55). Fast int8 matmul + coherent silu, one
    task, no memory intermediate. This sidesteps the entire memory-format wall.

## Next concrete step (fork)
(1) FUSED int8+int16-index (task #35 proper): in set_i8_silu, widen the acc->LUT-index precision
    (the int16 standalone silu proves the LUT is coherent at int16 input res). No memory handoff.
(2) Bank fp16-fused-silu as the coherent gate op now (works today; costs fp16-mm on that op).
(3) Unlock the fast fp16 memory-chain LATER via an RKNN INT8_TO_FP16 capture (.239 toolkit) — no
    more board bitfield-guessing.
