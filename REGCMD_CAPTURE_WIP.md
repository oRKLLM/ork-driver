# Regcmd-capture campaign — WIP recovery doc

Goal (user, 2026-07-08): **programmatically capture AND document EVERY regcmd program the vendor
libraries (librknnrt / librkllmrt) can emit** — a comprehensive RKNPU regcmd catalog. D_BANK (the CBUF
weight/data bank-split register; NVDLA CDMA 0x50bc / CSC 0x605c) is one specific target within that
broader map — the lever that could grow int4 batch rows and enable int8/fp16 batch beyond the M-scheduler.

## Campaign plan (three vendor surfaces)
1. **RKNN matmul API** (rknn_matmul_api.h, on board, NO model file): enumerate all 13 rknn_matmul_type ×
   a shape grid (vary M,K,N) → capture+decode+auto-classify each register (const / f(M) / f(K) / f(N) /
   f(dtype) / address). STATUS: int8 + fp16 done; int4/w4a16 blocked on B-layout/quant (fixing next).
2. **RKNN model API** (rknn_api.h): run single-op .rknn models (conv, activation, pool, softmax, add/mul,
   ...) → captures CNA/DPU/PPU/SDP programs incl. CBUF banking (D_BANK) and activations. Needs the RKNN
   toolkit to BUILD the single-op .rknn (models/build_act.py exists; toolkit on .239). This is where
   D_BANK + all non-matmul ops live.
3. **RKLLM** (librkllmrt): full-model inference → the real op graph rkllm uses (partially done: see
   [[rkllm-prefill-regcmd-re]] — all ops int8-input regcfg=108).

Output = a documented regcmd catalog on the wiki (regcmd-ISA-Reference): per-op register map + semantics.

## Surfaces (all on board 10.3.0.236, ~/rknn_sdk)
- `librknnrt.so` + `rknn_api.h` + `rknn_matmul_api.h` (RKNN matmul API — no model file needed).
- `rknpu_dump.so` = compiled LD_PRELOAD capture shim (regcmd_capture.c). Dumps every SUBMIT's regcmd.
  **regcmd dump is NOT truncated** (rcwords = regcfg_amount*2+16, exact). The 4KB/RKDUMP_WORDS cap only
  limits the correlation *buffer* dumps (matters for LUT RE, not register discovery). RKDUMP_WORDS=0 =
  regcmd-only, small dumps.
- `decode_reg` (built at ~/rknn_sdk/decode_reg from tools/re/decode_reg.c) — decodes hex → (dom,addr,val).
- `dbank_cap.c` (NEW, ~/rknn_sdk) — creates a matmul of (type,M,K,N) WITH quant params, runs once.
  int8 works; **int4 (type 10) + w4a16 (type 7) abort ("Unsupport type bits 0" / rc=1)** — need vendor
  B prepack / B_layout=1 / int4-specific quant setup (TODO).
- `rknn_mm_dtypes` (M K N iters) — sweeps 4 dtypes but sets NO quant params → aborts after w8a8 (still
  captures w8a8 cleanly, which is all the batch sweep needed).

## Regcmd (value,target) encoding (CALIBRATED from regcmd_i8.h)
Each entry = 2 u32 words (word0, word1):
- `addr  = word0 & 0xffff`
- `value = ((word1 & 0xffff) << 16) | (word0 >> 16)`
- `domain= word1 >> 16`   (0x0201=CNA, 0x0801=DPU, 0x1001=the matmul/PPU engine)

## FINDINGS (this session)
> **⚠️ decode_reg BUG (fixed 02802a1):** its filter kept only domains 1001/2001, silently dropping the
> 0201 CNA + 0801 DPU blocks. The three struck-through claims below were artifacts of that. Re-decode
> any .dump with the fixed decode_reg. Corrected picture: matmul regcmd = 0201 CNA (52) + 0801 DPU (5) +
> 1001 PPU (51) = 108 writes (same as regcmd_i8.h); K IS in 0201 (0x1024/0x1088/0x107c); the 0x40xx block
> is the PPU OUTPUT stage. Genuine new findings: 0x4010=mode reg, the two batch mechanisms, symmetric-only
> platform matrix, and the D_BANK answer (below).

~~RKNN matmul-API path = ONE self-contained engine block: domain 1001, addrs 0x4004–0x412c (~47 regs).
No 0x10xx (CNA), no 0x50xx/0x60xx (CDMA/CSC) in the per-run regcmd.~~ (ARTIFACT — see correction above.)

Register semantics (from M-sweep and N/K-sweep diffs, int8 w8a8):
- `0x4034 = M-1`                     (row count / "batch")
- `0x405c[31:16] = M-1`              (batch dim; SAME reg as int4 "batch stride")
- `0x403c = (N-1)|((N-1)<<16)`       (output width)
- `0x4058 = N-1`
- `0x4038 = (N/4-1)|((N/4-1)<<16)`   (output-width tiling; nonzero only when M>1)
- `0x4020`/`0x4010` = A base address
- **K changes NOTHING in the regcmd** (only the A base addr) → weight STREAMS; reduction is carried by
  the task descriptor, not a register. So this engine does NOT partition CBUF per-run.
- Static (shape-invariant here): 0x4004=0xe, 0x400c=0x1e4, 0x4024=0x10, 0x4040=0x53, 0x4050=0x7fc,
  0x4060=0x53, 0x4070=0x383, 0x40c0=0x80.

**RECONCILIATION: our regcmd_i8.h ALREADY emits this exact 0x40xx block** (template baked at M-tile 4,
N=32: our 0x4034=3, 0x405c=0x30000, 0x4058=0x1f, 0x403c=0x1f001f, 0x4038=7). So the vendor's int8
"batch" (0x4034/0x405c row count) **IS our int8 M-scheduler** — same block, same encoding. No new int8
batch capability; vendor stayed within our mg_max*64 cap at M=128. fp16 (type 1) uses the same block.

**⇒ This REVISES the just-committed mode-matrix note only in framing:** int8/fp16 "batch" is not a
separate toggle — it's row-count in 0x4034/0x405c, which we already drive as the M-scheduler.

## D_BANK STATUS — ANSWERED (corrected, conv .rknn captured)
- Built single-conv .rknn (build_conv.py, rkllm-converter container on .239), captured on the board,
  full-decoded (fixed decode_reg). Conv = 0201 CNA + 0801 DPU + 1001 PPU + **domain 2001 (0x50xx CDMA)**.
- **No explicit NVDLA-style D_BANK (bank-count) register on RK3588.** Swept weight up to 4.7MB (>>1MB
  CBUF): no discrete bank reg; conv_big still 1 submit / 3-core split. CBUF residency is controlled by the
  (already-documented) 0201 size regs: 0x1030 (weight bytes), 0x107c (entries-per-slice = K/16),
  0x1010/0x1040 (M-tile scheduler) — continuous dim-derived, not a bank knob.
- ⇒ int4 batch cap (rows×K ≤ 16384) is gated by **0x107c**, not a reallocatable bank register. The lever
  is dense int4 K/32 (halve 0x107c) — the existing roadmap RE item — and/or the multi-task submit.

## UPDATE (later 2026-07-08): dtype/platform matrix + two batch mechanisms
- `mm_cap.c` (per-type quant/layout aware) built. Probed ALL enum types via create():
  **SUPPORTED (symmetric only):** 1 fp16→f32, 2 int8→i32, 3 int8→i8, 4 fp16→f16, 10 int4→i16 (needs Bl=1).
  **UNSUPPORTED (-5 "unsupported matmul dtype in this platform"):** 5,6 (fp16×int8), 7,8,12 (fp16×int4=
  **w4a16**), 11 (int8×int4=**w4a8**), 15. ⇒ RK3588 matmul is SYMMETRIC-ONLY; NO native w4a16/w4a8.
- `0x4010` = precision/mode reg: fp16→f32 a8000002, fp16→f16 48000002, int8→i32 80000000, int8→i8 0,
  int4→i16 38000006.
- **Two batch mechanisms:** int8/fp16 = IN-TASK (M rows/task, 0x4034=M-1, 0x405c=(M-1)<<16, 1 submit).
  int4 = **MULTI-TASK submit** (M=32 → task_number=96 = 32/core×3, 1 row/task, task[2+] = 12-reg
  incremental). **REFUTES batch_probe "kernel rejects task_number>1"** — vendor uses 96.
  Lever: pack many matmuls' tasks into one submit to amortize the M=1-decode/MoE-GEMV submit floor.
- w4a16 for sensitive tensors: NOT native on RK3588 → use native fp16×fp16 (type 1/4); int4 storage is a
  separate software-decompress scheme.

## Multi-task-submit + incremental-task prototype (2026-07-08, post-campaign)
- **Multi-task chain (run_chain_i8) does NOT beat separate submits.** Clean probe (chain_bench, DMA A/C to
  kill run_chain's per-call bcreate confound): M=1 chain 0.63×, M=16 0.95×. The submit "floor" is small;
  full-regcmd PC-chain adds per-program overhead. Per-call tmp_A/tmp_C bcreate costs ~60us/matmul (hygiene
  only, doesn't rescue it).
- **Vendor incremental-task mechanism DECODED** (from cap_i4.dump w4a4 task_number=96): task[0,1]=full
  (108 regs), task[2+]=12-CONFIG incremental. Incremental advances ONLY A(0x1070 += K/2) and C(0x4020 +=
  N*2); weight 0x1110 CONSTANT (loaded once). Chain amount (0x0014) = **0x0007 for EVERY transition**
  (full->compact and compact->compact), core-ctrl 0x000d. Compact regcmd = 16 pairs (32 words): 12 config
  + (0x0101 0x0010 next, 0x0101 0x0014=7) + (0x0041 0x0000=0) + (0x0081 0x0008=0x000d). regcfg_amount=12.
- **ork_mm_run_i4_incr WORKS (commit bbaf1e3) — STRATEGY C, a real win.** Differential capture (shim on my
  probe vs cap_i4.dump) found the bug: the vendor uses **2 FULL priming tasks** then compacts; chaining a
  full task straight into a compact (my NFULL=1) returned errno 110. Fix: NFULL=2; chain amount 0x0037 into
  a full target / 0x0007 into a compact; + a warmup rep (cold-start stale-output). Now **BIT-EXACT vs CPU
  int4** and beats both baselines: **7–28× vs per-row separate, 1.3–2.7× vs STRATEGY-A** (widens past
  STRATEGY-A's Hcap cap: M=64/K512 = 2.71×; per-row cost drops 9.6→2.6 µs as M grows — weight loaded ONCE).
  Probe: tools/i4_incr_probe.c. NEXT (optional): wire STRATEGY C into ork_mm_run_i4 for M>1 (replacing/
  complementing STRATEGY A), multi-core split, and end-to-end validation (matmul-level win so far — dummy
  data, correctness-checked; needs a model run per AGENTS benchmarking rules before any model-level claim).
- Board npu.c backed up: ~/ork-driver/src/npu.c.bak.* (board had uncommitted fuzz-infra edits = subset of
  local main; local synced over it, superset verified — nothing lost). Local main is authoritative.

## Board ops / safety
- Capture is OBSERVE-ONLY (LD_PRELOAD over vendor's known-good programs) → board-safe. Still use
  `sudo timeout -s INT 60`. Single-stream NPU: one run at a time.
- Governors: pin DDR performance@2112MHz before any TIMING (irrelevant for capture).
- scp each source to an explicit path (multi-source scp flattens).

## Artifacts on board (~/rknn_sdk)
dbank_M{1,8,32,128}.dump/.dec (int8 M-sweep), w_M32_K512_N{64,256,1024}.dump/.dec + w_M32_K{2048,4096}_N64
(weight sweep), t_{1,2}.dec (fp16/int8), dbank_cap.c, decode_reg.
