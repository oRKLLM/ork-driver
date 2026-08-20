# tools/re — NPU reverse-engineering toolkit

The workflow used to reverse-engineer a **regcmd** (register-command program) for a new op or a new
Rockchip/NVDLA-derived NPU, so it can be templatized into `src/regcmd_*.h` and driven by the matching precision module under
`src/npu/` (`i8/regcmd.c`, `f16/regcmd.c`, …; see AGENTS.md §4).
These are **RE / calibration tools only** — they are NOT part of the `ork-driver` library, `make`, or
`make test`. Some depend on the proprietary `librknnrt.so` / `rknn-toolkit2`, which are used only to
*capture* reference programs (never to build or run the library). Keep that boundary.

As more NPUs are added to the platform, everything needed to onboard one lives here.

## The capture → decode → templatize loop

1. **Build a single-op reference model** (`models/build_act.py`, etc.) inside the RKNN-toolkit container.
   A tiny ONNX graph with just the target op (e.g. `SiLU(x)`, `Mul(a,b)`, `Add(a,b)`) exposes the
   standalone SDP/PPU program for that op. Match the tensor shape to the 2D `M=W × N=C, H=1` convention
   the library uses (`[1, N, 1, M]`), so the captured geometry lines up with `set_mul_geom()`.

   ```sh
   # in the rknn-toolkit container (x86 or emulated):
   python3 build_act.py silu i8 /tmp/silu_i8.rknn 64 1 8      # op prec out.rknn [C H W]
   ```

2. **Capture the regcmd on the board** with the LD_PRELOAD interposer (`regcmd_capture.c`). It tracks
   DRM buffers and, on `RKNPU_SUBMIT`, dumps each task's regcmd words + descriptors. `RKDUMP_WORDS`
   caps per-buffer hexdump length (use a big value to also dump the input/output/LUT buffers).

   ```sh
   # on the board (needs librknnrt + rknn_api.h, e.g. /home/michael/rknn_sdk):
   cc -shared -fPIC -O2 -I$HOME/ork-driver/src -I/path/to/rknn_sdk -o regcmd_capture.so regcmd_capture.c -ldl
   cc -O2 -I/path/to/rknn_sdk -o run_rknn run_rknn.c -L/path/to/rknn_sdk -lrknnrt
   sudo env LD_PRELOAD=$PWD/regcmd_capture.so LD_LIBRARY_PATH=/path/to/rknn_sdk RKDUMP_WORDS=64 \
       timeout 60 ./run_rknn /tmp/silu_i8.rknn 2> silu.dump
   ```

3. **Decode + diff** the captured regcmd (`decode_reg.c`). Feed it the region between a
   `--- regcmd (N u32 words) ---` header and the trailer. Diff a new op against a known-good one
   (e.g. the element-wise MUL) to isolate exactly which registers are op-specific vs shared geometry.

   ```sh
   cc -O2 -o decode_reg decode_reg.c
   sed -n '616,653p' silu.dump | ./decode_reg                 # decode one op's regcmd
   ./decode_reg silu_op.txt mul_op.txt                        # diff two ops (RE)
   ```

4. **Templatize** the decoded words into `src/regcmd_<op>.h` as a `static const uint32_t REGCMD_<OP>[]`,
   and add a marshaling function in the op's precision module under `src/npu/` that patches the
   addresses + geometry (`orki_set_mul_geom`)
   + any per-scale fields, submits it, and de-marshals. Validate bit-exact vs a CPU reference in an
   example under `examples/` so `make test` covers it.

5. **Calibrate** any LUT/scale-dependent op on silicon (`silu_std_probe.c` is the model): run a ramp-LUT
   pass to measure the op's index/transfer function, then build the real curve at the measured points.

## Files
- `regcmd_capture.c` — LD_PRELOAD interposer; dumps regcmd + buffers on SUBMIT (`RKDUMP_WORDS`, `ORK_SUBMIT_ONLY`).
- `run_rknn.c` — minimal RKNN-API runner; loads a `.rknn`, fills inputs deterministically, runs once.
- `decode_reg.c` — decode a regcmd dump into (domain, addr, value); diff two dumps to isolate op-specific regs.
- `parse_mfold.py` — tabulate the M-fold matmul's schedule registers per M from a verbose rkllm dump.
- `analyze_schedule.py` — the richer successor to `parse_mfold.py`: parses a verbose dump, tabulates the
  schedule regs per group with a CONSTANT-vs-VARIES marker, fits each reg as `a·M+b`, factors by `(rows,
  K-slice)`, and reads `0x100c`. Its header records the DEFINITIVE regcmd word encoding (value = `w0>>16`;
  NOT the 32-bit form, which corrupts 16-bit regs). Use it to answer "is reg X a function of shape?".
- `submit_extract.py` — group a verbose dump's `--- regcmd` blocks under their `=== SUBMIT` header and print
  one complete multi-task submit (per-task M / DATA_ENTRIES / K-slice / CBUF / N), to spec a chain replay.
- `silu_std_probe.c` — RE/calibration harness for the standalone SiLU activation op (ramp-measure idx, build curve).
- `models/build_act.py` — build a single-op activation `.rknn` (silu/sigmoid/gelu/relu/tanh; i8/fp16/i16).
- `ork_bench.cpp` — open-stack perf harness. Drives the llama.cpp C API directly (the `ggml-ork` backend
  intercepts `MUL_MAT`, so `ORK_FFN_CHAIN`/`ORK_PERSIST`/`ORK_PROFILE`/… all apply), exposing the exact
  levers `llama-bench` hides: prefill batch size (`n_ubatch`), warmup, and one-clock timing. Also a coherency
  smoke test (prints the generated text). This is the canonical open-stack bench, not `llama-bench`.
- `rkllm_bench.cpp` — the closed-baseline mirror: same prompt/shape via the public `librkllmrt` API, reporting
  the runtime's own prefill/decode tok/s. Same model on both runtimes = the AGENTS apples-to-apples rule.
- `ork_ppl.cpp` — the QUALITY companion to `ork_bench`: teacher-forced perplexity via the llama.cpp C API
  (ggml-ork engaged, reads the same `ORK_*` knobs), so any q4/int4/int8 matmul-path change gets a PPL number
  next to the tok/s. One all-position-logits decode over a text window; `PPL = exp(mean NLL)`. Compare
  `ORK_OFF=1 ork_ppl …` (CPU baseline) vs the NPU config. `ork_ppl <model.gguf> <textfile> [window=256] [ubatch]`.

## Benchmarking (open vs closed)

`ork_bench` / `rkllm_bench` are built out-of-tree against their respective runtimes (they need the llama.cpp
C API / `librkllmrt`, so they are NOT `ork-driver` Makefile targets — same boundary as the capture tools):

```sh
# open stack (built alongside the ggml-ork llama.cpp build; run against its libs):
LD_LIBRARY_PATH=<llama-build>/bin ORK_PROFILE=1 ./ork_bench model.gguf prompt.txt 128 64 [ubatch=P]
# closed baseline (same prompt, matching .rkllm):
g++ -O2 -I. -o rkllm_bench rkllm_bench.cpp -L. -lrkllmrt && ./rkllm_bench model.rkllm 64 prompt.txt
```

## Board/container notes
- The RKNN toolkit runs in an x86 Ubuntu docker container (`rkllm-converter`); `docker cp` models in/out.
- NPU is single-stream; never `kill -9` an in-flight submit. Use `timeout` and `ORK_EW_TIMEOUT` to fail fast.
- A malformed task shows `task counter: 0x0` / `raw status: 0xc0000000` in `dmesg` (the CP rejected it).

## Mode-mapping / batch / D_BANK RE (matmul-API + conv capture)

Two capture surfaces beyond the single-op activation flow, used to map the NPU's precision/batch
modes and the CBUF banking (2026-07; findings on the wiki `regcmd-ISA-Reference` +
`Reverse-Engineering-Roadmap`):

- **`mm_cap.c` — RKNN matmul-API enumeration** (no model file needed). Runs any `rknn_matmul_type`
  at any `(M,K,N)` so you can capture + diff the emitted regcmd across dtype/shape:
  ```sh
  cc -O2 -I$RKNN_SDK -o mm_cap mm_cap.c -L$RKNN_SDK -lrknnrt
  for M in 1 8 32 128; do
    sudo env LD_PRELOAD=$PWD/rknpu_dump.so LD_LIBRARY_PATH=$RKNN_SDK RKDUMP_WORDS=0 \
        ./mm_cap 2 $M 512 64 2> m$M.dump                       # int8, sweep batch M
    ./decode_reg < <(awk '/regcmd \(/{f=1;next}/--- /{f=0}f&&/^  \[/' m$M.dump) > m$M.dec
  done
  diff m1.dec m128.dec        # regs that move with M => batch encoding (0x4034, 0x405c)
  ```
  Established: RK3588 matmul is symmetric-only (fp16/int8/int4; all mixed types reject at create);
  `0x4010`=precision/mode; int8/fp16 batch in-task (=our M-scheduler), int4 batches multi-task
  (`task_number`=rows) — which refuted the old `batch_probe` "kernel rejects task_number>1".

## M-fold matmul schedule RE (the rkllm "fold", task #39)

RE-ing rkllm's fast int8 prefill matmul (the "M-fold": tokens folded into the CNA WIDTH). Full record on the
wiki + `MFOLD_RECAPTURE_WIP.md`; the decisive result: **the fold schedule is NOT a function of the matmul
shape** — the registers `0x1040/0x107c/0x1080/0x4024/0x40c0` are outputs of rkllm's internal per-sub-block
tiling planner and vary even at fixed `(M,N)`. So it cannot be *synthesized* from shape; only *captured*.

Workflow (all offline once a dump exists — no wedge risk; rkllm is the reference, not ork's synth):
```sh
# 1. capture rkllm's real programs (native C++ bench; the node harness crashes — do NOT use it):
gcc -shared -fPIC -O2 -Itools/re -Isrc -o /tmp/rknpu_dump.so tools/re/regcmd_capture.c -ldl
cd ~/rkbench && sudo env LD_PRELOAD=/tmp/rknpu_dump.so LD_LIBRARY_PATH=. RKDUMP_MM=1 \
    RKDUMP_MM_K=3584 RKDUMP_MM_N=1216 ./rk_bench_short model.rkllm 512 4 3 2>cap.log
#   -> /tmp/mm_{regcmd,meta,chain_meta}.txt + mm_{A,weight,C}.bin (exact operands for a bit-exact replay).
#   (omit ORK_SUBMIT_ONLY to get EVERY task's regcmd in cap.log; keep it to only get the RKDUMP_MM files.)
# 2. characterize offline:
python3 tools/re/analyze_schedule.py cap.log 3584 1216     # is each reg f(shape)? -> answers: NO for the DMA regs
python3 tools/re/submit_extract.py   cap.log 21            # one full task_number=21 submit's per-task tiling
```
Findings so far: one matmul = a `task_number=21` chain per core (3-core round-robin); every task is FULL-K
(`DATA_ENTRIES=56·M`), a complete `A[M,3584]·W[3584,1216]`; rkllm tiles the prefill M into VARIABLE row-tiles
(widths 1,2,4,6,8,10,12,14,20,24,36); shape-clean regs are `0x100c=0` (`=CONV1_PLAIN`), `0x104c=0xb`,
`0x1044=56·M`, WIDTH=`M-1`.

### ★ FOLD LAYOUT SOLVED + BIT-EXACT (2026-07-29) — and the weight layout is a DOCUMENTED vendor format

The full fold data layout is confirmed on silicon (`tools/re/validate_layout.c`: ork packs its OWN A/W, replays
rkllm's captured regcmd, de-tiles output, compares CPU ref → **0/9728 bit-exact @ 609 µs, M=8**):
- **A (input)**  = `nc16` : `(k/16)*(M*16) + m*16 + (k%16)`  — NC1HWC2 **C2=16**, M folded into width.
- **W (weight)** = `woff` : `((n/32)*(K/32) + (k/32))*1024 + (n%32)*32 + (k%32)` — the RK3588 int8 matmul-native
  weight format (32×32 tiles, K innermost).
- **C (output)** = `c4`   : `(n/4)*(M*4) + m*4 + (n%4)`  — NC1HWC2 **C2=4** int32.

**The weight layout did NOT need board RE — it is published in the RKNN toolkit header** and this is the
reusable asset for porting to other Rockchip NPUs: read the per-chip native (subN,subK) directly, no probing.
Source: `airockchip/rknn-toolkit2 → rknpu2/runtime/Linux/librknn_api/include/rknn_matmul_api.h`
(function `rknn_B_normal_layout_to_native_layout`). Native B layout `(N/subN, K/subK, subN, subK)`:

| chip | dtype | native (N/subN, K/subK, subN, subK) | subN | subK | align |
|---|---|---|---|---|---|
| RK3566/3568/3562 | int8 | (N/16, K/32, 16, 32) | 16 | 32 | 32B |
| **RK3588 / RK3576** | **int8** | **(N/32, K/32, 32, 32)** | **32** | **32** | **32B** |
| RK3588 / RK3576 | int4 | (N/64, K/32, 64, 32) | 64 | 32 | 64B |
| (all) | fp16 | — | — | — | 16B |

General weight offset: `woff(k,n) = ((n/subN)*(K/subK) + (k/subK))*(subN*subK) + (n%subN)*subK + (k%subK)`.
When adding a new Rockchip SoC, grab its (subN,subK) from that header and this formula gives the weight pack —
the activation stays NC1HWC2 (C2 = 16 int8 / 8 fp16) and output C2 = 4 (int32). GOTCHA when RE-decoding an
injection ramp: `A[i]=(i%251)-125` aliases (val -120 → i∈{5,256,…}); pick the i that is a valid `nc16` index.

Capture-replay is thus implementable: pack A=`nc16`/W=`woff`, replay rkllm's per-M captured regcmd
(`ork_npu_replay_i8`; do NOT set `ORK_REPLAY_RESET` — its `ACT_RESET` pre-submit is a wedge trigger), read
C=`c4`. Remaining is the weight-resident `task_number=N` HW-chain + benchmark vs ork's normal path.

- **`models/build_conv.py` — single-Conv `.rknn`** (build in the rknn-toolkit2 container, capture on
  the board with `run_rknn`). Conv emits the domain-2001 (`0x50xx`) CDMA block the matmul path lacks.
  Sweep Cout/Cin (weight) and H/W (data) and diff to hunt the CBUF bank-split. Finding so far: on
  RK3588 every moving reg tracks a dimension linearly — no discrete bank-count register in-regcmd,
  i.e. CBUF partitioning looks implicit/driver-computed (unlike NVDLA's explicit `D_BANK`); confirm
  with int8 + a bank-crossing conv.
