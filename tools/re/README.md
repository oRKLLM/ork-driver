# tools/re — NPU reverse-engineering toolkit

The workflow used to reverse-engineer a **regcmd** (register-command program) for a new op or a new
Rockchip/NVDLA-derived NPU, so it can be templatized into `src/regcmd_*.h` and driven by `src/npu.c`.
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
   and add a marshaling function in `src/npu.c` that patches the addresses + geometry (`set_mul_geom`)
   + any per-scale fields, submits it, and de-marshals. Validate bit-exact vs a CPU reference in an
   example under `examples/` so `make test` covers it.

5. **Calibrate** any LUT/scale-dependent op on silicon (`silu_std_probe.c` is the model): run a ramp-LUT
   pass to measure the op's index/transfer function, then build the real curve at the measured points.

## Files
- `regcmd_capture.c` — LD_PRELOAD interposer; dumps regcmd + buffers on SUBMIT (`RKDUMP_WORDS`, `ORK_SUBMIT_ONLY`).
- `run_rknn.c` — minimal RKNN-API runner; loads a `.rknn`, fills inputs deterministically, runs once.
- `decode_reg.c` — decode a regcmd dump into (domain, addr, value); diff two dumps to isolate op-specific regs.
- `silu_std_probe.c` — RE/calibration harness for the standalone SiLU activation op (ramp-measure idx, build curve).
- `models/build_act.py` — build a single-op activation `.rknn` (silu/sigmoid/gelu/relu/tanh; i8/fp16/i16).
- `ork_bench.cpp` — open-stack perf harness. Drives the llama.cpp C API directly (the `ggml-ork` backend
  intercepts `MUL_MAT`, so `ORK_FFN_CHAIN`/`ORK_PERSIST`/`ORK_PROFILE`/… all apply), exposing the exact
  levers `llama-bench` hides: prefill batch size (`n_ubatch`), warmup, and one-clock timing. Also a coherency
  smoke test (prints the generated text). This is the canonical open-stack bench, not `llama-bench`.
- `rkllm_bench.cpp` — the closed-baseline mirror: same prompt/shape via the public `librkllmrt` API, reporting
  the runtime's own prefill/decode tok/s. Same model on both runtimes = the AGENTS apples-to-apples rule.

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
