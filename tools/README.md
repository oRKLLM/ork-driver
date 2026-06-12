# tools

## regcmd_capture.c — calibration capture shim
An `LD_PRELOAD` shim that intercepts the `rknpu` DRM ioctls and dumps the regcmd stream,
task descriptors, and buffers for a matmul. Used to **capture a new SoC's scheduler
parameters** (rows-per-M-tile `0x1010`, schedule `0x1040`, output-width cap) when adding
support — see [../docs/ADDING_AN_SOC.md](../docs/ADDING_AN_SOC.md).

Requires Rockchip's proprietary `librknnrt.so` *at capture time only* (it intercepts that
runtime). It is NOT a build/runtime dependency of ork-driver itself — purely a dev aid.

```sh
gcc -shared -fPIC -O2 -Isrc -o regcmd_capture.so tools/regcmd_capture.c -ldl
sudo env LD_PRELOAD=$PWD/regcmd_capture.so LD_LIBRARY_PATH=. ./your_librknnrt_matmul_probe 2>capture.txt
```

## rkllm_bench.c — closed-runtime comparison benchmark
Times Rockchip's closed **`librkllmrt`** LLM runtime on a `.rkllm` model, for an apples-to-apples
comparison against ork-driver's own `examples/bench.c` (same model + quant + board). It `dlopen`s
librkllmrt at **runtime** and reads the runtime's own prefill/decode tok/s — so there's no build
dependency and no Rockchip code here (the RKLLM ABI structs are reverse-engineered, same status as
the regcmd headers). The runtime + a model live on the board, not in this repo.

```sh
make rkllm_bench
sudo ./rkllm_bench /var/lib/orkllm/runtimes/librkllmrt-aarch64-v1.2.3.so /path/model-w8a8.rkllm 128
# then compare with ork-driver at the model's config:
make bench && sudo ./bench <layers> 16 32 i8
```

See the [Performance wiki](https://github.com/oRKLLM/ork-driver/wiki/Performance) for the
Qwen3-1.7B w8a8 head-to-head produced this way.
