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
