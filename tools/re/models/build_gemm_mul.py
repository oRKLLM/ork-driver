#!/usr/bin/env python3
"""build_gemm_mul.py — Gemm followed by a PER-CHANNEL Mul, fp16, for regcmd capture.

Goal: capture how the vendor emits a fp16 MATMUL with a FUSED per-channel scale — the exact output-stage
config (0x40xx EW/BS + 0x50xx operand) that our ork_npu_mm_perchan_f16_fused needs (it hangs on guessed
fp16-EW bits). The scale is a SECOND INPUT of shape [1,1,N] (broadcast per output channel) so the compiler
can't constant-fold it into the Linear weights — it must emit a real fused matmul->per-channel-scale op.

Run INSIDE rknn-toolkit2 2.3.2 (Colima VM ~/rknnenv):  python3 build_gemm_mul.py
Produces gemm_mul_f16.rknn. Copy to board, capture under regcmd_capture.so (RKDUMP_WORDS=400), decode task[0].
RE/calibration only — proprietary rknn-toolkit2 (never part of the library; AGENTS.md).
"""
import sys, torch, torch.nn as nn
from rknn.api import RKNN

class GemmMul(nn.Module):
    def __init__(self, K, N):
        super().__init__(); self.fc = nn.Linear(K, N, bias=False)
    def forward(self, x, scale):
        return self.fc(x) * scale                    # [1,M,N] * [1,1,N] -> per-channel scale

def make(K=32, N=64, M=8):
    name = "gemm_mul_f16"
    m = GemmMul(K, N).eval()
    x = torch.randn(1, M, K); scale = torch.randn(1, 1, N)
    onnx = f"{name}.onnx"
    torch.onnx.export(m, (x, scale), onnx, input_names=["x", "scale"], output_names=["out"], opset_version=12)
    r = RKNN(verbose=False); r.config(target_platform="rk3588")
    if r.load_onnx(model=onnx) != 0: print("LOAD FAIL", name); return
    if r.build(do_quantization=False) != 0: print("BUILD FAIL", name); r.release(); return   # fp16, no quant
    if r.export_rknn(f"{name}.rknn") != 0: print("EXPORT FAIL", name); r.release(); return
    r.release(); print("BUILT", name, "M", M, "K", K, "N", N, "(fp16, per-channel Mul)")

if __name__ == "__main__":
    make()
