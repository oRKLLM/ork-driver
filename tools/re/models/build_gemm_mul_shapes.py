#!/usr/bin/env python3
"""build_gemm_mul_shapes.py — parameterized Gemm*[1,1,N] per-channel-Mul fp16 rknn for the RESHAPE geometry decode.
Usage (in rknnenv):  python3 build_gemm_mul_shapes.py M N   ->  gemm_mul_M{M}_N{N}.rknn
Build 2-3 shapes (e.g. 8 64 / 8 128 / 16 64) to solve the CNA reshape LINE_STRIDE(M,N)/SURF_STRIDE(M,N) + the
per-group output-base delta (task4-10 in the capture). K fixed at 32. RE/calibration only (proprietary toolkit)."""
import sys, torch, torch.nn as nn
from rknn.api import RKNN
K = 32
class GemmMul(nn.Module):
    def __init__(self, K, N):
        super().__init__(); self.fc = nn.Linear(K, N, bias=False)
    def forward(self, x, scale):
        return self.fc(x) * scale
def make(M, N):
    name = f"gemm_mul_M{M}_N{N}"
    m = GemmMul(K, N).eval()
    x = torch.randn(1, M, K); scale = torch.randn(1, 1, N)
    onnx = f"{name}.onnx"
    torch.onnx.export(m, (x, scale), onnx, input_names=["x","scale"], output_names=["out"], opset_version=12)
    r = RKNN(verbose=False); r.config(target_platform="rk3588")
    if r.load_onnx(model=onnx) != 0: print("LOAD FAIL", name); return
    if r.build(do_quantization=False) != 0: print("BUILD FAIL", name); r.release(); return
    if r.export_rknn(f"{name}.rknn") != 0: print("EXPORT FAIL", name); r.release(); return
    r.release(); print("BUILT", name, "M", M, "K", K, "N", N)
if __name__ == "__main__":
    M = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    N = int(sys.argv[2]) if len(sys.argv) > 2 else 64
    make(M, N)
