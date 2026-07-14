#!/usr/bin/env python3
"""build_gemm.py — build a pure MatMul/Gemm .rknn (fp16, NO quantization) for regcmd capture.

Goal (option a): find how the vendor emits FP16 OUTPUT from a matmul-class op. Our fp16 matmul (synth,
enable=0xd) HANGS on any 2-byte fp16 DPU writeout, while the vendor CONV emits fp16 fine (enable=0x1d).
A non-quantized MatMul/Gemm compiles to the vendor's fp16 matmul datapath — capture task[0] and decode
its output stage (0x40xx) + front-end (0x1xxx CNA) to learn the fp16-writeout config a matmul needs
(does it stay a matmul enable=0xd, or lower to a conv enable=0x1d?). That closes the single-submit
attention A·V-normalize chain (matmul must feed the chain-safe fp16 2-input SDP in fp16).

Run INSIDE rknn-toolkit2 2.3.2 (Colima VM ~/rknnenv):  python3 build_gemm.py
Produces gemm_f16.rknn. Copy to board, capture under regcmd_capture.so, decode task[0].
RE/calibration only — proprietary rknn-toolkit2 (never part of the library; AGENTS.md).
"""
import sys, torch, torch.nn as nn
from rknn.api import RKNN

class Gemm(nn.Module):
    def __init__(self, K, N):
        super().__init__(); self.fc = nn.Linear(K, N, bias=False)
    def forward(self, x):
        return self.fc(x)

def make(K=32, N=64, M=8):
    name = "gemm_f16"
    m = Gemm(K, N).eval()
    x = torch.randn(1, M, K)                      # (1, M, K) -> Linear over last dim -> (1, M, N)
    onnx = f"{name}.onnx"
    torch.onnx.export(m, (x,), onnx, input_names=["x"], output_names=["out"], opset_version=12)
    r = RKNN(verbose=False); r.config(target_platform="rk3588")
    if r.load_onnx(model=onnx) != 0: print("LOAD FAIL", name); return
    if r.build(do_quantization=False) != 0: print("BUILD FAIL", name); r.release(); return   # fp16, no quant
    if r.export_rknn(f"{name}.rknn") != 0: print("EXPORT FAIL", name); r.release(); return
    r.release(); print("BUILT", name, "M", M, "K", K, "N", N, "(fp16, no quant)")

if __name__ == "__main__":
    make()
