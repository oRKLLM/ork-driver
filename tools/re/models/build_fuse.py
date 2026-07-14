#!/usr/bin/env python3
"""build_fuse.py — build Conv -> 2-input-elementwise (Mul/Add) FUSED .rknn models for regcmd capture.

The RKNN compiler fuses a Conv/Gemm immediately followed by a 2-input Mul/Add (residual add, elementwise
scale) into ONE fused op = a matmul->2-input-SDP chain. Capturing that regcmd reveals how the vendor
programs the ERDMA second-operand read (0x5034 ERDMA_CFG / 0x5038 EW_BASE) in the CHAINED, no-per-op-reset
context — i.e. the ERDMA enable bits that ACT_RESET provides for a standalone op but that our chain_progs
2-input SDP task is missing (the M4 single-submit-chain blocker: 2-input SDP hangs as a chained task).

Run INSIDE the rknn-toolkit2 env (needs torch + onnx + rknn-toolkit2 2.3.2):
    python3 build_fuse.py                 # builds conv_mul.rknn + conv_add.rknn
Then copy to the board and capture under regcmd_capture.so (README), decode the 2nd (SDP) task's regcmd.
RE/calibration only — proprietary rknn-toolkit2 (never part of the library; AGENTS.md).
"""
import sys, torch, torch.nn as nn
from rknn.api import RKNN

class Fuse(nn.Module):
    def __init__(self, cin, cout, k, op):
        super().__init__(); self.conv = nn.Conv2d(cin, cout, k, padding=k // 2); self.op = op
    def forward(self, x, y):
        c = self.conv(x)
        return c * y if self.op == "mul" else c + y

def make(op, cin=32, cout=64, hw=8, k=3, quant=False):
    name = f"conv_{op}"
    m = Fuse(cin, cout, k, op).eval()
    x = torch.randn(1, cin, hw, hw); y = torch.randn(1, cout, hw, hw)   # y = full 2nd tensor (fused via ERDMA)
    onnx = f"{name}.onnx"
    torch.onnx.export(m, (x, y), onnx, input_names=["x", "y"], output_names=["out"], opset_version=12)
    r = RKNN(verbose=False); r.config(target_platform="rk3588")
    if r.load_onnx(model=onnx) != 0: print("LOAD FAIL", name); return
    if r.build(do_quantization=quant) != 0: print("BUILD FAIL", name); r.release(); return
    if r.export_rknn(f"{name}.rknn") != 0: print("EXPORT FAIL", name); r.release(); return
    r.release(); print("BUILT", name, "cin", cin, "cout", cout, "hw", hw, "quant", quant)

if __name__ == "__main__":
    q = "--int8" in sys.argv
    make("mul", quant=q)
    make("add", quant=q)
