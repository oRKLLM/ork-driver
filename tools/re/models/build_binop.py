#!/usr/bin/env python3
# build_binop.py — build a tiny 2-input elementwise binary-op .rknn to RE the standalone SDP element-wise op's
# ALU mode (add / mul / sub). Used to add on-NPU Residual Add etc. Run in the RKNN-toolkit container.
#   python3 build_binop.py <op> <prec> <out.rknn> [C H W]
#     op   : add | mul | sub
#     prec : i8 | fp16 | i16
# RE tool only (needs rknn-toolkit2). NOT part of the ork-driver build/test.
import sys, torch, torch.nn as nn, numpy as np
from rknn.api import RKNN

op, prec, out = sys.argv[1], sys.argv[2], sys.argv[3]
C, H, W = (int(sys.argv[4]), int(sys.argv[5]), int(sys.argv[6])) if len(sys.argv) > 6 else (64, 1, 8)

class M(nn.Module):
    def forward(self, a, b):
        if op == "add": return a + b
        if op == "sub": return a - b
        return a * b

m = M().eval()
a = torch.randn(1, C, H, W); b = torch.randn(1, C, H, W)
torch.onnx.export(m, (a, b), "/tmp/binop.onnx", opset_version=12, input_names=["a", "b"], output_names=["y"])
print("onnx exported", op, (1, C, H, W))

np.save("/tmp/cal_a.npy", np.random.randn(1, C, H, W).astype(np.float32))
np.save("/tmp/cal_b.npy", np.random.randn(1, C, H, W).astype(np.float32))
open("/tmp/dataset.txt", "w").write("/tmp/cal_a.npy /tmp/cal_b.npy\n")

r = RKNN(verbose=False)
if prec == "i16":
    r.config(target_platform="rk3588", quantized_dtype="w16a16i"); quant = True
elif prec == "fp16":
    r.config(target_platform="rk3588"); quant = False
else:
    r.config(target_platform="rk3588"); quant = True
assert r.load_onnx(model="/tmp/binop.onnx") == 0, "load_onnx failed"
assert r.build(do_quantization=quant, dataset="/tmp/dataset.txt") == 0, "build failed"
assert r.export_rknn(out) == 0, "export failed"
print("EXPORTED", out)
