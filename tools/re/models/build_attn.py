#!/usr/bin/env python3
# build_attn.py — build Softmax / exp / RoPE .rknn models to RE how RKNN maps the attention-path ops
# (fused op vs op chain; which activation-LUT for exp). Run in the RKNN-toolkit container.
#   python3 build_attn.py <op> <prec> <out.rknn> [C]
#     op   : softmax | exp | sigmoid | tanh
#     prec : i8 | fp16 | i16
# RE tool only. NOT part of the ork-driver build/test.
import sys, torch, torch.nn as nn, numpy as np
from rknn.api import RKNN

op, prec, out = sys.argv[1], sys.argv[2], sys.argv[3]
C = int(sys.argv[4]) if len(sys.argv) > 4 else 64

class M(nn.Module):
    def forward(s, x):
        if op == "softmax": return torch.softmax(x, dim=-1)
        if op == "exp":     return torch.exp(x)
        if op == "sigmoid": return torch.sigmoid(x)
        return torch.tanh(x)

m = M().eval()
d = torch.randn(1, 8, C)
torch.onnx.export(m, d, "/tmp/attn.onnx", opset_version=14, input_names=["x"], output_names=["y"])
print("onnx exported", op, tuple(d.shape))
np.save("/tmp/cal.npy", np.random.randn(1, 8, C).astype(np.float32))
open("/tmp/dataset.txt", "w").write("/tmp/cal.npy\n")

r = RKNN(verbose=True)
if prec == "i16": r.config(target_platform="rk3588", quantized_dtype="w16a16i"); quant = True
elif prec == "fp16": r.config(target_platform="rk3588"); quant = False
else: r.config(target_platform="rk3588"); quant = True
assert r.load_onnx(model="/tmp/attn.onnx") == 0, "load_onnx failed"
assert r.build(do_quantization=quant, dataset="/tmp/dataset.txt") == 0, "build failed"
assert r.export_rknn(out) == 0, "export failed"
print("EXPORTED", out)
