#!/usr/bin/env python3
# build_rmsnorm.py — build an RMSNorm .rknn to RE how RKNN decomposes it (which ops: the cross-channel reduce,
# rsqrt, scale). Also a bare ReduceMean model to isolate the reduction op. Run in the RKNN-toolkit container.
#   python3 build_rmsnorm.py <op> <prec> <out.rknn> [C]
#     op   : rms | reducemean | reducesum
#     prec : i8 | fp16 | i16
# RE tool only. NOT part of the ork-driver build/test.
import sys, torch, torch.nn as nn, numpy as np
from rknn.api import RKNN

op, prec, out = sys.argv[1], sys.argv[2], sys.argv[3]
C = int(sys.argv[4]) if len(sys.argv) > 4 else 64

class RMS(nn.Module):
    def __init__(s): super().__init__(); s.w = nn.Parameter(torch.ones(C))
    def forward(s, x):
        v = x.pow(2).mean(-1, keepdim=True)
        return x * torch.rsqrt(v + 1e-6) * s.w

class RMean(nn.Module):
    def forward(s, x): return x.pow(2).mean(-1, keepdim=True)
class RSum(nn.Module):
    def forward(s, x): return x.sum(-1, keepdim=True)

m = {"rms": RMS(), "reducemean": RMean(), "reducesum": RSum()}[op].eval()
d = torch.randn(1, 8, C)                                   # [batch, tokens, channels] — reduce over channels
torch.onnx.export(m, d, "/tmp/rms.onnx", opset_version=14, input_names=["x"], output_names=["y"])
print("onnx exported", op, tuple(d.shape))

np.save("/tmp/cal.npy", np.random.randn(1, 8, C).astype(np.float32))
open("/tmp/dataset.txt", "w").write("/tmp/cal.npy\n")

r = RKNN(verbose=True)   # verbose prints the op graph RKNN builds — the decomposition we want to see
if prec == "i16":
    r.config(target_platform="rk3588", quantized_dtype="w16a16i"); quant = True
elif prec == "fp16":
    r.config(target_platform="rk3588"); quant = False
else:
    r.config(target_platform="rk3588"); quant = True
assert r.load_onnx(model="/tmp/rms.onnx") == 0, "load_onnx failed"
assert r.build(do_quantization=quant, dataset="/tmp/dataset.txt") == 0, "build failed"
assert r.export_rknn(out) == 0, "export failed"
print("EXPORTED", out)
