#!/usr/bin/env python3
# build_act.py — build a tiny single-op activation .rknn to RE how the RK NPU (NVDLA-derived SDP) lays out a
# STANDALONE activation reading its input from memory (vs fused into a conv output). Used to reverse-engineer
# the regcmd for ork's on-NPU SiLU/GELU/etc. Run inside the RKNN-toolkit container (see tools/re/README.md).
#
#   python3 build_act.py <op> <prec> <out.rknn> [C H W]
#     op   : silu | sigmoid | gelu | relu | tanh
#     prec : i8 (do_quantization) | fp16 (no quant) | i16 (w16a16i)
#     C H W: NCHW dims (default 64 1 8 = the 2D M=W=8 x N=C=64 convention matching the EW-mul capture)
#
# NOTE: RE tool only. Requires the proprietary rknn-toolkit2 (Python). NOT part of the ork-driver build/test.
import sys, torch, torch.nn as nn, numpy as np
from rknn.api import RKNN

op, prec, out = sys.argv[1], sys.argv[2], sys.argv[3]
C, H, W = (int(sys.argv[4]), int(sys.argv[5]), int(sys.argv[6])) if len(sys.argv) > 6 else (64, 1, 8)
ACT = {
    "silu":    torch.nn.functional.silu,
    "sigmoid": torch.sigmoid,
    "gelu":    torch.nn.functional.gelu,
    "relu":    torch.nn.functional.relu,
    "tanh":    torch.tanh,
}[op]

class M(nn.Module):
    def forward(self, x):
        return ACT(x)           # pure elementwise activation, no conv — expose the standalone SDP op

m = M().eval()
d = torch.randn(1, C, H, W)     # NCHW
torch.onnx.export(m, d, "/tmp/act.onnx", opset_version=12, input_names=["x"], output_names=["y"])
print("onnx exported", (1, C, H, W))

np.save("/tmp/cal.npy", np.random.randn(1, C, H, W).astype(np.float32))
open("/tmp/dataset.txt", "w").write("/tmp/cal.npy\n")

r = RKNN(verbose=False)
if prec == "i16":
    r.config(target_platform="rk3588", quantized_dtype="w16a16i")
    quant = True
elif prec == "fp16":
    r.config(target_platform="rk3588")
    quant = False
else:  # i8
    r.config(target_platform="rk3588")
    quant = True
assert r.load_onnx(model="/tmp/act.onnx") == 0, "load_onnx failed"
assert r.build(do_quantization=quant, dataset="/tmp/dataset.txt") == 0, "build failed"
assert r.export_rknn(out) == 0, "export failed"
print("EXPORTED", out)
