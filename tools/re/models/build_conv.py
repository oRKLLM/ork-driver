#!/usr/bin/env python3
"""build_conv.py — build single-Conv .rknn models for regcmd capture (D_BANK / CNA-path RE).

Companion to build_act.py. A single Conv2d exposes the conv (CNA + CDMA/CBUF) program, which the
RKNN matmul-API path never emits. Sweeping the shape axes that drive CBUF residency — Cout/Cin
(weight banks) and H/W (data banks) — and diffing the captured regcmd is how you hunt the CBUF
bank-split ("D_BANK", NVDLA CDMA 0x50bc) and map the domain-2001 (0x50xx) CDMA register block.

Run INSIDE the rknn-toolkit2 container (x86/aarch64 Linux; needs torch + onnx + rknn-toolkit2):
    python3 build_conv.py                 # builds the default sweep into ./*.rknn
Then copy the .rknn to the board and capture each under rknpu_dump.so (see run_rknn.c / README).

RE/calibration only — uses the proprietary rknn-toolkit2 (never part of the library; AGENTS.md).
"""
import sys
import torch
import torch.nn as nn
from rknn.api import RKNN


def make(cin, cout, hw, k=3, quant=False, name="conv"):
    m = nn.Sequential(nn.Conv2d(cin, cout, k, padding=k // 2)).eval()
    x = torch.randn(1, cin, hw, hw)
    onnx = f"{name}.onnx"
    torch.onnx.export(m, x, onnx, input_names=["in"], output_names=["out"], opset_version=12)
    r = RKNN(verbose=False)
    r.config(target_platform="rk3588")
    if r.load_onnx(model=onnx) != 0:
        print("LOAD FAIL", name); return
    # do_quantization=False -> fp16 conv (still exercises CNA+CBUF). Set True (with a dataset) for int8.
    if r.build(do_quantization=quant) != 0:
        print("BUILD FAIL", name); r.release(); return
    if r.export_rknn(f"{name}.rknn") != 0:
        print("EXPORT FAIL", name); r.release(); return
    r.release()
    print("BUILT", name, "cin", cin, "cout", cout, "hw", hw, "quant", quant)


if __name__ == "__main__":
    q = "--int8" in sys.argv
    # baseline + weight sweep (Cout, Cin) + data/spatial sweep (H/W)
    make(32, 32, 8, quant=q, name="conv_base")
    make(32, 128, 8, quant=q, name="conv_cout128")
    make(32, 512, 8, quant=q, name="conv_cout512")
    make(128, 32, 8, quant=q, name="conv_cin128")
    make(32, 32, 32, quant=q, name="conv_hw32")
    make(32, 32, 64, quant=q, name="conv_hw64")
