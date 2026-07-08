#!/usr/bin/env python3
"""build_ops.py — build single-op .rknn models for the regcmd-capture campaign.

Companion to build_act.py / build_conv.py. Each op exposes which NPU engine block(s) it uses, so
capturing + decoding them (run_rknn + rknpu_dump.so + decode_reg on the board) builds an op->engine
map. Results (2026-07-08) are on the wiki: regcmd-ISA-Reference "Op -> engine-block map".

Run INSIDE the rknn-toolkit2 container (x86/aarch64 Linux; torch + onnx + rknn-toolkit2):
    python3 build_ops.py
then copy *.rknn to the board and capture each. RE/calibration only (proprietary toolkit; AGENTS.md).
"""
import torch
import torch.nn as nn
from rknn.api import RKNN

C, H = 64, 16
x = torch.randn(1, C, H, H)
x2 = torch.randn(1, C, H, H)


def build(mod, inputs, name):
    ni = len(inputs) if isinstance(inputs, tuple) else 1
    torch.onnx.export(mod, inputs, f"{name}.onnx",
                      input_names=[f"in{i}" for i in range(ni)], output_names=["out"], opset_version=13)
    r = RKNN(verbose=False)
    r.config(target_platform="rk3588")
    if r.load_onnx(model=f"{name}.onnx") != 0: print("LOAD FAIL", name); return
    if r.build(do_quantization=False) != 0: print("BUILD FAIL", name); r.release(); return
    if r.export_rknn(f"{name}.rknn") != 0: print("EXPORT FAIL", name); r.release(); return
    r.release()
    print("BUILT", name)


class Add(nn.Module):
    def forward(s, a, b): return a + b


class Mul(nn.Module):
    def forward(s, a, b): return a * b


class Cat(nn.Module):
    def forward(s, a, b): return torch.cat([a, b], 1)


class Transpose(nn.Module):
    def forward(s, a): return a.permute(0, 1, 3, 2).contiguous()


class ReduceMean(nn.Module):
    def forward(s, a): return a.mean(1, keepdim=True)


class ReduceMax(nn.Module):
    def forward(s, a): return a.amax(1, keepdim=True)


if __name__ == "__main__":
    OPS = [
        # elementwise / structural
        (nn.Softmax(dim=1), x, "op_softmax"),
        (Add(), (x, x2), "op_add"), (Mul(), (x, x2), "op_mul"),
        (Cat(), (x, x2), "op_concat"), (Transpose(), x, "op_transpose"),
        (nn.Upsample(scale_factor=2, mode="nearest"), x, "op_resize"),
        (nn.LayerNorm([C, H, H]), x, "op_layernorm"), (nn.BatchNorm2d(C).eval(), x, "op_batchnorm"),
        (nn.LocalResponseNorm(5), x, "op_lrn"),
        # activations: relu-family = PPU clamp (no LUT); smooth = SDP LUT (1001:1079)
        (nn.ReLU(), x, "op_relu"), (nn.ReLU6(), x, "op_relu6"), (nn.LeakyReLU(0.1), x, "op_leakyrelu"),
        (nn.Sigmoid(), x, "op_sigmoid"), (nn.GELU(), x, "op_gelu"), (nn.SiLU(), x, "op_silu"),
        (nn.Tanh(), x, "op_tanh"), (nn.Hardswish(), x, "op_hardswish"),
        # pooling (PDP 0x4001/0x8001) + reductions (CNA+PDP)
        (nn.AvgPool2d(2), x, "op_avgpool"), (nn.MaxPool2d(2), x, "op_maxpool"),
        (nn.AdaptiveAvgPool2d(1), x, "op_globalavgpool"),
        (ReduceMean(), x, "op_reducemean"), (ReduceMax(), x, "op_reducemax"),
        # conv variants + matmul
        (nn.Conv2d(C, C, 3, padding=1, groups=C).eval(), x, "op_dwconv"),
        (nn.Linear(C, 128).eval(), torch.randn(1, C), "op_gemm"),
    ]
    for mod, inp, name in OPS:
        build(mod.eval() if hasattr(mod, "eval") else mod, inp, name)
