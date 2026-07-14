#!/usr/bin/env python3
# Decode an ork regcmd (list of u32 words) against rocket_registers.h: name each register + break out fields.
import re, sys
RH = open("rocket_regs.h").read()
reg_by_off = {}
for m in re.finditer(r'#define\s+REG_(\w+)\s+(0x[0-9a-fA-F]+)', RH):
    reg_by_off[int(m.group(2),16) & 0xffff] = m.group(1)          # low 16 bits = the regcmd addr
fields = {}  # regname -> [(field, mask, shift)]
masks = dict(re.findall(r'#define\s+(\w+)__MASK\s+(0x[0-9a-fA-F]+)', RH))
shifts = dict(re.findall(r'#define\s+(\w+)__SHIFT\s+(\d+)', RH))
for fq, mk in masks.items():
    # fq = REGNAME_FIELD ; find the longest REG_<name> prefix that matches
    for rn in reg_by_off.values():
        if fq.startswith(rn + "_"):
            fields.setdefault(rn, []).append((fq[len(rn)+1:], int(mk,16), int(shifts.get(fq,0))))
# words: read from a file of "0xXXXXXXXX," tokens
words = [int(w,16) for w in re.findall(r'0x[0-9a-fA-F]{8}', open(sys.argv[1]).read())]
i=0
while i+1 < len(words):
    w0,w1 = words[i], words[i+1]
    reg = w0 & 0xffff
    if reg==0 and w0==0: i+=2; continue
    val = ((w0>>16)&0xffff) | ((w1&0xffff)<<16)
    name = reg_by_off.get(reg, "?")
    fb=[]
    for fn,mk,sh in sorted(fields.get(name,[]), key=lambda x:-x[1]):
        fv=(val&mk)>>sh
        if fv and "RESERVED" not in fn: fb.append(f"{fn}={fv:#x}")
    print(f"  0x{reg:04x} {name:32s} = 0x{val:08x}  {' '.join(fb)}")
    i+=2
