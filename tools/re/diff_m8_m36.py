#!/usr/bin/env python3
# Diff two captured tile regcmds (M8 self-contained/runs vs M36 wedges): which (blk,reg) does M8 SET that M36
# OMITS (=> M36 inherits it), which does M36 set that M8 doesn't, and which differ. Path-1: to make M36 run
# standalone, inject the OMITTED (M8-only) regs.
import sys
def load(p):
    w=[int(x,16) for x in open(p).read().split()]
    r={}
    for j in range(0,min(len(w),216)-1,2):   # 108-reg body
        reg=w[j]&0xffff; blk=(w[j+1]>>16)&0xffff; val=(w[j]>>16)|((w[j+1]&0xffff)<<16)
        if (blk,reg) not in r: r[(blk,reg)]=val
    return r
a=load(sys.argv[1]); b=load(sys.argv[2])  # a=M8, b=M36
ka=set(a); kb=set(b)
print("== regs M8 SETS but M36 OMITS (M36 inherits these; inject to self-contain M36) ==")
for k in sorted(ka-kb): print("  blk=%#x reg=%#06x  M8val=%#x"%(k[0],k[1],a[k]))
print("== regs M36 SETS but M8 OMITS ==")
for k in sorted(kb-ka): print("  blk=%#x reg=%#06x  M36val=%#x"%(k[0],k[1],b[k]))
print("== common regs that DIFFER (geometry etc.) ==")
for k in sorted(ka&kb):
    if a[k]!=b[k]: print("  blk=%#x reg=%#06x  M8=%#x M36=%#x"%(k[0],k[1],a[k],b[k]))
