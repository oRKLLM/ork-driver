#!/usr/bin/env python3
# Extract the FIRST `ntask` K=3584 fold tiles in ORDER from a prefill capture (the true prefill-start sequence,
# whose first tile is the real weight-loader with a clean predecessor = model init). Robust block read (skip
# [dump] lines). Writes one tile/line (232 hex words) + prints M/mode order.
import re,sys
lines=open(sys.argv[1],"r",errors="replace").read().splitlines()
out=sys.argv[2]; ntask=int(sys.argv[3]) if len(sys.argv)>3 else 12
hx=re.compile(r"^[0-9a-fA-F]{8}$"); tre=re.compile(r"task\[(\d+)\]:")
K1024=(3583<<16)|3584   # 0x1024 = ((K-1)<<16)|K for K=3584
blocks=[]; i=0
while i<len(lines):
    if "--- regcmd" in lines[i]:
        need=int(re.search(r"regcmd \((\d+)",lines[i]).group(1)); w=[]; i+=1
        while i<len(lines) and len(w)<need:
            x=lines[i]
            if "--- regcmd" in x or "=== SUBMIT" in x or tre.search(x): break
            for t in x.replace("["," ").replace("]"," ").split():
                if hx.match(t): w.append(int(t,16))
            i+=1
        blocks.append(w); continue
    i+=1
def fld(w,off,blk=0x201):
    for j in range(0,min(len(w),216)-1,2):
        if (w[j]&0xffff)==off and ((w[j+1]>>16)&0xffff)==blk: return (w[j]>>16)|((w[j+1]&0xffff)<<16)
    return None
f=open(out,"w"); n=0; seq=[]
for w in blocks:
    if len(w)<216: continue
    if fld(w,0x1024)!=K1024: continue        # K=3584 tiles only
    M=fld(w,0x102c); c=fld(w,0x100c); mode="GLO" if c==0x20000000 else "PLN"
    seq.append(f"{M}:{mode}"); f.write(" ".join("%08x"%x for x in w[:232])+"\n"); n+=1
    if n>=ntask: break
f.close()
print(f"first {n} K=3584 fold tiles (prefill start): "+"  ".join(seq))
print(f"wrote {out}")
