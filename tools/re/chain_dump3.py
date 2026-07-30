#!/usr/bin/env python3
# Robust chain extractor: read each "--- regcmd (N words)" block reading EXACTLY N hex words, SKIPPING interleaved
# [dump] log lines (the bug that truncated the M=2 tile). Associate each block with its preceding task[i]
# descriptor (index, regcfg_amount, regcmd_addr). Output the first `ntask` blocks in order (one per line, hex),
# annotated with task-index / M / mode / regcfg / addr so a clean same-submit chain can be picked.
import re,sys
lines=open(sys.argv[1],"r",errors="replace").read().splitlines()
out=sys.argv[2]; ntask=int(sys.argv[3]) if len(sys.argv)>3 else 9
hx=re.compile(r"^[0-9a-fA-F]{8}$")
tre=re.compile(r"task\[(\d+)\]:.*regcfg_amount=(\d+).*regcmd_addr=(0x[0-9a-fA-F]+)")
blocks=[]; pend=None; i=0
while i<len(lines):
    l=lines[i]
    mt=tre.search(l)
    if mt: pend=(int(mt.group(1)),int(mt.group(2)),mt.group(3))
    elif "--- regcmd" in l:
        need=int(re.search(r"regcmd \((\d+)",l).group(1)); w=[]; i+=1
        while i<len(lines) and len(w)<need:
            x=lines[i]
            if "--- regcmd" in x or "=== SUBMIT" in x or tre.search(x): break   # next block/task/submit
            for t in x.replace("["," ").replace("]"," ").split():               # [dump] lines contribute no 8-hex token -> skipped
                if hx.match(t): w.append(int(t,16))
            i+=1
        blocks.append((pend,w)); continue
    i+=1
def field(w,off,blk=0x201):
    for j in range(0,min(len(w),216)-1,2):
        if (w[j]&0xffff)==off and ((w[j+1]>>16)&0xffff)==blk: return (w[j]>>16)|((w[j+1]&0xffff)<<16)
    return None
f=open(out,"w"); n=0
print("idx | task[i] | words | M | mode | regcfg | regcmd_addr")
for (pend,w) in blocks[:ntask]:
    if len(w)<216: continue
    M=field(w,0x102c); c=field(w,0x100c); mode="GLO" if c==0x20000000 else ("PLAIN" if c==0 else hex(c) if c is not None else "?")
    ti,rcf,addr = pend if pend else ("?","?","?")
    print(f" {n:2} | task[{ti}] | {len(w):3} | {M} | {mode:5} | {rcf} | {addr}")
    f.write(" ".join(f"{x:08x}" for x in w[:232])+"\n"); n+=1
f.close(); print(f"wrote {n} tasks to {out}")
