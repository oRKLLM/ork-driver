#!/usr/bin/env python3
# Parse rkllm.dump -> per-regcmd-block register maps -> tabulate the fold's per-M SCHEDULE registers.
# regcmd word pairs: word0 = (val<<16)|reg ; word1 = (block<<16)|extra.  We key regs by (block,reg).
import re, sys
from collections import defaultdict

lines = open('rkllm.dump','r',errors='replace').read().splitlines()
blocks=[]           # list of dict {(block,reg): val32}
i=0
hexword=re.compile(r'^[0-9a-fA-F]{8}$')
while i < len(lines):
    ln=lines[i]
    if '--- regcmd' in ln:
        # collect hex words until the next '---' section header
        words=[]
        i+=1
        while i < len(lines):
            l=lines[i]
            if l.strip().startswith('---') or l.strip().startswith('[dump]'):
                break
            toks=l.replace('[',' ').replace(']',' ').split()
            for t in toks:
                if hexword.match(t): words.append(int(t,16))
            i+=1
        # pair consecutive words: (w0,w1) -> reg=w0&0xffff, val=w0>>16, block=w1>>16, extra=w1&0xffff
        regs={}
        for j in range(0,len(words)-1,2):
            w0,w1=words[j],words[j+1]
            reg=w0&0xffff; val=w0>>16; block=w1>>16; extra=w1&0xffff
            # 32-bit value regs (e.g. 0x1080 stride) put the high bits in extra
            full = val | (extra<<16) if extra not in (0,0x0101) else val
            regs[(block,reg)] = (val, extra, full)
        blocks.append(regs)
        continue
    i+=1

print(f"parsed {len(blocks)} regcmd blocks")

# helpers to read a reg value (search across likely blocks)
def rv(regs, reg):
    for (b,r),(val,extra,full) in regs.items():
        if r==reg: return val
    return None

# For each block, derive K, N, M and collect schedule regs. K from 0x1024 or 0x1088; N from 0x1038; DATA_ENTRIES 0x1044.
rows=[]
for regs in blocks:
    K = rv(regs,0x1024) or rv(regs,0x1088)
    N = rv(regs,0x1038)
    de = rv(regs,0x1044)          # (K/64)*M
    if not K or not N or not de: continue
    if K%64: continue
    M = de*64/K
    if M!=int(M): continue
    M=int(M)
    rows.append((M,K,N,{
        '1040':rv(regs,0x1040),'104c':rv(regs,0x104c),'107c':rv(regs,0x107c),
        '1080':rv(regs,0x1080),'1044':de,'100c':rv(regs,0x100c),
        '4024':rv(regs,0x4024),'40c0':rv(regs,0x40c0),
    }))

# focus on the fold shape K=3584 N=1216, tabulate schedule regs per M
print("\n== fold K=3584 N=1216: schedule registers per M-tile ==")
print(f"{'M':>4} {'1040':>6} {'104c':>6} {'107c':>6} {'1080':>8} {'100c':>8} {'4024':>6} {'40c0':>6}  count")
seen=defaultdict(list)
for M,K,N,s in rows:
    if K==3584 and N==1216: seen[M].append(s)
for M in sorted(seen):
    ss=seen[M]; s=ss[0]
    def h(x): return '-' if x is None else f'0x{x:x}'
    print(f"{M:>4} {h(s['1040']):>6} {h(s['104c']):>6} {h(s['107c']):>6} {h(s['1080']):>8} {h(s['100c']):>8} {h(s['4024']):>6} {h(s['40c0']):>6}  x{len(ss)}")
print("\n== all (K,N) shapes seen (top) ==")
shp=defaultdict(int)
for M,K,N,s in rows: shp[(K,N)]+=1
for (K,N),c in sorted(shp.items(),key=lambda x:-x[1])[:12]:
    print(f"  K={K} N={N}: {c} blocks")
