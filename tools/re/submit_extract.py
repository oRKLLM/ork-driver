#!/usr/bin/env python3
# Extract ONE complete SUBMIT (all its task regcmds) from a verbose rknpu dump, to spec the replay executor.
# Groups `--- regcmd` blocks under the preceding `=== SUBMIT ... task_number=T ===` header.
import re,sys
from collections import defaultdict
path=sys.argv[1] if len(sys.argv)>1 else 'rkllm.dump'
want_tn=int(sys.argv[2]) if len(sys.argv)>2 else 21
lines=open(path,'r',errors='replace').read().splitlines()
hexw=re.compile(r'^[0-9a-fA-F]{8}$')
def decode(words):
    r={}
    for j in range(0,len(words)-1,2):
        w0,w1=words[j],words[j+1]; reg=w0&0xffff; val=w0>>16; blk=(w1>>16)&0xffff
        r[(blk,reg)]=val
    return r
submits=[]; cur=None; i=0
while i<len(lines):
    l=lines[i]
    m=re.search(r'task_number=(\d+).*core=0x([0-9a-f]+)',l)
    if '=== SUBMIT' in l and m:
        cur={'tn':int(m.group(1)),'core':m.group(2),'tasks':[]}; submits.append(cur)
    elif '--- regcmd' in l and cur is not None:
        words=[]; i+=1
        while i<len(lines):
            x=lines[i]
            if '---' in x or '===' in x or '[dump]' in x: break
            for t in x.replace('[',' ').replace(']',' ').split():
                if hexw.match(t): words.append(int(t,16))
            i+=1
        cur['tasks'].append(decode(words)); continue
    i+=1
cand=[s for s in submits if s['tn']==want_tn and len(s['tasks'])>=1]
print(f"{len(submits)} submits total; {len(cand)} with task_number={want_tn}")
if not cand: sys.exit(0)
s=cand[len(cand)//2]      # a mid-run one (steady state)
print(f"\n== SUBMIT task_number={s['tn']} core=0x{s['core']}  ({len(s['tasks'])} regcmd blocks captured) ==")
def g(t,blk,reg): return t.get((blk,reg))
print(f"  {'#':>2} {'0x102c(M)':>9} {'0x1044(de)':>10} {'Ks=64de/M':>9} {'0x1024(K)':>9} {'0x1038(N)':>8} {'0x1040':>7} {'0x100c':>7} {'0x4030(w-1)':>11} {'0x4020?':>8}")
for i,t in enumerate(s['tasks']):
    M=g(t,0x201,0x102c); de=g(t,0x201,0x1044); K=g(t,0x201,0x1024); N=g(t,0x201,0x1038)
    cb=g(t,0x201,0x1040); cc=g(t,0x201,0x100c); w=g(t,0x1001,0x4030)
    ks=(64*de//M) if (M and de) else None
    def h(x): return '-' if x is None else (hex(x))
    print(f"  {i:>2} {h(M):>9} {h(de):>10} {h(ks) if ks else '-':>9} {h(K):>9} {h(N):>8} {h(cb):>7} {h(cc):>7} {h(w):>11}")
