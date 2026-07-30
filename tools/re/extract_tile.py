#!/usr/bin/env python3
# extract_tile.py — pull ONE task's RAW regcmd words (in order) from a verbose rknpu dump, so it can be
# replayed/chained by ork_npu_mfold_chain_cap. Picks a task_number=T submit (mid-run = steady state) and the
# first task matching a target fold M (0x102c) at full K=3584 / N=1216. Writes space-separated 8-hex words.
#   python3 extract_tile.py <dump> <targetM> <outfile> [task_number=21] [K=3584] [N=1216]
import re,sys
path=sys.argv[1]; targetM=int(sys.argv[2]); outfile=sys.argv[3]
want_tn=int(sys.argv[4]) if len(sys.argv)>4 else 21
K=int(sys.argv[5]) if len(sys.argv)>5 else 3584
N=int(sys.argv[6]) if len(sys.argv)>6 else 1216
lines=open(path,'r',errors='replace').read().splitlines()
hexw=re.compile(r'^[0-9a-fA-F]{8}$')
def decode(words):
    r={}
    for j in range(0,len(words)-1,2):
        reg=words[j]&0xffff; val=words[j]>>16; blk=(words[j+1]>>16)&0xffff
        r[(blk,reg)]=val
    return r
submits=[]; cur=None; i=0
while i<len(lines):
    l=lines[i]
    m=re.search(r'task_number=(\d+).*core=0x([0-9a-f]+)',l)
    if '=== SUBMIT' in l and m:
        cur={'tn':int(m.group(1)),'tasks':[]}; submits.append(cur)
    elif '--- regcmd' in l and cur is not None:
        words=[]; i+=1
        while i<len(lines):
            x=lines[i]
            if '---' in x or '===' in x or '[dump]' in x: break
            for t in x.replace('[',' ').replace(']',' ').split():
                if hexw.match(t): words.append(int(t,16))
            i+=1
        cur['tasks'].append(words); continue
    i+=1
cand=[s for s in submits if s['tn']==want_tn and s['tasks']]
print(f"{len(submits)} submits; {len(cand)} with task_number={want_tn}")
if not cand: sys.exit(1)
for s in cand[len(cand)//2:]+cand:            # prefer a steady-state submit, fall back to any
    for ti,words in enumerate(s['tasks']):
        r=decode(words)
        M=r.get((0x201,0x102c)); de=r.get((0x201,0x1044)); Nf=r.get((0x201,0x1038)); Kf=r.get((0x201,0x1024))
        if M==targetM and de==(K//64)*targetM and Nf==N:
            open(outfile,'w').write(' '.join(f'{w:08x}' for w in words)+'\n')
            def h(k): v=r.get(k); return '-' if v is None else hex(v)
            print(f"wrote {outfile}: task#{ti} M={M} de={h((0x201,0x1044))} N={h((0x201,0x1038))} K={h((0x201,0x1024))} "
                  f"words={len(words)} 0x1040={h((0x201,0x1040))} 0x100c={h((0x201,0x100c))} 0x107c={h((0x201,0x107c))} "
                  f"0x1080={h((0x201,0x1080))} 0x4024={h((0x1001,0x4024))} 0x40c0={h((0x1001,0x40c0))}")
            sys.exit(0)
print(f"no task with M={targetM} full K/N found"); sys.exit(1)
