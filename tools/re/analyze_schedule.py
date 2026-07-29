#!/usr/bin/env python3
# analyze_schedule.py — settle the mfold per-M schedule from a fresh rkllm capture.
# Consumes a verbose rknpu_dump (rkllm.dump) with `--- regcmd (N u32 words)` blocks, tabulates the
# schedule registers per M for the fold shape (default K=3584 N=1216), and — unlike parse_mfold.py —
# also (a) reports reg 0x100c (CONV_CON1 = OKV_CONV1_MFOLD, the lost constant), (b) FITS each schedule
# reg as val = a*M + b across ALL captured M and prints the max residual (a formula that fits every M
# has residual 0), and (c) prints the M=36 row explicitly so the "formula (16*36=576) vs M=36-literal
# (0x600=1536)" inconsistency is resolved straight from silicon data.
#   python3 analyze_schedule.py [rkllm.dump] [K] [N]
import sys, re
from collections import defaultdict

path = sys.argv[1] if len(sys.argv)>1 else 'rkllm.dump'
WK   = int(sys.argv[2]) if len(sys.argv)>2 else 3584
WN   = int(sys.argv[3]) if len(sys.argv)>3 else 1216

lines = open(path,'r',errors='replace').read().splitlines()
hexword = re.compile(r'^[0-9a-fA-F]{8}$')
blocks = []
i=0
while i < len(lines):
    if '--- regcmd' in lines[i]:
        words=[]; i+=1
        while i < len(lines):
            l=lines[i]
            if l.strip().startswith('---') or l.strip().startswith('[dump]') or '===' in l: break
            for t in l.replace('[',' ').replace(']',' ').split():
                if hexword.match(t): words.append(int(t,16))
            i+=1
        # === regcmd word-pair encoding (DEFINITIVE — verified against rkllm_ffn_capture_2026-07-27.dump) ===
        # Each register write is a 64-bit (value,target) pair of u32 words (w0,w1):
        #   w0 = (val16 << 16) | reg16     -> reg  = w0 & 0xffff ;  val = w0 >> 16   (the 16-bit value)
        #   w1 = (block16 << 16) | extra16 -> block= w1 >> 16    ;  extra = w1 & 0xffff
        # The VALUE IS w0>>16 (16-bit). It is NOT (w0>>16)|((w1&0xffff)<<16) — that mis-reads `extra` as a
        # high half and corrupts 16-bit regs (this is the trap: regcmd_capture.c's chain-walk uses the 32-bit
        # form for a few genuinely-32-bit regs, but the general read is 16-bit). A handful of regs ARE 32-bit
        # (addresses 0x1070/0x1110/0x4020, and the >0xffff cases of 0x1044/0x40c0 at very large M); for those
        # the high half is in `extra`. All fold schedule values here fit in 16 bits, so val=w0>>16 is exact.
        regs={}
        for j in range(0,len(words)-1,2):
            w0,w1=words[j],words[j+1]
            reg=w0&0xffff; val=w0>>16; extra=w1&0xffff; block=(w1>>16)&0xffff
            val32 = val | (extra<<16)                 # only meaningful for the known-32-bit regs
            regs[(block,reg)] = (val, val32)
        blocks.append(regs); continue
    i+=1
print(f"parsed {len(blocks)} regcmd blocks from {path}")

# shape inventory: which (K,N) matmuls does the model actually run, and how many tasks each?
def rv0(regs,reg):
    for (b,r),(v,v32) in regs.items():
        if r==reg: return v
    return None
shape_hist=defaultdict(int)
for regs in blocks:
    K = rv0(regs,0x1024) or rv0(regs,0x1088); N=rv0(regs,0x1038)
    if K and N: shape_hist[(K,N)]+=1
print("\n== matmul shape inventory (K,N) -> task count (top 20) ==")
for (K,N),c in sorted(shape_hist.items(), key=lambda x:-x[1])[:20]:
    print(f"  K={K:>6} N={N:>6}  x{c}")

def rv(regs, reg, wide=False):
    for (b,r),(v,v32) in regs.items():
        if r==reg: return v32 if wide else v
    return None

# schedule regs of interest + geometry regs (to test shape-function vs address-offset)
SCHED = [0x100c,0x1040,0x1044,0x104c,0x107c,0x1080,0x4024,0x40c0]
GEOM  = [0x1020,0x1084,0x1028,0x102c,0x1010,0x4030,0x4034,0x405c]  # DATAIN W/H, batch, M, grains, out cube W/H, wdma
# collect ALL values per (M, reg) so we can tell constant-at-M (shape fn) from varies-at-M (address/position)
vals=defaultdict(lambda: defaultdict(set))   # vals[M][reg] = {values...}
ntask=defaultdict(int)
for regs in blocks:
    K = rv(regs,0x1024) or rv(regs,0x1088)
    N = rv(regs,0x1038)
    de= rv(regs,0x1044)
    if not K or not N or not de or K%64: continue
    if K==WK and N==WN:
        # GROUP BY TRUE OUTPUT ROW COUNT = 0x4030+1 (out cube WIDTH = rows-1), NOT by DATA_ENTRIES/56
        # (DATA_ENTRIES = (Ks/64)*rows is a K-slice x row PRODUCT — different factorizations collide).
        w4030 = rv(regs,0x4030)
        M = (w4030+1) if w4030 is not None else (de*64//K)
        ntask[M]+=1
        for r in SCHED+GEOM+[0x1044,0x1024]:
            v=rv(regs,r)
            if v is not None: vals[M][r].add(v)
Ms=sorted(vals)
def cell(M,r):
    s=vals[M].get(r)
    if not s: return '-'
    if len(s)==1: return hex(next(iter(s)))
    return f"{hex(min(s))}..{hex(max(s))}#{len(s)}"     # VARIES at this M -> address/position, not shape
byM={M:{r:(next(iter(vals[M][r])) if len(vals[M].get(r,set()))==1 else None) for r in SCHED} for M in Ms}
print(f"\n== fold K={WK} N={WN}: schedule per M ({sum(ntask.values())} tasks, {len(Ms)} distinct M) ==")
print("   (a#k cell = reg VARIES across the k tasks at that M -> address/position offset, NOT a shape fn)")
print("  M  cnt " + "".join(f"{hex(r):>12}" for r in SCHED))
for M in Ms:
    print(f"{M:>4} {ntask[M]:>4} " + "".join(f"{cell(M,r):>12}" for r in SCHED))
print("\n  M  cnt " + "".join(f"{hex(r):>10}" for r in GEOM))
for M in Ms:
    print(f"{M:>4} {ntask[M]:>4} " + "".join(f"{cell(M,r):>10}" for r in GEOM))

def signed16(v): return v-0x10000 if v>=0x8000 else v

# fit val = a*M + b across all M (integer least-ish: solve from 2 extreme M, report max residual)
print("\n== per-reg fit  val = a*M + b  (residual 0 => formula holds at EVERY captured M) ==")
for r in SCHED:
    pts=[(M,byM[M][r]) for M in Ms if byM[M][r] is not None]
    if len(pts)<2:
        print(f"  {hex(r)}: <2 points"); continue
    # 0x1080 is signed 16-bit in the -3*M theory; fit both raw and signed
    for label,f in (("raw",lambda v:v),) + ((("signed16",signed16),) if r==0x1080 else ()):
        (m0,v0),(m1,v1)=pts[0],pts[-1]
        vv=[(M,f(v)) for M,v in pts]
        (m0,v0),(m1,v1)=vv[0],vv[-1]
        a=(v1-v0)/(m1-m0) if m1!=m0 else 0
        b=v0-a*m0
        res=max(abs(v-(a*M+b)) for M,v in vv)
        # also test pure ratio val = c*M (b==0) and val = c/M
        cM = all(v!=0 and abs(v/ M - vv[0][1]/vv[0][0])<1e-9 for M,v in vv) if vv[0][0] else False
        cover = all(v*M==vv[0][1]*vv[0][0] for M,v in vv)  # val*M constant => val = C/M
        note=""
        if abs(a-round(a))<1e-9 and res<1e-9: note=f"  => {int(round(a))}*M + {int(round(b))} EXACT"
        if cover: note+=f"  => {vv[0][1]*vv[0][0]}/M"
        print(f"  {hex(r)} [{label}]: a={a:.4g} b={b:.4g} maxres={res:.4g}{note}")

print("\n== THEORY CHECK at M=36 (formula 16*M=576/0x240, 128*M=4608/0x1200, 4*M=144, -3*M=0xff94"
      "  vs  literal 0x600, 0x3000, 0x60, 2160/36=60) ==")
if 36 in byM:
    s=byM[36]
    for r,fF,fL in ((0x4024,16*36,0x600),(0x40c0,128*36,0x3000),(0x107c,4*36,0x60),(0x1080,0x10000-3*36,60)):
        v=s.get(r)
        verdict = 'FORMULA' if v==fF else ('LITERAL' if v==fL else 'NEITHER')
        print(f"  {hex(r)}: captured={hex(v) if v is not None else '-'}  formula={hex(fF)}  literal={hex(fL)}  => {verdict}")
else:
    print("  (no M=36 task captured — run a prefill whose chunk hits M=36, or fit will still settle it)")

# === factor by (rows, K-slice): Ks = 64*DATA_ENTRIES/rows. Is the SHAPE group {0x1040,0x107c,0x1080}
# a clean function of (rows,Ks)?  If yes -> per-subblock synth is viable (revive #39). If it still
# co-varies -> the schedule is rkllm's internal per-subblock tiling choice, not shape-derivable. ===
print("\n== factor by (rows, Ks=64*DATA_ENTRIES/rows): shape regs 0x1040/0x107c/0x1080 (addr regs 0x4024/0x40c0 excluded) ==")
fac=defaultdict(lambda: defaultdict(set))
for regs in blocks:
    K = rv(regs,0x1024) or rv(regs,0x1088); N=rv(regs,0x1038); de=rv(regs,0x1044); w=rv(regs,0x4030)
    if not (K==WK and N==WN) or de is None or w is None: continue
    rows=w+1; Ks=64*de//rows if rows else 0
    if 64*de % rows: Ks=-1                                  # non-integer => rows/DATA_ENTRIES mismatch
    for r in (0x1040,0x107c,0x1080):
        v=rv(regs,r)
        if v is not None: fac[(rows,Ks)][r].add(v)
print(f"  {'rows':>4} {'Ks':>6}  {'0x1040':>14} {'0x107c':>12} {'0x1080':>12}   ntiles")
for (rows,Ks) in sorted(fac)[:40]:
    d=fac[(rows,Ks)]
    def c(r):
        s=d.get(r)
        if not s: return '-'
        return hex(next(iter(s))) if len(s)==1 else f"{hex(min(s))}..{hex(max(s))}#{len(s)}"
    print(f"  {rows:>4} {Ks:>6}  {c(0x1040):>14} {c(0x107c):>12} {c(0x1080):>12}")

print("\n== OKV_CONV1_MFOLD candidate = reg 0x100c ==")
vals=set(byM[M][0x100c] for M in Ms if byM[M][0x100c] is not None)
print(f"  0x100c values across M: {[hex(v) for v in sorted(vals)]}  (expect ONE constant => that IS OKV_CONV1_MFOLD)")
