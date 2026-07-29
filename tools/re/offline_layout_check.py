#!/usr/bin/env python3
# Decisive OFFLINE layout check (no board): does rkllm's OWN A*W (de-tiled via our confirmed fold layouts)
# equal rkllm's OWN captured output C?  If yes -> layout formulas are correct against real rkllm data, and any
# replay mismatch is an execution/context problem. If no -> our layout understanding of the operands is wrong.
#   python3 offline_layout_check.py [M=36] [K=3584] [N=1216]   (reads /tmp-pulled mm_{A,weight,C}.bin here)
import sys,struct
M=int(sys.argv[1]) if len(sys.argv)>1 else 36
K=int(sys.argv[2]) if len(sys.argv)>2 else 3584
N=int(sys.argv[3]) if len(sys.argv)>3 else 1216
A=open('mm_A.bin','rb').read()
W=open('mm_weight.bin','rb').read()
Cb=open('mm_C.bin','rb').read()
C=struct.unpack('<%di'%(len(Cb)//4), Cb[:4*(len(Cb)//4)])
ncap=len(C)
def s8(b): return b-256 if b>=128 else b
def nc16(m,k,w): return (k//16)*(w*16)+m*16+(k%16)          # C2-16 input
def woff(n,k,K):
    KT=(K+31)//32; return ((n//32)*KT+(k//32))*1024+(n%32)*32+(k%32)   # ork weight
def c4(m,n,w): return (n//4)*(w*4)+m*4+(n%4)                # C2-4 output
print(f"M={M} K={K} N={N}  A={len(A)}B W={len(W)}B  C={ncap} int32 captured")

# sample (m,n) whose c4 index is within the captured C, compare A*W to rkllm's C
import itertools
samples=[(0,0),(0,1),(0,2),(0,3),(1,0),(0,4),(2,0),(0,8),(1,1),(3,5),(0,100),(5,200),(0,1215),(7,7)]
ok=0; bad=0
for (m,n) in samples:
    ci=c4(m,n,M)
    if ci>=ncap: print(f"  (m={m},n={n}) c4={ci} beyond captured C, skip"); continue
    acc=0
    for k in range(K):
        ai=nc16(m,k,M); wi=woff(n,k,K)
        if ai>=len(A) or wi>=len(W): acc=None; break
        acc += s8(A[ai])*s8(W[wi])
    got=C[ci]
    tag = "OK" if acc==got else "MISMATCH"
    if acc==got: ok+=1
    else: bad+=1
    print(f"  m={m:>3} n={n:>4}  A*W={acc}  rkllmC={got}  {tag}")
print(f"\n{ok} match / {bad} mismatch  ->  {'LAYOUT CONFIRMED vs rkllm data' if bad==0 and ok>0 else 'LAYOUT WRONG (or wrong tile/partial-C indexing)'}")
