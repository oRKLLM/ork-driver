#!/usr/bin/env python3
# Offline layout solver: given rkllm's exact A,W,C for one tile, search simple layout hypotheses for A/W/C
# until A*W == C. Tests width-padding (M -> aligned) + a few weight/input/output orderings.
import struct,sys
M=24; K=3584; N=1216
A=open('mm_A.bin','rb').read(); W=open('mm_weight.bin','rb').read()
Cb=open('mm_C.bin','rb').read(); C=struct.unpack('<%di'%(len(Cb)//4),Cb[:4*(len(Cb)//4)])
def s8(b): return b-256 if b>=128 else b
KT=(K+31)//32

# candidate index fns, parameterized by width pad wp
def A_c216(m,k,wp): return (k//16)*(wp*16)+m*16+(k%16)
def A_rowmaj(m,k,wp): return m*K+k
def A_c216w(m,k,wp): return (k//16)*(wp*16)+(k%16)*wp+m      # transposed within group
def W_woff(n,k):  return ((n//32)*KT+(k//32))*1024+(n%32)*32+(k%32)
def W_kn(n,k):    return k*N+n
def W_nk(n,k):    return n*K+k
def C_c4(m,n,wp): return (n//4)*(wp*4)+m*4+(n%4)
def C_rowmaj(m,n,wp): return m*N+n

def dot(m,n,Af,Wf,wp):
    acc=0
    for k in range(K):
        ai=Af(m,k,wp); wi=W_woff(n,k) if Wf=='woff' else (W_kn(n,k) if Wf=='kn' else W_nk(n,k))
        if ai>=len(A) or wi>=len(W): return None
        acc+=s8(A[ai])*s8(W[wi])
    return acc

pads=[24,32,16,48]
Afns={'c216':A_c216,'row':A_rowmaj,'c216w':A_c216w}
Wfns=['woff','kn','nk']
Cfns={'c4':C_c4,'row':C_rowmaj}
probes=[(0,0),(0,1),(1,0),(2,3)]
best=None
for wp in pads:
 for an,Af in Afns.items():
  for Wf in Wfns:
   for cn,Cf in Cfns.items():
     okc=0; tot=0; detail=[]
     for (m,n) in probes:
        ci=Cf(m,n,wp)
        if ci>=len(C): detail.append('oob'); continue
        d=dot(m,n,Af,Wf,wp)
        if d is None: detail.append('Aoob'); continue
        tot+=1;
        if d==C[ci]: okc+=1
        detail.append(f"{d}/{C[ci]}")
     if okc>0:
        print(f"pad={wp} A={an} W={Wf} C={cn}: {okc}/{tot} match  {detail}")
     if best is None or okc>best[0]: best=(okc,wp,an,Wf,cn)
print("best:",best)
