#!/usr/bin/env python3
# Build a big-feature conv ONNX (256x256x64 -> forces CNA spatial tiling; weight tiny -> resident, reuse candidate)
# and convert to an RK3588 .rknn via rknn-toolkit2. int8 (w8a8) to match ork's fold path; fp16 fallback.
import numpy as np, onnx
from onnx import helper, TensorProto, numpy_helper
C,H,W,K,ks = 64,256,256,64,3
X =helper.make_tensor_value_info('X', TensorProto.FLOAT,[1,C,H,W])
Y =helper.make_tensor_value_info('Y', TensorProto.FLOAT,[1,K,H,W])
Wt=numpy_helper.from_array((((np.random.rand(K,C,ks,ks)-0.5))*0.1).astype(np.float32), name='W')
Bt=numpy_helper.from_array(np.zeros(K,dtype=np.float32), name='B')
node=helper.make_node('Conv',['X','W','B'],['Y'],kernel_shape=[ks,ks],pads=[1,1,1,1],strides=[1,1])
gph =helper.make_graph([node],'bigconv',[X],[Y],[Wt,Bt])
m=helper.make_model(gph, opset_imports=[helper.make_opsetid('',12)]); m.ir_version=8
onnx.save(m,'/work/bigconv.onnx'); print('onnx saved  feature=%dx%dx%d weight=%dx%dx%dx%d'%(C,H,W,K,C,ks,ks))
np.save('/work/cal.npy', np.random.rand(1,C,H,W).astype(np.float32)); open('/work/dataset.txt','w').write('/work/cal.npy\n')
from rknn.api import RKNN
def build(q):
    r=RKNN(verbose=False); r.config(target_platform='rk3588')
    if r.load_onnx(model='/work/bigconv.onnx')!=0: print('load_onnx FAIL'); r.release(); return None
    if r.build(do_quantization=q, dataset='/work/dataset.txt' if q else None)!=0: print('build FAIL q=%s'%q); r.release(); return None
    out='/work/bigconv_%s.rknn'%('i8' if q else 'f16')
    ok=r.export_rknn(out); r.release()
    if ok!=0: print('export FAIL'); return None
    print('EXPORTED', out); return out
build(True) or build(False)
