"""Map the released hybrid checkpoint to logical BF16/F32 tensor inputs."""
import math
import numpy as np
from codec import nvfp4,mxfp8

def logical_tensors(specs):
    out={};consumed=set()
    def add(name,s,used):
        if name in out:raise RuntimeError('Duplicate decoded tensor')
        out[name]=dict(s,name=name);consumed.update(used)
    for name,s in specs.items():
        if name.endswith('.weight_packed'):
            base=name.removesuffix('.weight_packed')
            names={suffix:base+suffix for suffix in ('.weight_packed','.weight_scale','.weight_global_scale','.input_global_scale')}
            if not all(n in specs for n in names.values()):raise RuntimeError('Incomplete NVFP4 source group')
            if s['dtype']!='U8' or specs[names['.weight_scale']]['dtype']!='F8_E4M3':raise RuntimeError('Unexpected NVFP4 layout')
            rows,packed=s['shape'];scale=specs[names['.weight_scale']]
            if scale['shape']!=[rows,packed//8]:raise RuntimeError('NVFP4 scale shape')
            add(base+'.weight',{'kind':'nvfp4','dtype':'F32','shape':[rows,packed*2],'parts':names,'path':s['path']},names.values())
        elif s['dtype']=='F8_E4M3' and name.endswith('.weight'):
            scale=name+'_scale_inv'
            if scale not in specs or specs[scale]['dtype']!='U8':raise RuntimeError('Missing MXFP8 scales')
            rows,cols=s['shape']
            if specs[scale]['shape']!=[rows,cols//32]:raise RuntimeError('MXFP8 scale shape')
            add(name,{'kind':'mxfp8','dtype':'F32','shape':s['shape'],'parts':{'weight':name,'scale':scale},'path':s['path']},(name,scale))
    for name,s in specs.items():
        if name in consumed:continue
        if s['dtype'] not in ('BF16','F32'):raise RuntimeError('Unconsumed quantized source: '+name)
        add(name,dict(s,kind='raw',physical=name),(name,))
    if consumed!=set(specs):raise RuntimeError('Logical source partition incomplete')
    return out

class Decoder:
    def __init__(self,specs,read):self.specs=specs;self.read=read
    def __call__(self,s):
        if s['kind']=='raw':return self.read(self.specs[s['physical']])
        if s['kind']=='nvfp4':
            parts=s['parts'];rows,cols=s['shape']
            p=np.frombuffer(self.read(self.specs[parts['.weight_packed']]),np.uint8).reshape(rows,cols//2)
            scale=np.frombuffer(self.read(self.specs[parts['.weight_scale']]),np.uint8).reshape(rows,cols//16)
            g=np.frombuffer(self.read(self.specs[parts['.weight_global_scale']]),'<f4')[0]
            return bytearray(nvfp4(p,scale,g).tobytes())
        parts=s['parts'];rows,cols=s['shape']
        w=np.frombuffer(self.read(self.specs[parts['weight']]),np.uint8).reshape(rows,cols)
        scale=np.frombuffer(self.read(self.specs[parts['scale']]),np.uint8).reshape(rows,cols//32)
        return bytearray(mxfp8(w,scale).tobytes())
