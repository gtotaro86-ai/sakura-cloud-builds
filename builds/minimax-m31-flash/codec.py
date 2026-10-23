"""CPU decoding of released NVFP4 expert and MXFP8 attention weight rows."""
import numpy as np

def e4m3(raw):
    u=np.asarray(raw,dtype=np.uint8)
    exponent=((u>>3)&15).astype(np.int16);fraction=(u&7).astype(np.float32)
    out=np.where(exponent==0,np.ldexp(fraction,-9),np.ldexp(1+fraction/8,exponent-7)).astype(np.float32)
    out=np.where(u&128,-out,out).astype(np.float32)
    out=np.where((u&127)==127,np.float32(np.nan),out)
    return out

FP4=np.array([0,.5,1,1.5,2,3,4,6,0,-.5,-1,-1.5,-2,-3,-4,-6],np.float32)

def nvfp4(packed,scales,global_scale):
    p=np.asarray(packed,dtype=np.uint8);s=e4m3(scales)
    if p.ndim!=2 or s.shape!=(p.shape[0],p.shape[1]//8):raise ValueError('NVFP4 shape/group mismatch')
    g=np.float32(global_scale)
    if not np.isfinite(g) or g<=0 or not np.isfinite(s).all() or (s<0).any():raise ValueError('Invalid NVFP4 scales')
    n=np.empty((len(p),p.shape[1]*2),np.uint8);n[:,::2]=p&15;n[:,1::2]=p>>4
    out=FP4[n].reshape(len(p),-1,16)*s[:,:,None]/g
    out=np.ascontiguousarray(out.reshape(len(p),-1),np.float32)
    if not np.isfinite(out).all():raise ValueError('Nonfinite decoded NVFP4')
    return out

def mxfp8(weight,scale):
    w=np.asarray(weight,dtype=np.uint8);u=np.asarray(scale,dtype=np.uint8)
    if w.ndim!=2 or u.shape!=(len(w),w.shape[1]//32):raise ValueError('MXFP8 shape/group mismatch')
    if (u==255).any():raise ValueError('NaN UE8M0 scale')
    s=np.ldexp(np.ones(u.shape,np.float32),u.astype(np.int16)-127)
    out=(e4m3(w).reshape(len(w),-1,32)*s[:,:,None]).reshape(w.shape)
    if not np.isfinite(out).all():raise ValueError('Nonfinite decoded MXFP8')
    return np.ascontiguousarray(out,np.float32)

def pack_ggml_nvfp4(packed,scales):
    # Independent native ggml dequantizer reference uses the same source bytes
    # repacked into its 64-value superblocks; global weight scale is separate.
    p=np.asarray(packed,np.uint8);s=np.asarray(scales,np.uint8)
    rows=len(p);blocks=s.shape[1]
    if blocks%4:raise ValueError('Native NVFP4 reference alignment')
    vals=np.empty((rows,blocks,16),np.uint8)
    raw=p.reshape(rows,blocks,8);vals[:,:,::2]=raw&15;vals[:,:,1::2]=raw>>4
    qs=vals[:,:,:8]|(vals[:,:,8:]<<4)
    return np.ascontiguousarray(np.concatenate((s.reshape(rows,blocks//4,4),qs.reshape(rows,blocks//4,32)),axis=-1))
