"""Bounded true-row decoder and native Q2/Q3/Q4/Q5 construction probes."""
import argparse,collections,ctypes,hashlib,json,struct,time
from pathlib import Path
import numpy as np
from codec import e4m3,nvfp4,mxfp8,pack_ggml_nvfp4
from source import ROOT,fetch,inventory,headers,digest

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--work',type=Path,required=True);ap.add_argument('--library',type=Path,required=True);args=ap.parse_args()
    specs=headers(args.work);files={x['path']:x for x in inventory()['weights']}
    lib=ctypes.CDLL(str(args.library.resolve()))
    lib.dequantize_row_nvfp4.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_int64]
    lib.ggml_quantize_chunk.argtypes=[ctypes.c_int,ctypes.c_void_p,ctypes.c_void_p,ctypes.c_int64,ctypes.c_int64,ctypes.c_int64,ctypes.c_void_p]
    lib.ggml_quantize_chunk.restype=ctypes.c_size_t
    def raw(name,rows=None):
        s=specs[name];count=s['size'] if rows is None else rows*s['size']//s['shape'][0]
        if count>2_000_000:raise RuntimeError('Probe tensor byte cap')
        cache=args.work/'decoder-probes';cache.mkdir(exist_ok=True);p=cache/(digest(f'{name}:{count}'.encode())+'.bin')
        if p.exists():b=p.read_bytes()
        else:b=fetch(s['path'],s['start'],count,files[s['path']]['size'],args.work);p.write_bytes(b)
        if len(b)!=count:raise RuntimeError('Probe size mismatch')
        return b
    import torch
    allbits=np.arange(256,dtype=np.uint8)
    ref=torch.from_numpy(allbits.copy()).view(torch.float8_e4m3fn).float().numpy()
    assert np.array_equal(e4m3(allbits),ref,equal_nan=True)
    results=[]
    for layer,eid,proj in ((3,0,'w1'),(3,127,'w2'),(59,0,'w3')):
        name=f'language_model.model.layers.{layer}.block_sparse_moe.experts.{eid}.{proj}';rows=8
        s=specs[name+'.weight_packed'];cols=s['shape'][1]
        p=np.frombuffer(raw(name+'.weight_packed',rows),np.uint8).reshape(rows,cols)
        scale=np.frombuffer(raw(name+'.weight_scale',rows),np.uint8).reshape(rows,cols//8)
        g=struct.unpack('<f',raw(name+'.weight_global_scale'))[0]
        x=nvfp4(p,scale,g);packed=pack_ggml_nvfp4(p,scale);native=np.empty_like(x)
        lib.dequantize_row_nvfp4(packed.ctypes.data,native.ctypes.data,x.size)
        native/=np.float32(g)
        if not np.array_equal(x,native):raise RuntimeError('True NVFP4 row differs from native ggml reference')
        measurements=[]
        for typ,label,bpr in ((10,'Q2_K',84),(11,'Q3_K',110),(12,'Q4_K',144),(13,'Q5_K',176)):
            out=np.empty((rows,x.shape[1]//256*bpr),np.uint8);digests=[];times=[]
            for _ in range(2):
                tick=time.perf_counter();n=lib.ggml_quantize_chunk(typ,x.ctypes.data,out.ctypes.data,0,rows,x.shape[1],None);times.append(time.perf_counter()-tick)
                assert n==out.size;digests.append(digest(out.tobytes()))
            assert digests[0]==digests[1]
            measurements.append({'type':label,'bytes':out.size,'sha256':digests[0],'rows_per_second_compute_only':rows/min(times)})
        results.append({'name':name,'rows':rows,'decoded_sha256':digest(x.tobytes()),'global_scale':g,'native_decode_exact':True,'quantizers':measurements})
    for layer,proj in ((0,'q_proj'),(59,'o_proj')):
        name=f'language_model.model.layers.{layer}.self_attn.{proj}';rows=8;cols=specs[name+'.weight']['shape'][1]
        w=np.frombuffer(raw(name+'.weight',rows),np.uint8).reshape(rows,cols)
        s=np.frombuffer(raw(name+'.weight_scale_inv',rows),np.uint8).reshape(rows,cols//32)
        x=mxfp8(w,s)
        tw=torch.from_numpy(w.copy()).view(torch.float8_e4m3fn).float().reshape(rows,-1,32)
        ts=torch.ldexp(torch.ones(s.shape),torch.from_numpy(s.astype(np.int32))-127)
        reference=(tw*ts[:,:,None]).reshape(rows,cols).numpy()
        assert np.array_equal(x,reference)
        results.append({'name':name,'rows':rows,'decoded_sha256':digest(x.tobytes()),'torch_reference_exact':True})
    result={'source_revision':inventory()['source_revision'],'decoder_probes':results,'all_256_e4m3_encodings_match_torch':True,
            'range_budget':json.loads((args.work/'range-budget.json').read_text()),'full_source_files_read':False,'model_inference_tested':False,'model_uploaded':False}
    (args.work/'decoder-results.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2),flush=True)

if __name__=='__main__':main()
