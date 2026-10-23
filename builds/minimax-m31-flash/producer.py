"""Decode released hybrid weights, apply official conversion/type policy, stream on CPU."""
import argparse, collections, copy, ctypes, gc, hashlib, json, logging, math, os, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from source import ROOT, LLAMA, headers, inventory, metadata, digest, VerifiedShardCache
from logical import logical_tensors,Decoder

def setup(llama):
    if subprocess.check_output(['git','-C',str(llama),'rev-parse','HEAD'],text=True).strip()!=LLAMA:
        raise RuntimeError('Pinned llama.cpp revision required')
    sys.path.insert(0,str(llama)); sys.path.insert(0,str(llama/'gguf-py'))

def make_model(llama, work, reader):
    import gguf, torch
    from conversion import get_model_class, LazyTorchTensor
    physical=headers(work);specs=logical_tensors(physical);decode=Decoder(physical,reader)
    base=get_model_class('MiniMaxM3SparseForConditionalGeneration')
    class PinnedMiniMax(base):
        model_arch=base.model_arch
        def index_tensors(self, remote_hf_model_id=None):
            type(self)._experts_cache={}
            result={}
            for name,s in sorted(specs.items(), key=lambda pair:(pair[1]['path'],pair[0])):
                remote=SimpleNamespace(dtype=s['dtype'],shape=tuple(s['shape']),data=lambda s=s:decode(s))
                gen=lambda r=remote:LazyTorchTensor.from_remote_tensor(r)
                item=self.filter_tensors((name,gen))
                if item:
                    if item[0] in result: raise RuntimeError('Duplicate filtered tensor')
                    result[item[0]]=item[1]
            return result
    torch.set_num_threads(4)
    folder=metadata(work)
    cfg=json.loads((folder/'config.json').read_text());cfg.pop('quantization_config',None)
    model=PinnedMiniMax(folder,gguf.LlamaFileType.MOSTLY_BF16,Path(work)/'unused.gguf',
                     hparams=cfg,model_name='MiniMax-M3.1-Flash')
    model.prepare_tensors(); model.prepare_metadata(vocab_only=False)
    model.gguf_writer.add_uint32('minimax-m3.attention.indexer.first_layer',0)
    model.logical_specs=specs;model.source_decoder=decode
    if len(model.gguf_writer.tensors)!=1:raise RuntimeError('Unexpected converter split')
    return model

class Quantizer:
    def __init__(self, build):
        import numpy as np
        self.np=np
        self.bridge=ctypes.CDLL(str((build/'lib_minimax_quant.so').resolve()))
        self.lib=ctypes.CDLL(str((build/'bin/libggml-base.so').resolve()))
        self.bridge.sage_types.argtypes=[ctypes.c_int,ctypes.c_int,ctypes.POINTER(ctypes.c_char_p),ctypes.c_void_p,ctypes.c_void_p,ctypes.c_void_p,ctypes.c_void_p]
        self.bridge.sage_error.restype=ctypes.c_char_p
        self.lib.ggml_quantize_chunk.argtypes=[ctypes.c_int,ctypes.c_void_p,ctypes.c_void_p,ctypes.c_int64,ctypes.c_int64,ctypes.c_int64,ctypes.c_void_p]
        self.lib.ggml_quantize_chunk.restype=ctypes.c_size_t
        self.lib.ggml_validate_row_data.argtypes=[ctypes.c_int,ctypes.c_void_p,ctypes.c_size_t]
        self.lib.ggml_validate_row_data.restype=ctypes.c_bool
        self.pool=ThreadPoolExecutor(max_workers=4)

    def plan(self, tensors, variant):
        import gguf
        np=self.np; names=list(tensors)
        c_names=(ctypes.c_char_p*len(names))(*(x.encode() for x in names))
        shapes=np.ones((len(names),4),dtype=np.int64);dims=np.empty(len(names),np.int32)
        types=np.array([int(tensors[n].dtype) for n in names],np.int32);out=np.empty_like(types)
        for i,n in enumerate(names):
            dims[i]=len(tensors[n].shape);shapes[i,:dims[i]]=list(reversed(tensors[n].shape))
        ftype=getattr(gguf.LlamaFileType,'MOSTLY_'+variant)
        if self.bridge.sage_types(ftype,len(names),c_names,shapes.ctypes.data,dims.ctypes.data,types.ctypes.data,out.ctypes.data):
            raise RuntimeError(self.bridge.sage_error().decode())
        return {n:gguf.GGMLQuantizationType(int(t)) for n,t in zip(names,out)}

    def chunks(self, data, shape, input_type, output_type):
        import gguf
        np=self.np
        # This is exactly the BF16/F32 intermediate selected by the official converter.
        if output_type==input_type:
            raw=np.ascontiguousarray(data).view(np.uint8).reshape(-1)
            for start in range(0,len(raw),8*1024**2):
                block=raw[start:start+8*1024**2]
                if not self.lib.ggml_validate_row_data(input_type,block.ctypes.data,len(block)):
                    raise RuntimeError('Unquantized tensor validation failed')
                yield block.tobytes()
            return
        cols=shape[-1];rows=math.prod(shape)//cols
        block,blockbytes=gguf.GGML_QUANT_SIZES[output_type]
        if cols%block:raise RuntimeError('Unsupported quantizer block shape')
        rowbytes=cols//block*blockbytes
        data=data.reshape((rows,-1))
        rows_per=max(1,(8*1024**2)//(cols*4))
        for start in range(0,rows,rows_per):
            x=np.ascontiguousarray(gguf.quants.dequantize(data[start:start+rows_per],input_type),dtype=np.float32)
            if x.shape[1]!=cols or not np.isfinite(x).all():raise RuntimeError('Invalid BF16/F32 source values')
            result=np.empty((len(x),rowbytes),dtype=np.uint8)
            def run(a,b):
                written=self.lib.ggml_quantize_chunk(output_type,x[a:b].ctypes.data,result[a:b].ctypes.data,0,b-a,cols,None)
                if written!=(b-a)*rowbytes or not self.lib.ggml_validate_row_data(output_type,result[a:b].ctypes.data,written):
                    raise RuntimeError('Quantizer row size/data validation failed')
            stride=max(1,(len(x)+3)//4)
            futures=[self.pool.submit(run,a,min(a+stride,len(x))) for a in range(0,len(x),stride)]
            for f in futures:f.result()
            yield result.tobytes()

def assemble_plan(model, q, variant, work):
    import gguf, numpy as np
    src=model.gguf_writer; tensors=src.tensors[0]; types=q.plan(tensors,variant)
    dst=gguf.GGUFWriter(Path(work)/'header.gguf','minimax-m3')
    dst.kv_data=copy.deepcopy(src.kv_data)
    dst.add_string('general.architecture','minimaxm31_sakura')
    dst.add_file_type(getattr(gguf.LlamaFileType,'MOSTLY_'+variant))
    dst.add_quantization_version(gguf.GGML_QUANT_VERSION)
    entries=[]; offset=0
    for name,ti in tensors.items():
        typ=types[name];block,size=gguf.GGML_QUANT_SIZES[typ]
        if ti.shape[-1]%block:raise RuntimeError('Unaligned planned tensor')
        nbytes=math.prod(ti.shape)//block*size
        dst.add_tensor_info(name,ti.shape,np.dtype(np.float32),nbytes,raw_dtype=typ)
        entries.append({'name':name,'shape':list(ti.shape),'source_type':int(ti.dtype),'type':int(typ),
                        'nbytes':nbytes,'offset':offset})
        offset+=gguf.GGUFWriter.ggml_pad(nbytes,dst.data_alignment)
    dst.write_header_to_file();dst.write_kv_data_to_file();dst.write_ti_data_to_file()
    dst.write_padding(dst.fout[0],dst.fout[0].tell());dst.close()
    header=(Path(work)/'header.gguf').read_bytes()
    plan={'variant':variant,'source_revision':inventory()['source_revision'],'llama_revision':LLAMA,
          'header_size':len(header),'header_sha256':digest(header),'size':len(header)+offset,
          'tensor_count':len(entries),'tensors':entries,
          'types':dict(collections.Counter(gguf.GGMLQuantizationType(t['type']).name for t in entries))}
    if model.block_count!=60:raise RuntimeError('MiniMax block count mismatch')
    if sum(t['name'].endswith('indexer.q_proj.weight') for t in entries)!=60:raise RuntimeError('All 60 sparse indexers must be retained')
    frozen_path=ROOT/'expected-plans.json'
    if variant in ('Q2_K','Q3_K_M','Q4_K_M','Q5_K_M') and frozen_path.exists():
        expected=json.loads(frozen_path.read_text())[variant]
        summary={k:v for k,v in plan.items() if k!='tensors'}
        expected={k:v for k,v in expected.items() if k!='native_full_model_type_plan_verified'}
        if summary!=expected:raise RuntimeError('Pinned release plan/header changed')
    return plan,header

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--llama',type=Path,required=True);ap.add_argument('--build',type=Path,required=True)
    ap.add_argument('--work',type=Path,required=True);ap.add_argument('--variant',choices=['Q2_K','Q3_K_M','Q4_K_M','Q5_K_M'],required=True)
    ap.add_argument('--mode',choices=['plan','stream'],required=True);args=ap.parse_args()
    logging.basicConfig(level=logging.INFO,stream=sys.stderr)
    setup(args.llama.resolve());args.work.mkdir(parents=True,exist_ok=True)
    cache=VerifiedShardCache(args.work) if args.mode=='stream' else None
    def refuse(_):raise RuntimeError('Plan must not read weight payloads')
    model=make_model(args.llama,args.work,cache.read if cache else refuse)
    q=Quantizer(args.build)
    plan,header=assemble_plan(model,q,args.variant,args.work)
    planpath=args.work/'plan.json'
    if args.mode=='plan':
        planpath.write_text(json.dumps(plan,indent=2)+'\n');print(json.dumps({k:v for k,v in plan.items() if k!='tensors'}));return
    if plan!=json.loads(planpath.read_text()):raise RuntimeError('Plan changed between passes')
    h=hashlib.sha256();count=0;proofs=[];sink=sys.stdout.buffer;tick=time.perf_counter()
    def emit(raw):
        nonlocal count
        sink.write(raw);h.update(raw);count+=len(raw)
    emit(header)
    import gguf
    for entry in plan['tensors']:
        name=entry['name'];ti=model.gguf_writer.tensors[0][name]
        def arrays():
            import re,numpy as np
            match=re.fullmatch(r'blk[.](\d+)[.]ffn_(gate|down|up)_exps[.]weight',name)
            if match:
                bid=int(match[1]);wid={'gate':'w1','down':'w2','up':'w3'}[match[2]]
                for eid in range(128):
                    key=f'language_model.model.layers.{bid}.block_sparse_moe.experts.{eid}.{wid}.weight'
                    spec=model.logical_specs[key]
                    x=np.frombuffer(model.source_decoder(spec),'<f4').reshape(spec['shape'])
                    converted=gguf.quants.quantize(x,gguf.GGMLQuantizationType(ti.dtype))
                    yield converted,tuple(spec['shape'])
                return
            direct={'token_embd.weight':'language_model.model.embed_tokens.weight','output.weight':'language_model.lm_head.weight'}
            if name in direct:
                spec=model.logical_specs[direct[name]]
                if spec['kind']!='raw' or spec['dtype']!='BF16' or int(ti.dtype)!=int(gguf.GGMLQuantizationType.BF16):raise RuntimeError('Unexpected embedding source encoding')
                cols=spec['shape'][1];step=max(1,(8*1024**2)//(cols*2))
                for row in range(0,spec['shape'][0],step):
                    rows=min(step,spec['shape'][0]-row)
                    part=dict(spec,start=spec['start']+row*cols*2,size=rows*cols*2)
                    data=np.frombuffer(cache.read(part),np.uint8).reshape(rows,cols*2)
                    yield data,(rows,cols)
                return
            yield gguf.LazyNumpyTensor.to_eager(ti.tensor),ti.shape
        tensorhash=hashlib.sha256();n=0
        for data,shape in arrays():
            for raw in q.chunks(data,shape,ti.dtype,gguf.GGMLQuantizationType(entry['type'])):
                emit(raw);tensorhash.update(raw);n+=len(raw)
            del data
        if n!=entry['nbytes']:raise RuntimeError('Tensor stream size mismatch')
        emit(b'\0'*(-n%32));proofs.append({'name':name,'sha256':tensorhash.hexdigest(),'bytes':n})
        ti.tensor=None;gc.collect()
        print(json.dumps({'quantized':name,'bytes':count,'total':plan['size'],'seconds':time.perf_counter()-tick}),file=sys.stderr,flush=True)
    sink.flush()
    if count!=plan['size']:raise RuntimeError('Final GGUF size mismatch')
    expected={x['path'] for x in inventory()['weights']}
    # Vision-only shards still contribute other language tensors, but verify all source files explicitly.
    if set(cache.verified)!=expected:
        for spec in inventory()['weights']:
            if spec['path'] not in cache.verified:cache.read({'path':spec['path'],'start':0,'size':8})
    receipt={'size':count,'sha256':h.hexdigest(),'tensors':proofs,'source_files_verified':cache.verified,
             'source_shard_loads':cache.loads,'seconds':time.perf_counter()-tick,'full_inference_test_performed':False}
    (args.work/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps({'stream_complete':True,'size':count,'sha256':h.hexdigest()}),file=sys.stderr,flush=True)

if __name__=='__main__':main()
