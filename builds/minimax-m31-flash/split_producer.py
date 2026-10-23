"""Seek-free CPU stream of standard split GGUFs; one expert at a time."""
import argparse,copy,gc,hashlib,json,logging,re,sys,time
from pathlib import Path
from producer import setup,make_model,Quantizer,assemble_plan
from lazy_memory import bounded_eager
from source import ROOT,inventory,VerifiedShardCache,digest

VARIANTS=('Q2_K','Q3_K_M','Q4_K_M','Q5_K_M')
def split_plan(model,q,variant,work):
    import gguf,numpy as np
    whole,_=assemble_plan(model,q,variant,work)
    groups=[];current=[];size=0
    for t in whole['tensors']:
        padded=(t['nbytes']+31)//32*32
        if current and size+padded>47_900_000_000:groups.append(current);current=[];size=0
        current.append(t);size+=padded
    if current:groups.append(current)
    files=[]
    for i,group in enumerate(groups):
        name=f'Sakura-MiniMax-M3.1-Flash-{variant}-{i+1:05d}-of-{len(groups):05d}.gguf'
        header=Path(work)/(name+'.header')
        w=gguf.GGUFWriter(header,'minimax-m3');w.kv_data=copy.deepcopy(model.gguf_writer.kv_data)
        w.add_string('general.architecture','minimaxm31_sakura')
        w.add_file_type(getattr(gguf.LlamaFileType,'MOSTLY_'+variant));w.add_quantization_version(gguf.GGML_QUANT_VERSION)
        w.add_uint16('split.no',i);w.add_uint16('split.count',len(groups));w.add_int32('split.tensors.count',len(whole['tensors']))
        entries=[];offset=0
        for original in group:
            t=dict(original,offset=offset);entries.append(t)
            w.add_tensor_info(t['name'],tuple(t['shape']),np.dtype(np.float32),t['nbytes'],raw_dtype=gguf.GGMLQuantizationType(t['type']))
            offset+=(t['nbytes']+31)//32*32
        w.write_header_to_file();w.write_kv_data_to_file();w.write_ti_data_to_file();w.write_padding(w.fout[0],w.fout[0].tell());w.close()
        raw=header.read_bytes()
        f={'path':name,'header_size':len(raw),'header_sha256':digest(raw),'size':len(raw)+offset,'tensors':entries}
        if f['size']>=48_000_000_000:raise RuntimeError('Output shard budget exceeded')
        files.append(f)
    result={'variant':variant,'source_revision':whole['source_revision'],'llama_revision':whole['llama_revision'],
            'architecture':'minimaxm31_sakura','runtime_patch_sha256':digest((ROOT/'runtime.patch').read_bytes()),
            'tensor_count':whole['tensor_count'],'types':whole['types'],'size':sum(f['size'] for f in files),'files':files}
    frozen=ROOT/'split-plans.json'
    if frozen.exists() and result!=json.loads(frozen.read_text())[variant]:raise RuntimeError('Frozen split output plan differs')
    return result

def arrays(model,cache,entry,ti):
    import numpy as np,gguf
    name=entry['name'];match=re.fullmatch(r'blk[.](\d+)[.]ffn_(gate|down|up)_exps[.]weight',name)
    if match:
        bid=int(match[1]);wid={'gate':'w1','down':'w2','up':'w3'}[match[2]]
        for eid in range(128):
            key=f'language_model.model.layers.{bid}.block_sparse_moe.experts.{eid}.{wid}.weight'
            spec=model.logical_specs[key]
            x=np.frombuffer(model.source_decoder(spec),'<f4').reshape(spec['shape'])
            yield gguf.quants.quantize(x,gguf.GGMLQuantizationType(ti.dtype)),tuple(spec['shape'])
        return
    direct={'token_embd.weight':'language_model.model.embed_tokens.weight','output.weight':'language_model.lm_head.weight'}
    if name in direct:
        spec=model.logical_specs[direct[name]]
        if spec['kind']!='raw' or spec['dtype']!='BF16' or int(ti.dtype)!=int(gguf.GGMLQuantizationType.BF16):raise RuntimeError('Unexpected embedding encoding')
        cols=spec['shape'][1];step=max(1,(8*1024**2)//(cols*2))
        for row in range(0,spec['shape'][0],step):
            rows=min(step,spec['shape'][0]-row)
            part=dict(spec,start=spec['start']+row*cols*2,size=rows*cols*2)
            yield np.frombuffer(cache.read(part),np.uint8).reshape(rows,cols*2),(rows,cols)
        return
    with bounded_eager(ti.tensor) as data:yield data,ti.shape

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--llama',type=Path,required=True);ap.add_argument('--build',type=Path,required=True)
    ap.add_argument('--work',type=Path,required=True);ap.add_argument('--variant',choices=VARIANTS,required=True);ap.add_argument('--mode',choices=('plan','stream'),required=True);ap.add_argument('--file');args=ap.parse_args()
    logging.basicConfig(level=logging.WARNING,stream=sys.stderr);setup(args.llama.resolve());args.work.mkdir(parents=True,exist_ok=True)
    cache=VerifiedShardCache(args.work) if args.mode=='stream' else None
    def refuse(_):raise RuntimeError('Metadata planning must not read source payloads')
    model=make_model(args.llama,args.work,cache.read if cache else refuse);q=Quantizer(args.build)
    plan=split_plan(model,q,args.variant,args.work);saved=args.work/'split-plan.json'
    if args.mode=='plan':saved.write_text(json.dumps(plan,indent=2)+'\n');print(json.dumps({'variant':args.variant,'size':plan['size'],'shards':len(plan['files'])}));return
    if plan!=json.loads(saved.read_text()):raise RuntimeError('Split plan changed between passes')
    import gguf
    sink=sys.stdout.buffer;tick=time.perf_counter();receipts=[]
    files=[f for f in plan['files'] if args.file is None or f['path']==args.file]
    if not files:raise RuntimeError('Unknown output shard')
    for file in files:
        h=hashlib.sha256();count=0;proofs=[]
        def emit(raw):
            nonlocal count
            sink.write(raw);h.update(raw);count+=len(raw)
        emit((args.work/(file['path']+'.header')).read_bytes())
        for t in file['tensors']:
            ti=model.gguf_writer.tensors[0][t['name']];th=hashlib.sha256();n=0
            for data,shape in arrays(model,cache,t,ti):
                for raw in q.chunks(data,shape,ti.dtype,gguf.GGMLQuantizationType(t['type'])):emit(raw);th.update(raw);n+=len(raw)
                del data
            if n!=t['nbytes']:raise RuntimeError('Tensor size differs')
            emit(b'\0'*(-n%32));proofs.append({'name':t['name'],'bytes':n,'sha256':th.hexdigest()});ti.tensor=None;gc.collect()
            print(json.dumps({'file':file['path'],'quantized':t['name'],'bytes':count,'total':file['size'],'seconds':time.perf_counter()-tick}),file=sys.stderr,flush=True)
        if count!=file['size']:raise RuntimeError('File size differs')
        sink.flush();receipts.append({'path':file['path'],'size':count,'sha256':h.hexdigest(),'tensors':proofs})
    receipt={'files':receipts,'source_files_verified':cache.verified,'source_shard_loads':cache.loads,'seconds':time.perf_counter()-tick,'full_inference_test_performed':False}
    (args.work/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps({'stream_complete':True,'variant':args.variant,'files':[{'path':f['path'],'size':f['size'],'sha256':f['sha256']} for f in receipts]}),file=sys.stderr,flush=True)

if __name__=='__main__':main()
