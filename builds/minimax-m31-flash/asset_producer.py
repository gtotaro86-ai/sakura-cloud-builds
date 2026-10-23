"""Pinned standard F16 vision projector conversion, streamed on CPU."""
import argparse,gc,hashlib,json,logging,subprocess,sys
from pathlib import Path
from types import SimpleNamespace
from source import ROOT,metadata,headers,digest,VerifiedShardCache,inventory
from producer import setup
from lazy_memory import bounded_eager

NAME='MiniMax-M3.1-Flash-mmproj-f16.gguf'
def make_model(llama,work,read):
    import torch,gguf
    from conversion import get_model_class,LazyTorchTensor
    specs=headers(work);base=get_model_class('MiniMaxM3SparseForConditionalGeneration',mmproj=True)
    class Projector(base):
        model_arch=base.model_arch
        def index_tensors(self,remote_hf_model_id=None):
            result={}
            for name,s in sorted(specs.items(),key=lambda x:(x[1]['path'],x[0])):
                if not name.startswith(('vision_tower.','multi_modal_projector.','patch_merge_mlp.')):continue
                if s['dtype'] not in ('BF16','F32'):raise RuntimeError('Unexpected quantized vision source')
                remote=SimpleNamespace(dtype=s['dtype'],shape=tuple(s['shape']),data=lambda s=s:read(s))
                item=self.filter_tensors((name,lambda r=remote:LazyTorchTensor.from_remote_tensor(r)))
                if item:result[item[0]]=item[1]
            return result
    folder=metadata(work);cfg=json.loads((folder/'config.json').read_text());cfg.pop('quantization_config',None)
    model=Projector(folder,gguf.LlamaFileType.MOSTLY_F16,Path(work)/'unused-projector.gguf',hparams=cfg,model_name='MiniMax-M3.1-Flash Vision')
    torch.set_num_threads(4);model.prepare_tensors();model.prepare_metadata(vocab_only=False)
    return model

def plan(model,work):
    writer=model.gguf_writer;path=Path(work)/'projector.header';writer.path=path
    entries=[];offset=0
    for name,t in writer.tensors[0].items():
        entries.append({'name':name,'shape':list(t.shape),'type':int(t.dtype),'nbytes':t.nbytes,'offset':offset});offset+=(t.nbytes+31)//32*32
    writer.write_header_to_file();writer.write_kv_data_to_file();writer.write_ti_data_to_file();writer.write_padding(writer.fout[0],writer.fout[0].tell());writer.close()
    raw=path.read_bytes();p={'path':NAME,'source_revision':inventory()['source_revision'],'header_size':len(raw),'header_sha256':digest(raw),'size':len(raw)+offset,'tensors':entries}
    frozen=ROOT/'asset-plan.json'
    if frozen.exists() and p!=json.loads(frozen.read_text()):raise RuntimeError('Frozen projector plan differs')
    return p

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--llama',type=Path,required=True);ap.add_argument('--work',type=Path,required=True);ap.add_argument('--mode',choices=('plan','stream'),required=True);args=ap.parse_args()
    logging.basicConfig(level=logging.WARNING,stream=sys.stderr);setup(args.llama.resolve());args.work.mkdir(parents=True,exist_ok=True)
    cache=VerifiedShardCache(args.work) if args.mode=='stream' else None
    def refuse(_):raise RuntimeError('Projector plan may not read weights')
    model=make_model(args.llama,args.work,cache.read if cache else refuse);p=plan(model,args.work)
    if args.mode=='plan':(args.work/'asset-plan.json').write_text(json.dumps(p,indent=2)+'\n');print(json.dumps({k:v for k,v in p.items() if k!='tensors'}));return
    import gguf,numpy as np
    if p!=json.loads((args.work/'asset-plan.json').read_text()):raise RuntimeError('Changed projector plan')
    h=hashlib.sha256();count=0;proof=[];sink=sys.stdout.buffer
    def emit(raw):
        nonlocal count
        h.update(raw);count+=len(raw);sink.write(raw)
    emit((args.work/'projector.header').read_bytes())
    for t in p['tensors']:
        ti=model.gguf_writer.tensors[0][t['name']]
        with bounded_eager(ti.tensor) as eager:data=np.ascontiguousarray(eager);raw=data.tobytes()
        if len(raw)!=t['nbytes'] or not np.isfinite(gguf.quants.dequantize(data,ti.dtype)).all():raise RuntimeError('Invalid vision tensor')
        emit(raw);emit(b'\0'*(-len(raw)%32));proof.append({'name':t['name'],'bytes':len(raw),'sha256':digest(raw)});ti.tensor=None;del data,raw;gc.collect()
    sink.flush()
    if count!=p['size']:raise RuntimeError('Projector stream size')
    (args.work/'asset-receipt.json').write_text(json.dumps({'path':NAME,'size':count,'sha256':h.hexdigest(),'tensors':proof,'source_files_verified':cache.verified},indent=2)+'\n')

if __name__=='__main__':main()
