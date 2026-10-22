"""Small real BF16 CPU checks; no full model download, no inference, no upload."""
import argparse, ctypes, gc, hashlib, json, logging, math, os, re, subprocess, sys, tempfile
from pathlib import Path
from unittest import mock
from source import fetch, inventory, headers, VerifiedShardCache, digest
from producer import setup, make_model, Quantizer, assemble_plan
from publish import authentication_token

def range_checks(work):
    for raw in ('dummy-token',' dummy-token ','dummy-token\r\n'):
        with mock.patch.dict(os.environ,{'HF_TOKEN':raw}):assert authentication_token()=='dummy-token'
    for raw in ('',' ','dummy\ntoken','dummy token','dumm\u00fcy'):
        with mock.patch.dict(os.environ,{'HF_TOKEN':raw}):
            try:authentication_token()
            except RuntimeError as error:assert raw not in str(error) if raw.strip() else True
            else:raise AssertionError('Malformed credential accepted')
    class Response:
        def __init__(self,status,contentrange,body):
            self.status=status;self.headers={'Content-Range':contentrange};self.body=body;self.reads=0
        def __enter__(self):return self
        def __exit__(self,*args):pass
        def read(self,n):self.reads+=1;return self.body[:n]
    for status,cr,body in [(200,'bytes 0-3/8',b'abcd'),(206,'bytes 1-4/8',b'abcd')]:
        response=Response(status,cr,body)
        with mock.patch('source.urllib.request.urlopen',return_value=response):
            try:fetch('fake',0,4,8,work/'http-checks')
            except RuntimeError:pass
            else:raise AssertionError('Bad HTTP response accepted')
        assert response.reads==0
    good=Response(206,'bytes 0-3/8',b'abcd')
    with mock.patch('source.urllib.request.urlopen',return_value=good):assert fetch('fake',0,4,8,work/'http-checks')==b'abcd'
    with mock.patch.dict(os.environ,{},clear=True):
        try:VerifiedShardCache(work)
        except RuntimeError:pass
        else:raise AssertionError('Non-GitHub full-source read guard failed')

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--llama',type=Path,required=True);ap.add_argument('--build',type=Path,required=True)
    ap.add_argument('--work',type=Path,required=True);args=ap.parse_args()
    args.work.mkdir(parents=True,exist_ok=True);logging.basicConfig(level=logging.WARNING,stream=sys.stderr)
    setup(args.llama.resolve());import numpy as np,gguf
    range_checks(args.work);specs=headers(args.work);files={s['path']:s for s in inventory()['weights']}
    q=Quantizer(args.build)
    def sample_fetch(s,start,count):
        folder=args.work/'small-probes';folder.mkdir(exist_ok=True)
        path=folder/(digest(f"{s['path']}:{start}:{count}".encode())+'.bin')
        checksum=path.with_suffix('.sha256')
        if path.exists() and checksum.exists():
            raw=path.read_bytes()
            if len(raw)!=count or digest(raw)!=checksum.read_text():raise RuntimeError('Cached small probe hash mismatch')
            return raw
        raw=fetch(s['path'],start,count,files[s['path']]['size'],args.work)
        path.write_bytes(raw);checksum.write_text(digest(raw));return raw
    # One true BF16 expert matrix row range; no fused expert matrix needs to be loaded.
    name='model.language_model.layers.0.mlp.experts.0.down_proj.weight';s=specs[name]
    cols=s['shape'][-1];rows=16;count=rows*cols*2
    raw=sample_fetch(s,s['start'],count)
    bf16=np.frombuffer(raw,dtype=np.uint8).reshape(rows,cols*2)
    x=np.ascontiguousarray(gguf.quants.dequantize(bf16,gguf.GGMLQuantizationType.BF16),dtype=np.float32)
    assert np.array_equal(x,(np.frombuffer(raw,dtype='<u2').astype('<u4')<<16).view('<f4').reshape(rows,cols))
    hashes={}
    for typ in (gguf.GGMLQuantizationType.Q4_K,gguf.GGMLQuantizationType.Q5_K,gguf.GGMLQuantizationType.Q6_K):
        streamed=b''.join(q.chunks(bf16,(rows,cols),gguf.GGMLQuantizationType.BF16,typ))
        twice=b''.join(q.chunks(bf16,(rows,cols),gguf.GGMLQuantizationType.BF16,typ));assert streamed==twice
        out=np.empty(len(streamed),np.uint8)
        n=q.lib.ggml_quantize_chunk(typ,x.ctypes.data,out.ctypes.data,0,rows,cols,None)
        assert n==len(streamed) and streamed==out.tobytes()
        hashes[typ.name]=digest(streamed)
    def small_read(s):
        if s['size']>1024**2:raise RuntimeError('Small probe source cap')
        return bytearray(sample_fetch(s,s['start'],s['size']))
    from concurrent.futures import ThreadPoolExecutor
    probes=[s for n,s in specs.items() if (n.startswith('model.language_model.layers.0.') or n.startswith('mtp.')) and math.prod(s['shape'])<=2048]
    with ThreadPoolExecutor(max_workers=4) as pool:
        for result in pool.map(small_read,probes):del result
    model=make_model(args.llama,args.work,small_read)
    tensors=model.gguf_writer.tensors[0]
    # Exercise real norm transforms, delta-time/head reorder, and MTP tensor export.
    selected=[n for n,t in tensors.items() if math.prod(t.shape)<=2048 and (n.startswith('blk.0.') or n.startswith('blk.40.'))]
    if not selected:raise AssertionError('No small real conversion probes')
    real={}
    for n in selected:
        ti=tensors[n];data=gguf.LazyNumpyTensor.to_eager(ti.tensor)
        assert np.isfinite(gguf.quants.dequantize(data,ti.dtype)).all();real[n]=digest(np.ascontiguousarray(data).tobytes())
    plans={}
    for variant in ('Q4_K_M','Q5_K_M'):
        plan,head=assemble_plan(model,q,variant,args.work)
        assert head[:4]==b'GGUF' and len(head)%32==0 and plan['tensor_count']==len(tensors)
        assert any(t['name'].startswith('blk.40.') for t in plan['tensors'])
        assert set(plan['types'])<={'F32','BF16','F16','Q4_K','Q5_K','Q6_K','Q8_0'}
        assert plan['size']==len(head)+sum((t['nbytes']+31)//32*32 for t in plan['tensors'])
        plans[variant]={k:v for k,v in plan.items() if k!='tensors'}
        (args.work/(variant+'-plan.json')).write_text(json.dumps(plan,indent=2)+'\n')
        (args.work/(variant+'-header.gguf')).write_bytes(head)
    assert plans['Q4_K_M']['size']<plans['Q5_K_M']['size']<37_802_150_112
    # A sparse file contains only metadata and zero holes, never actual full model weights.
    # Native --dry-run reads no tensor payload. Compare every planned type independently.
    bfplan,bfheader=assemble_plan(model,q,'BF16',args.work)
    sparse=args.work/'metadata-only-sparse.gguf'
    try:
        with sparse.open('wb') as out:out.write(bfheader);out.truncate(bfplan['size'])
        for variant in plans:
            log=args.work/(variant+'-native.log')
            with log.open('wb') as err:
                subprocess.run([sys.executable,str(Path(__file__).with_name('native_plan.py')),
                                str(args.build/'lib_sage_quant.so'),str(sparse),
                                str(int(getattr(gguf.LlamaFileType,'MOSTLY_'+variant)))],stderr=err,stdout=err,check=True)
            native={}
            for line in log.read_text().splitlines():
                match=re.search(r'\[\s*\d+/\s*\d+\]\s+(\S+)\s+-.*?type =\s+(\S+),\s*size =.*',line)
                if match:
                    target=re.search(r'\((\w+)\)\s*$',line)
                    native[match[1]]=(target[1] if target else match[2]).upper()
            planned=json.loads((args.work/(variant+'-plan.json')).read_text())
            expected={t['name']:gguf.GGMLQuantizationType(t['type']).name for t in planned['tensors']}
            if native!=expected:
                diff={n:(native.get(n),v) for n,v in expected.items() if native.get(n)!=v}
                raise AssertionError('Native full-model type plan differs: '+json.dumps(diff))
            plans[variant]['native_full_model_type_plan_verified']=True
    finally:sparse.unlink(missing_ok=True)
    result={'context':'bounded cloud CPU development check','source_revision':inventory()['source_revision'],
            'source_probe':{'tensor':name,'rows':rows,'bytes':len(raw),'sha256':digest(raw)},'quantized_sha256':hashes,
            'small_converted_tensors':real,'plans':plans,'range_budget':json.loads((args.work/'range-budget.json').read_text()),
            'model_uploaded':False,'full_inference_test_performed':False,'full_source_files_read':False}
    (args.work/'checks.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))

if __name__=='__main__':main()
