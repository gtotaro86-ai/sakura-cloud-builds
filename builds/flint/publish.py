"""CPU producer -> native Xet -> independently verified atomic model release."""
import argparse,base64,hashlib,json,os,subprocess,time,zlib
from pathlib import Path
from source import ROOT,inventory,digest
from credentials import authentication_token,check_context
from release_card import readme

PREFIX='FLINT_CHECKPOINT_V1 '

def target(variant):
    if variant not in ('Q4_K_M','Q5_K_M'):raise RuntimeError('Unauthorized quantization')
    return f'webmp3/Sakura-Flint-{variant}-GGUF',f'Sakura-Flint-{variant}.gguf'

def validate_manifest(manifest):
    plan=manifest['output'];variant=plan['variant'];target(variant)
    expected=json.loads((ROOT/'expected-plans.json').read_text())[variant]
    summary={k:v for k,v in plan.items() if k!='tensors'}
    if summary!={k:v for k,v in expected.items() if k!='native_full_model_type_plan_verified'}:
        raise RuntimeError('Frozen output plan changed')
    pinned=json.loads((ROOT/'evidence'/f'{variant}-plan.json').read_text())
    if plan!=pinned or manifest['source']!=inventory():raise RuntimeError('Pinned provenance differs')
    receipt=manifest['receipt'];sha=manifest['sha256']
    if len(sha)!=64 or len(bytes.fromhex(sha))!=32:raise RuntimeError('Invalid final checksum')
    if receipt['size']!=plan['size'] or receipt['sha256']!=sha:raise RuntimeError('Producer proof mismatch')
    if receipt['source_files_verified']!={x['path']:x['sha256'] for x in inventory()['weights']}:
        raise RuntimeError('Incomplete original source verification')
    if len(receipt['tensors'])!=len(plan['tensors']):raise RuntimeError('Incomplete tensor proof')
    for proof,t in zip(receipt['tensors'],plan['tensors']):
        if proof['name']!=t['name'] or proof['bytes']!=t['nbytes'] or len(bytes.fromhex(proof['sha256']))!=32:
            raise RuntimeError('Tensor proof differs')
    if manifest.get('upload_native_sha256_verified') is not True:raise RuntimeError('Native Xet proof missing')
    return plan,sha

def verify_remote(api,repo,revision,plan,sha):
    _,path=target(plan['variant']);info=api.model_info(repo,revision=revision,files_metadata=True)
    f=next(x for x in info.siblings if x.rfilename==path)
    if f.size!=plan['size'] or not f.lfs or f.lfs.sha256!=sha:raise RuntimeError('Hub file SHA256/size differs')
    return info

def commit_existing(api,manifest,sample):
    from huggingface_hub import CommitOperationAdd,hf_hub_download
    from huggingface_hub.lfs import UploadInfo
    plan,sha=validate_manifest(manifest);repo,path=target(plan['variant'])
    if len(sample)!=512 or sample[:4]!=b'GGUF':raise RuntimeError('Bad GGUF sample')
    info=api.model_info(repo,files_metadata=True)
    contents={f.rfilename:f for f in info.siblings}
    if any(n!='.gitattributes' for n in contents):
        if path not in contents or 'build-manifest.json' not in contents:raise RuntimeError('Nonempty unrelated target')
        verify_remote(api,repo,info.sha,plan,sha)
        entry=contents['build-manifest.json']
        if entry.size is None or entry.size>2_000_000:raise RuntimeError('Unbounded existing manifest')
        saved=json.loads(Path(hf_hub_download(repo_id=repo,filename='build-manifest.json',revision=info.sha,token=authentication_token())).read_text())
        if saved!=manifest:raise RuntimeError('Existing release has different proof')
        print(json.dumps({'already_published':repo,'revision':info.sha,'sha256':sha}),flush=True)
        return info.sha
    op=CommitOperationAdd(path_in_repo=path,path_or_fileobj=b'')
    op.upload_info=UploadInfo(size=plan['size'],sample=sample,sha256=bytes.fromhex(sha))
    op._upload_mode='lfs';op._should_ignore=False;op._is_uploaded=True
    docs={'README.md':readme(plan,sha),'LICENSE':(ROOT/'LICENSE').read_text(),
          'NOTICE':(ROOT/'NOTICE').read_text(),'SHA256SUMS':f'{sha}  {path}\n',
          'build-manifest.json':json.dumps(manifest,indent=2)+'\n'}
    ops=[op]+[CommitOperationAdd(path_in_repo=n,path_or_fileobj=v.encode()) for n,v in docs.items()]
    result=api.create_commit(repo_id=repo,operations=ops,parent_commit=info.sha,
        commit_message=f'Publish Flint {plan["variant"]} directly quantized from original BF16')
    verify_remote(api,repo,result.oid,plan,sha)
    print(json.dumps({'published':repo,'revision':result.oid,'bytes':plan['size'],'sha256':sha}),flush=True)
    if os.environ.get('GITHUB_STEP_SUMMARY'):
        Path(os.environ['GITHUB_STEP_SUMMARY']).write_text(f'Published https://huggingface.co/{repo}\n\nMain file: {plan["size"]:,} bytes\n\nSHA256: `{sha}`\n')
    return result.oid

def emit_checkpoint(manifest,sample,work):
    # Only small provenance is retained in the job log. It contains no model
    # payload beyond the first 512 GGUF header bytes and no credentials.
    packet={'manifest':manifest,'sample':base64.b64encode(sample).decode()}
    raw=json.dumps(packet,separators=(',',':')).encode()
    (work/'publication-checkpoint.json').write_bytes(raw)
    enc=base64.b64encode(zlib.compress(raw,9)).decode();checksum=digest(raw)
    parts=[enc[i:i+4000] for i in range(0,len(enc),4000)]
    for i,part in enumerate(parts):
        print(PREFIX+json.dumps({'sha256':checksum,'part':i,'parts':len(parts),'data':part}),flush=True)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--variant',choices=['Q4_K_M','Q5_K_M'],required=True)
    ap.add_argument('--converter-python',required=True);ap.add_argument('--llama',type=Path,required=True)
    ap.add_argument('--build',type=Path,required=True);ap.add_argument('--work',type=Path,required=True);args=ap.parse_args()
    check_context()
    import hf_xet
    from huggingface_hub import HfApi
    from huggingface_hub.utils._xet import XetTokenType,xet_connection_info_refresh_url,xet_headers_without_auth
    args.work.mkdir(parents=True,exist_ok=True)
    cmd=[args.converter_python,str(ROOT/'producer.py'),'--llama',str(args.llama.resolve()),'--build',str(args.build.resolve()),'--work',str(args.work.resolve()),'--variant',args.variant]
    env={k:v for k,v in os.environ.items() if k not in ('HF_TOKEN','HUGGING_FACE_HUB_TOKEN','HUGGINGFACE_TOKEN','GITHUB_TOKEN','GH_TOKEN')}
    env['HF_HUB_DISABLE_IMPLICIT_TOKEN']='1'
    subprocess.run(cmd+['--mode','plan'],check=True,env=env)
    plan=json.loads((args.work/'plan.json').read_text());header=(args.work/'header.gguf').read_bytes()
    if digest(header)!=plan['header_sha256']:raise RuntimeError('Header mismatch')
    api=HfApi(token=authentication_token());repo,path=target(args.variant)
    if api.whoami()['name']!='webmp3':raise RuntimeError('Wrong publisher account')
    if api.repo_exists(repo):
        if any(x!='.gitattributes' for x in api.list_repo_files(repo)):
            raise RuntimeError('Target has content; finalizer must inspect it instead of rebuilding')
    else:api.create_repo(repo_id=repo,repo_type='model',private=False)
    config=hf_xet.XetConfig().with_config({'data.max_concurrent_file_ingestion':1,'client.ac_max_upload_concurrency':4,'client.ac_initial_upload_concurrency':2,'shard.cache_size_limit':'256mb','shard.chunk_index_table_max_size':1048576,'telemetry.enabled':False})
    session=hf_xet.XetSession(config=config);headers_=api._build_hf_headers()
    connection={'token_refresh_url':xet_connection_info_refresh_url(token_type=XetTokenType.WRITE,repo_id=repo,repo_type='model',revision='main',endpoint=api.endpoint),'token_refresh_headers':headers_,'custom_headers':xet_headers_without_auth(headers_)}
    count=0;h=hashlib.sha256();tick=last=time.perf_counter();process=None
    with session.new_upload_commit(**connection) as commit:
        stream=commit.start_upload_stream(name=path,sha256=hf_xet.COMPUTE_SHA256)
        try:
            process=subprocess.Popen(cmd+['--mode','stream'],stdout=subprocess.PIPE,env=env)
            while count<plan['size']:
                raw=process.stdout.read(min(8*1024**2,plan['size']-count))
                if not raw:raise RuntimeError('Producer ended early')
                h.update(raw);stream.write(raw);count+=len(raw);now=time.perf_counter()
                if now-last>=30:
                    print(json.dumps({'variant':args.variant,'stream_bytes':count,'total':plan['size'],'seconds':now-tick}),flush=True);last=now
            if process.stdout.read(1) or process.wait()!=0:raise RuntimeError('Producer failed or exceeded plan')
            receipt=json.loads((args.work/'receipt.json').read_text());sha=h.hexdigest()
            if receipt['sha256']!=sha or receipt['size']!=count:raise RuntimeError('Independent stream hash mismatch')
            meta=stream.finish().xet_info
            if meta.sha256!=sha or meta.file_size!=count:raise RuntimeError('Native Xet hash/size mismatch')
        except BaseException:
            stream.abort()
            if process and process.poll() is None:process.terminate();process.wait(timeout=30)
            raise
    manifest={'source':inventory(),'output':plan,'sha256':sha,'receipt':receipt,'upload_native_sha256_verified':True,
              'builder_commit':os.environ.get('GITHUB_SHA'),'build_run_id':os.environ.get('GITHUB_RUN_ID')}
    validate_manifest(manifest)
    emit_checkpoint(manifest,header[:512],args.work)
    commit_existing(api,manifest,header[:512])

if __name__=='__main__':main()
