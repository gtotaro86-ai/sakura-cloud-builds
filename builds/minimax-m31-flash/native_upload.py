"""Bounded pipe ingestion into finalized native Xet files, with independent SHA256."""
import hashlib,json,os,subprocess,time

def ingest(api,repo,revision,files,cmd,env):
    import hf_xet
    from huggingface_hub.utils._xet import XetTokenType,xet_connection_info_refresh_url,xet_headers_without_auth
    config=hf_xet.XetConfig().with_config({'data.max_concurrent_file_ingestion':1,'client.ac_max_upload_concurrency':4,'client.ac_initial_upload_concurrency':2,'shard.cache_size_limit':'256mb','shard.chunk_index_table_max_size':1048576,'telemetry.enabled':False})
    session=hf_xet.XetSession(config=config);headers=api._build_hf_headers()
    connection={'token_refresh_url':xet_connection_info_refresh_url(token_type=XetTokenType.WRITE,repo_id=repo,repo_type='model',revision=revision,endpoint=api.endpoint),'token_refresh_headers':headers,'custom_headers':xet_headers_without_auth(headers)}
    process=subprocess.Popen(cmd,stdout=subprocess.PIPE,env=env);outputs=[];tick=last=time.perf_counter()
    try:
        for f in files:
            count=0;h=hashlib.sha256();sample=b''
            with session.new_upload_commit(**connection) as commit:
                stream=commit.start_upload_stream(name=f['path'],sha256=hf_xet.COMPUTE_SHA256)
                try:
                    while count<f['size']:
                        raw=process.stdout.read(min(8*1024**2,f['size']-count))
                        if not raw:raise RuntimeError('Producer ended early')
                        if len(sample)<512:sample=(sample+raw)[:512]
                        h.update(raw);count+=len(raw);stream.write(raw);now=time.perf_counter()
                        if now-last>=30:print(json.dumps({'path':f['path'],'bytes':count,'total':f['size'],'seconds':now-tick}),flush=True);last=now
                    native=stream.finish().xet_info
                    if native.file_size!=count or native.sha256!=h.hexdigest():raise RuntimeError('Independent native Xet hash/size differs')
                except BaseException:stream.abort();raise
            outputs.append({'path':f['path'],'size':count,'sha256':h.hexdigest(),'sample':sample})
        if process.stdout.read(1) or process.wait()!=0:raise RuntimeError('Producer failed or exceeded planned bytes')
        return outputs
    finally:
        if process.poll() is None:process.terminate();process.wait(timeout=30)

def public_source_env():
    env={k:v for k,v in os.environ.items() if k not in ('HF_TOKEN','HUGGING_FACE_HUB_TOKEN','HUGGINGFACE_TOKEN','GH_TOKEN','GITHUB_TOKEN')}
    env['HF_HUB_DISABLE_IMPLICIT_TOKEN']='1'
    return env

def pointer(f):
    from huggingface_hub import CommitOperationAdd
    from huggingface_hub.lfs import UploadInfo
    op=CommitOperationAdd(path_in_repo=f['path'],path_or_fileobj=b'')
    op.upload_info=UploadInfo(size=f['size'],sample=f['sample'],sha256=bytes.fromhex(f['sha256']))
    op._upload_mode='lfs';op._should_ignore=False;op._is_uploaded=True
    return op
