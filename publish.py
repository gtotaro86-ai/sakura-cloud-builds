"""Authenticated GitHub-runner-only CPU producer -> native Xet -> verified Hub commit."""
import argparse, contextlib, hashlib, json, os, subprocess, sys, time
from pathlib import Path
from importlib.metadata import version
from source import ROOT, inventory, digest
from release_card import readme

def check_context():
    required={'GITHUB_ACTIONS':'true','RUNNER_ENVIRONMENT':'github-hosted','RUNNER_OS':'Linux','REPO_VISIBILITY':'public','SAGE_STANDARD_RUNNER':'ubuntu-24.04'}
    if any(os.environ.get(k)!=v for k,v in required.items()):
        raise RuntimeError('Actual public standard GitHub-hosted Linux runner required')
    if not os.environ.get('HF_TOKEN'):raise RuntimeError('HF_TOKEN Actions secret missing')
    if version('huggingface-hub')!='2.2.0' or version('hf-xet')!='1.7.0':raise RuntimeError('Pinned upload dependencies required')

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--variant',choices=['Q4_K_M','Q5_K_M'],required=True)
    ap.add_argument('--converter-python',required=True);ap.add_argument('--llama',type=Path,required=True)
    ap.add_argument('--build',type=Path,required=True);ap.add_argument('--work',type=Path,required=True);args=ap.parse_args()
    check_context()
    import hf_xet
    from huggingface_hub import HfApi, CommitOperationAdd, CommitOperationCopy
    from huggingface_hub.lfs import UploadInfo
    from huggingface_hub.utils._xet import XetTokenType,xet_connection_info_refresh_url,xet_headers_without_auth
    args.work.mkdir(parents=True,exist_ok=True)
    cmd=[args.converter_python,str(ROOT/'producer.py'),'--llama',str(args.llama.resolve()),'--build',str(args.build.resolve()),
         '--work',str(args.work.resolve()),'--variant',args.variant]
    # The converter has no Hub credentials. All source reads are public and revision pinned.
    env={k:v for k,v in os.environ.items() if k not in ('HF_TOKEN','HUGGING_FACE_HUB_TOKEN','HUGGINGFACE_TOKEN','GITHUB_TOKEN','GH_TOKEN')}
    env['HF_HUB_DISABLE_IMPLICIT_TOKEN']='1'
    subprocess.run(cmd+['--mode','plan'],check=True,env=env)
    plan=json.loads((args.work/'plan.json').read_text());header=(args.work/'header.gguf').read_bytes()
    if digest(header)!=plan['header_sha256'] or plan['variant']!=args.variant:raise RuntimeError('Invalid output plan')
    api=HfApi(token=os.environ['HF_TOKEN']);repo=f'webmp3/Sakura-Sage-A1.5-{args.variant}-GGUF';path=f'Sakura-Sage-A1.5-{args.variant}.gguf'
    owner=api.whoami()['name']
    if owner!='webmp3':raise RuntimeError('HF account is not the authorized publisher')
    if api.repo_exists(repo):
        files=api.list_repo_files(repo)
        if any(x not in ('.gitattributes',) for x in files):raise RuntimeError('Target already has content; inspect instead of overwriting')
    else:api.create_repo(repo_id=repo,private=False,repo_type='model')
    info=api.model_info(repo);parent=info.sha
    h=hashlib.sha256();count=0;sample=b'';tick=last=time.perf_counter()
    config=hf_xet.XetConfig().with_config({'data.max_concurrent_file_ingestion':1,'client.ac_max_upload_concurrency':4,
             'client.ac_initial_upload_concurrency':2,'shard.cache_size_limit':'256mb','shard.chunk_index_table_max_size':1048576,'telemetry.enabled':False})
    session=hf_xet.XetSession(config=config);headers=api._build_hf_headers()
    connection={'token_refresh_url':xet_connection_info_refresh_url(token_type=XetTokenType.WRITE,repo_id=repo,repo_type='model',revision='main',endpoint=api.endpoint),
                'token_refresh_headers':headers,'custom_headers':xet_headers_without_auth(headers)}
    process=None
    with session.new_upload_commit(**connection) as commit:
        stream=commit.start_upload_stream(name=path,sha256=hf_xet.COMPUTE_SHA256)
        try:
            process=subprocess.Popen(cmd+['--mode','stream'],stdout=subprocess.PIPE,env=env)
            while count<plan['size']:
                raw=process.stdout.read(min(8*1024**2,plan['size']-count))
                if not raw:raise RuntimeError('Producer ended before the planned output length')
                if len(sample)<512:sample=(sample+raw)[:512]
                h.update(raw);stream.write(raw);count+=len(raw)
                now=time.perf_counter()
                if now-last>=30:
                    print(json.dumps({'variant':args.variant,'stream_bytes':count,'total':plan['size'],'seconds':now-tick}),flush=True);last=now
            if process.stdout.read(1):raise RuntimeError('Producer exceeded planned output')
            if process.wait()!=0:raise RuntimeError('Producer failed; no model commit')
            receipt=json.loads((args.work/'receipt.json').read_text());sha=h.hexdigest()
            expected={x['path']:x['sha256'] for x in inventory()['weights']}
            if receipt['sha256']!=sha or receipt['size']!=count or receipt['source_files_verified']!=expected:
                raise RuntimeError('Independent producer/source verification failed')
            meta=stream.finish().xet_info
            if meta.sha256!=sha or meta.file_size!=count:raise RuntimeError('Native Xet size/SHA256 mismatch')
        except BaseException:
            stream.abort()
            if process and process.poll() is None:process.terminate();process.wait(timeout=30)
            raise
    op=CommitOperationAdd(path_in_repo=path,path_or_fileobj=b'')
    op.upload_info=UploadInfo(size=count,sample=sample,sha256=bytes.fromhex(sha))
    op._upload_mode='lfs';op._should_ignore=False;op._is_uploaded=True
    projector=inventory()['projector']
    copied=CommitOperationCopy(src_repo_id=projector['repo'],src_revision=projector['revision'],src_path_in_repo=projector['path'],path_in_repo=projector['path'])
    manifest={'source':inventory(),'output':plan,'sha256':sha,'receipt':receipt,'upload_native_sha256_verified':True}
    sums=f"{sha}  {path}\n{projector['sha256']}  {projector['path']}\n"
    docs={'README.md':readme(plan,sha),'SHA256SUMS':sums,'build-manifest.json':json.dumps(manifest,indent=2)+'\n',
          'LICENSE':(ROOT/'LICENSE').read_text()}
    ops=[op,copied]+[CommitOperationAdd(path_in_repo=n,path_or_fileobj=v.encode()) for n,v in docs.items()]
    result=api.create_commit(repo_id=repo,operations=ops,commit_message=f'Publish Sage-A1.5 {args.variant} from original BF16 weights',parent_commit=parent)
    remote=api.model_info(repo,revision=result.oid,files_metadata=True)
    expected_outputs={path:(count,sha),projector['path']:(projector['size'],projector['sha256'])}
    for name,(size,expectedsha) in expected_outputs.items():
        f=next(x for x in remote.siblings if x.rfilename==name)
        if f.size!=size or not f.lfs or f.lfs.sha256!=expectedsha:raise RuntimeError('Published file size/SHA256 mismatch')
    print(json.dumps({'published':repo,'revision':result.oid,'bytes':count,'sha256':sha,'seconds':time.perf_counter()-tick}),flush=True)
    if os.environ.get('GITHUB_STEP_SUMMARY'):
        Path(os.environ['GITHUB_STEP_SUMMARY']).write_text(f'Published https://huggingface.co/{repo}\n\nMain file: {count:,} bytes\n\nSHA256: `{sha}`\n')

if __name__=='__main__':main()
