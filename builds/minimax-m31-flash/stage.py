"""One independently recoverable GGUF shard per free CPU job; no partial main release."""
import argparse,base64,json,os,re,subprocess,zlib
from pathlib import Path
from source import ROOT,inventory,digest
from credentials import authentication_token,check_context
from storage_preflight import public_target
from native_upload import ingest,pointer,public_source_env
from assets import verify_assets
from publish import target,checksum
PREFIX='MINIMAX_M31_SHARD_V1 '

def file_plan(variant,index):
    plan=json.loads((ROOT/'split-plans.json').read_text())[variant]
    if not 0<=index<len(plan['files']):raise RuntimeError('Invalid shard index')
    return plan,plan['files'][index]

def validate(packet):
    p,f=file_plan(packet['variant'],packet['index']);r=packet['receipt'];n=packet['native_file']
    checksum(n['sha256'])
    header=base64.b64decode(packet['header_sample'],validate=True)
    if len(header)!=512 or header[:4]!=b'GGUF':raise RuntimeError('Shard GGUF header sample missing')
    if n['path']!=f['path'] or n['size']!=f['size'] or packet['source_revision']!=inventory()['source_revision'] or packet.get('native_sha256_verified') is not True:raise RuntimeError('Shard identity differs')
    if re.fullmatch('[0-9a-f]{40}',packet['builder_commit']) is None or re.fullmatch('[0-9a-f]{40}',packet['asset_revision']) is None or not str(packet['build_run_id']).isdigit():raise RuntimeError('Shard build identity missing')
    if len(r['files'])!=1:raise RuntimeError('One output shard required')
    proof=r['files'][0]
    if {k:proof[k] for k in ('path','size','sha256')}!=n or len(proof['tensors'])!=len(f['tensors']):raise RuntimeError('Producer shard differs')
    for a,t in zip(proof['tensors'],f['tensors']):
        checksum(a['sha256'])
        if a['name']!=t['name'] or a['bytes']!=t['nbytes']:raise RuntimeError('Shard tensor proof mismatch')
    expected={x['path']:x['sha256'] for x in inventory()['weights']}
    if not r['source_files_verified'] or any(expected.get(k)!=v for k,v in r['source_files_verified'].items()):raise RuntimeError('Invalid shard source verification')
    return p,f

def branch(packet):return 'build-'+packet['builder_commit']
def manifest_path(index):return f'staging/shard-{index:05d}.json'

def verified_staged(api,variant,index,commit,run_id,asset_revision):
    from huggingface_hub import hf_hub_download
    from huggingface_hub.utils import RevisionNotFoundError
    repo=target(variant)
    try:info=api.model_info(repo,revision='build-'+commit,files_metadata=True)
    except RevisionNotFoundError:return None
    files={f.rfilename:f for f in info.siblings};name=manifest_path(index)
    if name not in files:return None
    if files[name].size is None or files[name].size>2_000_000:raise RuntimeError('Staging proof size budget')
    packet=json.loads(Path(hf_hub_download(repo_id=repo,revision=info.sha,filename=name,token=authentication_token())).read_text());validate(packet)
    if (packet['variant'],packet['index'],packet['builder_commit'],packet['build_run_id'],packet['asset_revision'])!=(variant,index,commit,str(run_id),asset_revision):raise RuntimeError('Staged build provenance differs')
    n=packet['native_file'];f=files.get(n['path'])
    if not f or f.size!=n['size'] or not f.lfs or f.lfs.sha256!=n['sha256']:raise RuntimeError('Staged Hub shard identity differs')
    return packet

def commit_staged(api,packet,sample):
    from huggingface_hub import CommitOperationAdd
    from huggingface_hub.utils import RevisionNotFoundError,HfHubHTTPError
    validate(packet);repo=target(packet['variant']);ref=branch(packet)
    if len(sample)!=512 or sample[:4]!=b'GGUF':raise RuntimeError('Invalid GGUF sample')
    try:api.model_info(repo,revision=ref)
    except RevisionNotFoundError:api.create_branch(repo_id=repo,branch=ref,exist_ok=True)
    # Different shard jobs share one staging branch. Optimistic lease retries never rebuild weights.
    for attempt in range(8):
        existing=verified_staged(api,packet['variant'],packet['index'],packet['builder_commit'],packet['build_run_id'],packet['asset_revision'])
        if existing:
            if existing!=packet:raise RuntimeError('Conflicting staged proof')
            return
        head=api.model_info(repo,revision=ref).sha
        try:
            api.create_commit(repo_id=repo,revision=ref,parent_commit=head,
                operations=[pointer(dict(packet['native_file'],sample=sample)),CommitOperationAdd(path_in_repo=manifest_path(packet['index']),path_or_fileobj=(json.dumps(packet,indent=2)+'\n').encode())],
                commit_message=f'Stage verified {packet["variant"]} shard {packet["index"]+1}')
            if verified_staged(api,packet['variant'],packet['index'],packet['builder_commit'],packet['build_run_id'],packet['asset_revision'])!=packet:raise RuntimeError('Staging verification differs')
            return
        except HfHubHTTPError as e:
            if e.response is None or e.response.status_code!=409 or attempt==7:raise
    raise RuntimeError('Staging lease retry exhausted')

def emit(packet,sample,work):
    envelope={'packet':packet,'sample':base64.b64encode(sample).decode()};raw=json.dumps(envelope,separators=(',',':')).encode()
    if len(raw)>2_000_000:raise RuntimeError('Shard checkpoint budget')
    (work/'shard-checkpoint.json').write_bytes(raw);encoded=base64.b64encode(zlib.compress(raw,9)).decode();parts=[encoded[i:i+4000] for i in range(0,len(encoded),4000)]
    for i,data in enumerate(parts):print(PREFIX+json.dumps({'sha256':digest(raw),'part':i,'parts':len(parts),'data':data}),flush=True)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--variant',required=True);ap.add_argument('--index',type=int,required=True);ap.add_argument('--asset-revision',required=True);ap.add_argument('--converter-python',required=True);ap.add_argument('--llama',type=Path,required=True);ap.add_argument('--build',type=Path,required=True);ap.add_argument('--work',type=Path,required=True);a=ap.parse_args()
    check_context()
    from huggingface_hub import HfApi
    api=HfApi(token=authentication_token())
    if api.whoami()['name']!='webmp3':raise RuntimeError('Wrong HF publisher')
    verify_assets(api,a.asset_revision);repo=target(a.variant)
    if not api.repo_exists(repo):api.create_repo(repo_id=repo,repo_type='model',private=False,exist_ok=True)
    public_target(api,repo)
    existing=verified_staged(api,a.variant,a.index,os.environ['GITHUB_SHA'],os.environ['GITHUB_RUN_ID'],a.asset_revision)
    if existing:print(json.dumps({'already_staged':existing['native_file']}),flush=True);return
    if any(n!='.gitattributes' for n in api.list_repo_files(repo)):raise RuntimeError('Main already has content; do not rebuild')
    p,f=file_plan(a.variant,a.index);a.work.mkdir(parents=True,exist_ok=True);env=public_source_env()
    cmd=[a.converter_python,str(ROOT/'split_producer.py'),'--llama',str(a.llama.resolve()),'--build',str(a.build.resolve()),'--work',str(a.work.resolve()),'--variant',a.variant,'--file',f['path']]
    subprocess.run(cmd+['--mode','plan'],env=env,check=True)
    actual=json.loads((a.work/'split-plan.json').read_text())
    if actual!=p:raise RuntimeError('Frozen split plan differs')
    from huggingface_hub.utils import RevisionNotFoundError
    ref='build-'+os.environ['GITHUB_SHA']
    try:api.model_info(repo,revision=ref)
    except RevisionNotFoundError:api.create_branch(repo_id=repo,branch=ref,exist_ok=True)
    n=ingest(api,repo,ref,[f],cmd+['--mode','stream'],env)[0]
    packet={'variant':a.variant,'index':a.index,'source_revision':inventory()['source_revision'],'asset_revision':a.asset_revision,
        'builder_commit':os.environ['GITHUB_SHA'],'build_run_id':os.environ['GITHUB_RUN_ID'],
        'header_sample':base64.b64encode(n['sample']).decode(),'native_file':{k:n[k] for k in ('path','size','sha256')},'receipt':json.loads((a.work/'receipt.json').read_text()),'native_sha256_verified':True}
    validate(packet);emit(packet,n['sample'],a.work);commit_staged(api,packet,n['sample'])
    print(json.dumps({'verified_staged':packet['native_file']}),flush=True)
if __name__=='__main__':main()
