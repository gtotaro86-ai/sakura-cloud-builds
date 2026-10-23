"""Bounded split-GGUF native Xet ingestion and atomic verified HF release."""
import argparse,base64,json,os,re,subprocess,zlib
from pathlib import Path
from source import ROOT,inventory,digest,metadata
from credentials import authentication_token,check_context
from storage_preflight import public_target
from native_upload import ingest,pointer,public_source_env
from assets import verify_assets,validate_asset_manifest,REPO as ASSET_REPO
from release_card import readme
from split_producer import VARIANTS
PREFIX='MINIMAX_M31_CHECKPOINT_V1 '

def target(variant):
    if variant not in VARIANTS:raise RuntimeError('Unauthorized target policy')
    return f'webmp3/Sakura-MiniMax-M3.1-Flash-{variant}-GGUF'

def checksum(sha):
    if not isinstance(sha,str) or re.fullmatch('[0-9a-f]{64}',sha) is None:raise RuntimeError('Invalid SHA256')

def validate_manifest(m):
    p=m['output'];v=p['variant'];target(v)
    if p!=json.loads((ROOT/'split-plans.json').read_text())[v] or m['source']!=inventory():raise RuntimeError('Frozen source/output differs')
    if m.get('upload_native_sha256_verified') is not True:raise RuntimeError('Native upload proof missing')
    validate_asset_manifest(m['assets'])
    if re.fullmatch('[0-9a-f]{40}',m['asset_revision']) is None:raise RuntimeError('Asset revision must be pinned')
    r=m['receipt']
    if r['source_files_verified']!={x['path']:x['sha256'] for x in inventory()['weights']}:raise RuntimeError('Incomplete original source verification')
    if len(r['files'])!=len(p['files']) or len(m['native_files'])!=len(p['files']):raise RuntimeError('Incomplete output set')
    for f,proof,native in zip(p['files'],r['files'],m['native_files']):
        checksum(proof['sha256'])
        identity={k:proof[k] for k in ('path','size','sha256')}
        if native!=identity or proof['path']!=f['path'] or proof['size']!=f['size']:raise RuntimeError('Independent stream proof differs')
        if len(proof['tensors'])!=len(f['tensors']):raise RuntimeError('Incomplete tensor proof')
        for a,t in zip(proof['tensors'],f['tensors']):
            checksum(a['sha256'])
            if a['name']!=t['name'] or a['bytes']!=t['nbytes']:raise RuntimeError('Tensor proof differs')
    return p

def verify_remote(api,repo,revision,m):
    from asset_producer import NAME
    info=api.model_info(repo,revision=revision,files_metadata=True);files={f.rfilename:f for f in info.siblings}
    wanted=m['native_files']+[{'path':NAME,'size':m['assets']['output']['size'],'sha256':m['assets']['sha256']},inventory()['original_dspark']]
    for w in wanted:
        f=files.get(w['path'])
        if not f or f.size!=w['size'] or not f.lfs or f.lfs.sha256!=w['sha256']:raise RuntimeError('Hub file size/hash differs: '+w['path'])
    for n in ('README.md','LICENSE','NOTICE','SHA256SUMS','build-manifest.json','runtime.patch','build-runtime.sh','dspark/config.json','dspark/model.safetensors.index.json'):
        if n not in files:raise RuntimeError('Missing release documentation: '+n)
    return info

def commit_existing(api,m,samples):
    from huggingface_hub import CommitOperationAdd,CommitOperationCopy,hf_hub_download
    p=validate_manifest(m);repo=target(p['variant']);public_target(api,repo);info=api.model_info(repo,files_metadata=True)
    if len(samples)!=len(p['files']) or any(len(s)!=512 or s[:4]!=b'GGUF' for s in samples):raise RuntimeError('Invalid split GGUF header samples')
    if any(f.rfilename!='.gitattributes' for f in info.siblings):
        verify_remote(api,repo,info.sha,m)
        entry=next(f for f in info.siblings if f.rfilename=='build-manifest.json')
        if entry.size is None or entry.size>5_000_000:raise RuntimeError('Existing manifest exceeds budget')
        old=json.loads(Path(hf_hub_download(repo_id=repo,filename='build-manifest.json',revision=info.sha,token=authentication_token())).read_text())
        if old!=m:raise RuntimeError('Existing release provenance differs')
        print(json.dumps({'already_published':repo,'revision':info.sha}),flush=True);return info.sha
    asset=verify_assets(api,m['asset_revision'])
    if asset!=m['assets']:raise RuntimeError('Auxiliary provenance differs')
    ops=[pointer(dict(f,sample=s)) for f,s in zip(m['native_files'],samples)]
    ds=inventory()['original_dspark'];projector=asset['output'];folder=metadata(Path(os.environ.get('RUNNER_TEMP','/tmp'))/'minimax-docs')
    ops += [CommitOperationCopy(src_repo_type='model',src_repo_id=ASSET_REPO,src_revision=m['asset_revision'],src_path_in_repo=projector['path'],path_in_repo=projector['path']),
            CommitOperationCopy(src_repo_type='model',src_repo_id=inventory()['repo'],src_revision=inventory()['source_revision'],src_path_in_repo=ds['path'],path_in_repo=ds['path'])]
    weightfiles=m['native_files']+[{'path':projector['path'],'sha256':asset['sha256']},ds]
    docs={'README.md':readme(m),'LICENSE':(ROOT/'LICENSE').read_text(),'NOTICE':(ROOT/'NOTICE').read_text(),
          'runtime.patch':(ROOT/'runtime.patch').read_text(),'build-runtime.sh':(ROOT/'build-runtime.sh').read_text(),
          'SHA256SUMS':''.join(f"{f['sha256']}  {f['path']}\n" for f in weightfiles),
          'build-manifest.json':json.dumps(m,indent=2)+'\n'}
    for n in ('dspark/config.json','dspark/model.safetensors.index.json'):docs[n]=(folder/n).read_text()
    ops += [CommitOperationAdd(path_in_repo=n,path_or_fileobj=v.encode()) for n,v in docs.items()]
    result=api.create_commit(repo_id=repo,parent_commit=info.sha,operations=ops,commit_message=f'Publish MiniMax M3.1 Flash {p["variant"]} with all-layer sparse runtime and verified splits')
    verify_remote(api,repo,result.oid,m)
    print(json.dumps({'published':repo,'revision':result.oid,'main_bytes':p['size'],'files':m['native_files']}),flush=True)
    if os.environ.get('GITHUB_STEP_SUMMARY'):
        Path(os.environ['GITHUB_STEP_SUMMARY']).write_text(f'Published https://huggingface.co/{repo}\n\nMain GGUF: {p["size"]:,} bytes in {len(p["files"])} splits. Original hybrid source, required runtime patch and auxiliary files documented.\n')
    return result.oid

def emit_checkpoint(m,samples,work):
    packet={'manifest':m,'samples':[base64.b64encode(s).decode() for s in samples]}
    raw=json.dumps(packet,separators=(',',':')).encode()
    if len(raw)>4_000_000:raise RuntimeError('Publication checkpoint budget exceeded')
    (work/'publication-checkpoint.json').write_bytes(raw)
    enc=base64.b64encode(zlib.compress(raw,9)).decode();parts=[enc[i:i+4000] for i in range(0,len(enc),4000)]
    for i,part in enumerate(parts):print(PREFIX+json.dumps({'sha256':digest(raw),'part':i,'parts':len(parts),'data':part}),flush=True)

