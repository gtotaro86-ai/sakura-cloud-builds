"""Prepare verified auxiliary files on an assets branch; no partial main release."""
import argparse,json,os,subprocess
from pathlib import Path
from source import ROOT,inventory,metadata,digest
from credentials import authentication_token,check_context
from storage_preflight import public_target
from native_upload import ingest,pointer,public_source_env
from asset_producer import NAME

REPO='webmp3/Sakura-MiniMax-M3.1-Flash-Q2_K-GGUF'
BRANCH='assets'
def validate_asset_manifest(manifest):
    import re
    plan=json.loads((ROOT/'asset-plan.json').read_text())
    if manifest['source_revision']!=inventory()['source_revision'] or manifest['output']!=plan or manifest['native_sha256_verified'] is not True:raise RuntimeError('Unexpected asset provenance')
    receipt=manifest['receipt'];sha=manifest['sha256']
    if not isinstance(sha,str) or re.fullmatch('[0-9a-f]{64}',sha) is None:raise RuntimeError('Invalid vision checksum')
    if receipt['sha256']!=sha or receipt['size']!=plan['size'] or receipt['path']!=plan['path'] or len(receipt['tensors'])!=len(plan['tensors']):raise RuntimeError('Invalid vision proof')
    for proof,t in zip(receipt['tensors'],plan['tensors']):
        if proof['name']!=t['name'] or proof['bytes']!=t['nbytes'] or re.fullmatch('[0-9a-f]{64}',proof['sha256']) is None:raise RuntimeError('Vision tensor proof mismatch')
    expected_sources={x['path']:x['sha256'] for x in inventory()['weights']}
    if not receipt['source_files_verified'] or any(expected_sources.get(k)!=v for k,v in receipt['source_files_verified'].items()):raise RuntimeError('Invalid vision source proof')
    return manifest

def verify_assets(api,revision):
    from huggingface_hub import hf_hub_download
    info=api.model_info(REPO,revision=revision,files_metadata=True);entry=next(x for x in info.siblings if x.rfilename=='asset-manifest.json')
    if entry.size is None or entry.size>2_000_000:raise RuntimeError('Asset manifest size budget')
    manifest=json.loads(Path(hf_hub_download(repo_id=REPO,revision=revision,filename='asset-manifest.json',token=authentication_token())).read_text())
    validate_asset_manifest(manifest)
    plan=json.loads((ROOT/'asset-plan.json').read_text());ds=inventory()['original_dspark'];sha=manifest['sha256']
    for f in info.siblings:
        if f.rfilename in (NAME,ds['path']):
            expected=(plan['size'],sha) if f.rfilename==NAME else (ds['size'],ds['sha256'])
            if f.size!=expected[0] or not f.lfs or f.lfs.sha256!=expected[1]:raise RuntimeError('Asset file identity mismatch')
    if not {NAME,ds['path']}.issubset({x.rfilename for x in info.siblings}):raise RuntimeError('Missing auxiliary output')
    return manifest

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--converter-python',required=True);ap.add_argument('--llama',type=Path,required=True);ap.add_argument('--work',type=Path,required=True);args=ap.parse_args()
    check_context()
    from huggingface_hub import HfApi,CommitOperationAdd,CommitOperationCopy
    from huggingface_hub.utils import RevisionNotFoundError
    api=HfApi(token=authentication_token())
    if api.whoami()['name']!='webmp3':raise RuntimeError('Wrong HF account')
    if not api.repo_exists(REPO):api.create_repo(repo_id=REPO,repo_type='model',private=False)
    public_target(api,REPO)
    try:info=api.model_info(REPO,revision=BRANCH,files_metadata=True)
    except RevisionNotFoundError:api.create_branch(repo_id=REPO,branch=BRANCH);info=api.model_info(REPO,revision=BRANCH,files_metadata=True)
    if any(x.rfilename!='.gitattributes' for x in info.siblings):
        verify_assets(api,info.sha);revision=info.sha
    else:
        args.work.mkdir(parents=True,exist_ok=True);env=public_source_env()
        cmd=[args.converter_python,str(ROOT/'asset_producer.py'),'--llama',str(args.llama.resolve()),'--work',str(args.work.resolve())]
        subprocess.run(cmd+['--mode','plan'],env=env,check=True)
        plan=json.loads((args.work/'asset-plan.json').read_text());out=ingest(api,REPO,BRANCH,[plan],cmd+['--mode','stream'],env)[0]
        receipt=json.loads((args.work/'asset-receipt.json').read_text())
        if receipt['sha256']!=out['sha256'] or receipt['size']!=out['size']:raise RuntimeError('Independent vision checksum mismatch')
        manifest={'source_revision':inventory()['source_revision'],'output':plan,'receipt':receipt,'sha256':out['sha256'],'native_sha256_verified':True}
        validate_asset_manifest(manifest)
        folder=metadata(args.work);ds=inventory()['original_dspark']
        ops=[pointer(out),CommitOperationCopy(src_repo_type='model',src_repo_id=inventory()['repo'],src_revision=inventory()['source_revision'],src_path_in_repo=ds['path'],path_in_repo=ds['path'])]
        docs={'asset-manifest.json':json.dumps(manifest,indent=2)+'\n'}
        for n in ('dspark/config.json','dspark/model.safetensors.index.json'):docs[n]=(folder/n).read_text()
        ops += [CommitOperationAdd(path_in_repo=n,path_or_fileobj=v.encode()) for n,v in docs.items()]
        commit=api.create_commit(repo_id=REPO,revision=BRANCH,parent_commit=info.sha,operations=ops,commit_message='Stage F16 vision projector and unchanged original DSpark checkpoint')
        revision=commit.oid;verify_assets(api,revision)
    print(json.dumps({'assets_verified':True,'revision':revision}),flush=True)
    if os.environ.get('GITHUB_OUTPUT'):
        with Path(os.environ['GITHUB_OUTPUT']).open('a') as f:f.write('asset_revision='+revision+'\n')

if __name__=='__main__':main()
