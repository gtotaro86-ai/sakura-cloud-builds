"""Update documentation only after both immutable model outputs are verified."""
import hashlib,json,os
from pathlib import Path
from source import ROOT,inventory
from publish import authentication_token,check_context
from release_card import readme

def verify_release(variant,manifest,info):
    frozen=json.loads((ROOT/'expected-plans.json').read_text())[variant]
    expected={k:v for k,v in frozen.items() if k!='native_full_model_type_plan_verified'}
    plan=manifest['output'];summary={k:v for k,v in plan.items() if k!='tensors'}
    if summary!=expected or manifest['source']!=inventory():raise RuntimeError('Pinned model provenance differs')
    if not manifest['upload_native_sha256_verified']:raise RuntimeError('Native Xet proof missing')
    receipt=manifest['receipt'];sha=manifest['sha256']
    if len(sha)!=64 or len(bytes.fromhex(sha))!=32:raise RuntimeError('Invalid recorded output checksum')
    if receipt['sha256']!=sha or receipt['size']!=plan['size']:raise RuntimeError('Producer proof mismatch')
    sources={x['path']:x['sha256'] for x in inventory()['weights']}
    if receipt['source_files_verified']!=sources:raise RuntimeError('Complete original source checksums missing')
    projector=inventory()['projector'];path=f'Sakura-Sage-A1.5-{variant}.gguf'
    expected_files={path:(plan['size'],sha),projector['path']:(projector['size'],projector['sha256'])}
    files={x.rfilename:x for x in info.siblings}
    for name,(size,digest) in expected_files.items():
        f=files[name]
        if f.size!=size or not f.lfs or f.lfs.sha256!=digest:raise RuntimeError('Remote weights differ from verified release')
    return plan,sha,expected_files

def main():
    check_context()
    from huggingface_hub import HfApi,CommitOperationAdd,hf_hub_download
    token=authentication_token();api=HfApi(token=token)
    if api.whoami()['name']!='webmp3':raise RuntimeError('Incorrect HF publisher account')
    local=(ROOT/'local-reference.json').read_bytes()
    json.loads(local)
    active=[]
    # Verify both releases before writing any model card. No GGUF payload is downloaded.
    for variant in ('Q4_K_M','Q5_K_M'):
        repo=f'webmp3/Sakura-Sage-A1.5-{variant}-GGUF'
        info=api.model_info(repo,files_metadata=True)
        manifest_file=next(x for x in info.siblings if x.rfilename=='build-manifest.json')
        if manifest_file.size is None or not 0<manifest_file.size<=2_000_000:raise RuntimeError('Unbounded/missing build manifest')
        path=hf_hub_download(repo_id=repo,filename='build-manifest.json',revision=info.sha,token=token)
        manifest=json.loads(Path(path).read_text())
        plan,sha,expected=verify_release(variant,manifest,info)
        card=readme(plan,sha).encode()
        active.append((repo,info,card,expected))
    for repo,info,card,expected in active:
        current=Path(hf_hub_download(repo_id=repo,filename='README.md',revision=info.sha,token=token)).read_bytes()
        if current==card and any(x.rfilename=='local-reference.json' for x in info.siblings):
            previous=Path(hf_hub_download(repo_id=repo,filename='local-reference.json',revision=info.sha,token=token)).read_bytes()
            if previous==local:
                print(json.dumps({'model_card_current':repo}),flush=True);continue
        result=api.create_commit(repo_id=repo,parent_commit=info.sha,
            operations=[CommitOperationAdd(path_in_repo='README.md',path_or_fileobj=card),
                        CommitOperationAdd(path_in_repo='local-reference.json',path_or_fileobj=local)],
            commit_message='Document owner-reported local Q8-to-Q5 imatrix performance reference')
        after=api.model_info(repo,revision=result.oid,files_metadata=True)
        for name,(size,sha) in expected.items():
            f=next(x for x in after.siblings if x.rfilename==name)
            if f.size!=size or not f.lfs or f.lfs.sha256!=sha:raise RuntimeError('Weight identity changed during documentation-only commit')
        saved=Path(hf_hub_download(repo_id=repo,filename='README.md',revision=result.oid,token=token)).read_bytes()
        if saved!=card:raise RuntimeError('Published model card differs')
        print(json.dumps({'model_card_updated':repo,'revision':result.oid,'readme_sha256':hashlib.sha256(card).hexdigest(),'weight_hashes_unchanged':True}),flush=True)
    if os.environ.get('GITHUB_STEP_SUMMARY'):
        Path(os.environ['GITHUB_STEP_SUMMARY']).write_text('Updated both Sage model cards with the owner-reported local Q8-to-Q5 imatrix reference, English and Chinese. All main and projector GGUF hashes remain unchanged. No model weights were downloaded or requantized.\n')

if __name__=='__main__':main()
