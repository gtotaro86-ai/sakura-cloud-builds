"""Metadata-only publication of the complete Q2 variant before other variants finish."""
import argparse,json,os,subprocess,sys
from pathlib import Path

REPO='gtotaro86-ai/sakura-cloud-builds'
COMMIT='35fe24364aede18b80e7789cff74ae5a4f2bb231'
RUN=39908350992
VARIANT='Q2_K'
EXPECTED={'build_run_id':RUN,'build_commit':COMMIT,'variant':VARIANT,'publish':True,'paid_resources':False}

def check_run(run,jobs,checkout):
    if run['head_repository']['full_name']!=REPO or run['path']!='.github/workflows/minimax-m31.yml' or run['head_sha']!=COMMIT or checkout!=COMMIT:
        raise RuntimeError('Original build identity mismatch')
    names={j['name']:j for j in jobs}
    if len(names)!=len(jobs):raise RuntimeError('Ambiguous build job set')
    for n in ('qualify','assets',*[f'quantize-{VARIANT}-{i}' for i in range(4)]):
        if n not in names or names[n]['conclusion']!='success':raise RuntimeError('Complete successful Q2 job set is required: '+n)

def check_packet(p,index,frozen,remote):
    if (p['variant'],p['index'],p['builder_commit'],p['build_run_id'])!=(VARIANT,index,COMMIT,str(RUN)):
        raise RuntimeError('Staging provenance differs')
    n=p['native_file']
    if n['path']!=frozen['path'] or n['size']!=frozen['size'] or remote is None or remote.size!=n['size'] or not remote.lfs or remote.lfs.sha256!=n['sha256']:
        raise RuntimeError('Hub main shard identity differs')

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--original-build',type=Path,required=True);a=ap.parse_args()
    if json.loads((Path(__file__).parent/'request.json').read_text())!=EXPECTED:raise RuntimeError('Unapproved release request')
    original=a.original_build.resolve();checkout=subprocess.check_output(['git','-C',str(original),'rev-parse','HEAD'],text=True).strip()
    sys.path.insert(0,str(original/'builds/minimax-m31-flash'))
    from credentials import check_context,authentication_token
    from finalize import github,assemble
    from stage import validate
    from publish import commit_existing,verify_remote,target
    from source import ROOT,digest
    from huggingface_hub import HfApi,hf_hub_download,CommitOperationAdd
    check_context();api=HfApi(token=authentication_token())
    if api.whoami()['name']!='webmp3':raise RuntimeError('Wrong HF publisher')
    run=github('/actions/runs/'+str(RUN));jobs=github('/actions/runs/'+str(RUN)+'/jobs?per_page=100')['jobs'];check_run(run,jobs,checkout)
    repo=target(VARIANT);info=api.model_info(repo,revision='build-'+COMMIT,files_metadata=True)
    if info.private is not False:raise RuntimeError('Only a public HF target is authorized')
    files={f.rfilename:f for f in info.siblings};plan=json.loads((ROOT/'split-plans.json').read_text())[VARIANT]
    if len(plan['files'])!=4:raise RuntimeError('Unexpected Q2 split count')
    packets=[]
    for i,frozen in enumerate(plan['files']):
        name=f'staging/shard-{i:05d}.json';entry=files.get(name)
        if entry is None or entry.size is None or not 0<entry.size<=2_000_000:raise RuntimeError('Missing/unbounded staging proof')
        path=hf_hub_download(repo_id=repo,revision=info.sha,filename=name,token=authentication_token());raw=Path(path).read_bytes()
        if len(raw)!=entry.size:raise RuntimeError('Staging proof byte count differs')
        p=json.loads(raw);validate(p);check_packet(p,i,frozen,files.get(p['native_file']['path']));packets.append(p)
    manifest,samples=assemble(api,VARIANT,packets)
    print(json.dumps({'all_four_shards_verified':True,'main_bytes':plan['size'],'source_shards_verified':len(manifest['receipt']['source_files_verified']),'files':manifest['native_files']}),flush=True)
    revision=commit_existing(api,manifest,samples);verify_remote(api,repo,revision,manifest)
    # Include the exact pinned upstream runtime license alongside its derivative patch.
    license_path=Path(__file__).parent/'RUNTIME-LICENSE';license_bytes=license_path.read_bytes()
    latest=api.model_info(repo,files_metadata=True);existing=next((f for f in latest.siblings if f.rfilename=='RUNTIME-LICENSE'),None)
    if existing:
        old=Path(hf_hub_download(repo_id=repo,revision=latest.sha,filename='RUNTIME-LICENSE',token=authentication_token())).read_bytes()
        if old!=license_bytes:raise RuntimeError('Existing runtime license differs')
        revision=latest.sha
    else:
        result=api.create_commit(repo_id=repo,parent_commit=latest.sha,operations=[CommitOperationAdd(path_in_repo='RUNTIME-LICENSE',path_or_fileobj=license_bytes)],commit_message='Include the exact pinned upstream llama.cpp MIT license')
        revision=result.oid
    verify_remote(api,repo,revision,manifest)
    print(json.dumps({'published':repo,'revision':revision,'main_bytes':plan['size'],'runtime_license_sha256':digest(license_bytes),'files':manifest['native_files']}),flush=True)
    if os.environ.get('GITHUB_STEP_SUMMARY'):Path(os.environ['GITHUB_STEP_SUMMARY']).write_text('Published https://huggingface.co/'+repo+'\n\nAll four Q2 GGUF shards and auxiliary files are verified. English/Chinese README, comparison, original license, pinned runtime patch/script, SHA256SUMS and build provenance are included. No new model inference/quality result is claimed. Q3–Q5 continue separately.\n')
if __name__=='__main__':main()
