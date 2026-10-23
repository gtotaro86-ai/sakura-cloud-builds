"""Autonomous metadata-only publication recovery and bounded transient retry."""
import argparse,base64,json,os,subprocess,urllib.request,urllib.error,zlib
from pathlib import Path
from credentials import authentication_token,check_context
from source import digest
from publish import commit_existing,target,validate_manifest
from stage import PREFIX,commit_staged,verified_staged,validate
from assets import verify_assets
from source import ROOT,inventory
from split_producer import VARIANTS

REPO='gtotaro86-ai/sakura-cloud-builds';LIMIT=8_000_000
class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self,*args,**kwargs):return None

def github(path,logs=False,post=False):
    token=os.environ['GITHUB_TOKEN'].strip()
    req=urllib.request.Request('https://api.github.com/repos/'+REPO+path,
        headers={'Authorization':'Bearer '+token,'Accept':'application/vnd.github+json','X-GitHub-Api-Version':'2022-11-28'},
        data=b'{}' if post else None,method='POST' if post else 'GET')
    try:r=urllib.request.build_opener(NoRedirect).open(req,timeout=90)
    except urllib.error.HTTPError as e:
        if not logs or e.code not in (301,302,303,307,308):raise
        url=e.headers['Location']
        if not url.startswith('https://'):raise RuntimeError('Invalid log redirect')
        r=urllib.request.urlopen(url,timeout=90) # no GitHub token on signed blob host
    with r:raw=r.read(LIMIT+1)
    if len(raw)>LIMIT:raise RuntimeError('Bounded job-log budget exceeded')
    return raw if logs else (json.loads(raw) if raw else {})

def checkpoint(log):
    pieces={};total=None;checksum=None
    for line in log.decode().splitlines():
        if PREFIX not in line:continue
        p=json.loads(line.split(PREFIX,1)[1])
        if not 1<=p['parts']<=500 or not 0<=p['part']<p['parts']:raise RuntimeError('Invalid checkpoint parts')
        if total is not None and (total!=p['parts'] or checksum!=p['sha256']):raise RuntimeError('Ambiguous checkpoint')
        total=p['parts'];checksum=p['sha256']
        if p['part'] in pieces:raise RuntimeError('Duplicate checkpoint part')
        pieces[p['part']]=p['data']
    if total is None:return None
    if len(pieces)!=total:raise RuntimeError('Truncated checkpoint; publication refused')
    compressed=base64.b64decode(''.join(pieces[i] for i in range(total)),validate=True)
    dec=zlib.decompressobj();raw=dec.decompress(compressed,4_000_001)
    if len(raw)>4_000_000 or not dec.eof or dec.unused_data or dec.unconsumed_tail:raise RuntimeError('Checkpoint budget/format')
    if digest(raw)!=checksum:raise RuntimeError('Checkpoint checksum mismatch')
    envelope=json.loads(raw);packet=envelope['packet'];validate(packet)
    sample=base64.b64decode(envelope['sample'],validate=True)
    if sample!=base64.b64decode(packet['header_sample'],validate=True):raise RuntimeError('Checkpoint header sample differs')
    return packet,sample

def transient(log):
    s=log.decode(errors='replace').lower()
    if any(x in s for x in ('quota','billing','storage limit','payment required','402 client error','403 client error','disk quota','no space left','byte budget exhausted')):return False
    return any(x in s for x in ('readtimeout','connecttimeout','timed out','connection reset','connection aborted','remoteprotocolerror','http error 502','http error 503','http error 504','502 bad gateway','503 service unavailable','504 gateway timeout'))

def assemble(api,variant,packets):
    import base64
    plan=json.loads((ROOT/'split-plans.json').read_text())[variant]
    if len(packets)!=len(plan['files']) or [p['index'] for p in packets]!=list(range(len(packets))):raise RuntimeError('Incomplete variant shard set')
    first=packets[0]
    if any((p['asset_revision'],p['builder_commit'],p['build_run_id'])!=(first['asset_revision'],first['builder_commit'],first['build_run_id']) for p in packets):raise RuntimeError('Mixed shard build identities')
    asset=verify_assets(api,first['asset_revision']);verified=dict(asset['receipt']['source_files_verified'])
    for p in packets:validate(p);verified.update(p['receipt']['source_files_verified'])
    manifest={'source':inventory(),'output':plan,'assets':asset,'asset_revision':first['asset_revision'],
        'native_files':[p['native_file'] for p in packets],
        'receipt':{'files':[p['receipt']['files'][0] for p in packets],'source_files_verified':verified,
                   'source_shard_loads':sum(p['receipt']['source_shard_loads'] for p in packets),
                   'seconds_sum_cpu_jobs':sum(p['receipt']['seconds'] for p in packets),'full_inference_test_performed':False},
        'upload_native_sha256_verified':True,'builder_commit':first['builder_commit'],'build_run_id':first['build_run_id']}
    validate_manifest(manifest)
    return manifest,[base64.b64decode(p['header_sample'],validate=True) for p in packets]

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run-id',type=int,required=True);a=ap.parse_args();check_context()
    from huggingface_hub import HfApi
    api=HfApi(token=authentication_token())
    if api.whoami()['name']!='webmp3':raise RuntimeError('Wrong HF publisher')
    run=github('/actions/runs/'+str(a.run_id))
    if run['head_repository']['full_name']!=REPO or run['path']!='.github/workflows/minimax-m31.yml' or run['status']!='completed':raise RuntimeError('Not an authorized completed MiniMax build')
    checkout=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    if checkout!=run['head_sha']:raise RuntimeError('Exact original build checkout required')
    jobs=github('/actions/runs/'+str(a.run_id)+'/jobs?per_page=100')['jobs'];plans=json.loads((ROOT/'split-plans.json').read_text());retry=False;unresolved=[];results=[]
    prerequisites=[j for j in jobs if j['name'] in ('qualify','assets') and j['conclusion'] not in ('success','skipped')]
    for j in prerequisites:
        log=github('/actions/jobs/'+str(j['id'])+'/logs',logs=True)
        if j['conclusion']=='failure' and transient(log) and run['run_attempt']<2:retry=True
        else:unresolved.append({'prerequisite':j['name'],'conclusion':j['conclusion']})
    byname={j['name']:j for j in jobs}
    if len(byname)!=len(jobs):raise RuntimeError('Duplicate job names')
    for v,plan in plans.items():
        packets=[]
        for i,f in enumerate(plan['files']):
            job=byname.get(f'quantize-{v}-{i}')
            if not job:
                if prerequisites:continue
                raise RuntimeError('Missing expected shard job')
            if job['conclusion']=='skipped' and prerequisites:continue
            log=github('/actions/jobs/'+str(job['id'])+'/logs',logs=True);state=checkpoint(log)
            if state:
                p,sample=state
                if (p['variant'],p['index'],p['builder_commit'],p['build_run_id'])!=(v,i,run['head_sha'],str(a.run_id)):raise RuntimeError('Checkpoint job identity mismatch')
                commit_staged(api,p,sample)
            # Successful skipped-rebuild paths have the same checked remote proof.
            from huggingface_hub.utils import RevisionNotFoundError,RepositoryNotFoundError
            ref='build-'+run['head_sha']
            try:
                info=api.model_info(target(v),revision=ref,files_metadata=True)
                entry=next((x for x in info.siblings if x.rfilename==f'staging/shard-{i:05d}.json'),None)
                if entry:
                    from huggingface_hub import hf_hub_download
                    if entry.size is None or entry.size>2_000_000:raise RuntimeError('Shard proof budget')
                    raw=json.loads(Path(hf_hub_download(repo_id=target(v),revision=info.sha,filename=entry.rfilename,token=authentication_token())).read_text())
                    p=verified_staged(api,v,i,run['head_sha'],str(a.run_id),raw['asset_revision']);packets.append(p);continue
            except (RevisionNotFoundError,RepositoryNotFoundError):pass
            if job['conclusion']=='failure' and transient(log) and run['run_attempt']<2:retry=True
            else:unresolved.append({'variant':v,'index':i,'conclusion':job['conclusion']})
        if len(packets)==len(plan['files']):
            m,samples=assemble(api,v,packets);oid=commit_existing(api,m,samples);results.append({'variant':v,'revision':oid,'files':m['native_files']})
    if retry:
        github('/actions/runs/'+str(a.run_id)+'/rerun-failed-jobs',post=True)
        print(json.dumps({'retry_failed_jobs':True,'max_total_attempts':2,'run_id':a.run_id}),flush=True)
    report={'verified_releases':results,'unresolved_jobs':unresolved,'retry_queued':retry}
    print(json.dumps(report),flush=True)
    if os.environ.get('GITHUB_STEP_SUMMARY'):Path(os.environ['GITHUB_STEP_SUMMARY']).write_text('MiniMax M3.1 autonomous finalization\n\n```json\n'+json.dumps(report,indent=2)+'\n```\n')
    if unresolved:raise RuntimeError('Unproven or non-transient failure; incomplete model not released')
if __name__=='__main__':main()
