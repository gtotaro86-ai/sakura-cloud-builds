"""Autonomous metadata-only publication recovery and bounded transient retry."""
import argparse,base64,json,os,subprocess,urllib.request,urllib.error,zlib
from pathlib import Path
from credentials import authentication_token,check_context
from source import digest
from publish import PREFIX,commit_existing,target,validate_manifest

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
    dec=zlib.decompressobj();raw=dec.decompress(compressed,2_000_001)
    if len(raw)>2_000_000 or not dec.eof or dec.unused_data or dec.unconsumed_tail:raise RuntimeError('Checkpoint budget/format')
    if digest(raw)!=checksum:raise RuntimeError('Checkpoint checksum mismatch')
    packet=json.loads(raw);validate_manifest(packet['manifest'])
    sample=base64.b64decode(packet['sample'],validate=True)
    if len(sample)!=512 or sample[:4]!=b'GGUF':raise RuntimeError('Checkpoint header sample')
    return packet,sample

def transient(log):
    s=log.decode(errors='replace').lower()
    if any(x in s for x in ('quota','billing','storage limit','payment required','402 client error','403 client error','disk quota','no space left','byte budget exhausted')):return False
    return any(x in s for x in ('readtimeout','connecttimeout','timed out','connection reset','connection aborted','remoteprotocolerror','http error 502','http error 503','http error 504','502 bad gateway','503 service unavailable','504 gateway timeout'))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run-id',type=int,required=True);args=ap.parse_args()
    check_context()
    from huggingface_hub import HfApi
    run=github('/actions/runs/'+str(args.run_id))
    if run['head_repository']['full_name']!=REPO or run['path']!='.github/workflows/flint-bf16.yml' or run['status']!='completed':raise RuntimeError('Not an authorized completed Flint build')
    checkout=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    if run['head_sha']!=checkout:raise RuntimeError('Recovery must use exact original build commit')
    api=HfApi(token=authentication_token())
    if api.whoami()['name']!='webmp3':raise RuntimeError('Wrong HF publisher')
    jobs=github('/actions/runs/'+str(args.run_id)+'/jobs')['jobs'];found=set();retry=False;unresolved=[];results=[]
    for job in jobs:
        variant=next((q for q in ('Q4_K_M','Q5_K_M') if job['name']==f'quantize ({q})'),None)
        if variant is None:continue
        if variant in found:raise RuntimeError('Ambiguous build job')
        found.add(variant)
        log=github('/actions/jobs/'+str(job['id'])+'/logs',logs=True)
        state=checkpoint(log)
        if state:
            packet,sample=state;m=packet['manifest']
            if m['output']['variant']!=variant or m['builder_commit']!=run['head_sha'] or m['build_run_id']!=str(args.run_id):raise RuntimeError('Checkpoint build identity differs')
            oid=commit_existing(api,m,sample);results.append({'variant':variant,'revision':oid,'sha256':m['sha256']})
        elif job['conclusion']=='failure' and transient(log) and run['run_attempt']<2:
            retry=True
        else:unresolved.append({'variant':variant,'job_id':job['id'],'conclusion':job['conclusion']})
    if found!={'Q4_K_M','Q5_K_M'}:raise RuntimeError('Both original build jobs must be present')
    if retry:
        github('/actions/runs/'+str(args.run_id)+'/rerun-failed-jobs',post=True)
        print(json.dumps({'retry_failed_jobs':True,'max_total_attempts':2,'run_id':args.run_id}),flush=True)
    print(json.dumps({'verified_releases':results,'unresolved_jobs':unresolved,'retry_queued':retry}),flush=True)
    if unresolved:raise RuntimeError('Non-transient or unproven build failure; no guessed release')
    if os.environ.get('GITHUB_STEP_SUMMARY'):
        Path(os.environ['GITHUB_STEP_SUMMARY']).write_text('Flint autonomous finalization\n\n```json\n'+json.dumps({'verified_releases':results,'retry_queued':retry},indent=2)+'\n```\n')

if __name__=='__main__':main()
