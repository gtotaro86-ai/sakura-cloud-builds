"""Route bounded streaming jobs without materializing large models on Oracle."""
from __future__ import annotations
import json, os, re, urllib.request, urllib.error
from pathlib import Path
from hfqa.heretic import release_plan
from hfqa.security import redact
from hfqa.stream_sources import select_stream_sources

RUNTIME_REVISION='620d2225cc436ab0b001e3200e590e3b286ef9ab'
GITHUB_REPO='gtotaro86-ai/sakura-cloud-builds'
WORKFLOW='hfqa-native-stream.yml'


def github_token():
    path=Path(os.environ.get('HFQA_GITHUB_TOKEN_FILE','/etc/hf-quant-agent/github.token'))
    if not path.is_file():return ''
    if os.name!='nt' and path.stat().st_mode & 0o077:raise RuntimeError('GitHub token file must be private to its owner')
    return path.read_text(encoding='utf-8').strip()


def github_request(path, *, body=None, token=None):
    token=token if token is not None else github_token()
    if not token:raise RuntimeError('GitHub actions:write credential is not configured on Oracle')
    request=urllib.request.Request('https://api.github.com/repos/'+GITHUB_REPO+path,
        data=None if body is None else json.dumps(body).encode(),
        headers={'Authorization':'Bearer '+token,'Accept':'application/vnd.github+json','Content-Type':'application/json','User-Agent':'hf-quant-agent'})
    with urllib.request.urlopen(request,timeout=30) as response:
        raw=response.read(8*1024**2+1)
        if len(raw)>8*1024**2:raise RuntimeError('GitHub metadata response exceeds bound')
        return json.loads(raw) if raw else {'accepted':response.status==204}


def worker_status():
    revision=os.environ.get('HFQA_GITHUB_WORKER_SHA','')
    try:configured=bool(github_token())
    except OSError:configured=False
    return {'oracle_stream':{'configured':True,'full_source_materialization':False,'cpu_quota_percent_of_one_core':50,'memory_max_bytes':2*1024**3},
            'github_stream':{'credential_configured':configured,'worker_revision':revision or None,
                'ready':configured and bool(re.fullmatch('[a-f0-9]{40}',revision)),
                'repository':GITHUB_REPO,'workflow':WORKFLOW,'paid_runners_allowed':False}}


def queue_stream(store,job_id,work_dir):
    job=store.get(job_id)
    if not job or job['state']!='WAITING_FOR_WORKER':return job
    if job.get('variants',['normal'])!=['normal']:
        store.transition(job_id,'BLOCKED','Heretic requires a separately qualified GPU worker')
        return store.get(job_id)
    backend=job.get('execution_backend','auto')
    backend='github' if backend=='github' else 'oracle'
    selected=select_stream_sources(job['analysis'])
    plan=dict(job.get('plan')or{})
    plan.update(route='native_gguf_stream',execution_backend=backend,full_source_materialization=False,
                total_source_bytes=selected['total_source_bytes'],source_shard_count=len(selected['sources']),
                publish_allowed=False,stream_stage_only=True)
    store.update_fields(job_id,plan_json=plan,expected_state='WAITING_FOR_WORKER')
    if not job['publish']:
        store.transition(job_id,'BLOCKED','streaming output requires authorized HF staging; publication was not requested')
        return store.get(job_id)
    if ((job['analysis'].get('license_decision')or{}).get('redistributable') is not True
            or (selected.get('license_decision')or{}).get('redistributable') is not True):
        store.transition(job_id,'BLOCKED','original and GGUF source licenses must both permit automatic HF staging')
        return store.get(job_id)
    manifest={'sources':selected['sources'],'quants':job['quants'],'runtime_revision':RUNTIME_REVISION,
              'source_license':selected['license_decision'],'original_source':{'repo_id':job['repo_id'],'revision':job['revision']},
              'target_repo_id':release_plan(job['repo_id'],['normal'])[0]['repo_id'],'target_revision':'main'}
    if backend=='oracle':
        directory=Path(os.environ.get('HFQA_STREAM_QUEUE_DIR','/var/lib/hf-quant-agent/stream-queue'))
        directory.mkdir(parents=True,exist_ok=True)
        ticket={'job_id':job_id,'execution_backend':'oracle','manifest':manifest}
        current=directory/(job_id+'.json')
        if current.is_file():
            if json.loads(current.read_text(encoding='utf-8'))!=ticket:raise RuntimeError('existing stream ticket differs from pinned job')
            return store.get(job_id)
        temporary=directory/(job_id+'.tmp');temporary.write_text(json.dumps(ticket),encoding='utf-8')
        os.replace(temporary,directory/(job_id+'.json'))
        store.event(job_id,job['state'],job['state'],'Oracle streaming ticket queued',{'source_bytes':selected['total_source_bytes'],'source_shards':len(selected['sources']),'cpu_quota_percent':50,'memory_max_bytes':2*1024**3,'full_source_materialization':False})
    else:
        result=dict(job.get('result')or{})
        previous=result.get('github_stream')or{}
        if previous.get('dispatch_intent_recorded'):
            if github_token():return monitor_github(store,job_id)
            return store.get(job_id)
        revision=os.environ.get('HFQA_GITHUB_WORKER_SHA','')
        if not github_token() or not re.fullmatch('[a-f0-9]{40}',revision):
            store.event(job_id,job['state'],job['state'],'GitHub worker waiting for one-time server credential and pinned worker configuration',{'full_source_materialization':False,'repository':GITHUB_REPO})
            return store.get(job_id)
        intent={'dispatch_intent_recorded':True,'worker_revision':revision,'job_id':job_id,'manifest':manifest,'state':'DISPATCHING'}
        result.update(published=False,github_stream=intent)
        store.update_fields(job_id,result_json=result)
        try:
            github_request('/actions/workflows/'+WORKFLOW+'/dispatches',body={'ref':revision,'inputs':{'job_id':job_id,'mode':'stage','manifest_json':json.dumps(manifest,separators=(',',':'))}})
            intent['state']='DISPATCHED'
        except Exception as exc:
            intent.update(state='DISPATCH_OUTCOME_REQUIRES_CHECK',error=redact(type(exc).__name__))
        result['github_stream']=intent;store.update_fields(job_id,result_json=result)
        store.event(job_id,job['state'],job['state'],'GitHub streaming dispatch recorded',{'state':intent['state'],'worker_revision':revision})
    return store.get(job_id)


def monitor_github(store,job_id):
    job=store.get(job_id);result=dict(job.get('result')or{});dispatch=result.get('github_stream')or{}
    if not dispatch.get('dispatch_intent_recorded'):return job
    data=github_request('/actions/workflows/'+WORKFLOW+'/runs?event=workflow_dispatch&per_page=100')
    matched=[r for r in data.get('workflow_runs',[]) if r.get('head_sha')==dispatch.get('worker_revision') and r.get('display_title')=='HFQA stream '+job_id]
    if not matched:return job
    run=max(matched,key=lambda r:r['id']);dispatch.update(run_id=run['id'],url=run['html_url'],state=run['status'],conclusion=run.get('conclusion'))
    result['github_stream']=dispatch;store.update_fields(job_id,result_json=result)
    if run['status']=='completed' and job['state']=='WAITING_FOR_WORKER':
        message='GitHub streaming stage finished; validation receipts require verification before publication' if run.get('conclusion')=='success' else 'GitHub streaming worker failed; previous receipts retained'
        store.transition(job_id,'BLOCKED',message,{'run_url':run['html_url'],'conclusion':run.get('conclusion')})
    return store.get(job_id)
