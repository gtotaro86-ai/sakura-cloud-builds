"""Exercise the real Tailscale API without printing authentication material."""
import json, pathlib, sys, urllib.request
BASE='http://100.76.161.25:8787'
TOKEN=pathlib.Path('/etc/hf-quant-agent/api.token').read_text().strip()

def call(path,body=None):
    req=urllib.request.Request(BASE+path,data=None if body is None else json.dumps(body).encode(),headers={'Authorization':'Bearer '+TOKEN,'Content-Type':'application/json'})
    with urllib.request.urlopen(req,timeout=120) as r:return json.load(r)

def main():
    action=sys.argv[1]
    if action=='create':
        created=call('/jobs',{'url':'https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct','quants':['Q2_K','Q3_K_M','Q4_K_M','Q5_K_M'],'variants':['normal'],'publish':True})
        jid=created['job']['id'];pathlib.Path('/var/lib/hf-quant-agent/e2e-job-id').write_text(jid)
        job=call('/jobs/'+jid+'/run',{})['job']
        print(json.dumps({'id':jid,'state':job['state'],'route':(job.get('plan')or{}).get('route'),'publish_allowed':(job.get('plan')or{}).get('publish_allowed'),'error':job.get('error')}))
    else:
        jid=pathlib.Path('/var/lib/hf-quant-agent/e2e-job-id').read_text().strip()
        data=call('/jobs/'+jid)
        job=data['job']; result=job.get('result')or{}
        print(json.dumps({'id':jid,'state':job['state'],'error':job.get('error'),'published':result.get('published'),'release':result.get('release'),'artifacts':[{k:a.get(k) for k in ('quant','bytes','sha256','runtime_compatible','inference_tested')} for a in result.get('artifacts',[])],'last_events':[{'message':e['message'],'data':e.get('data')} for e in data['events'][-5:]],'shards':[{'quant':s['quant'],'state':s['state']} for s in data['shards']]}))
if __name__=='__main__':main()
