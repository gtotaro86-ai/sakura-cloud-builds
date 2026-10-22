"""Finalize already-uploaded outputs after the pinned projector-copy argument failure.

No model payload reads or quantization. Evidence is reconstructed from the completed
GitHub job log; the original per-tensor checksum receipt is not recoverable.
"""
import argparse,base64,hashlib,json,os,urllib.request,urllib.error
from pathlib import Path
from source import ROOT,inventory,digest
from publish import check_context,authentication_token
from release_card import readme

REPOSITORY='gtotaro86-ai/sakura-cloud-builds'
RUN=39845179392
BUILD='75b0f0e41b304bbffc417dedab4d4c1f7a783827'
JOBS={'Q4_K_M':120056447241,'Q5_K_M':120056447830}
LIMIT=5_000_000

class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self,*args,**kwargs):return None

def github(path,logs=False):
    url='https://api.github.com/repos/'+REPOSITORY+path
    token=os.environ['GITHUB_TOKEN'].strip()
    req=urllib.request.Request(url,headers={'Authorization':'Bearer '+token,'Accept':'application/vnd.github+json'})
    try:r=urllib.request.build_opener(NoRedirect).open(req,timeout=60)
    except urllib.error.HTTPError as e:
        if not logs or e.code not in (301,302,303,307,308):raise
        target=e.headers['Location']
        if not target.startswith('https://'):raise RuntimeError('Invalid log redirect')
        # Signed blob URL is read without sending the GitHub token across hosts.
        r=urllib.request.urlopen(target,timeout=60)
    with r:raw=r.read(LIMIT+1)
    if len(raw)>LIMIT:raise RuntimeError('GitHub evidence budget exceeded')
    return raw if logs else json.loads(raw)

def evidence(log,plan):
    text=log.decode()
    if 'ValueError: `src_repo_type` is required when `src_repo_id` is set.' not in text:
        raise RuntimeError('Not the recoverable projector-copy failure')
    # Reaching this exact statement in the pinned publisher means producer receipt,
    # independent pipe hash, native Xet hash and upload context all passed first.
    if "copied=CommitOperationCopy(src_repo_id=projector['repo']" not in text:
        raise RuntimeError('Pinned publisher failure location missing')
    events=[]
    for line in text.splitlines():
        pos=line.find('{')
        if pos<0:continue
        try:events.append(json.loads(line[pos:]))
        except ValueError:pass
    sources={x['source_verified'] for x in events if 'source_verified' in x}
    expected={x['path']:x['sha256'] for x in inventory()['weights']}
    if sources!=set(expected):raise RuntimeError('Incomplete verified source evidence')
    tensors=[x for x in events if 'quantized' in x]
    if [x['quantized'] for x in tensors]!=[x['name'] for x in plan['tensors']]:
        raise RuntimeError('Incomplete or reordered tensor evidence')
    for x,t in zip(tensors,plan['tensors']):
        end=plan['header_size']+t['offset']+t['nbytes']
        end+=-t['nbytes']%32
        if x['bytes']!=end or x['total']!=plan['size']:raise RuntimeError('Tensor size evidence differs')
    final=[x for x in events if x.get('stream_complete')]
    if len(final)!=1 or final[0]['size']!=plan['size']:raise RuntimeError('No complete producer stream')
    sha=final[0]['sha256']
    if len(sha)!=64 or len(bytes.fromhex(sha))!=32:raise RuntimeError('Invalid stream hash')
    return {'size':plan['size'],'sha256':sha,'source_files_verified':expected,
            'reconstructed_from_completed_job_log':True,'original_per_tensor_hash_receipt_available':False,
            'tensor_count':len(tensors),'full_inference_test_performed':False}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--variant',choices=list(JOBS),required=True);args=ap.parse_args()
    check_context()
    from huggingface_hub import HfApi,CommitOperationAdd,CommitOperationCopy
    from huggingface_hub.lfs import UploadInfo
    run=github('/actions/runs/'+str(RUN))
    if run['head_sha']!=BUILD or run['head_repository']['full_name']!=REPOSITORY:raise RuntimeError('Unexpected build source')
    job=github('/actions/jobs/'+str(JOBS[args.variant]))
    if job['run_id']!=RUN or job['status']!='completed' or job['conclusion']!='failure':raise RuntimeError('Build job not yet recoverable')
    log=github('/actions/jobs/'+str(JOBS[args.variant])+'/logs',logs=True)
    plan=json.loads((ROOT/'evidence'/f'{args.variant}-plan.json').read_text())
    receipt=evidence(log,plan);sha=receipt['sha256']
    samples=json.loads((ROOT/'evidence/recovery-header-samples.json').read_text())
    item=samples[args.variant];sample=base64.b64decode(item['base64'],validate=True)
    if len(sample)!=512 or digest(sample)!=item['sha256'] or sample[:4]!=b'GGUF':raise RuntimeError('Invalid GGUF sample')
    api=HfApi(token=authentication_token())
    if api.whoami()['name']!='webmp3':raise RuntimeError('Incorrect HF publisher')
    repo=f'webmp3/Sakura-Sage-A1.5-{args.variant}-GGUF';path=f'Sakura-Sage-A1.5-{args.variant}.gguf'
    info=api.model_info(repo,files_metadata=True)
    if any(x.rfilename!='.gitattributes' for x in info.siblings):raise RuntimeError('Inspect nonempty target before recovery')
    op=CommitOperationAdd(path_in_repo=path,path_or_fileobj=b'')
    op.upload_info=UploadInfo(size=plan['size'],sample=sample,sha256=bytes.fromhex(sha))
    op._upload_mode='lfs';op._should_ignore=False;op._is_uploaded=True
    p=inventory()['projector']
    copy=CommitOperationCopy(src_repo_type='model',src_repo_id=p['repo'],src_revision=p['revision'],src_path_in_repo=p['path'],path_in_repo=p['path'])
    manifest={'source':inventory(),'output':plan,'sha256':sha,'receipt':receipt,
              'upload_native_sha256_verified':True,
              'publication_recovery':{'evidence':'completed job log and pinned publisher control flow',
                  'log_sha256':hashlib.sha256(log).hexdigest(),'build_commit':BUILD,
                  'job_id':JOBS[args.variant],'run_id':RUN,
                  'original_receipt_file_retained':False,'model_payload_read_during_recovery':False}}
    docs={'README.md':readme(plan,sha),'LICENSE':(ROOT/'LICENSE').read_text(),
          'SHA256SUMS':f"{sha}  {path}\n{p['sha256']}  {p['path']}\n",
          'build-manifest.json':json.dumps(manifest,indent=2)+'\n',
          'local-reference.json':(ROOT/'local-reference.json').read_text()}
    ops=[op,copy]+[CommitOperationAdd(path_in_repo=n,path_or_fileobj=v.encode()) for n,v in docs.items()]
    result=api.create_commit(repo_id=repo,parent_commit=info.sha,operations=ops,
        commit_message=f'Publish completed BF16 {args.variant} stream and original projector')
    after=api.model_info(repo,revision=result.oid,files_metadata=True)
    for name,size,checksum in ((path,plan['size'],sha),(p['path'],p['size'],p['sha256'])):
        f=next(x for x in after.siblings if x.rfilename==name)
        if f.size!=size or not f.lfs or f.lfs.sha256!=checksum:raise RuntimeError('Recovered Hub output differs')
    print(json.dumps({'published':repo,'revision':result.oid,'sha256':sha,'bytes':plan['size'],'requantized':False}),flush=True)
    if os.environ.get('GITHUB_STEP_SUMMARY'):
        Path(os.environ['GITHUB_STEP_SUMMARY']).write_text(f'Published https://huggingface.co/{repo}\n\nRecovered the completed native Xet stream. No model payload was downloaded or requantized.\n\nSHA256: `{sha}`\n')

if __name__=='__main__':main()
