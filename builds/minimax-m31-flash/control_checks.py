"""Offline synthetic publication guard checks; never model-quality evidence."""
import contextlib,copy,io,json,tempfile
from pathlib import Path
from source import ROOT,inventory
from publish import validate_manifest
from stage import validate,emit
from finalize import checkpoint,transient

def fixture(v):
    p=json.loads((ROOT/'split-plans.json').read_text())[v];a=json.loads((ROOT/'asset-plan.json').read_text());inv=inventory()
    asset={'source_revision':inv['source_revision'],'output':a,'sha256':'0'*64,'native_sha256_verified':True,
           'receipt':{'path':a['path'],'size':a['size'],'sha256':'0'*64,'tensors':[{'name':t['name'],'bytes':t['nbytes'],'sha256':'0'*64} for t in a['tensors']],
           'source_files_verified':{inv['weights'][0]['path']:inv['weights'][0]['sha256']}}}
    files=[{'path':f['path'],'size':f['size'],'sha256':'0'*64,'tensors':[{'name':t['name'],'bytes':t['nbytes'],'sha256':'0'*64} for t in f['tensors']]} for f in p['files']]
    return {'source':inv,'output':p,'assets':asset,'asset_revision':'0'*40,'native_files':[{k:f[k] for k in ('path','size','sha256')} for f in files],
            'receipt':{'files':files,'source_files_verified':{x['path']:x['sha256'] for x in inv['weights']}},'upload_native_sha256_verified':True}

def main():
    for v in ('Q2_K','Q3_K_M','Q4_K_M','Q5_K_M'):
        m=fixture(v);validate_manifest(m)
        for kind in ('source','size','tensor','native','asset','sha','plan'):
            b=copy.deepcopy(m)
            if kind=='source':b['receipt']['source_files_verified'].pop(next(iter(b['receipt']['source_files_verified'])))
            if kind=='size':b['native_files'][0]['size']-=1
            if kind=='tensor':b['receipt']['files'][0]['tensors'][0]['name']='wrong'
            if kind=='native':b['upload_native_sha256_verified']=False
            if kind=='asset':b['assets']['receipt']['tensors'][0]['bytes']-=1
            if kind=='sha':b['receipt']['files'][0]['sha256']='z'*64
            if kind=='plan':b['output']['files'][0]['size']-=1
            try:validate_manifest(b)
            except RuntimeError:pass
            else:raise AssertionError('Invalid proof accepted: '+kind)
        for i,f in enumerate(m['receipt']['files']):
            sample=b'GGUF'+bytes(508)
            import base64
            packet={'variant':v,'index':i,'source_revision':inventory()['source_revision'],'asset_revision':'0'*40,
                'builder_commit':'1'*40,'build_run_id':'123','header_sample':base64.b64encode(sample).decode(),
                'native_file':m['native_files'][i],'native_sha256_verified':True,
                'receipt':{'files':[f],'source_files_verified':m['receipt']['source_files_verified']}}
            validate(packet);capture=io.StringIO()
            with tempfile.TemporaryDirectory() as tmp,contextlib.redirect_stdout(capture):emit(packet,sample,Path(tmp))
            log=capture.getvalue().encode();saved,sample2=checkpoint(log);assert saved==packet and sample2==sample
            lines=log.splitlines()
            badlogs=[log+b'\n'+lines[0]]
            if len(lines)>1:badlogs.append(b'\n'.join(lines[:-1]))
            for bad in badlogs:
                try:checkpoint(bad)
                except RuntimeError:pass
                else:raise AssertionError('Invalid checkpoint accepted')
    assert transient(b'HTTP Error 503: Service Unavailable')
    for s in (b'HTTP Error 503: storage quota exhausted',b'No space left',b'Native Xet hash mismatch'):assert not transient(s)
    assert checkpoint(b'No complete upload') is None
    print('All four synthetic publication/checkpoint guards passed; no network or model data.')
if __name__=='__main__':main()
