"""Offline synthetic control-flow checks; never presented as model evidence."""
import base64,contextlib,copy,io,json,tempfile
from pathlib import Path
from source import ROOT,inventory
from publish import validate_manifest,emit_checkpoint
from finalize import checkpoint,transient

def main():
    plans=json.loads((ROOT/'expected-plans.json').read_text())
    for variant in plans:
        p=json.loads((ROOT/'evidence'/f'{variant}-plan.json').read_text())
        receipt={'size':p['size'],'sha256':'0'*64,
            'source_files_verified':{x['path']:x['sha256'] for x in inventory()['weights']},
            'tensors':[{'name':t['name'],'bytes':t['nbytes'],'sha256':'0'*64} for t in p['tensors']]}
        m={'source':inventory(),'output':p,'sha256':'0'*64,'receipt':receipt,'upload_native_sha256_verified':True}
        validate_manifest(m)
        for kind in ('source','size','tensor','native'):
            broken=copy.deepcopy(m)
            if kind=='source':broken['receipt']['source_files_verified'].pop(next(iter(broken['receipt']['source_files_verified'])))
            if kind=='size':broken['receipt']['size']-=1
            if kind=='tensor':broken['receipt']['tensors'][0]['name']='wrong'
            if kind=='native':broken['upload_native_sha256_verified']=False
            try:validate_manifest(broken)
            except RuntimeError:pass
            else:raise AssertionError('Unproven publication accepted: '+kind)
        capture=io.StringIO();sample=b'GGUF'+bytes(508)
        with tempfile.TemporaryDirectory() as tmp,contextlib.redirect_stdout(capture):emit_checkpoint(m,sample,Path(tmp))
        log=capture.getvalue().encode();packet,saved=checkpoint(log)
        assert packet['manifest']==m and saved==sample
        lines=log.splitlines()
        assert len(lines)>1
        for bad in (b'\n'.join(lines[:-1]),log+b'\n'+lines[0]):
            try:checkpoint(bad)
            except RuntimeError:pass
            else:raise AssertionError('Incomplete or duplicated checkpoint accepted')
    assert transient(b'HTTP Error 503: Service Unavailable')
    assert not transient(b'HTTP Error 503: storage quota exhausted')
    assert not transient(b'ValueError: unrecognized model architecture')
    assert checkpoint(b'No complete upload checkpoint') is None
    print('Offline synthetic control checks passed: provenance, tensor proofs, checkpoint roundtrip, truncation/duplicate refusal, bounded retry classifier. No model data or network used.')

if __name__=='__main__':main()
