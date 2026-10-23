"""Synthetic publication guard checks; no network, credentials or model payload."""
import copy
from types import SimpleNamespace as N
import publish_ready as p

def fails(fn):
    try:fn()
    except RuntimeError:return
    raise AssertionError('Invalid release state accepted')

def main():
    run={'head_repository':{'full_name':p.REPO},'path':'.github/workflows/minimax-m31.yml','head_sha':p.COMMIT,'status':'in_progress'}
    jobs=[{'name':n,'conclusion':'success'} for n in ('qualify','assets',*[f'quantize-{p.VARIANT}-{i}' for i in range(p.SHARD_COUNT)])]
    p.check_run(run,jobs,p.COMMIT)
    fails(lambda:p.check_run(run,jobs[:-1],p.COMMIT))
    fails(lambda:p.check_run(run,jobs+jobs[:1],p.COMMIT))
    broken=copy.deepcopy(jobs);broken[-1]['conclusion']='in_progress';fails(lambda:p.check_run(run,broken,p.COMMIT))
    fails(lambda:p.check_run(run,jobs,'0'*40))
    frozen={'path':'second.gguf','size':100};native=dict(frozen,sha256='a'*64)
    packet={'variant':p.VARIANT,'index':1,'builder_commit':p.COMMIT,'build_run_id':str(p.RUN),'native_file':native}
    remote=N(size=100,lfs=N(sha256='a'*64));p.check_packet(packet,1,frozen,remote)
    fails(lambda:p.check_packet(packet,0,frozen,remote))
    fails(lambda:p.check_packet(packet,1,frozen,N(size=100,lfs=N(sha256='b'*64))))
    class Copy:
        def __init__(self,repo):self.src_repo_id=repo
    same=Copy('target');external=Copy('source');other=N(src_repo_id='target')
    ops=p.same_repo_operations('target',[same,external,other],Copy)
    assert same.src_repo_id is None and external.src_repo_id=='source' and other.src_repo_id=='target' and len(ops)==3
    print('Synthetic guards passed: successful complete job set, exact build identity and matching Hub size/hash required.')
if __name__=='__main__':main()
