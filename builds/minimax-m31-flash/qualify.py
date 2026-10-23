"""CPU construction qualification: bounded real rows and synthetic/metadata fixtures."""
import argparse,json,os,subprocess,sys
from pathlib import Path
from source import ROOT,digest

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--llama',type=Path,required=True);ap.add_argument('--build',type=Path,required=True);ap.add_argument('--work',type=Path,required=True);a=ap.parse_args();a.work.mkdir(parents=True,exist_ok=True)
    def run(script,*args,**kw):subprocess.run([sys.executable,str(ROOT/script),*map(str,args)],check=True,**kw)
    if digest((ROOT/'runtime.patch').read_bytes())!=json.loads((ROOT/'evidence/runtime-checks.json').read_text())['runtime_patch_sha256']:raise RuntimeError('Runtime patch proof differs')
    run('probe_codec.py','--work',a.work,'--library',a.build/'bin/libggml-base.so')
    run('check_plans.py','--llama',a.llama,'--build',a.build,'--work',a.work)
    sys.path.insert(0,str(a.llama/'gguf-py'))
    plans=json.loads((ROOT/'split-plans.json').read_text())
    for v in plans:
        run('split_producer.py','--mode','plan','--variant',v,'--llama',a.llama,'--build',a.build,'--work',a.work)
        generated=json.loads((a.work/'split-plan.json').read_text());paths=[]
        try:
            for f in generated['files']:
                path=a.work/f['path'];paths.append(path)
                with path.open('wb') as out:out.write((a.work/(f['path']+'.header')).read_bytes());out.truncate(f['size'])
            import gguf
            log=a.work/(v+'-split-native.log')
            with log.open('wb') as stream:run('native_plan.py',a.build/'lib_minimax_quant.so',paths[0],int(getattr(gguf.LlamaFileType,'MOSTLY_'+v)),stdout=stream,stderr=stream)
            print('Native split metadata loader passed:',v,flush=True)
        finally:
            for p in paths:p.unlink(missing_ok=True)
    run('asset_producer.py','--mode','plan','--llama',a.llama,'--work',a.work)
    env=dict(os.environ,PYTHONPATH=str(a.llama/'gguf-py'))
    toy=a.work/'synthetic.gguf';run('make_toy.py',a.llama,toy,env=env)
    subprocess.run([str(a.build/'runtime_probe'),str(toy),str(a.build/'bin')],check=True);toy.unlink()
    print('Qualification passed. Real-row construction, metadata-only native plans, synthetic runtime. No full-model inference.',flush=True)
if __name__=='__main__':main()
