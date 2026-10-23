"""Deploy the tested continuation; preserve existing state and stock runtime."""
from pathlib import Path
import datetime, json, os, shutil, sqlite3, subprocess, tarfile

ROOT=Path('/opt/hf-quant-agent')
STAGE=Path('/home/ubuntu/hfqa/deploy-20261023')
SOURCE=STAGE/'source'
STAMP=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
BACKUP=Path('/var/backups/hf-quant-agent')/STAMP

def run(args,**kwargs): return subprocess.run(args,check=True,**kwargs)

def active(name): return subprocess.run(['systemctl','is-active','--quiet',name]).returncode==0

def main():
    assert os.geteuid()==0
    if active('hf-quant-worker.service'):
        raise RuntimeError('worker is busy; deploy later without interrupting it')
    for name in ('AdGuardHome','unbound','tailscaled'):
        assert active(name), name+' was not healthy before deployment'
    tmp=STAGE/'test-tmp';tmp.mkdir(exist_ok=True)
    env=os.environ.copy();env.update({'TMPDIR':str(tmp),'HFQA_SPARK_ENABLED':'0','PYTHONDONTWRITEBYTECODE':'1'})
    run([str(ROOT/'.venv/bin/python'),'-m','unittest','discover','-s','tests','-q'],cwd=SOURCE,env=env)
    BACKUP.mkdir(parents=True,exist_ok=False)
    run(['systemctl','stop','hf-quant-worker.timer','hf-quant-agent.service'])
    try:
        with tarfile.open(BACKUP/'code.tar.gz','w:gz') as archive:
            for name in ('hfqa','converters','scripts','docs','oracle','README.md','requirements.txt'):
                if (ROOT/name).exists(): archive.add(ROOT/name,arcname=name)
        state=Path('/var/lib/hf-quant-agent/data/jobs.sqlite')
        with sqlite3.connect(state) as src,sqlite3.connect(BACKUP/'jobs.sqlite') as dst: src.backup(dst)
        for name in ('hf-quant-agent.service','hf-quant-worker.service','hf-quant-worker.timer'):
            shutil.copy2('/etc/systemd/system/'+name,BACKUP/name)
        for name in ('hfqa','converters','scripts','docs','oracle','tests'):
            shutil.copytree(SOURCE/name,ROOT/name,dirs_exist_ok=True)
        for name in ('README.md','requirements.txt'): shutil.copy2(SOURCE/name,ROOT/name)
        runtime=ROOT/'runtime/llama-b11669'
        shutil.copytree('/home/ubuntu/hfqa/runtime/llama-b11669',runtime,dirs_exist_ok=True)
        # Remove world-write inherited from the previous installer, only in this project's code.
        for name in ('hfqa','converters','scripts','docs','oracle','tests','runtime'):
            for path in [ROOT/name,*list((ROOT/name).rglob('*'))]:
                if path.is_symlink(): continue
                os.chown(path,0,0)
                os.chmod(path,0o755 if path.is_dir() or path.name.startswith('llama-') or '.so' in path.name else 0o644)
        os.chmod(ROOT,0o755)
        state_root=Path('/var/lib/hf-quant-agent')
        for name in ('sources','outputs','queue','huggingface','data','work'):
            (state_root/name).mkdir(exist_ok=True)
        run(['chown','-R','hfquant:hfquant',str(state_root)])
        run(['chmod','750',str(state_root)])
        for name in ('hf-quant-agent.service','hf-quant-worker.service','hf-quant-worker.timer'):
            shutil.copy2(ROOT/'oracle/systemd'/name,'/etc/systemd/system/'+name)
        run(['systemd-analyze','verify','/etc/systemd/system/hf-quant-agent.service','/etc/systemd/system/hf-quant-worker.service'])
        run(['systemctl','daemon-reload'])
        run(['systemctl','start','hf-quant-agent.service'])
        for name in ('AdGuardHome','unbound','tailscaled'): assert active(name)
        run(['systemctl','enable','--now','hf-quant-worker.timer'])
        print(json.dumps({'deployed':True,'backup':str(BACKUP),'dns_active':True}))
    except BaseException:
        with tarfile.open(BACKUP/'code.tar.gz') as archive: archive.extractall(ROOT,filter='data')
        for name in ('hf-quant-agent.service','hf-quant-worker.service','hf-quant-worker.timer'):
            if (BACKUP/name).exists():shutil.copy2(BACKUP/name,'/etc/systemd/system/'+name)
        run(['systemctl','daemon-reload'])
        run(['systemctl','start','hf-quant-agent.service','hf-quant-worker.timer'])
        raise
if __name__=='__main__':main()
