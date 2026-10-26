"""Build a small project adapter against a pinned, untouched stock llama runtime."""
from pathlib import Path
import argparse, hashlib, json, re, subprocess, urllib.request
REVISION='620d2225cc436ab0b001e3200e590e3b286ef9ab'

def build(runtime, destination, cpp, revision=REVISION):
    if not re.fullmatch('[a-f0-9]{40}',revision):raise ValueError('runtime revision must be a full commit SHA')
    runtime=Path(runtime).resolve();destination=Path(destination).resolve();cpp=Path(cpp).resolve()
    if not (runtime/'libllama.so').is_file():raise ValueError('stock Linux runtime library missing')
    destination.mkdir(parents=True,exist_ok=True)
    headers=destination/'headers'
    todo=['src/llama-ext.h','src/llama-model-loader.h','src/llama-arch.h','ggml/include/ggml.h'];seen={}
    while todo:
        name=todo.pop()
        if name in seen:continue
        with urllib.request.urlopen('https://raw.githubusercontent.com/ggml-org/llama.cpp/'+revision+'/'+name,timeout=30) as response:
            raw=response.read(2*1024**2+1)
        if len(raw)>2*1024**2:raise ValueError('upstream header exceeds bound')
        path=headers/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(raw)
        seen[name]=hashlib.sha256(raw).hexdigest()
        for dependency in re.findall(r'^\s*#\s*include\s*"([^"]+)"',raw.decode(),re.M):
            if '/' in dependency or '..' in dependency:raise ValueError('unsupported header include')
            prefix='ggml/include/' if dependency.startswith('ggml') or dependency=='gguf.h' else ('include/' if dependency=='llama.h' else 'src/')
            todo.append(prefix+dependency)
        if len(seen)>100:raise ValueError('upstream include closure exceeds build budget')
    output=destination/'libhfqa_plan.so'
    subprocess.run(['g++','-std=c++17','-O2','-shared','-fPIC',str(cpp),'-I'+str(headers/'include'),'-I'+str(headers/'src'),'-I'+str(headers/'ggml/include'),'-L'+str(runtime),'-Wl,-rpath,'+str(runtime),'-lllama','-lggml','-lggml-base','-o',str(output)],check=True,timeout=180)
    proof={'runtime_revision':revision,'runtime_patched':False,'headers':seen,'bridge_sha256':hashlib.sha256(output.read_bytes()).hexdigest()}
    (destination/'build-receipt.json').write_text(json.dumps(proof,indent=2))
    print(json.dumps({'bridge':str(output),'upstream_headers':len(seen),'runtime_patched':False}))
    return output

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runtime',required=True);p.add_argument('--destination',required=True)
    p.add_argument('--cpp',default=str(Path(__file__).resolve().parents[1]/'github/native/generic_quant_bridge.cpp'))
    p.add_argument('--manifest',help='build against the runtime revision this manifest pins (default: the baseline)')
    a=p.parse_args();revision=REVISION
    if a.manifest:
        import sys;sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
        from hfqa import runtimes
        revision=runtimes.descriptor_for_manifest(json.loads(Path(a.manifest).read_text(encoding='utf-8-sig')))['revision']
    build(a.runtime,a.destination,a.cpp,revision)
