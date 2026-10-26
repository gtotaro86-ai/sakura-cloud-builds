"""Fetch and hash-check a fixed CPU runtime for a free public GitHub runner.

Without a manifest the baseline build is used. With a manifest the runtime named in it is used, but only an official
llama.cpp release asset with the exact SHA-256 that Oracle pinned (see hfqa/runtimes.py).
"""
from pathlib import Path
import argparse, hashlib, json, sys, tarfile, urllib.request
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hfqa import runtimes
URL = runtimes.asset_url(runtimes.BASELINE_TAG)
DIGEST = runtimes.BASELINE_SHA256

def fetch(destination, descriptor=None):
    descriptor = descriptor or runtimes.baseline()
    destination=Path(destination);destination.mkdir(parents=True,exist_ok=True)
    archive=destination/'runtime.tar.gz';sha=hashlib.sha256();count=0
    with urllib.request.urlopen(descriptor['url'],timeout=60) as response,archive.open('wb') as handle:
        while chunk:=response.read(1024**2):
            count+=len(chunk)
            if count>runtimes.MAX_ASSET_BYTES:raise ValueError('runtime asset exceeds fixed download budget')
            sha.update(chunk);handle.write(chunk)
    if sha.hexdigest()!=descriptor['sha256']:raise ValueError('stock runtime asset SHA-256 mismatch')
    with tarfile.open(archive) as bundle:bundle.extractall(destination,filter='data')
    found=list(destination.rglob('libllama.so'))
    if len(found)!=1:raise ValueError('ambiguous stock runtime layout')
    print(found[0].parent.resolve())
    return found[0].parent
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--destination',required=True);p.add_argument('--manifest')
    a=p.parse_args()
    chosen=None
    if a.manifest:
        raw=Path(a.manifest)
        if raw.stat().st_size>1024**2:raise ValueError('manifest exceeds metadata budget')
        chosen=runtimes.descriptor_for_manifest(json.loads(raw.read_text(encoding='utf-8-sig')))
    fetch(a.destination,chosen)
