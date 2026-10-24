"""Fetch and hash-check a fixed CPU runtime for a free public GitHub runner."""
from pathlib import Path
import argparse, hashlib, tarfile, urllib.request
URL='https://github.com/ggml-org/llama.cpp/releases/download/b11669/llama-b11669-bin-ubuntu-x64.tar.gz'
DIGEST='cf07c1c112d709bd67e522017471de57ed0ec8bea031d732c5422351c07f65ac'

def fetch(destination):
    destination=Path(destination);destination.mkdir(parents=True,exist_ok=True)
    archive=destination/'runtime.tar.gz';sha=hashlib.sha256();count=0
    with urllib.request.urlopen(URL,timeout=60) as response,archive.open('wb') as handle:
        while chunk:=response.read(1024**2):
            count+=len(chunk)
            if count>32*1024**2:raise ValueError('runtime asset exceeds fixed download budget')
            sha.update(chunk);handle.write(chunk)
    if sha.hexdigest()!=DIGEST:raise ValueError('stock runtime asset SHA-256 mismatch')
    with tarfile.open(archive) as bundle:bundle.extractall(destination,filter='data')
    found=list(destination.rglob('libllama.so'))
    if len(found)!=1:raise ValueError('ambiguous stock runtime layout')
    print(found[0].parent.resolve())
    return found[0].parent
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--destination',required=True)
    fetch(p.parse_args().destination)
