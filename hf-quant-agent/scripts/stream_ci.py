"""Cloud qualification/staging driver: no complete source file on the runner."""
from __future__ import annotations
import argparse, json, os, sys
from pathlib import Path
from types import SimpleNamespace
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from hfqa.gguf import validate_gguf
from hfqa.quantize import test_inference
from hfqa.streaming import _sha256 as sha256_file
from scripts.run_stream_worker import execute, read_manifest
from scripts.build_stream_bridge import REVISION


def run(manifest_path,runtime,bridge,work,mode='proof'):
    manifest,source=read_manifest(manifest_path)
    if manifest['runtime_revision']!=REVISION:raise ValueError('manifest does not match the fixed stock runtime build')
    work=Path(work);work.mkdir(parents=True,exist_ok=True)
    runtime=Path(runtime);bridge=Path(bridge)
    ggml=runtime/'libggml-base.so'
    if mode=='stage':
        from hfqa.inspect import inspect_repo
        descriptors=source.get('sources') or [source]
        checked=inspect_repo(f"https://huggingface.co/{descriptors[0]['repo_id']}/tree/{descriptors[0]['revision']}")
        if (checked.get('license_decision') or {}).get('redistributable') is not True:
            raise ValueError('live pinned GGUF source license blocks automatic Xet staging')
        target=manifest.get('target_repo_id')
        if not target:raise ValueError('staging target must be explicitly pinned')
        args=SimpleNamespace(manifest=Path(manifest_path),bridge=bridge,ggml=ggml,receipt_dir=work/'receipts',
                             output_dir=None,xet_repo=target,shard_bytes=4*1024**3)
        result=execute(args)
        result['worker_state']='STAGED_AWAITING_RUNTIME_VALIDATION'
    else:
        # A real source stream is exercised for every quant; only small fixture
        # outputs are materialized so the stock runtime can load actual bytes.
        reports=[]
        for quant in manifest['quants']:
            folder=work/quant;folder.mkdir(exist_ok=False)
            chosen={**manifest,'quants':[quant]}
            request=folder/'request.json';request.write_text(json.dumps(chosen),encoding='utf-8')
            args=SimpleNamespace(manifest=request,bridge=bridge,ggml=ggml,receipt_dir=folder/'receipts',
                                 output_dir=folder/'output',xet_repo=None,shard_bytes=256*1024**2)
            produced=execute(args);output=produced['outputs'][0]
            proofs=output['receipt']['shards'];files=output['files']
            if len(files)!=len(proofs) or not files:raise ValueError('streamed output shard set incomplete')
            if not output['receipt']['source_sha256_verified']:raise ValueError('source proof incomplete')
            tensors=0
            for item,proof in zip(files,proofs):
                path=Path(item['path']);report=validate_gguf(path)
                if not report['structure_valid'] or sha256_file(path)!=proof['sha256']:raise ValueError('streamed shard structural/hash validation failed')
                tensors+=report['tensor_count']
            if tensors!=output['receipt']['tensor_count']:raise ValueError('streamed shard tensor coverage incomplete')
            inference=test_inference(runtime/'llama-cli',files[0]['path'],folder/'logs')
            if 'paris' not in inference['completion'].lower():raise ValueError('known small-model smoke prompt did not return Paris')
            validation={'quant':quant,'source_sha256_verified':True,'shard_count':len(files),'tensor_count':tensors,
                        'runtime_compatible':True,'inference_tested':True,'inference':inference,'published':False,
                        'source_materialized':False,'output_files_materialized_for_small_fixture_only':True,
                        'receipt':output['receipt']}
            (folder/'validation.json').write_text(json.dumps(validation,indent=2),encoding='utf-8')
            reports.append(validation)
        result={'worker_state':'QUALIFIED_SMALL_STREAM_PROOF','source_materialized':False,'published':False,'validations':reports}
    (work/'worker-result.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({'worker_state':result['worker_state'],'published':False,'source_materialized':False}))
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',required=True);p.add_argument('--runtime',required=True);p.add_argument('--bridge',required=True)
    p.add_argument('--work',required=True);p.add_argument('--mode',choices=['proof','stage'],default='proof')
    a=p.parse_args();run(a.manifest,a.runtime,a.bridge,a.work,a.mode)
