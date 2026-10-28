"""Strict public, immutable inputs for a read-only GGUF comparison."""
from __future__ import annotations
import hashlib,json,math,re
from hfqa import runner_quality,source_builds
from hfqa.remote_gguf import descriptor_url

MAX_CANDIDATES=12
MAX_BYTES=40*10**9
MAX_SOURCE_BYTES=8*10**9
SCHEMA='hfqa-measure-1'

def fingerprint(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()

def file_descriptor(item):
    if not isinstance(item,dict):raise ValueError('measurement file descriptor must be an object')
    if type(item.get('size')) is not int or not 24<=item['size']<=MAX_SOURCE_BYTES:raise ValueError('measurement file size missing or exceeds 8 GB')
    if not re.fullmatch('[a-f0-9]{64}',str(item.get('sha256',''))):raise ValueError('measurement file requires immutable SHA-256')
    descriptor_url(item)
    return item

def validate_manifest(manifest):
    if not isinstance(manifest,dict):raise ValueError('measurement manifest must be an object')
    runtime=source_builds.validate(manifest.get('runtime'))
    if manifest.get('runtime_revision')!=runtime['revision']:raise ValueError('measurement runtime revisions differ')
    file_descriptor(manifest.get('source'))
    candidates=manifest.get('candidates')
    if not isinstance(candidates,list) or not 1<=len(candidates)<=MAX_CANDIDATES:raise ValueError('between one and twelve candidates are required')
    labels=[]
    for item in candidates:
        file_descriptor(item)
        label=item.get('label')
        if not isinstance(label,str) or not re.fullmatch('[A-Za-z0-9][A-Za-z0-9_-]{0,63}',label):raise ValueError('candidate label must be a safe unique identifier')
        labels.append(label)
    if len(set(labels))!=len(labels):raise ValueError('candidate labels must be unique')
    if sum(item['size'] for item in [manifest['source'],*candidates])>MAX_BYTES:raise ValueError('measurement input set exceeds 40 GB')
    if manifest.get('quality',runner_quality.SPEC)!=runner_quality.SPEC:raise ValueError('evaluation specification differs from the qualified fixed text/context')
    return manifest

def validate_result(result,manifest):
    validate_manifest(manifest)
    if not isinstance(result,dict) or result.get('schema')!=SCHEMA or result.get('manifest_sha256')!=fingerprint(manifest):raise ValueError('measurement result is not bound to the request')
    build=result.get('runtime') or {}
    if build.get('repo')!=source_builds.REPO or build.get('head')!=manifest['runtime']['revision'] or build.get('static') is not True:raise ValueError('measurement source-build identity differs')
    binaries=build.get('binaries') or {}
    if any(not re.fullmatch('[a-f0-9]{64}',str(binaries.get(name,''))) for name in source_builds.TARGETS):raise ValueError('measurement binary hashes are incomplete')
    for wanted,actual in [(manifest['source'],result.get('source') or {})]+[(item,(result.get('files') or {}).get(item['label']) or {}) for item in manifest['candidates']]:
        if any(actual.get(k)!=wanted[k] for k in ('repo_id','revision','filename','sha256')) or actual.get('bytes')!=wanted['size']:raise ValueError('measurement input size/hash/provenance differs')
    labels={item['label'] for item in manifest['candidates']}
    if set(result.get('files') or {})!=labels or result.get('spec')!=runner_quality.SPEC:raise ValueError('measurement candidate/specification set differs')
    compatibility=result.get('compatibility') or {};signature=compatibility.get('source') or {}
    if (not isinstance(signature.get('architecture'),str) or not signature['architecture']
        or any(not re.fullmatch('[a-f0-9]{64}',str(signature.get(key,''))) for key in ('tensor_shapes_sha256','model_metadata_sha256'))
        or set(compatibility.get('candidates') or {})!=labels
        or any(item!=signature for item in compatibility['candidates'].values())):raise ValueError('measurement tokenizer/model compatibility is not established')
    quality=result.get('quality') or {}
    if quality.get('skipped'):return {'measured':False,'reason':str(quality['skipped'])[:500]}
    evidence=quality.get('evidence') or {}
    texts=[{k:t[k] for k in ('name','sha256','bytes')} for t in runner_quality.SPEC['texts']]
    if (evidence.get('texts')!=texts or evidence.get('context')!=runner_quality.CONTEXT or evidence.get('chunks')!=runner_quality.chunks_for(manifest['source']['size'])
        or evidence.get('tool_sha256')!=binaries['llama-perplexity'] or evidence.get('source_bytes')!=manifest['source']['size']
        or not re.fullmatch('[a-f0-9]{64}',str(evidence.get('file_sha256','')))):raise ValueError('measurement evaluation/runtime evidence differs')
    if set(quality.get('per_quant') or {})!=labels:raise ValueError('measurement statistics do not cover every candidate')
    for stats in quality['per_quant'].values():
        for key in ('ppl_quant','ppl_source','ppl_ratio','mean_kld','same_top_token_percent'):
            pair=stats.get(key)
            if not isinstance(pair,dict) or any(type(pair.get(k)) not in (int,float) or not math.isfinite(pair[k]) for k in ('value','error')):raise ValueError('measurement statistics are missing or non-finite')
            if pair['error']<0:raise ValueError('negative measurement uncertainty')
            if pair['value']<0 or key=='same_top_token_percent' and pair['value']>100:raise ValueError('measurement statistic outside physical bounds')
            if key in {'ppl_quant','ppl_source','ppl_ratio'} and pair['value']<=0:raise ValueError('perplexities and their ratio must be positive')
    return {'measured':True,'candidates':sorted(labels)}
