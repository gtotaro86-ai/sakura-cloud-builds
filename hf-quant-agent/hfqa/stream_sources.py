"""Immutable source selection for remote streaming. Metadata only on Oracle."""
from __future__ import annotations
import re
import urllib.parse
from hfqa.inspect import _json


def select_stream_sources(analysis: dict, timeout: float = 30, *, component: str | None = None) -> dict:
    if component not in {None, 'image_transformer'}:
        raise ValueError('unsupported stream component')
    native = analysis.get('native_gguf_repo') or analysis
    repo,revision=native.get('repo_id'),native.get('revision')
    if not repo or not re.fullmatch('[a-fA-F0-9]{40}',revision or ''):
        raise ValueError('stream source must have a pinned full revision')
    info=_json(f'https://huggingface.co/api/models/{urllib.parse.quote(repo,safe="/")}/revision/{revision}?blobs=true',timeout)
    groups={}
    for item in info.get('siblings') or []:
        name=item.get('rfilename') or ''
        if component == 'image_transformer':
            from hfqa.image_components import is_image_weight_candidate
            if not is_image_weight_candidate(name):continue
        if not name.lower().endswith('.gguf') or not re.search(r'(?:^|[-_.])(q8_0|f16|bf16)(?:[-_.]|$)',name.lower()):continue
        split=re.search(r'-(\d{5})-of-(\d{5})(?=\.gguf$)',name,re.I)
        key=re.sub(r'-\d{5}-of-\d{5}(?=\.gguf$)','',name,flags=re.I)
        lfs=item.get('lfs') or {};digest=lfs.get('sha256') or lfs.get('oid') or ''
        digest=digest.removeprefix('sha256:')
        size=item.get('size') or lfs.get('size')
        if not isinstance(size,int) or isinstance(size,bool) or size<24 or not re.fullmatch('[a-fA-F0-9]{64}',digest):continue
        descriptor={'repo_id':repo,'revision':revision.lower(),'filename':name,'size':size,'sha256':digest.lower(),
                    'split_index':int(split.group(1)) if split else 1,'split_count':int(split.group(2)) if split else 1}
        groups.setdefault(key,[]).append(descriptor)
    valid=[]
    for name,files in groups.items():
        expected=files[0]['split_count']
        if expected>1024 or expected<1 or len(files)!=expected or any(f['split_count']!=expected for f in files):continue
        files.sort(key=lambda f:f['split_index'])
        if [f['split_index'] for f in files]!=list(range(1,expected+1)):continue
        valid.append((name,files))
    if not valid:raise ValueError('no complete verified Q8_0/F16/BF16 GGUF source set found')
    valid.sort(key=lambda item:('q8_0' not in item[0].lower(),sum(f['size'] for f in item[1]),item[0]))
    name,files=valid[0]
    return {'repo_id':repo,'revision':revision.lower(),'sources':files,'total_source_bytes':sum(f['size'] for f in files),
            'source_set':name,'weights_downloaded_on_oracle':False,
            'license_decision':native.get('license_decision') or analysis.get('license_decision')}
