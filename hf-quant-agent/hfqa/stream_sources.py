"""Immutable source selection for remote streaming. Metadata only on Oracle (plus a ranged header read for files whose name does not say what they hold)."""
from __future__ import annotations
import re
import urllib.parse
from hfqa.inspect import _json

PRECISION_TAG = r'(?:^|[-_.])(q8_0|f16|bf16)(?:[-_.]|$)'
# any other quantization or precision tag in a file name: such a file is what its name says and is not a candidate for header classification
OTHER_TAG = r'(?:^|[-_.])(?:i?q\d[a-z0-9_]*|ud-[a-z0-9_]+|f32|fp16|fp32|bf32|mxfp\d[a-z0-9_]*|tq\d[a-z0-9_]*)(?:[-_.]|$)'


def _header_precision(descriptor: dict) -> str | None:
    """'bf16', 'f16' or 'q8_0' when most of the weights of a single GGUF file are stored in that type, read from the header only (ranged reads), else None."""
    from hfqa.gguf import TENSOR_TYPES
    from hfqa.remote_gguf import BufferedRangeReader, RangeReader, inspect_remote_gguf
    remote = inspect_remote_gguf(descriptor, reader=BufferedRangeReader(RangeReader(descriptor)))
    weight: dict = {}
    for tensor in remote.tensors:
        count = 1
        for dimension in tensor.shape:
            count *= int(dimension)
        kind = TENSOR_TYPES.get(tensor.type)
        weight[kind] = weight.get(kind, 0) + count
    if not weight:
        return None
    dominant = max(weight, key=weight.get)
    return {'BF16': 'bf16', 'F16': 'f16', 'Q8_0': 'q8_0'}.get(dominant)


def select_stream_sources(analysis: dict, timeout: float = 30, *, component: str | None = None, classify=None, only_precision: str | None = None) -> dict:
    if component not in {None, 'image_transformer'}:
        raise ValueError('unsupported stream component')
    classify = classify or _header_precision
    native = analysis.get('native_gguf_repo') or analysis
    repo,revision=native.get('repo_id'),native.get('revision')
    if not repo or not re.fullmatch('[a-fA-F0-9]{40}',revision or ''):
        raise ValueError('stream source must have a pinned full revision')
    info=_json(f'https://huggingface.co/api/models/{urllib.parse.quote(repo,safe="/")}/revision/{revision}?blobs=true',timeout)
    groups={};untagged=[];forced={}
    for item in info.get('siblings') or []:
        name=item.get('rfilename') or ''
        if component == 'image_transformer':
            from hfqa.image_components import is_image_weight_candidate
            if not is_image_weight_candidate(name):continue
        lowered=name.lower()
        if not lowered.endswith('.gguf'):continue
        # Companion files are no model weights: the vision projector (mmproj) is far smaller than the model and would otherwise win "smallest complete set".
        if re.search(r'mmproj|(?:^|[-_./])imatrix',lowered):continue
        tagged=bool(re.search(PRECISION_TAG,lowered))
        split=re.search(r'-(\d{5})-of-(\d{5})(?=\.gguf$)',name,re.I)
        if not tagged and (component or split or re.search(OTHER_TAG,lowered)):continue
        key=re.sub(r'-\d{5}-of-\d{5}(?=\.gguf$)','',name,flags=re.I)
        lfs=item.get('lfs') or {};digest=lfs.get('sha256') or lfs.get('oid') or ''
        digest=digest.removeprefix('sha256:')
        size=item.get('size') or lfs.get('size')
        if not isinstance(size,int) or isinstance(size,bool) or size<24 or not re.fullmatch('[a-fA-F0-9]{64}',digest):continue
        descriptor={'repo_id':repo,'revision':revision.lower(),'filename':name,'size':size,'sha256':digest.lower(),
                    'split_index':int(split.group(1)) if split else 1,'split_count':int(split.group(2)) if split else 1}
        if tagged:groups.setdefault(key,[]).append(descriptor)
        else:untagged.append((key,descriptor))
    # A single file whose name carries no tag (Spark-X2.6-1.7B.gguf) is classified from its own header, so a full-precision file is not passed over for a Q8_0 next to it.
    for key,descriptor in untagged:
        try:found=classify(descriptor)
        except Exception:found=None  # unreadable header: the file is simply not a candidate
        if found in {'bf16','f16','q8_0'}:
            groups.setdefault(key,[]).append(descriptor);forced[key]=found
    valid=[]
    for name,files in groups.items():
        expected=files[0]['split_count']
        if expected>1024 or expected<1 or len(files)!=expected or any(f['split_count']!=expected for f in files):continue
        files.sort(key=lambda f:f['split_index'])
        if [f['split_index'] for f in files]!=list(range(1,expected+1)):continue
        valid.append((name,files))
    if not valid:raise ValueError('no complete verified Q8_0/F16/BF16 GGUF source set found')
    if only_precision is not None:
        wanted=('bf16','f16','q8_0').index(only_precision)
        def kind(name):
            if name in forced:return ('bf16','f16','q8_0').index(forced[name])
            lowered=name.lower()
            return 0 if 'bf16' in lowered else 1 if re.search(r'(?:^|[-_.])f16(?:[-_.]|$)',lowered) else 2
        valid=[item for item in valid if kind(item[0])==wanted]
        if not valid:raise ValueError('no complete verified %s GGUF source set found'%only_precision.upper())
    # Own quants should come from the highest precision available: BF16, then F16, then (already quantized) Q8_0.
    def precision(name):
        if name in forced:return ('bf16','f16','q8_0').index(forced[name])
        lowered=name.lower()
        return 0 if 'bf16' in lowered else 1 if re.search(r'(?:^|[-_.])f16(?:[-_.]|$)',lowered) else 2
    valid.sort(key=lambda item:(precision(item[0]),sum(f['size'] for f in item[1]),item[0]))
    name,files=valid[0]
    from hfqa.vision_projector import discover as discover_projectors
    return {'repo_id':repo,'revision':revision.lower(),'sources':files,'total_source_bytes':sum(f['size'] for f in files),
            'source_set':name,'source_precision':('bf16','f16','q8_0')[precision(name)],'weights_downloaded_on_oracle':False,
            'vision_projector':discover_projectors(repo,revision,info),
            'license_decision':native.get('license_decision') or analysis.get('license_decision')}
