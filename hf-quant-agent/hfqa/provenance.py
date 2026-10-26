"""Factual model assignment from pinned Hub metadata; never infer identity from a filename."""
from __future__ import annotations
import json
import re


def model_id(value):
    return isinstance(value,str) and len(value)<=160 and bool(re.fullmatch(r'[A-Za-z0-9_.-]+(?:/[A-Za-z0-9_.-]+)?',value)) and all(x not in {'.','..'} for x in value.split('/'))


def identify(info, repo_id, revision):
    card=info.get('cardData') or {}
    raw=card.get('base_model')
    bases=[raw] if isinstance(raw,str) else raw if isinstance(raw,list) else []
    bases=[x for x in bases if model_id(x)][:8]
    relation=card.get('base_model_relation')
    known={'quantized','finetune','adapter','merge'}
    if relation not in known:relation=None
    for tag in info.get('tags') or []:
        match=re.fullmatch(r'base_model:(quantized|finetune|adapter|merge):(.+)',str(tag))
        if match and model_id(match.group(2)):
            if match.group(2) not in bases and len(bases)<8:bases.append(match.group(2))
            if relation is None:relation=match.group(1)
    bases=list(dict.fromkeys(bases))
    primary=bases[0] if relation=='quantized' and len(bases)==1 else repo_id
    return {'input_repo_id':repo_id,'input_revision':revision,'model_repo_id':primary,
            'declared_base_models':bases,'relation':relation or 'undeclared',
            'assignment_basis':'pinned_model_card_metadata','metadata_source':f'https://huggingface.co/{repo_id}/blob/{revision}/README.md',
            'base_model_revisions_resolved':False,'runtime_identity_claimed':False}


def assignment(analysis):
    value=analysis.get('model_provenance') or {}
    if value.get('input_repo_id')!=analysis.get('repo_id') or value.get('input_revision')!=analysis.get('revision'):
        return identify({},analysis.get('repo_id') or 'unknown',analysis.get('revision') or 'unpinned')
    if not model_id(value.get('model_repo_id')):raise ValueError('invalid model provenance')
    return value


def markdown(analysis):
    value=assignment(analysis)
    bases=', '.join(f'[{name}](https://huggingface.co/{name})' for name in value.get('declared_base_models') or [] if model_id(name)) or 'not declared'
    repo=value['input_repo_id'];primary=value['model_repo_id']
    lines = [f"- Original model: [{primary}](https://huggingface.co/{primary}), commit `{value['input_revision']}` ([model card at that commit]({value['metadata_source']}))"]
    if repo != primary:
        lines.append(f"- Submitted repository: [{repo}](https://huggingface.co/{repo})")
    if value.get('declared_base_models'):
        lines.append(f"- Declared base models: {bases} (relation: `{value['relation']}`)")
    return "\n".join(lines) + "\n"


def yaml_base_model(analysis):
    return json.dumps(assignment(analysis)['model_repo_id'])
