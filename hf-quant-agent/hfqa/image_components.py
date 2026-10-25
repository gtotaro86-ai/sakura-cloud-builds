"""Image repositories may contain only the diffusion model; auxiliaries are references, not jobs."""
from __future__ import annotations
from pathlib import PurePosixPath
import re
from urllib.parse import urlsplit, unquote, quote

ROLES = ('transformer', 'unet', 'text_encoder', 'text_encoder_2', 'vae')
WEIGHTS = ('.gguf', '.safetensors', '.bin')


def auxiliary_role(value):
    text = str(value).lower().replace('-', '_')
    if re.search(r'(?:^|[/\s_.])(?:text_encoders?_2)(?:[/\s_.]|$)', text):
        return 'text_encoder_2'
    if re.search(r'text[ _]?encoders?|qwen[\d._]*vl|(?:^|[/\s_.])(?:clip|clip_l|clip_g|t5|t5xxl)(?:[/\s_.]|$)', text):
        return 'text_encoder'
    if re.search(r'(?:^|[/\s_.])(?:vae|autoencoder)(?:[/\s_.]|$)', text):
        return 'vae'
    return None


def is_image_weight_candidate(filename):
    """Exclude identified auxiliary files; tensor signatures still prove the image architecture."""
    return auxiliary_role(filename) is None


def component_files(files):
    result = {role: [] for role in ROLES}
    for name in files:
        if not isinstance(name, str) or not name.lower().endswith(WEIGHTS):
            continue
        role = auxiliary_role(name)
        if role is None:
            first = name.split('/', 1)[0]
            role = first if first in ('transformer', 'unet') else None
        if role and len(result[role]) < 16:
            result[role].append(name)
    return result


def hf_reference(value):
    """Record a public HF pointer only. Never follow it or copy query credentials."""
    if not isinstance(value, str) or len(value) > 2048:
        return None
    try:
        url = urlsplit(value)
        if url.scheme != 'https' or url.netloc != 'huggingface.co':
            return None
        parts = [unquote(part) for part in url.path.strip('/').split('/')]
        if len(parts) < 2 or not all(re.fullmatch(r'[A-Za-z0-9._-]+', part or '') and part not in {'.','..'} for part in parts[:2]):
            return None
        if parts[0] in {'datasets','spaces','docs','settings','login'}:
            return None
        reference = {'repo_id':'/'.join(parts[:2]), 'source_url':'https://huggingface.co/'+quote('/'.join(parts), safe='/'),
                     'verified':False, 'downloaded':False, 'quantization_requested':False}
        if len(parts) > 2:
            if len(parts) < 4 or parts[2] not in {'resolve','blob','tree'}:
                return None
            revision = parts[3]
            if not revision or '/' in revision or '\\' in revision or revision in {'.','..'}:
                return None
            reference['requested_revision'] = revision
            reference['revision'] = revision.lower() if re.fullmatch('[a-fA-F0-9]{40}',revision) else None
            if len(parts) > 4:
                filename = '/'.join(parts[4:])
                path = PurePosixPath(filename)
                if path.is_absolute() or '..' in path.parts or '\\' in filename or ':' in filename:
                    return None
                reference['filename'] = filename
        return reference
    except ValueError:
        return None


def extract_component_references(readme='', card=None):
    refs = []
    def add(role, url, evidence):
        reference = hf_reference(url)
        if reference is None or role not in {'text_encoder','text_encoder_2','vae'}:
            return
        record = {'role':role, **reference, 'evidence':evidence}
        if not any(x['role'] == role and x['source_url'] == record['source_url'] for x in refs):
            refs.append(record)
    for role in ('text_encoder','text_encoder_2','vae'):
        value = (card or {}).get(role)
        if isinstance(value, str):
            add(role, value, 'model_card_field')
    # Bounded model-card metadata is never passed to an agent as instructions.
    for line in str(readme or '')[:262144].splitlines():
        if len(refs) >= 32:
            break
        markdown = re.findall(r'\[([^\]\n]{0,160})\]\((https://huggingface\.co/[^\s)<>]+)\)', line)
        consumed = set()
        for label, url in markdown:
            consumed.add(url)
            role = auxiliary_role(label) or auxiliary_role(urlsplit(url).path)
            # A generic label may use an explicit, unambiguous role on its own line.
            if role is None and len(markdown) == 1:
                role = auxiliary_role(line)
            add(role, url, 'model_card_link')
        urls = re.findall(r'https://huggingface\.co/[^\s)<>\]"\']+', line)
        for url in urls:
            if url not in consumed:
                add(auxiliary_role(urlsplit(url).path) or (auxiliary_role(line) if len(urls) == 1 else None), url, 'model_card_link')
    return refs[:32]


def plan_components(analysis):
    files = analysis.get('component_files') or component_files(analysis.get('files') or analysis.get('weight_files') or [])
    refs = analysis.get('component_references') or []
    result = []
    for role in ('transformer','text_encoder','text_encoder_2','vae'):
        local = list(files.get(role) or [])[:16]
        if role == 'transformer' and not local:
            local = [name for name in analysis.get('weight_files') or [] if is_image_weight_candidate(name)][:16]
        external = [x for x in refs if x.get('role') == role][:16]
        # A missing encoder is normal for standalone diffusion repositories.
        result.append({'role':role, 'local_files':local, 'external_references':external,
                       'availability':'local_unverified' if local else 'external_reference_unverified' if external else 'unresolved',
                       'action':'quantize_selected_image_model' if role == 'transformer' else 'copy_unchanged_into_release_repo',
                       'include_in_release_repo':True,
                       'quantize_in_image_job':role == 'transformer', 'automatic_download':False,
                       'quantization_requested':role == 'transformer',
                       'required_for_transformer_quantization':role == 'transformer'})
    return result
