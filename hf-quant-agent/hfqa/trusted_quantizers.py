"""Whose GGUF files may be used automatically as the INPUT of our own quantization.

Only established quantizers are trusted to have converted the ORIGINAL weights unchanged; a repository of anyone else, or one whose name signals changed weights
(see inspect.MODIFIED_WEIGHTS_HINT), is never picked automatically. The built-in list below is reviewed in code. The operator can widen or narrow it without a
deploy through `trusted-quantizers.json` in the data directory:

    {"add": ["SomeQuantizer"], "remove": ["TheBloke"], "add_scoped": [], "remove_scoped": []}

Three kinds of trust, from wide to narrow:
  * the model's OWN author: a GGUF repository of the same owner that the Hub lists as a quantization of exactly that model;
  * the built-in/operator list (`trusted_owners`): quantizers trusted for any model;
  * SCOPED publishers (`SCOPED`): publishers of modified weights (abliterated, uncensored, fine-tunes). They are trusted ONLY when the requested model is itself such a
    model, because on the Hub they list their modified GGUF as a "quantization" of the ORIGINAL (e.g. huihui-ai's abliterated GGUF names Qwen/Qwen3.8-27B as base).

Hub owner names only (letters, digits, `.`, `_`, `-`), compared case-insensitively. Anything else in the file is ignored. Trust is per owner, never per repository.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

FILE = 'trusted-quantizers.json'
OWNER = re.compile(r'[A-Za-z0-9][A-Za-z0-9._-]{0,95}')
# (The dpo/sft word boundaries (an underscore counts as a boundary) were stored as backspace characters in hfqa/inspect.py from 0.11.0 until 0.23.14, so those two hints never matched.)
# A name that signals changed weights: such a GGUF is never taken for the ORIGINAL model (it could pass off a modified model as the original).
MODIFIED_WEIGHTS_HINT = re.compile(r"abliterat|uncensor|heretic|merge|merged|finetune|fine-tune|lora|(?<![a-z0-9])(?:dpo|sft)(?![a-z0-9])|rp[-_]|roleplay|distill", re.I)

# owner -> why it is on the list (Hub evidence, 2026-11-04: GGUF repositories tagged as quantizations of many different base models, plain quantizations)
BUILT_IN = {
    'bartowski': 'established LLM quantizer', 'unsloth': 'established LLM quantizer', 'lmstudio-community': 'LM Studio staff picks', 'mradermacher': 'established bulk quantizer',
    'QuantFactory': 'bulk quantizer', 'ggml-org': 'the llama.cpp project', 'second-state': 'bulk quantizer', 'TheBloke': 'historic quantizer (inactive since early 2024)',
    'MaziyarPanahi': 'bulk quantizer',
    'AtomicChat': 'operator decision 2026-11-04; ~97 GGUF repositories of ~75 base models since 2026-05, 3.5 M downloads',
    'QuantStack': 'ComfyUI/diffusion GGUF quantizer; 34 repositories of 30 base models since 2025-05, 3.1 M downloads',
    'city96': 'author of ComfyUI-GGUF; 30 repositories of 27 base models since 2024-08, 1.8 M downloads',
    'bullerwins': 'long-standing LocalLLaMA quantizer; 43 repositories of 29 base models since 2024-06',
    'mudler': 'author of LocalAI; 98 repositories of 63 base models since 2024-05, 4.4 M downloads',
}


# Publishers of MODIFIED weights, trusted only for requests that are modified models themselves (operator decision 2026-11-04). Hub evidence: huihui-ai (abliterations),
# orcarouter (uncensored/abliterated variants, since 2026-07), peculiar-ragdoll (fine-tunes such as Dirk/Tiel/Sharp, listed on the Hub as quantizations of the originals).
SCOPED = {
    'huihui-ai': 'abliterated models and their GGUF; 30 GGUF repositories since 2024-12',
    'orcarouter': 'uncensored/abliterated variants (reads "oracrouter" in the operator request; the Hub owner is orcarouter)',
    'peculiar-ragdoll': 'fine-tunes (Dirk, Tiel-Coder, Sharp-*) that the Hub lists as quantizations of the original models',
}


def _data() -> Path:
    return Path(os.environ.get('HFQA_DATA', 'data'))


def _names(value) -> set[str]:
    return {item.lower() for item in value if isinstance(item, str) and OWNER.fullmatch(item)} if isinstance(value, list) else set()


def overrides() -> dict:
    """{'add': set, 'remove': set} from the operator's file; a missing or broken file changes nothing."""
    try:
        value = json.loads((_data() / FILE).read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return {key: set() for key in ('add', 'remove', 'add_scoped', 'remove_scoped')}
    value = value if isinstance(value, dict) else {}
    return {key: _names(value.get(key)) for key in ('add', 'remove', 'add_scoped', 'remove_scoped')}


def trusted_owners() -> frozenset:
    """Lower-case owner names, built-in list plus the operator's additions minus the operator's removals."""
    changes = overrides()
    return frozenset((({name.lower() for name in BUILT_IN} | changes['add']) - changes['remove']))


def is_trusted(owner: str) -> bool:
    return isinstance(owner, str) and owner.lower() in trusted_owners()


def scoped_owners() -> frozenset:
    changes = overrides()
    return frozenset(((({name.lower() for name in SCOPED}) | changes['add_scoped']) - changes['remove_scoped']))


def request_is_modified(repo_id: str) -> bool:
    """The requested model is itself a modified one: its name says so, or a scoped publisher owns it."""
    owner, _, name = str(repo_id or '').partition('/')
    return bool(MODIFIED_WEIGHTS_HINT.search(name or owner)) or owner.lower() in scoped_owners()


def may_supply(candidate_owner: str, requested_repo_id: str) -> bool:
    """May a GGUF repository of `candidate_owner`, listed on the Hub as a quantization of `requested_repo_id`, be the input of our quantization?"""
    if not isinstance(candidate_owner, str) or not candidate_owner:
        return False
    owner = candidate_owner.lower()
    if owner == str(requested_repo_id or '').partition('/')[0].lower():
        return True  # the model's own author
    if is_trusted(owner):
        return True
    return owner in scoped_owners() and request_is_modified(requested_repo_id)


def name_filter_applies(requested_repo_id: str) -> bool:
    """The changed-weights name filter protects ORIGINAL models. For a request that is itself a modified model, a GGUF named alike is the same model, not a substitute."""
    return not request_is_modified(requested_repo_id)
