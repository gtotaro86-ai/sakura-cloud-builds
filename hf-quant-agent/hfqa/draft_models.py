"""Draft models (drafters for speculative decoding: DFlash, DSpine, EAGLE, Medusa ...) are not language models HFQA can release.

A drafter only predicts the next block of tokens from the hidden states of its TARGET model; on its own it cannot produce text, so the runner's real inference
test cannot pass, and DSpine files additionally need a llama.cpp fork (upstream refuses to load them). On 2026-11-09 two jobs (tencent/Qwen3.8-27B-DSpine-b16 and
aj9o9/Qwen3.8-27B-DSpine-b16-GGUF) ran on GitHub for 4 and 18 minutes and failed; the planner had called the GGUF "eligible" because the architecture name `dflash`
exists in the stock runtime. What the runner log said (all four quant jobs, step "Quantize, verify, run a real inference"): the quantizer wrote the files from the author's BF16 GGUF, but the stock
runtime b11669 refused to load them: "done_getting_tensors: wrong number of tensors; expected 73, got 58" (DSpine's tensor set differs from the stock `dflash`).
Only signals that are specific to a standalone drafter count:
a Hub tag `draft-model`, a drafter architecture in the GGUF or in config.json. The tag `speculative-decoding` alone does NOT count: full models that ship MTP heads
or sidecars (handled by stream_sources) carry it too.
"""
from __future__ import annotations

import re

DRAFT_TAGS = frozenset({'draft-model', 'draft_model', 'drafter'})
DRAFT_GGUF_ARCHITECTURES = frozenset({'dflash', 'dflash2', 'dspine', 'eagle', 'eagle2', 'eagle3', 'medusa'})
DRAFT_CLASS = re.compile(r'(draft[-_]?model|drafter|^eagle\d*(for|model)|^medusa)', re.I)

MESSAGE = ('Das ist ein Entwurfsmodell (Drafter) für spekulatives Dekodieren, kein eigenständiges Sprachmodell: Es sagt nur Textblöcke aus den inneren Zuständen seines '
           'Zielmodells voraus und erzeugt allein keinen Text. HFQA kann es deshalb weder mit einer echten Inferenz prüfen noch sinnvoll quantisieren'
           ' (DSpine braucht außerdem einen eigenen llama.cpp-Zweig). Nutze die vom Autor gelieferten Dateien zusammen mit dem Zielmodell.')


def detect(*, tags=None, architectures=None, gguf_architecture=None) -> dict | None:
    """{'signals': [...]} when the repository is a standalone drafter, else None. Pure function of Hub metadata."""
    signals = []
    for tag in tags or []:
        if isinstance(tag, str) and tag.lower() in DRAFT_TAGS:
            signals.append(f'Hub-Tag "{tag}"')
    for name in architectures or []:
        if isinstance(name, str) and DRAFT_CLASS.search(name):
            signals.append(f'Architektur "{name}"')
    if isinstance(gguf_architecture, str) and gguf_architecture.lower() in DRAFT_GGUF_ARCHITECTURES:
        signals.append(f'GGUF-Architektur "{gguf_architecture}"')
    return {'signals': signals[:6]} if signals else None


def blocker(draft: dict) -> str:
    return MESSAGE + ' Erkannt an: ' + ', '.join(draft.get('signals') or ['Metadaten']) + '.'
