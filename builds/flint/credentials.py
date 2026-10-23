"""Credential hygiene and actual free-runner guard; never print credentials."""
import os
from importlib.metadata import version

def authentication_token():
    token=os.environ.get('HF_TOKEN','').strip()
    if not token or not token.isascii() or any(c.isspace() for c in token):
        raise RuntimeError('HF_TOKEN must be one ASCII token; value suppressed')
    return token

def check_context():
    required={'GITHUB_ACTIONS':'true','RUNNER_ENVIRONMENT':'github-hosted','RUNNER_OS':'Linux',
              'REPO_VISIBILITY':'public','FLINT_STANDARD_RUNNER':'ubuntu-24.04'}
    if any(os.environ.get(k)!=v for k,v in required.items()):
        raise RuntimeError('Actual public standard GitHub-hosted CPU runner required')
    authentication_token()
    if version('huggingface-hub')!='2.2.0' or version('hf-xet')!='1.7.0':
        raise RuntimeError('Pinned upload dependencies required')
