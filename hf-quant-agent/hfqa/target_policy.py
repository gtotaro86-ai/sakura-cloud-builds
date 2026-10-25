"""Portable publisher identity/namespace checks; no local credentials or infrastructure."""
import re

def authorize_target(target,identity):
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]{0,95}/[A-Za-z0-9][A-Za-z0-9._-]{0,95}',target or ''):
        raise RuntimeError('invalid release target')
    namespaces={identity.get('name')}
    namespaces.update(org.get('name') for org in identity.get('orgs') or [] if org.get('roleInOrg') in {'admin','write','contributor'})
    if target.split('/')[0] not in namespaces:raise RuntimeError('HF credential is not a writer in the requested release namespace')
    return target

def ensure_repository(api,target,private=False):
    authorize_target(target,api.whoami())
    api.create_repo(repo_id=target,repo_type='model',exist_ok=True,private=private)
    info=api.model_info(target)
    visibility=getattr(info,'private',None)
    if type(visibility) is not bool or visibility!=private:
        raise RuntimeError('existing repository visibility differs from the requested release; choose another target')
    return info
