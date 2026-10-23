"""Read-only public-target/account storage assessment; never enables billing."""
import json
from source import ROOT,inventory
from credentials import authentication_token

def public_target(api,repo):
    info=api.model_info(repo)
    if info.private is not False:raise RuntimeError('Only public HF target repositories are authorized')
    return info

def main():
    from huggingface_hub import HfApi
    from publish import target
    api=HfApi(token=authentication_token());account=api.whoami()
    if account['name']!='webmp3':raise RuntimeError('Wrong HF publisher')
    plans=json.loads((ROOT/'split-plans.json').read_text());aux=json.loads((ROOT/'asset-plan.json').read_text())['size']+inventory()['original_dspark']['size']
    targets=[target(v) for v in plans]
    known=0;unknown=0;public=0
    for info in api.list_models(author='webmp3',full=True,limit=500):
        if info.private:continue
        public+=1;size=getattr(info,'usedStorage',None)
        if isinstance(size,int):known+=size
        else:unknown+=1
    for repo in targets:
        if api.repo_exists(repo):public_target(api,repo)
    # The supported whoami/model API exposes no guaranteed free public-byte
    # balance. Report that explicitly rather than inventing one from a plan name.
    report={'public_targets_only':True,'planned_logical_weight_bytes':sum(p['size']+aux for p in plans.values()),
        'public_model_scan_limit':500,'public_model_repos_observed':public,'reported_existing_public_model_storage_bytes':known,
        'repos_without_storage_measurement':unknown,'guaranteed_free_public_bytes_remaining':None,
        'quota_confirmed_sufficient':False,'public_storage_policy':'best-effort; server acceptance required',
        'deduplication_assumed_quota_neutral':False,'paid_storage_enabled_by_this_project':False,
        'on_quota_error':'stop; no billing change or quota retry'}
    print(json.dumps(report,indent=2),flush=True)
if __name__=='__main__':main()
