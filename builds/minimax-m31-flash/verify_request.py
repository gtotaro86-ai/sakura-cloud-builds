"""Validate exactly the user-authorized source, formats, resources and release path."""
import json
from source import ROOT,REV
expected={'model':'MiniMaxAI/MiniMax-M3.1-Flash','source_revision':REV,'variants':['Q2_K','Q3_K_M','Q4_K_M','Q5_K_M'],
    'publish':True,'paid_resources':False,'source_kind':'released hybrid NVFP4/MXFP8 checkpoint',
    'runtime':'pinned patched llama.cpp; minimaxm31_sakura','preserve_original_dspark':True,'max_parallel_shards':2,'automatic_transient_retry_limit':1}
if json.loads((ROOT/'request.json').read_text())!=expected:raise SystemExit('Request differs from authorized release')
print('Authorized pinned MiniMax M3.1 source, four policies, public standard CPU, native streaming release.')
