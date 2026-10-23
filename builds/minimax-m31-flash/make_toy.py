"""Small synthetic runtime fixture; not a model quality evaluation."""
import sys
from pathlib import Path
import numpy as np
llama=Path(sys.argv[1]).resolve();sys.path.insert(0,str(llama/'gguf-py'))
import gguf
out=Path(sys.argv[2]);w=gguf.GGUFWriter(out,'minimax-m3')
w.add_string('general.architecture','minimaxm31_sakura')
w.add_name('Synthetic M3.1 all-sparse runtime fixture')
w.add_context_length(512);w.add_embedding_length(128);w.add_block_count(3)
w.add_feed_forward_length(128);w.add_head_count(4);w.add_head_count_kv(4)
w.add_key_length(128);w.add_value_length(128);w.add_layer_norm_rms_eps(1e-6)
w.add_rope_dimension_count(64);w.add_rope_freq_base(10000)
w.add_leading_dense_block_count(1);w.add_expert_feed_forward_length(128)
w.add_expert_count(2);w.add_expert_used_count(1);w.add_expert_shared_count(1)
w.add_expert_weights_scale(2);w.add_expert_weights_norm(True)
w.add_expert_gating_func(gguf.ExpertGatingFuncType.SIGMOID)
w.add_indexer_head_count(4);w.add_indexer_key_length(128);w.add_indexer_top_k(2)
w.add_indexer_block_size(128);w.add_indexer_local_blocks(1)
w.add_uint32('minimax-m3.attention.indexer.first_layer',0)
w.add_tokenizer_model('llama');w.add_token_list([f't{i}' for i in range(32)])
w.add_token_scores([0.]*32);w.add_token_types([gguf.TokenType.NORMAL]*32)
w.add_bos_token_id(0);w.add_eos_token_id(1);w.add_add_bos_token(False)
rng=np.random.default_rng(74231)
def tensor(n,shape,norm=False,bias=False):
    x=np.ones(shape,np.float32) if norm else (np.zeros(shape,np.float32) if bias else rng.normal(0,.015,shape).astype(np.float32))
    w.add_tensor(n,x)
tensor('token_embd.weight',(32,128));tensor('output.weight',(32,128));tensor('output_norm.weight',(128,),True)
for i in range(3):
    p=f'blk.{i}.'
    for n in ('attn_norm','ffn_norm'):tensor(p+n+'.weight',(128,),True)
    for n,shape in (('attn_q',(512,128)),('attn_k',(512,128)),('attn_v',(512,128)),('attn_output',(128,512)),('indexer.q_proj',(512,128)),('indexer.k_proj',(128,128))):tensor(p+n+'.weight',shape)
    for n in ('attn_q_norm','attn_k_norm','indexer.q_norm','indexer.k_norm'):tensor(p+n+'.weight',(128,),True)
    if i==0:
        for n in ('ffn_gate','ffn_up','ffn_down'):tensor(p+n+'.weight',(128,128))
    else:
        tensor(p+'ffn_gate_inp.weight',(2,128));tensor(p+'exp_probs_b.bias',(2,),bias=True)
        for n in ('ffn_gate_exps','ffn_up_exps','ffn_down_exps'):tensor(p+n+'.weight',(2,128,128))
        for n in ('ffn_gate_shexp','ffn_up_shexp','ffn_down_shexp'):tensor(p+n+'.weight',(128,128))
w.write_header_to_file();w.write_kv_data_to_file();w.write_tensors_to_file();w.close()
print('Synthetic fixture written:',out)
