#include "llama.h"
#include "ggml-backend.h"
#include <cmath>
#include <cstdio>
#include <vector>

int main(int argc, char ** argv) {
    if (argc != 3) { return 2; }
    ggml_backend_load_all_from_path(argv[2]);
    llama_backend_init();
    auto mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    auto * model = llama_model_load_from_file(argv[1], mp);
    if (!model) { return 3; }
    auto cp = llama_context_default_params();
    cp.n_ctx = 512; cp.n_batch = 384; cp.n_ubatch = 384;
    cp.n_threads = 4; cp.n_threads_batch = 4;
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    auto * ctx = llama_init_from_model(model, cp);
    if (!ctx) { return 4; }
    std::vector<llama_token> tokens(260, 2);
    auto batch = llama_batch_get_one(tokens.data(), tokens.size());
    if (llama_decode(ctx, batch)) { return 5; }
    auto * logits = llama_get_logits_ith(ctx, -1);
    for (int i = 0; i < 32; ++i) { if (!std::isfinite(logits[i])) { return 6; } }
    llama_token token = 3;
    batch = llama_batch_get_one(&token, 1);
    if (llama_decode(ctx, batch)) { return 7; }
    logits = llama_get_logits_ith(ctx, -1);
    for (int i = 0; i < 32; ++i) { if (!std::isfinite(logits[i])) { return 8; } }
    std::puts("Synthetic M3.1 runtime: all-sparse leading dense layer, 260-token prefill and one-token decode passed with finite logits.");
    llama_free(ctx); llama_model_free(model); llama_backend_free();
    return 0;
}
