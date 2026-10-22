#include "llama-ext.h"
#include "llama-model-loader.h"
#include "ggml.h"
#include <algorithm>
#include <exception>
#include <string>
#include <stdexcept>
#include <vector>

static thread_local std::string last_error;
extern "C" const char * sage_error() { return last_error.c_str(); }

extern "C" int sage_dryrun(const char * input, int ftype) {
    auto params = llama_model_quantize_default_params();
    params.ftype = static_cast<llama_ftype>(ftype);
    params.dry_run = true;
    return llama_model_quantize(input, "/dev/null", &params);
}

extern "C" int sage_types(int ftype, int n, const char ** names, const int64_t * shapes,
                           const int * dims, const int * input_types, int * output_types) {
    ggml_context * ctx = nullptr;
    llama_model * model = nullptr;
    quantize_state_impl * state = nullptr;
    try {
        llama_quant_model_desc desc = {"qwen35moe", 2048, 512, 41, 16, 2, 256, 256, 256};
        model = llama_quant_model_from_metadata(&desc);
        auto params = llama_model_quantize_default_params();
        params.ftype = static_cast<llama_ftype>(ftype);
        state = llama_quant_init(model, &params);
        ctx = ggml_init({ggml_tensor_overhead() * (static_cast<size_t>(n)+1), nullptr, true});
        if (!ctx) throw std::runtime_error("ggml metadata allocation failed");
        std::vector<ggml_tensor *> tensors;
        std::vector<int> indices;
        std::vector<int> order(n);
        for (int i=0; i<n; ++i) order[i]=i;
        std::sort(order.begin(), order.end(), [names](int a, int b) {
            return llama_model_loader::weight_name_comparer{}(names[a], names[b]);
        });
        for (int i : order) {
            auto * t = ggml_new_tensor(ctx, static_cast<ggml_type>(input_types[i]), dims[i], shapes+4*i);
            ggml_set_name(t, names[i]);
            output_types[i] = input_types[i];
            if (llama_quant_tensor_allows_quantization(state, t)) {
                tensors.push_back(t); indices.push_back(i);
            }
        }
        std::vector<ggml_type> types(tensors.size());
        llama_quant_compute_types(state, params.ftype, tensors.data(), types.data(), tensors.size());
        for (size_t i=0; i<types.size(); ++i) output_types[indices[i]]=types[i];
        ggml_free(ctx); llama_quant_free(state); llama_model_free(model);
        return 0;
    } catch (const std::exception & e) {
        last_error=e.what();
        if (ctx) ggml_free(ctx);
        if (state) llama_quant_free(state);
        if (model) llama_model_free(model);
        return 1;
    }
}
