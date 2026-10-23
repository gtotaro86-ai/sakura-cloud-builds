#include "llama-ext.h"
#include "llama-model-loader.h"
#include "llama-arch.h"
#include "ggml.h"
#include <algorithm>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

#if defined(_WIN32)
#define HFQA_EXPORT __declspec(dllexport)
#else
#define HFQA_EXPORT
#endif

static thread_local std::string last_error;
extern "C" HFQA_EXPORT const char * hfqa_error() { return last_error.c_str(); }

// hparams: embedding, ffn, layers, heads, kv_heads, experts, key_width, value_width.
// shapes: four GGML dimensions per tensor; pad trailing dimensions with one.
extern "C" HFQA_EXPORT int hfqa_plan(
        int ftype, const char * architecture, const uint32_t * hparams, int n,
        const char ** names, const int64_t * shapes, const int * dims,
        const int * input_types, int * output_types) {
    ggml_context * ctx = nullptr;
    llama_model * model = nullptr;
    quantize_state_impl * state = nullptr;
    last_error.clear();
    try {
        if (!architecture || !hparams || !names || !shapes || !dims || !input_types || !output_types || n < 1 || n > 100000)
            throw std::runtime_error("invalid planner arguments");
        if (llm_arch_from_string(architecture) == LLM_ARCH_UNKNOWN)
            throw std::runtime_error("architecture is absent from the pinned stock runtime");
        if (!hparams[0] || !hparams[1] || !hparams[2] || hparams[2] > 4096 || !hparams[3] || !hparams[4] || !hparams[6] || !hparams[7])
            throw std::runtime_error("invalid model hyperparameters");
        llama_quant_model_desc desc = {architecture, hparams[0], hparams[1], hparams[2], hparams[3], hparams[4], hparams[5], hparams[6], hparams[7]};
        model = llama_quant_model_from_metadata(&desc);
        auto params = llama_model_quantize_default_params();
        params.ftype = static_cast<llama_ftype>(ftype);
        params.nthread = 1;
        if (llama_ftype_get_default_type(params.ftype) == GGML_TYPE_COUNT)
            throw std::runtime_error("invalid quantization preset");
        state = llama_quant_init(model, &params);
        ctx = ggml_init({ggml_tensor_overhead() * (static_cast<size_t>(n) + 1), nullptr, true});
        if (!ctx) throw std::runtime_error("metadata allocation failed");
        std::vector<int> order(n);
        std::unordered_set<std::string> unique_names;
        for (int i = 0; i < n; ++i) {
            if (!names[i] || !*names[i] || std::string(names[i]).size() >= GGML_MAX_NAME || !unique_names.insert(names[i]).second)
                throw std::runtime_error("invalid or duplicate tensor name");
            if (dims[i] < 1 || dims[i] > 4)
                throw std::runtime_error("invalid tensor dimensions");
            if (input_types[i] != 0 && input_types[i] != 1 && input_types[i] != 8 && input_types[i] != 10 && input_types[i] != 11 && input_types[i] != 12 && input_types[i] != 13 && input_types[i] != 14 && input_types[i] != 30)
                throw std::runtime_error("unsupported input tensor format");
            int64_t count = 1;
            for (int k = 0; k < 4; ++k) {
                const auto value = shapes[4 * i + k];
                if (value < 1 || count > std::numeric_limits<int64_t>::max() / value || (k >= dims[i] && value != 1))
                    throw std::runtime_error("invalid tensor shape");
                count *= value;
            }
            order[i] = i;
        }
        std::sort(order.begin(), order.end(), [names](int a, int b) {
            return llama_model_loader::weight_name_comparer{}(names[a], names[b]);
        });
        std::vector<ggml_tensor *> tensors;
        std::vector<int> indices;
        for (int i : order) {
            auto * tensor = ggml_new_tensor(ctx, static_cast<ggml_type>(input_types[i]), dims[i], shapes + 4 * i);
            ggml_set_name(tensor, names[i]);
            output_types[i] = input_types[i];
            if (llama_quant_tensor_allows_quantization(state, tensor)) {
                tensors.push_back(tensor);
                indices.push_back(i);
            }
        }
        std::vector<ggml_type> types(tensors.size());
        llama_quant_compute_types(state, params.ftype, tensors.data(), types.data(), tensors.size());
        for (size_t i = 0; i < types.size(); ++i) output_types[indices[i]] = types[i];
        ggml_free(ctx);
        llama_quant_free(state);
        llama_model_free(model);
        return 0;
    } catch (const std::exception & error) {
        last_error = error.what();
        if (ctx) ggml_free(ctx);
        if (state) llama_quant_free(state);
        if (model) llama_model_free(model);
        return 1;
    }
}
