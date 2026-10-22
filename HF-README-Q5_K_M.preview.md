---
license: apache-2.0
language:
- en
- zh
pipeline_tag: text-generation
base_model:
- InternScience/Sage-A1.5
base_model_relation: quantized
library_name: gguf
tags:
- gguf
- qwen3_5_moe
- mixture-of-experts
- sage
- sakura
- q5_k_m
---

# Sakura Sage-A1.5 Q5_K_M GGUF

Q5_K_M quantization of [InternScience/Sage-A1.5](https://huggingface.co/InternScience/Sage-A1.5), built directly from the original **BF16 safetensors**, including its built-in MTP head. The language weights were converted with llama.cpp and quantized on CPU.

## Source and format

| Item | Details |
|---|---|
| Original model | InternScience/Sage-A1.5 |
| Source revision | `0e3205cf441cf01df0c0d238cc4738d0e43b4ca7` |
| Source precision | BF16, with original F32 tensors where provided |
| Output | Q5_K_M, standard mixed tensor quantization |
| Architecture | Qwen3.5 MoE / `qwen35moe` |
| Blocks | 40 language blocks + 1 MTP block |
| Experts | 256 per MoE layer, 8 selected per token |
| Tokenizer and chat template | From the pinned original checkpoint |
| Conversion and quantizer | llama.cpp `970b51ce7c9ba5fff0726c5e9994f1c2f0e4eedc` |
| License | Apache 2.0 |

This is a full language-model quantization. The official Q8_0 GGUF was **not** used as the input to the language-weight quantizer. Norms, routing and other tensors follow the upstream converter and quantizer's precision rules; Q5_K_M does not mean every tensor uses the same bit width.

## Size comparison

| Distribution | Scope | Size |
|---|---|---:|
| Original BF16 checkpoint | Complete upstream safetensors, including vision and MTP | 71.908 GB / 66.970 GiB |
| Official Sage-A1.5 Q8_0 GGUF | Language model + MTP | 37.802 GB / 35.206 GiB |
| **Sakura Q5_K_M** | **Language model + MTP** | **25.348 GB / 23.607 GiB** |
| Sage-A1.5 mmproj | Separate, unchanged official projector | 0.899 GB / 0.838 GiB |

The main Q5_K_M file is **32.95% smaller** than the official Q8_0 main file, saving **12.455 GB**. With the optional projector, this package totals **26.247 GB**. File sizes are not a promise of runtime memory usage; context and backend buffers require additional memory.

## Files

| File | Purpose |
|---|---|
| `Sakura-Sage-A1.5-Q5_K_M.gguf` | Quantized language model, with MTP weights |
| `Sage-A1.5-mmproj.gguf` | Original official vision projector, copied unchanged |
| `SHA256SUMS` | SHA256 checksums for both GGUF files |
| `build-manifest.json` | Pinned provenance, tensor types, sizes and checksums |

Main file SHA256: `<generated-file-sha256>`.

## Usage

Use a recent llama.cpp build supporting Qwen3.5 MoE and its included MTP tensors. The conversion revision above is the exact format reference. Backend acceleration and MTP usage depend on the runtime and its settings.

```bash
hf download webmp3/Sakura-Sage-A1.5-Q5_K_M-GGUF Sakura-Sage-A1.5-Q5_K_M.gguf --local-dir .
llama-server -m Sakura-Sage-A1.5-Q5_K_M.gguf -c 32768
```

For image input, also download `Sage-A1.5-mmproj.gguf` from this repository and pass `--mmproj Sage-A1.5-mmproj.gguf` to a compatible multimodal runtime. Text-only use does not require the projector. The original model supports a longer context; choose a context size that fits your available memory.

The original model's recommended prompt format and capabilities are documented in the [upstream model card](https://huggingface.co/InternScience/Sage-A1.5). The [official Q8_0 distribution](https://huggingface.co/InternScience/Sage-A1.5-Q8_0-GGUF) supplies the unchanged projector and the Q8_0 size reference.

<details>
<summary>中文说明</summary>

## Sakura Sage-A1.5 Q5_K_M GGUF

本仓库基于 [InternScience/Sage-A1.5](https://huggingface.co/InternScience/Sage-A1.5) 的原始 **BF16 safetensors** 直接转换并进行 CPU 量化，包含模型自带的 MTP 预测头。语言模型权重并非从 Q8_0 文件再次量化。

源版本为 `0e3205cf441cf01df0c0d238cc4738d0e43b4ca7`，llama.cpp 版本为 `970b51ce7c9ba5fff0726c5e9994f1c2f0e4eedc`。架构为 `qwen35moe`，包含 40 个语言模型块及 1 个 MTP 块。Q5_K_M 使用上游标准混合精度规则，归一化、路由等张量保留上游规定的精度。

| 文件 | 大小 |
|---|---:|
| 官方 Q8_0 主模型（含 MTP） | 37.802 GB |
| 本仓库 Q5_K_M 主模型（含 MTP） | 25.348 GB |
| 原样保留的官方视觉投影器 | 0.899 GB |

主模型比官方 Q8_0 主文件减少 **32.95%**，节省 **12.455 GB**。这些数字表示文件大小；实际运行还需要上下文及后端缓冲区内存。

使用支持 Qwen3.5 MoE 的近期 llama.cpp 版本。纯文本推理只需主 GGUF 文件；图像输入还需要 `Sage-A1.5-mmproj.gguf`。MTP 的启用方式取决于所用运行时。分词器和聊天模板来自固定版本的原始模型。许可证为 Apache 2.0。

</details>
