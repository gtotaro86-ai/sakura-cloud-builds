"""Release documentation rendered from measured final file sizes."""
from source import inventory, REV, LLAMA
from source import ROOT
import json

def local_reference_section():
    path=ROOT/'local-reference.json'
    if not path.exists():return '', ''
    ref=json.loads(path.read_text());artifact=ref['artifact'];source=artifact['source']
    english=f'''## Local performance reference

The model owner reported the following results for a **separate local artifact**, `{artifact['file']}` ({artifact['size']:,} bytes / {artifact['size']/2**30:.3f} GiB). It was requantized from the official Q8_0 file with an importance matrix; its 20 block-40 MTP tensors were reported as byte-identical Q8_0/F32 copies. These figures apply to that reference artifact. They do not establish performance for the BF16-generated files in this repository.

| Build detail | This BF16 cloud release | Owner's local reference |
|---|---|---|
| Language-weight input | Original BF16 safetensors | Official Q8_0 GGUF |
| Importance matrix | None; standard llama.cpp quantization | 96 x 512-token mixed calibration |
| MTP precision | Standard mixed quantization policy, plus retained F32 tensors | Original Q8_0/F32, 20 tensors reported unchanged |
| Reference file size | See this release's size table | {artifact['size']/1e9:.3f} GB |

Reported configuration: **Ryzen AI Max+ 392 / Radeon 8060S, 64 GB unified memory, Windows, Vulkan**, llama.cpp build **11594** (`e0bcaccbe`, Clang 20.1.8), full GPU offload, Flash Attention, a **65,536-token context allocation**, one sequence, and `draft-mtp` with at most two draft tokens.

| Metric / workload | Owner-reported result |
|---|---:|
| Prompt processing | 850-1,050 tokens/s |
| TTFT, short prompts | 0.6-0.9 s |
| TTFT, 8,500-token input | 8-12 s |
| Reasoning decode | 59-65 tokens/s |
| Reasoning MTP acceptance | 58-64% |
| Python generation decode | 64-77 tokens/s |
| Python MTP acceptance | Up to 87.5% |
| Python decode gain over MTP off | 13-34% |
| Model and KV-cache UMA usage | Approximately 27.8 GB |

The owner reported successful loading, web UI, OpenAI-compatible chat API, streaming and tool/JSON calls for text workloads. The reported calibration covered all 40 main blocks and all 256 experts per MoE layer, with zero unseen experts.

<details>
<summary>Local reference provenance and exact commands</summary>

Q8_0 source: [{source['repo']}](https://huggingface.co/{source['repo']}/tree/{source['revision']}), revision `{source['revision']}`.

Source SHA256: `{source['sha256']}`.

Local reference SHA256: `{artifact['sha256']}`.

Importance matrix: `Sage-A1.5-Q8_0-calib-mix-96x512-main.imatrix.gguf`, 384,359,808 bytes, from `calib_mix_v1.txt` (96 chunks of 512 tokens). Commands below reproduce the reported local configuration and refer to that runtime build.

```cmd
{ref['quantizer']['command']}
```

```cmd
{ref['runtime']['command']}
```

</details>

'''
    chinese=f'''### 本地性能参考

模型所有者报告了独立本地文件 `{artifact['file']}` 的文本运行数据：该文件由官方 Q8_0 通过重要性矩阵重新量化，大小 {artifact['size']/1e9:.3f} GB，20 个 MTP 张量据报告保持原始 Q8_0/F32 字节不变。本仓库从原始 BF16 生成，未使用该重要性矩阵，MTP 按标准混合精度规则处理。下列数据仅对应该本地参考文件。

硬件为 Ryzen AI Max+ 392、Radeon 8060S、64 GB 统一内存，Windows/Vulkan，llama.cpp build 11594（`e0bcaccbe`），全 GPU 卸载、Flash Attention、65,536 token 上下文分配、单序列、`draft-mtp` 最多 2 个草稿 token。

| 指标 | 所有者报告 |
|---|---:|
| 提示处理 | 850-1,050 tokens/s |
| 短提示首 token 延迟 | 0.6-0.9 秒 |
| 8,500-token 输入首 token 延迟 | 8-12 秒 |
| 推理解码 | 59-65 tokens/s，MTP 接受率 58-64% |
| Python 代码生成 | 64-77 tokens/s，MTP 接受率最高 87.5% |
| Python 解码相对关闭 MTP 的提升 | 13-34% |
| 模型与 KV 缓存内存占用 | 约 27.8 GB UMA |

所有者报告文本加载、Web UI、聊天 API、流式输出和工具/JSON 调用正常。校准据报告覆盖全部 40 个主模型块、每个 MoE 层全部 256 个专家，未见专家数为 0。

'''
    return english,chinese

def readme(plan, sha256):
    variant=plan['variant']; repo=f'webmp3/Sakura-Sage-A1.5-{variant}-GGUF'
    filename=f'Sakura-Sage-A1.5-{variant}.gguf'
    original=sum(x['size'] for x in inventory()['weights'])
    q8=37_802_150_112; projector=inventory()['projector']['size']; size=plan['size']
    gb=lambda n:f'{n/1e9:,.3f} GB'
    gib=lambda n:f'{n/2**30:,.3f} GiB'
    saving=100*(1-size/q8)
    local_en,local_zh=local_reference_section()
    return f'''---
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
- {variant.lower()}
---

# Sakura Sage-A1.5 {variant} GGUF

{variant} quantization of [InternScience/Sage-A1.5](https://huggingface.co/InternScience/Sage-A1.5), built directly from the original **BF16 safetensors**, including its built-in MTP head. The language weights were converted with llama.cpp and quantized on CPU.

## Source and format

| Item | Details |
|---|---|
| Original model | InternScience/Sage-A1.5 |
| Source revision | `{REV}` |
| Source precision | BF16, with original F32 tensors where provided |
| Output | {variant}, standard mixed tensor quantization |
| Architecture | Qwen3.5 MoE / `qwen35moe` |
| Blocks | 40 language blocks + 1 MTP block |
| Experts | 256 per MoE layer, 8 selected per token |
| Tokenizer and chat template | From the pinned original checkpoint |
| Conversion and quantizer | llama.cpp `{LLAMA}` |
| License | Apache 2.0 |

This is a full language-model quantization. The official Q8_0 GGUF was **not** used as the input to the language-weight quantizer. Norms, routing and other tensors follow the upstream converter and quantizer's precision rules; {variant} does not mean every tensor uses the same bit width.

## Size comparison

| Distribution | Scope | Size |
|---|---|---:|
| Original BF16 checkpoint | Complete upstream safetensors, including vision and MTP | {gb(original)} / {gib(original)} |
| Official Sage-A1.5 Q8_0 GGUF | Language model + MTP | {gb(q8)} / {gib(q8)} |
| **Sakura {variant}** | **Language model + MTP** | **{gb(size)} / {gib(size)}** |
| Sage-A1.5 mmproj | Separate, unchanged official projector | {gb(projector)} / {gib(projector)} |

The main {variant} file is **{saving:.2f}% smaller** than the official Q8_0 main file, saving **{gb(q8-size)}**. With the optional projector, this package totals **{gb(size+projector)}**. File sizes are not a promise of runtime memory usage; context and backend buffers require additional memory.

## Files

| File | Purpose |
|---|---|
| `{filename}` | Quantized language model, with MTP weights |
| `Sage-A1.5-mmproj.gguf` | Original official vision projector, copied unchanged |
| `SHA256SUMS` | SHA256 checksums for both GGUF files |
| `build-manifest.json` | Pinned provenance, tensor types, sizes and checksums |

Main file SHA256: `{sha256}`.

## Usage

Use a recent llama.cpp build supporting Qwen3.5 MoE and its included MTP tensors. The conversion revision above is the exact format reference. Backend acceleration and MTP usage depend on the runtime and its settings.

```bash
hf download {repo} {filename} --local-dir .
llama-server -m {filename} -c 32768
```

For image input, also download `Sage-A1.5-mmproj.gguf` from this repository and pass `--mmproj Sage-A1.5-mmproj.gguf` to a compatible multimodal runtime. Text-only use does not require the projector. The original model supports a longer context; choose a context size that fits your available memory.

The original model's recommended prompt format and capabilities are documented in the [upstream model card](https://huggingface.co/InternScience/Sage-A1.5). The [official Q8_0 distribution](https://huggingface.co/InternScience/Sage-A1.5-Q8_0-GGUF) supplies the unchanged projector and the Q8_0 size reference.

{local_en}\
<details>
<summary>中文说明</summary>

## Sakura Sage-A1.5 {variant} GGUF

本仓库基于 [InternScience/Sage-A1.5](https://huggingface.co/InternScience/Sage-A1.5) 的原始 **BF16 safetensors** 直接转换并进行 CPU 量化，包含模型自带的 MTP 预测头。语言模型权重并非从 Q8_0 文件再次量化。

源版本为 `{REV}`，llama.cpp 版本为 `{LLAMA}`。架构为 `qwen35moe`，包含 40 个语言模型块及 1 个 MTP 块。{variant} 使用上游标准混合精度规则，归一化、路由等张量保留上游规定的精度。

| 文件 | 大小 |
|---|---:|
| 官方 Q8_0 主模型（含 MTP） | {gb(q8)} |
| 本仓库 {variant} 主模型（含 MTP） | {gb(size)} |
| 原样保留的官方视觉投影器 | {gb(projector)} |

主模型比官方 Q8_0 主文件减少 **{saving:.2f}%**，节省 **{gb(q8-size)}**。这些数字表示文件大小；实际运行还需要上下文及后端缓冲区内存。

使用支持 Qwen3.5 MoE 的近期 llama.cpp 版本。纯文本推理只需主 GGUF 文件；图像输入还需要 `Sage-A1.5-mmproj.gguf`。MTP 的启用方式取决于所用运行时。分词器和聊天模板来自固定版本的原始模型。许可证为 Apache 2.0。

{local_zh}\
</details>
'''
