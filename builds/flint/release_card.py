"""English and Chinese model cards from frozen source and measured output."""
import json
from source import ROOT,inventory,REV,LLAMA

def readme(plan,sha):
    allplans=json.loads((ROOT/'expected-plans.json').read_text())
    variant=plan['variant'];inv=inventory();size=plan['size'];source=inv['source_total_bytes']
    table='\n'.join(f"| Sakura {v} | {p['size']:,} | {p['size']/2**30:.3f} | {100*(1-p['size']/source):.2f}% |" for v,p in allplans.items())
    types=', '.join(f'{k}: {v}' for k,v in plan['types'].items())
    return f'''---
language:
- en
- zh
license: apache-2.0
library_name: gguf
pipeline_tag: text-generation
base_model:
- RecursiveSI/flint
base_model_relation: quantized
tags:
- gguf
- flint
- sakura
- mixture-of-experts
- code
- tool-use
---

# Sakura Flint {variant} GGUF

**Direct CPU quantization of [RecursiveSI/flint](https://huggingface.co/RecursiveSI/flint) from its original BF16 checkpoint.** This repository contains the complete text model as a single standard GGUF file, including all 49 layers and all 256 routed experts in each MoE layer. No expert pruning is applied. Quantization uses llama.cpp's standard mixed-precision **{variant}** policy, without an importance matrix.

## Files and comparison

| Model / format | Weight-file bytes | GiB | Reduction from original BF16 shards |
|---|---:|---:|---:|
| Original Recursive Flint BF16 safetensors | {source:,} | {source/2**30:.3f} | — |
{table}

Source totals include safetensors headers. GGUF sizes include tokenizer, configuration metadata and GGUF headers. These are file sizes, not total RAM/VRAM requirements; the runtime also needs KV cache and compute buffers. The smaller file size does not establish a quality or speed advantage.

| File | Bytes | SHA256 |
|---|---:|---|
| `Sakura-Flint-{variant}.gguf` | {size:,} | `{sha}` |

`SHA256SUMS` and `build-manifest.json` record the exact released output and source provenance. The manifest contains all original shard checksums, the complete tensor/type plan, per-tensor output checksums and the final producer checksum. The publisher independently hashes the stream, compares it with native Xet's checksum and verifies the committed Hub file's size and SHA256.

## Original model

| Property | Recursive Flint |
|---|---|
| Developer | Recursive Superintelligence, Inc. |
| Total parameters | 39,785,287,456; approximately 39.8B |
| Active parameters | Approximately 2.5B per token, using the developer's convention |
| Layers | 49: 2 dense, then 47 MoE |
| Experts | 256 routed experts per MoE layer, 6 selected per token, plus a shared expert |
| Attention | 35 sliding-window and 14 full-attention layers |
| Sliding window | 513 tokens |
| Configured maximum context | 262,144 tokens, shared by prompt and output |
| Modalities | Text input and output |

This checkpoint has its own Flint architecture, tokenizer, learned attention sinks, sliding/full-attention layout and tool-use chat template. It is not a Sage or Qwen checkpoint. No vision projector or external draft model is required.

## Provenance and method

- Original weights: [`RecursiveSI/flint`](https://huggingface.co/RecursiveSI/flint/tree/{REV}), revision `{REV}`.
- Original source: 15 BF16 safetensors shards. Each complete shard is SHA256-verified before its tensor data are used.
- Converter and CPU quantizer: llama.cpp revision `{LLAMA}`, using its native Flint converter and native quantization type planner.
- Output: {plan['tensor_count']} GGUF tensors; {types}.
- Input precision: original BF16, rather than requantization of a lower-precision GGUF.
- Importance matrix: none. `K_M` is a mixed-type policy; individual tensor types are recorded in the manifest.
- Construction: bounded source-shard cache and streaming GGUF output; no complete BF16 or GGUF intermediate file is materialized on the builder.

The publisher checks the exact planned tensor types against an independent unchanged native llama.cpp full-model metadata dry run before the full build. The source tensors and produced rows are validated during construction. No inference-quality or performance scores are claimed for these files.

## Run with llama.cpp

Use a current llama.cpp build with Flint support and hexadecimal Jinja string escapes, or build revision `{LLAMA}`. Older runtimes may not recognize this architecture or chat template.

```sh
llama-server -m Sakura-Flint-{variant}.gguf -ngl 99 -c 32768 --jinja --temp 1.0 --top-p 0.95 --top-k 20
```

On Windows, use `llama-server.exe`; a compatible Vulkan build can use `--device Vulkan0` on AMD hardware. Select a context that fits available memory. The 262,144-token source configuration is not a claim that maximum context fits every device.

For tool use, preserve the checkpoint's chat template and the returned reasoning content in conversation history. The developer recommends an output budget of at least 32,768 tokens for hard reasoning tasks when sufficient context and memory are available.

## License and attribution

Original model by Recursive Superintelligence, Inc.; [Apache 2.0](https://huggingface.co/RecursiveSI/flint/blob/{REV}/LICENSE). This quantization preserves the source license. Sakura identifies this redistribution and quantization, not a change to the original model's training or capabilities. See `LICENSE` and `NOTICE`.

<details>
<summary>中文说明（点击展开）</summary>

## Sakura Flint {variant}

本仓库由 [RecursiveSI/flint](https://huggingface.co/RecursiveSI/flint) 的**原始 BF16 权重直接进行 CPU 量化**，采用 llama.cpp 标准 {variant} 混合精度策略，未使用重要性矩阵（imatrix），未裁剪专家。保留全部 49 层及每个 MoE 层的 256 个路由专家。

| 格式 | 文件字节数 | GiB |
|---|---:|---:|
| 原始 BF16 safetensors | {source:,} | {source/2**30:.3f} |
| 本仓库 {variant} GGUF | {size:,} | {size/2**30:.3f} |

文件体积减少 **{100*(1-size/source):.2f}%**。这些数字是权重文件大小，不是完整运行内存需求；还需为 KV 缓存和计算缓冲区预留空间。

Flint 约有 398 亿总参数，每个 token 约激活 25 亿参数（按开发者口径）。包含 2 个稠密层、47 个 MoE 层；每个 MoE 层从 256 个路由专家中选择 6 个，并使用共享专家。注意力结构包括 35 个滑动窗口层和 14 个全注意力层，滑动窗口为 513 token。源配置最大上下文为 262,144 token，并不表示任意设备都能容纳最大上下文。

源版本：`{REV}`。llama.cpp 版本：`{LLAMA}`。全部 15 个原始分片先验证 SHA256，再用于构建。输出共 {plan['tensor_count']} 个张量（{types}）。构建时分别验证生产端、上传流、原生 Xet 和 Hugging Face 已提交文件的大小与校验值。具体来源、张量类型和校验值见 `build-manifest.json` 与 `SHA256SUMS`。

请使用支持 Flint 架构及十六进制 Jinja 转义的新版 llama.cpp。可参考上方启动命令；AMD Windows 用户可使用兼容的 Vulkan 版本。保留原始聊天模板及对话中的推理内容。无需视觉投影器或外部草稿模型。本仓库不声称新的推理质量或速度成绩。

原始模型由 Recursive Superintelligence, Inc. 发布，采用 Apache 2.0 许可证；本量化遵循该许可证。Sakura 为本次量化与分发的名称。

</details>
'''
