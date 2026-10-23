"""Model cards describe measured files, source quantization and required runtime."""
import json
from source import ROOT,inventory,LLAMA,REV

def readme(manifest):
    inv=inventory();plans=json.loads((ROOT/'split-plans.json').read_text());plan=manifest['output'];v=plan['variant']
    asset=manifest['assets'];projector=asset['output'];ds=inv['original_dspark'];aux=projector['size']+ds['size']
    source=inv['source_total_bytes']+ds['size']
    comparisons='\n'.join(f"| Sakura {q} | {len(p['files'])} | {p['size']/1e9:.3f} | {(p['size']+aux)/1e9:.3f} | {100*((p['size']+aux)/source-1):+.2f}% |" for q,p in plans.items())
    files='\n'.join(f"| `{f['path']}` | {f['size']:,} | `{f['sha256']}` |" for f in manifest['native_files'])
    first=plan['files'][0]['path'];types=', '.join(f'{k}: {n}' for k,n in plan['types'].items())
    return f'''---
language:
- en
- zh
license: other
license_name: minimax-community
license_link: LICENSE
library_name: gguf
pipeline_tag: image-text-to-text
base_model:
- MiniMaxAI/MiniMax-M3.1-Flash
base_model_relation: quantized
tags:
- gguf
- minimax
- sakura
- mixture-of-experts
- coding
- multimodal
---

# Sakura MiniMax-M3.1-Flash {v} GGUF

CPU requantization of the official [MiniMax-M3.1-Flash](https://huggingface.co/MiniMaxAI/MiniMax-M3.1-Flash) **hybrid NVFP4/MXFP8 checkpoint**, using llama.cpp's standard mixed-type **{v}** policy. This is not a quantization from an unquantized BF16 checkpoint. No importance matrix or expert pruning is applied.

**Runtime requirement:** use the included `runtime.patch` with llama.cpp revision `{LLAMA}`. These files use the explicit architecture identifier `minimaxm31_sakura`; unchanged stock runtimes reject it. M3.1 has sparse attention in all 60 layers, including its first three dense-MLP layers. The patch preserves that distinction in tensor loading, indexer KV caching and execution. A generic older M3 runtime must not silently substitute full attention in those layers.

## Size comparison

The original 48 main safetensors shards total **{inv['source_total_bytes']:,} bytes**. The original DSpark file adds **{ds['size']:,} bytes**. The original source is already quantized: NVFP4 routed experts, MXFP8 attention paths, and retained BF16/F32 tensors.

| Checkpoint | Main GGUF shards | Main language files, GB | Including F16 projector and original DSpark, GB | Change vs official main + DSpark |
|---|---:|---:|---:|---:|
| Official hybrid safetensors + DSpark | — | — | {source/1e9:.3f} | — |
{comparisons}

GB = 10^9 bytes. Auxiliary-file totals include the same {projector['size']/1e9:.3f}-GB F16 projector and {ds['size']/1e9:.3f}-GB unchanged DSpark file in each release. Small JSON, license and runtime source files are excluded. GGUF main sizes include their metadata. The official main checkpoint includes the vision tower, which is separated into `mmproj` here. Q4/Q5 may be larger than the released low-precision source; increasing stored precision does not recover information lost during the source's quantization. These are file sizes, not complete RAM/VRAM requirements or speed/quality results.

## Released files

| File | Bytes | SHA256 |
|---|---:|---|
{files}
| `{projector['path']}` | {projector['size']:,} | `{asset['sha256']}` |
| `{ds['path']}` | {ds['size']:,} | `{ds['sha256']}` |

Download every main GGUF shard and keep the names together; load the first shard. Each main shard is under 48 GB. `SHA256SUMS` records all released weight files. `build-manifest.json` records pinned source checksums, all target tensor types, every output tensor checksum, independent stream checksums and native Xet verification.

## Source and preserved components

| Property | MiniMax-M3.1-Flash |
|---|---|
| Developer | MiniMax |
| Total / active parameters | Approximately 428B / 23B per token, developer-stated |
| Layers | 60, all using MiniMax Sparse Attention |
| Dense / MoE MLP layers | 3 leading dense MLP layers, followed by 57 MoE layers |
| Routed experts | 128 per MoE layer; 4 selected per token, plus shared expert |
| Configured context | 1,048,576 tokens |
| Modalities | Text, image and video input; text output |
| Main output tensors | {plan['tensor_count']} ({types}) |

The standard upstream MiniMax vision converter produces the separate F16 projector. The complete original DSpark safetensors checkpoint, config and index are copied unchanged. **DSpark execution is not integrated into this patched llama.cpp runtime**; the original files are retained for compatible serving implementations and future integration. No speculative speedup is claimed.

## Provenance and construction

- Source revision: [`{REV}`](https://huggingface.co/MiniMaxAI/MiniMax-M3.1-Flash/tree/{REV}).
- NVFP4 source weights are decoded with per-group FP8 scales and the released global-scale convention; MXFP8 uses E4M3 weights and UE8M0 block scales. Conversion uses BF16/F32 intermediates selected by the pinned upstream converter.
- Every complete source shard is SHA256-verified before its payload is used. Tiny verified scale/norm tensors are cached in RAM to avoid repeatedly reloading unrelated large shards.
- Main expert matrices are encoded one expert at a time; embeddings are encoded in bounded row blocks. No full decoded model or full output GGUF is materialized on disk.
- Full-model tensor type assignments match the unchanged native quantization planner. Real-row decoder checks match independent native ggml/PyTorch references. The patched all-sparse runtime passes a small synthetic CPU prefill/decode check. Those checks do not establish full-model response quality; no new benchmark scores are claimed.
- All final source/producer/upload checks must pass before an atomic model commit is published. An independent finalizer can recover a completed upload from its checksum-protected provenance checkpoint.

## Build the compatible runtime and run

Use `build-runtime.sh` alongside `runtime.patch`. It checks out the pinned upstream revision and applies the included patch. CPU is the default; supported GPU backends can be selected when building for your own hardware. No model weights are downloaded by the runtime build script.

```sh
bash build-runtime.sh
./minimax-runtime-build/bin/llama-server -m {first} --mmproj {projector['path']} -ngl 0 -c 32768 -fa on --parallel 1 --jinja --temp 1.0 --top-p 0.95
```

Flash Attention and a supported stream layout are required for this M3.1 path; unsupported settings fail instead of silently using full attention. Omit `--mmproj` for text-only use. Set context and offloading for available memory; the source's maximum context is not a claim that it fits a particular device. Preserve the original chat template and returned reasoning content. Do not add guessed DSpark flags to this runtime.

## License and attribution

Original model: **MiniMax-M3.1-Flash, Copyright (c) 2026 MiniMax**. This redistribution retains the **MINIMAX COMMUNITY LICENSE**, including its attribution, commercial-use and prohibited-use conditions. It is not Apache-2.0 or MIT model licensing. Read `LICENSE` before use. llama.cpp runtime code is separately covered by its upstream MIT license. Sakura identifies this quantization and distribution, not original model training.

<details>
<summary>中文说明（点击展开）</summary>

## Sakura MiniMax-M3.1-Flash {v}

本仓库对 MiniMax 官方发布的 **NVFP4 专家权重与 MXFP8 注意力权重混合检查点**进行 CPU 再量化，采用标准 {v} 混合精度策略，未使用 imatrix，未裁剪专家。并非由未量化的 BF16 检查点生成。主语言权重合计 **{plan['size']/1e9:.3f} GB**，分为 **{len(plan['files'])} 个 GGUF 分片**，每片小于 48 GB。另附 F16 视觉投影器及保持原样的 DSpark 权重、配置与索引。

**必须使用本仓库附带的运行时补丁及指定 llama.cpp 版本 `{LLAMA}`。** 文件架构标识为 `minimaxm31_sakura`，未打补丁的原版运行时会拒绝加载。M3.1 的全部 60 层均使用稀疏注意力，包括前三个稠密 MLP 层；不能沿用旧版 M3 对这三层的全注意力假设。补丁同时修正索引器权重加载、KV 缓存分配和执行路径，并要求开启 Flash Attention。

原始模型约 4280 亿总参数，每个 token 约激活 230 亿参数。包含 3 个稠密 MLP 层和 57 个 MoE 层，每个 MoE 层有 128 个路由专家，选择 4 个并使用共享专家。源配置上下文为 1,048,576 token。主 GGUF 保留 {plan['tensor_count']} 个张量。文件体积不是完整运行内存需求；Q4/Q5 可能比原始低精度源更大，也不能恢复源量化已损失的信息。

下载全部分片并保持文件名，加载第一个分片即可。图片和视频输入另需 `mmproj`。原始 DSpark 文件保持字节不变，但本补丁版 llama.cpp 尚未集成其执行路径，不能据此声称推测解码加速。启动命令见英文部分。

源版本为 `{REV}`。构建前检查完整源分片 SHA256；按专家和行块流式编码；独立验证输出流、原生 Xet 及 HF 已提交文件的校验值。真实行解码与原生 ggml/PyTorch 参考一致，完整张量类型与原生规划器一致，小型合成 CPU 模型可完成 prefill/decode。这些构建检查不代表完整模型的回答质量评测；本仓库不声称新的质量或速度成绩。详情见 `SHA256SUMS` 与 `build-manifest.json`。

原始模型版权属于 MiniMax，继续遵循 **MINIMAX COMMUNITY LICENSE**。请阅读其署名、商业使用及禁止用途条款；模型许可证不是 Apache 或 MIT。llama.cpp 运行时代码单独遵循上游 MIT 许可证。

</details>
'''
