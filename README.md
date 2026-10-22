# Sakura Sage BF16 CPU builds

Build Q4_K_M and Q5_K_M directly from the pinned original InternScience/Sage-A1.5 BF16 checkpoint. This build repository contains small sources and manifests only. It contains no model weights or credentials.

The release cards are English with a collapsible Chinese translation. They include source revisions, architecture, MTP, exact final sizes, comparison with the official Q8_0 distribution, usage and licensing. They contain no links to a personal builder repository and no test-scope section.

## Execution

- An actual public standard GitHub-hosted `ubuntu-24.04` runner is required for complete source reads. No larger, GPU or self-hosted runners are selected. Each job has a 350-minute limit.
- Jobs run on CPU, separately for Q4_K_M and Q5_K_M. The original BF16 checkpoint is the language-weight source; the Q8_0 main file is not requantized.
- Only one original safetensors shard, at most about 5.37 GB, is materialized at a time on the runner. Every load checks its complete upstream SHA256 before tensor access. Exact HTTP 206 ranges are mandatory. The overall per-job source-read budget is 450 GB, including repeated shard loads and reserved failed reads.
- The pinned official converter performs name mapping, expert stacking, norm transforms, linear-attention head ordering, tokenizer export and MTP export lazily. An external adapter uses upstream llama.cpp's global mixed-type planner. Quantized chunks are validated before output.
- Generated GGUF bytes go through a pipe into bounded native Xet upload streams. The full GGUF is not materialized on runner disk or the user's computer.
- Conversion and upload use separate Python environments because the upstream conversion dependency and the pinned upload protocol require different huggingface-hub versions.
- The optional official vision projector is copied server-side, unchanged. No GPU inference or paid model API is used.
- Targets use public HF storage. The main model pointer and final card are committed only after the generated file, original-source receipts and native Xet checksum pass. Both GGUF hashes are checked again against the final Hub commit. Sufficient free public storage is not assumed; a quota refusal must stop the build.

## One-time repository setup

Create a dedicated **public** repository with an initial README, then grant the ChatGPT GitHub connector access to it. Add an Actions repository secret named `HF_TOKEN` whose token has permission to create and write the two `webmp3/Sakura-Sage-A1.5-*` target repositories. Never paste tokens into chat, code or workflow logs.

The dedicated repository's `request.json` push starts both jobs. Alternatively, use Actions -> Sage BF16 CPU quantization and release -> Run workflow. Do not add a scheduled trigger or run the request repeatedly. The publisher refuses to overwrite existing model contents.

The full Sage workflow must still be qualified in its actual GitHub runner context; preparation alone is not a completed model build. A failed job does not prove a successful release. Check the actual run logs, final model commits and recorded hashes.

## Model-card refresh

`Update Sage model cards` runs after a successful full build and can also be started manually. It verifies both published model manifests and remote GGUF size/SHA256 values, then updates only the English/Chinese README and `local-reference.json`. The original GGUF hashes must remain unchanged. No weights are downloaded or requantized.

The owner-reported local reference is a separate Q8-to-Q5_K_M importance-matrix build with Q8/F32 MTP tensors retained. Its performance numbers must not be assigned to the BF16 cloud builds. The local output and raw measurement logs have not been independently inspected here. The reported Q8 source file's size and LFS SHA256 match its pinned public HF revision.
