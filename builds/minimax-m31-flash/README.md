# MiniMax M3.1 Flash: CPU streaming quantization and HF release

This project is authorized to build and publish Q2_K, Q3_K_M, Q4_K_M and Q5_K_M from `MiniMaxAI/MiniMax-M3.1-Flash`, frozen at `54e4e69dfc452b1e94a426c24a4fdeeaa8e3ad5e`. Target repositories are `webmp3/Sakura-MiniMax-M3.1-Flash-<policy>-GGUF`. This is requantization of the released hybrid NVFP4/MXFP8 source, not an original BF16 source. All original experts remain.

## Output plan

| Policy | Main GGUF bytes | Shards |
|---|---:|---:|
| Q2_K | 155,983,334,400 | 4 |
| Q3_K_M | 203,571,051,264 | 5 |
| Q4_K_M | 258,095,221,280 | 6 |
| Q5_K_M | 302,930,383,168 | 7 |

Each shard is below 48 GB. Every release also has the 1,732,284,000-byte F16 vision projector and unchanged 2,301,867,426-byte original DSpark file/config/index. The official main source is already quantized and totals 250,116,419,880 bytes, including vision. Q4/Q5 are larger than that source; larger precision cannot restore information lost in source quantization. Sizes are not RAM or VRAM requirements. Original MiniMax Community License is retained; runtime code has separate upstream MIT licensing.

## Compatible runtime is required

The pinned llama.cpp revision is `970b51ce7c9ba5fff0726c5e9994f1c2f0e4eedc` with `runtime.patch`. M3.1 uses sparse attention in all 60 layers, independently of the first three dense MLP layers. The patch corrects first-sparse-layer tensor allocation, indexer KV cache and execution, and refuses unsupported Flash Attention/stream settings. The explicit `minimaxm31_sakura` architecture makes unmodified stock runtimes reject files instead of silently using old M3 full-attention assumptions. `build-runtime.sh` is distributed with every release. Original DSpark weights are preserved, but DSpark execution is not implemented in this runtime.

## Workflow and resource limits

Root workflow `.github/workflows/minimax-m31.yml` starts on a change to this project's `request.json` or a manual dispatch. It uses only public standard `ubuntu-24.04` CPU runners with repository secret `HF_TOKEN`; no paid runner/GPU/inference API and no model artifacts. Existing project credentials are reused. Model payloads cannot be downloaded by setting a local convenience flag: the cache requires the actual GitHub-hosted public Linux runner context. Do not manufacture that context in Work or on a user PC.

1. Qualification builds the pinned patch, compares genuine bounded NVFP4/MXFP8 row decodes to independent ggml/PyTorch references, checks deterministic native Q2–Q5 encoding, all 1,188 native target tensor types and split-header loading, and runs a small synthetic all-sparse CPU prefill/decode fixture. It reads no full original model. These are construction checks, not full-model inference benchmarks.
2. Assets stage the converted projector and unchanged original DSpark on an `assets` branch in the Q2 repository. Model `main` remains unreleased.
3. Twenty-two independently recoverable shard jobs run with at most two in parallel. Each has its own six-hour standard-job ceiling. One full original shard (largest 10,213,649,632 bytes) is SHA256-verified on runner disk before use; tiny verified scales/norms remain in RAM. Source buffers are bounded to four 64-MiB ranges. Each range requires exact HTTP 206/Content-Range/length, with HTTP 200 rejected before reading. Source request reservations, including failures/retries, are capped at 750 GB per job; qualification has a 100-MB header/row budget. Small source metadata is separately pinned and hash checked.
4. Experts are decoded and quantized individually; embeddings use row blocks. Final GGUF bytes flow through an 8-MiB pipe into native hf-xet 1.7.0 ingestion; no complete output GGUF is written to disk. Native Xet SHA256, independently computed pipe SHA256 and the producer's file/tensor receipts must agree. Fully verified shards and small provenance are committed only to the build staging branch. Quota/disk/correctness failures stop rather than enabling billing.
5. `.github/workflows/minimax-m31-finalize.yml` runs independently of the chat/PC after completion, using the exact original build commit. It reconstructs completed-upload checkpoints from bounded job logs if needed, verifies staged file hashes and the union of all 48 original source hashes, then atomically commits each complete main release with English/Chinese card, actual file comparison/hashes, license, provenance and required runtime. It may retry only transient failed jobs once. Successful staged shards are reused. Incomplete variants are not released.

The qualification job reports existing public-model storage metadata where the supported HF API exposes it, refuses private targets, and prints the exact additional logical weight bytes. The supported account API exposes no guaranteed remaining free-public byte balance; the report explicitly marks that balance as unknown and does not claim capacity is confirmed. Free HF public storage is best-effort. Approximately 937 GB of new logical weight-file content across the four repositories needs actual account storage acceptance; deduplication does not prove quota neutrality. Do not assume unlimited free storage. Nothing in preparation proves a full build or published release; report actual Actions IDs and final HF hashes.

## Code integrity and handoff

`SHA256SUMS` covers small project code, frozen plans and evidence. `source-inventory.json` pins original sizes/LFS hashes and small metadata hashes. `expected-plans.json`, `split-plans.json` and `asset-plan.json` freeze the exact constructor outputs. `control_checks.py` uses synthetic proofs to test guard rejection and checkpoint integrity; it never creates publication evidence. Runtime/decoder evidence under `evidence/` explicitly labels local CPU and synthetic context. No full original weights were read in Work.

New agents should read this file and `AGENTS.md`, verify hashes, inspect actual completed workflow jobs and staged/released manifests, and continue from their actual state. Do not repeat Sage/Flint releases, overwrite unrelated projects, expose credentials, add private builder links to public cards, or infer model quality from tiny row errors. Launching a request is an actual release action already authorized for these four MiniMax targets, not for other models.
