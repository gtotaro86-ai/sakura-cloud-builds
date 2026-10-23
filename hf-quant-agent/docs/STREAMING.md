# Native GGUF streaming

Oracle stores job state and immutable source manifests. Large source files are never materialized on Oracle. The worker receives pinned Hugging Face filenames, lengths and SHA-256 values; source data is read only through exact HTTP byte ranges.

The native worker obtains one complete tensor catalogue, asks stock llama.cpp b11669 (`620d2225cc436ab0b001e3200e590e3b286ef9ab`) for the global Q2/Q3/Q4/Q5 type plan, then processes bounded rows. The separate C++ adapter calls upstream staging APIs and native GGML kernels. It does not modify llama.cpp. A one-tensor-at-a-time *type planner* is deliberately avoided because it can change K_M selection.

Input: one GGUF or a complete numbered GGUF source set, F32/F16/BF16/Q8_0 tensors. Ordinary supported architectures must expose the required uniform native hyperparameters. Unknown architectures, unsupported tensor types and inconsistent source sets fail closed. This path does not convert arbitrary Safetensors checkpoints or implement Heretic.

HTTP reads are capped at 8 MiB. One in-memory read window is retained; old source-shard windows are released. The parser has a 64 MiB metadata budget per source and a 100,000-tensor aggregate limit. It rejects servers that ignore Range. The complete original SHA-256 is independently checked for every source file, including headers, gaps and trailing bytes.

Output is emitted directly in chunks. Native GGUF split metadata and tensor-boundary output shards keep individual output files bounded (default 4 GiB). A tensor itself cannot be split across separate GGUF files by this implementation; if one tensor plus metadata exceeds the configured per-file cap, the worker stops instead of producing an invalid split. The cap is never allowed above 50,000,000,000 bytes.

The Xet adapter ingests streamed bytes without an output model file. Native Xet SHA-256/size must agree with the independent producer. Ingested objects remain unreferenced CAS data: this worker **does not create a Hub commit or claim a validated release**. Publication requires separate validation receipts and a deliberate final commit. A source-integrity failure cannot produce a partial model repository.

## Real qualification completed on 2026-10-23

- Small Qwen2.5-0.5B Q8 source: all 291 tensor types and every tensor payload SHA-256 match stock b11669 for Q2_K, Q3_K_M, Q4_K_M and Q5_K_M.
- Real HTTPS source stream: 675,710,816 source bytes, every strict HTTP range checked, full SHA-256 verified. Q4 output was discarded without a source/output model file. Peak Python RSS about 78.4 MiB. Unbuffered qualification made 570 range requests; the production reader adds a bounded window to reduce request overhead.
- Two output shards under a 256 MiB test cap were loaded together by unchanged llama-cli b11669 and returned “Paris.” This small fixture test materialized output shards for runtime validation only.
- A real 72B source set was inspected: 77,530,000,800 bytes in 21 files, 963 tensors; only 27,262,976 bytes were transferred for metadata inspection. No 72B weights were quantized or published. Its `other` license blocks automatic redistribution.

The evidence is separate from mock/unit tests. No large-model runtime inference, end-to-end GitHub streaming release or automatic new-model converter qualification is claimed by these results.

## GitHub worker

The new workflow uses public `ubuntu-24.04` standard runners only, bounded concurrency and a 350-minute timeout. It cannot activate paid runners. `proof` mode uses the fixed small source and actually loads generated output shards. `stage` mode requires explicit source license clearance and an existing authorized HF publisher target; it ingests Xet objects and retains receipts, without a Hub commit.

The GitHub controller needs an `actions:write` credential configured on Oracle before automatic dispatch can work. Existing specialized Sakura workflows have fixed model requests and, in some cases, patched runtimes. They must not receive arbitrary model URLs or be relaunched merely because a historical run succeeded.

## Limits still requiring qualification

- Source-shard and output-shard handling is implemented; a real large source-to-Xet-to-final-release run remains to be qualified.
- An interrupted whole output stream may have to re-read data. Xet deduplicates already ingested data, but durable CPU-level per-shard skip/recovery is not yet qualified.
- Full runtime inference for models exceeding a free worker's RAM is not silently declared successful. Such staging results remain awaiting validation.
- GPU-backed Heretic and arbitrary Safetensors conversion are separate, unconfigured worker capabilities.
