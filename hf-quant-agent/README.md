# Native GGUF streaming worker

A portable worker for immutable Hugging Face GGUF sources. It uses exact HTTP ranges, a complete tensor catalogue, one global stock llama.cpp quantization plan and bounded native row kernels. No llama.cpp runtime patch is applied.

Complete numbered input sets are verified per original SHA-256. Output is split on tensor boundaries and streamed to native Xet ingestion without an output model file. Xet staging alone never creates a Hub release or claims runtime validation. The free public Ubuntu workflow also supports a fixed small qualification mode: every requested quant is streamed, actual generated shards are loaded by stock llama.cpp, and a real inference is checked.

Pinned runtime: b11669 / 620d2225cc436ab0b001e3200e590e3b286ef9ab. CPU runtime archives are verified by SHA-256; the small adapter is compiled against pinned upstream headers. Input tensor types F32/F16/BF16/Q8_0 are supported when required native model metadata is available. No arbitrary model repository code is executed.

A real qualification on Qwen2.5-0.5B matched all 291 tensor types and payload hashes against stock output for Q2/Q3/Q4/Q5. Separate strict HTTP streaming and split-model loading tests passed. A 77.5 GB, 21-file source catalogue was inspected through about27MB of bounded metadata reads; this was not a large-model quantization or runtime test.

The workflow uses public standard ubuntu-24.04 runners only. Staging requires an explicitly authorized HF_TOKEN secret, publisher namespace and redistributable source. Unknown formats, incomplete sets and failed integrity checks stop processing. Large-model publication requires independent validation; successful staging is not a completed release.

This public package contains the portable cloud worker. Private service deployment and client configuration are distributed separately.
