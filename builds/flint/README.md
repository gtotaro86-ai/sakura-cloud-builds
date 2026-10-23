# Flint BF16 CPU cloud releases

Authorized outputs: `webmp3/Sakura-Flint-Q4_K_M-GGUF` and `webmp3/Sakura-Flint-Q5_K_M-GGUF`.

The complete source is `RecursiveSI/flint`, revision `2a247be0c0aec895ccb0ac7ca92229ec6df37288`. Only actual public standard Ubuntu CPU runners may read full source shards. One bounded shard is materialized at a time, SHA256-verified, then read for the official lazy Flint conversion. Final GGUF output is a pipe into native Xet; no full model output file is stored on runner disk. No paid hardware or inference APIs are used.

The pinned llama.cpp revision is `970b51ce7c9ba5fff0726c5e9994f1c2f0e4eedc`. The C++ adapter only exposes its existing native mixed-type planner and CPU quantizer. Its complete type assignments were checked against unchanged native full-model metadata dry runs for all 779 output tensors. The checks use small genuine BF16 rows and small real norms/attention sinks. These are construction checks, not inference-quality claims.

The root workflow `.github/workflows/flint-bf16.yml` runs both variants after a push affecting this directory's `request.json`. It checks small-source checksums, verifies the HF account, builds the pinned CPU library, reruns the bounded correctness checks, and publishes complete model files with English/Chinese cards, attribution, source hashes and output manifests. The repository's existing `HF_TOKEN` secret is reused.

The root workflow `.github/workflows/flint-finalize.yml` runs automatically when the full build ends. It checks out the exact original build commit and reads only the completed jobs' logs and small Hub metadata. An authenticated, checksum-verified upload checkpoint in each build log preserves the complete source/plan/tensor receipt after native Xet commit. The finalizer can finish publication without requantizing or downloading model payloads. If a pre-checkpoint failure is a recognized transient connection error, at most one retry of failed jobs is requested. Quota, billing, disk, validation and unknown errors stop; billing is never enabled.

Both workflows are GitHub-hosted and run independently of ChatGPT usage limits, browser state and the user's PC. A ChatGPT quota reset time requires no special restart. No model artifacts or persistent model caches are uploaded to GitHub. The small provenance checkpoint contains checksums and 512 GGUF header bytes, not model weights or credentials.

`control_checks.py` uses clearly synthetic manifests solely to check refusal of missing proofs, wrong sizes, truncated/duplicated checkpoints and inappropriate retries. `evidence/cloud-cpu-checks.json` records the actual bounded Cloud development checks separately. The final release SHA256 is generated only from the actual completed output stream.

Source and generated header metadata were pinned without downloading full weights in the Work environment. The source shards total 79,575,138,056 bytes. Frozen output sizes are 24,166,841,312 bytes for Q4_K_M and 28,314,254,304 bytes for Q5_K_M, including GGUF metadata.
