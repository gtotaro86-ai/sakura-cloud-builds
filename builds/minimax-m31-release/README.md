# Publish a complete MiniMax variant independently

This metadata-only publisher can release a complete Q2_K, Q3_K_M, Q4_K_M or Q5_K_M variant from run `39908350992`, using exact frozen build commit `35fe24364aede18b80e7789cff74ae5a4f2bb231`. `request.json` selects one authorized policy; expected shard counts are 4, 5, 6 and 7 respectively. Q2 was already released and must not be rebuilt.

It requires successful qualification, assets and every selected shard job, reads bounded immutable HF staging receipts, verifies exact build identity and each Hub size/LFS SHA256, and invokes the original full-proof assembler/publisher. All 48 original source hashes, output tensor proofs, native upload identities, F16 projector and original DSpark must match. No model payload is downloaded or requantized. English/Chinese cards, provenance, comparisons and required runtime patch come from the original frozen release code; the exact upstream runtime MIT license is included.

Root workflow `minimax-release-ready.yml` runs on this request change or manual dispatch with the existing HF_TOKEN, a public standard ubuntu-24.04 CPU runner and a 20-minute ceiling. Active full-build jobs remain unchanged. The full-build finalizer can verify any early release idempotently.

The API adapter clears src_repo_id only for same-repository cross-branch copies; true cross-repository copies retain explicit repository and model type. Frozen revisions and all identity checks remain required.
