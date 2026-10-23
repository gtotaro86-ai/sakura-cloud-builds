# Publish the completed MiniMax Q2 build independently

The four Q2_K jobs in build run `39908350992` have succeeded. This small metadata-only publisher releases their complete model without waiting for Q3/Q4/Q5, using the exact original frozen build commit `35fe24364aede18b80e7789cff74ae5a4f2bb231`.

It verifies the original qualifier, assets and all four successful jobs; reads their bounded immutable HF staging receipts; checks original build identity and every staged file's Hub size/LFS SHA256; then invokes the original full-proof assembler and publisher. All 48 original source hashes, output tensor proofs, independent/native upload identities, projector and original DSpark identity must match. No model payload is read or requantized. The English/Chinese model card, comparisons, exact hashes, original license and required runtime patch/build script come from the frozen original release code. The exact pinned upstream runtime MIT license is also included.

Root workflow `minimax-release-ready.yml` runs once on this project's `request.json` change or a manual dispatch. It uses the existing HF_TOKEN on a public standard ubuntu-24.04 CPU runner, with a 20-minute limit and no paid resources or model artifacts. It does not alter/restart the active full build. A complete variant can be released while other variants continue; the original finalizer remains idempotent for it. A single completed shard cannot substitute for the four-file model.

The release API adapter leaves `src_repo_id` unset for same-repository cross-branch copies. This avoids the cross-repository LFS duplication endpoint while preserving the pinned source revision and all content hashes. Other repositories continue using explicit source repository/type.
