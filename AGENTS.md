# Build repository conventions

- Reuse this existing public build repository for subsequent model projects. Put new project code in `builds/<model-name>/`; keep GitHub Actions workflow YAML files in the root `.github/workflows/` directory. Reuse the repository Actions secret `HF_TOKEN`. Do not ask the user to create a repository and configure credentials again for each project.
- The existing Sage project remains at the repository root for its current release. Do not reorganize an active release while jobs are running.
- Only public standard `ubuntu-24.04` GitHub-hosted CPU runners are authorized. No paid runners, GPUs, inference API calls, or model artifact storage. Stop on quota failures rather than enabling billing.
- Full weights may be read only in the real GitHub-hosted build context. Never manufacture `GITHUB_ACTIONS` or runner identity variables to label a Work/local run as a GitHub run.
- Keep original source revisions pinned and validate exact HTTP 206 ranges before reading model response bodies. Reject whole-file HTTP 200 responses before reading their bodies. Source budgets include failed reserved reads.
- Keep model files off the user's PC and this Work environment. Use bounded source shards on runner disk and stream final GGUF output through native Xet. Verify original source SHA256, producer output SHA256, Xet SHA256 and final Hub file metadata.
- Do not log tokens, copy secrets into code, or publish personal builder GitHub links in HF model cards.
- Public model cards are English with collapsible Chinese, source provenance, exact final sizes, comparisons, usage and license. The user requested no test-scope section. Do not invent benchmark or inference-quality results.
- A model release requires user authorization. Current authorization covers the original BF16 Sage-A1.5 and RecursiveSI/flint sources and their Q4_K_M/Q5_K_M releases in the named webmp3 repositories. Flint targets: webmp3/Sakura-Flint-Q4_K_M-GGUF and webmp3/Sakura-Flint-Q5_K_M-GGUF; the user explicitly requested autonomous construction and release. Do not infer authorization for unrelated models.
- Preserve bundled MTP weights. Do not replace them with external draft models.
- Report actual run IDs, failures and final published file hashes. Prepared scripts and successful metadata checks are not completed full model builds.

- Additional current user authorization: original released MiniMaxAI/MiniMax-M3.1-Flash hybrid checkpoint to Q2_K, Q3_K_M, Q4_K_M and Q5_K_M and corresponding webmp3 Sakura HF releases; see builds/minimax-m31-flash/AGENTS.md.
