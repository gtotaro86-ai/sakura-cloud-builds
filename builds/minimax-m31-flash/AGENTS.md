# MiniMax M3.1 Flash release authorization

- The user explicitly authorized construction and HF publication of Q2, Q3, Q4 and Q5 for MiniMaxAI/MiniMax-M3.1-Flash. Interpret the typo a4 as Q4. Planned standard policies are Q2_K, Q3_K_M, Q4_K_M, Q5_K_M.
- Only existing public standard ubuntu-24.04 CPU resources, no paid GPUs/runners/APIs, and existing HF_TOKEN are authorized. Full source weights stay off Work and the user PC.
- Preserve source license, all experts and attention components. Publication must identify this as requantization of the released hybrid NVFP4/MXFP8 checkpoint, not original BF16.
- M3.1 uses sparse attention in all layers. Stock M3 code treats the leading three dense layers as full attention. This project uses a pinned patch and explicit minimaxm31_sakura architecture identifier so unpatched stock runtimes reject the files rather than silently using incorrect attention.
- The CPU synthetic runtime and real-row decoder checks do not constitute a full-model inference/quality benchmark. Do not fabricate one.
- User-facing HF cards are English with collapsible Chinese and have no private builder GitHub links or a test-scope section.
- Stop on quota, disk, correctness and unknown failures. Do not enable billing.
