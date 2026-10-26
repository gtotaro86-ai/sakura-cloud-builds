"""The quantization presets this system knows. The four core presets are qualified on every approved build; the extra ones are
stock llama.cpp presets that a build must have proven (tensor-for-tensor parity with its own llama-quantize) before the
GitHub runner path offers them. Per-tensor rules are not part of this list."""
from __future__ import annotations

CORE = ('Q2_K', 'Q3_K_M', 'Q4_K_M', 'Q5_K_M')
EXTRA = ('Q3_K_S', 'Q3_K_L', 'Q4_K_S', 'Q5_K_S', 'Q6_K', 'Q8_0')
ALL = CORE + EXTRA

# llama_ftype values of the stock library
FTYPE = {'Q2_K': 10, 'Q3_K_S': 11, 'Q3_K_M': 12, 'Q3_K_L': 13, 'Q4_K_S': 14, 'Q4_K_M': 15, 'Q5_K_S': 16, 'Q5_K_M': 17, 'Q6_K': 18, 'Q8_0': 7}
# Planning ratio against the BF16 size (generic, overestimates) and typical bits per weight (not measured).
RATIO = {'Q2_K': 0.28, 'Q3_K_S': 0.33, 'Q3_K_M': 0.36, 'Q3_K_L': 0.40, 'Q4_K_S': 0.43, 'Q4_K_M': 0.47, 'Q5_K_S': 0.52, 'Q5_K_M': 0.56, 'Q6_K': 0.62, 'Q8_0': 0.80}
BITS = {'Q2_K': 2.96, 'Q3_K_S': 3.50, 'Q3_K_M': 3.93, 'Q3_K_L': 4.30, 'Q4_K_S': 4.58, 'Q4_K_M': 4.85, 'Q5_K_S': 5.52, 'Q5_K_M': 5.68, 'Q6_K': 6.60, 'Q8_0': 8.50}
