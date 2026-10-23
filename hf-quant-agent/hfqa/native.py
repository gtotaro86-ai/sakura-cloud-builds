"""Native GGUF policy. A custom llama.cpp patch is not the default path."""

from __future__ import annotations

# Strings found in the stock Ubuntu ARM64 build b11669 installed on Oracle.
# A string proves the architecture name is compiled in. It does not by itself
# prove that convert_hf_to_gguf can read every source format.
STOCK_BUILD = "b11669"
STOCK_ARCH_STRINGS = frozenset({
    "spark2_5",
    "minimax_m3",
    "minimax-m3",
    "qwen2",
    "qwen3",
    "gemma2",
    "gpt2",
    "llama",
})

# model_type / GGUF architecture aliases that map onto those strings.
ALIASES = {
    "qwen2": "qwen2",
    "qwen3": "qwen3",
    "gemma2": "gemma2",
    "llama": "llama",
    "mistral": "llama",
    "gpt2": "gpt2",
    "spark2_5": "spark2_5",
    "spark2.5": "spark2_5",
    "minimax_m3": "minimax_m3",
    "minimax_m3_vl": "minimax_m3",
    "minimax-m3": "minimax_m3",
}


def classify(analysis: dict, adapter: dict | None = None) -> dict:
    raw = (analysis.get("gguf_architecture") or analysis.get("model_type") or "").lower()
    token = ALIASES.get(raw, raw.replace("-", "_"))
    in_stock = token in STOCK_ARCH_STRINGS or raw in STOCK_ARCH_STRINGS
    patched = bool((adapter or {}).get("requires_custom_runtime"))
    if in_stock and not patched:
        level = "native"
        reason = f"{token or raw} is present in stock llama.cpp {STOCK_BUILD}; no custom patch will be written"
    elif in_stock and patched:
        level = "native_preferred"
        reason = (
            f"{token or raw} is present in stock llama.cpp {STOCK_BUILD}. "
            "An older patched pipeline exists and must not be extended. "
            "New work uses stock llama.cpp only."
        )
    else:
        level = "absent"
        reason = (
            f"no confirmed native GGUF architecture in stock llama.cpp {STOCK_BUILD}. "
            "A custom patch is not started."
        )
    return {
        "level": level,
        "stock_build": STOCK_BUILD,
        "architecture": token or raw or None,
        "in_stock_binary": in_stock,
        "allow_new_patch": False,
        "publish_as_native": level == "native",
        "reason": reason,
    }
