#!/usr/bin/env bash
set -euo pipefail
patch_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [ -e minimax-runtime-source ] || [ -e minimax-runtime-build ]; then
  echo 'Use a new empty build directory; existing runtime directories will not be overwritten.' >&2
  exit 2
fi
git clone --filter=blob:none --no-checkout https://github.com/ggml-org/llama.cpp.git minimax-runtime-source
git -C minimax-runtime-source checkout 970b51ce7c9ba5fff0726c5e9994f1c2f0e4eedc
git -C minimax-runtime-source apply --check "$patch_dir/runtime.patch"
git -C minimax-runtime-source apply "$patch_dir/runtime.patch"
cmake -S minimax-runtime-source -B minimax-runtime-build -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=OFF -DGGML_VULKAN=OFF -DLLAMA_CURL=OFF "$@"
cmake --build minimax-runtime-build --target llama-server --parallel 4
