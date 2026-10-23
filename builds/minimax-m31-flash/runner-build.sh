#!/usr/bin/env bash
set -euo pipefail
llama="$GITHUB_WORKSPACE/llama"
build="$RUNNER_TEMP/minimax-native"
git -C "$llama" apply --check "$PWD/runtime.patch"
git -C "$llama" apply "$PWD/runtime.patch"
cmake -S "$llama" -B "$build" -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON -DGGML_NATIVE=OFF -DGGML_CUDA=OFF -DGGML_VULKAN=OFF -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_SERVER=OFF
cmake --build "$build" --target llama --parallel 4
g++ -std=c++17 -shared -fPIC quant_bridge.cpp -I "$llama/src" -I "$llama/include" -I "$llama/ggml/include" -L "$build/bin" -Wl,-rpath,'$ORIGIN/bin' -lllama -lggml-base -o "$build/lib_minimax_quant.so"
g++ -std=c++17 runtime_probe.cpp -I "$llama/include" -I "$llama/ggml/include" -L "$build/bin" -Wl,-rpath,'$ORIGIN/bin' -lllama -lggml -lggml-base -o "$build/runtime_probe"
