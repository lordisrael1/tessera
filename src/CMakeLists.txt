# Core library: everything the model, runtime, and simulator build on.
add_library(tessera_core
  core/safetensors.cpp
  model/config.cpp
  model/kv_cache.cpp
  model/qwen2.cpp
  ops/ops.cpp
  tokenizer/bpe.cpp
)
target_include_directories(tessera_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
