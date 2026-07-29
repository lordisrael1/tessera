# Core library: everything the model, runtime, and simulator build on.
add_library(tessera_core
  accel/simulator.cpp
  accel/lower.cpp
  core/safetensors.cpp
  core/threadpool.cpp
  model/config.cpp
  model/kv_cache.cpp
  model/qwen2.cpp
  ops/ops.cpp
  ops/simd.cpp
  ops/quant.cpp
  ops/gemm_avx2.cpp
  ops/linear_f32.cpp
  tokenizer/bpe.cpp
)
target_include_directories(tessera_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

find_package(Threads REQUIRED)
target_link_libraries(tessera_core PUBLIC Threads::Threads)
