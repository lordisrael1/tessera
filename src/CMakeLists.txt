# Core library: everything the model, runtime, and simulator build on.
add_library(tessera_core
  core/safetensors.cpp
  model/config.cpp
  ops/ops.cpp
)
target_include_directories(tessera_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
