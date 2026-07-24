# A tiny header-only assertion helper keeps tests dependency-free (project rule).
add_executable(test_core test_core.cpp)
target_link_libraries(test_core PRIVATE tessera_core)
add_test(NAME core COMMAND test_core)

add_executable(test_ops test_ops.cpp)
target_link_libraries(test_ops PRIVATE tessera_core)
add_test(NAME ops COMMAND test_ops)

add_executable(test_tokenizer test_tokenizer.cpp)
target_link_libraries(test_tokenizer PRIVATE tessera_core)
add_test(NAME tokenizer COMMAND test_tokenizer)
set_tests_properties(tokenizer PROPERTIES
  ENVIRONMENT "TESSERA_MODEL_DIR=${CMAKE_SOURCE_DIR}/models/qwen2.5-0.5b")
