# A tiny header-only assertion helper keeps tests dependency-free (project rule).
#
# Tests that need the 1 GB weights or the generated goldens SELF-SKIP when those
# are absent, so `ctest` is green on a fresh clone and in CI, and is a real gate
# on a machine that has fetched them.
set(TESSERA_TEST_ENV
  "TESSERA_MODEL_DIR=${CMAKE_SOURCE_DIR}/models/qwen2.5-0.5b"
  "TESSERA_GOLDEN_DIR=${CMAKE_SOURCE_DIR}/test/golden")

function(tessera_test name)
  add_executable(${name} ${name}.cpp)
  target_link_libraries(${name} PRIVATE tessera_core)
  add_test(NAME ${ARGV1} COMMAND ${name})
  set_tests_properties(${ARGV1} PROPERTIES ENVIRONMENT "${TESSERA_TEST_ENV}")
endfunction()

tessera_test(test_core core)
tessera_test(test_ops ops)
tessera_test(test_gemm gemm)
tessera_test(test_tokenizer tokenizer)
tessera_test(sim_isa sim_isa)
tessera_test(compiler_parity compiler_parity)
tessera_test(logit_parity logit_parity)
tessera_test(logit_parity_int8 logit_parity_int8)

# The parity tests load ~1 GB of weights and run ~200 forward passes. Generous
# timeouts; they are still the fastest way to know the stack is correct.
set_tests_properties(logit_parity logit_parity_int8 PROPERTIES TIMEOUT 1800)
