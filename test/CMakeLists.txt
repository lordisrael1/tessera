# A tiny header-only assertion helper keeps tests dependency-free (project rule).
add_executable(test_core test_core.cpp)
target_link_libraries(test_core PRIVATE tessera_core)
add_test(NAME core COMMAND test_core)

add_executable(test_ops test_ops.cpp)
target_link_libraries(test_ops PRIVATE tessera_core)
add_test(NAME ops COMMAND test_ops)
