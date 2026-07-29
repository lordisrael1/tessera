# Benchmarks. NEVER run in CI (shared runners + thermals = meaningless numbers,
# bible §10). Run locally before tagging a milestone; commit the CSV.
add_executable(stream_bench stream_bench.cpp)
target_link_libraries(stream_bench PRIVATE tessera_core)

add_executable(peak_bench peak_bench.cpp)
target_link_libraries(peak_bench PRIVATE tessera_core)

add_executable(gemm_bench gemm_bench.cpp)
target_link_libraries(gemm_bench PRIVATE tessera_core)
