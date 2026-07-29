add_executable(run_infer run_infer.cpp)
target_link_libraries(run_infer PRIVATE tessera_core)

# The full-model M3 gate: the real 0.5B compiled to T1 and simulated. Minutes
# per token and a 2 GB HBM image, so it is a tool you run, not a test that runs
# on every commit — test/compiler_parity.cpp is the fast gate.
add_executable(run_sim run_sim.cpp)
target_link_libraries(run_sim PRIVATE tessera_core)
