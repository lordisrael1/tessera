#pragma once
#include <cstddef>
#include <cstdint>
#include <condition_variable>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

// A fixed-size worker pool with one primitive: a blocking parallel_for.
//
// Deliberately not a general task system. Everything in this project that wants
// threads wants the same shape — "split this index range over the cores, then
// join" — and building only that keeps the pool small enough to reason about
// under TSan.
//
// DETERMINISM CONTRACT: parallel_for partitions [0, n) into contiguous chunks
// and each chunk is executed start-to-finish by exactly one thread. Callers that
// keep all accumulation for one output element inside one chunk therefore get
// bit-identical results regardless of thread count or scheduling. Every use in
// this repo obeys that, which is why the M1 oracle stays green with threads on.
class ThreadPool {
public:
    // 0 => hardware_concurrency/2 (physical cores on an SMT machine).
    explicit ThreadPool(unsigned n_threads = 0);
    ~ThreadPool();
    ThreadPool(const ThreadPool&) = delete;
    ThreadPool& operator=(const ThreadPool&) = delete;

    unsigned size() const { return static_cast<unsigned>(workers_.size()) + 1; }  // +caller

    // Calls body(begin, end) on disjoint contiguous sub-ranges covering [0, n).
    // Blocks until all sub-ranges have completed. The calling thread runs one of
    // them, so a pool of size 1 is a plain loop with no synchronisation at all.
    void parallel_for(int64_t n, const std::function<void(int64_t, int64_t)>& body);

    // Pin worker k to CPU (k+1) and the caller to CPU 0. No-op off Linux.
    // Returns false if pinning was attempted and failed.
    bool pin_to_cores();

    static unsigned default_threads();

private:
    void worker_loop(unsigned idx);

    std::vector<std::thread> workers_;
    std::mutex m_;
    std::condition_variable cv_start_, cv_done_;

    const std::function<void(int64_t, int64_t)>* body_ = nullptr;
    int64_t n_ = 0;
    unsigned epoch_ = 0;        // bumped once per parallel_for
    unsigned outstanding_ = 0;  // chunks not yet finished
    bool stop_ = false;
};

// One process-wide pool. Created on first use, sized from the machine.
ThreadPool& global_pool();
