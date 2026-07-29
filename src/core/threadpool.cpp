#include "threadpool.h"

#include <algorithm>

#if defined(__linux__)
#  include <pthread.h>
#  include <sched.h>
#endif

unsigned ThreadPool::default_threads() {
    unsigned hw = std::thread::hardware_concurrency();
    if (hw == 0) return 1;
    // Halve on SMT machines: the target laptop reports 12 for 6 physical cores,
    // and for FMA/AVX2 work the two SMT siblings contend for the same pair of
    // vector pipes, so 12 threads buys ~nothing and costs cache. bench/ measures
    // this rather than assuming it — see docs/01-roofline.md.
    return std::max(1u, hw / 2);
}

ThreadPool::ThreadPool(unsigned n_threads) {
    unsigned n = n_threads ? n_threads : default_threads();
    // The calling thread is one of the n workers, so we spawn n-1.
    workers_.reserve(n - 1);
    for (unsigned i = 0; i + 1 < n; ++i)
        workers_.emplace_back([this, i] { worker_loop(i); });
}

ThreadPool::~ThreadPool() {
    {
        std::lock_guard<std::mutex> lk(m_);
        stop_ = true;
        ++epoch_;
    }
    cv_start_.notify_all();
    for (auto& t : workers_) t.join();
}

void ThreadPool::worker_loop(unsigned idx) {
    unsigned seen = 0;
    while (true) {
        const std::function<void(int64_t, int64_t)>* body;
        int64_t n;
        unsigned nthreads;
        {
            std::unique_lock<std::mutex> lk(m_);
            cv_start_.wait(lk, [&] { return epoch_ != seen; });
            seen = epoch_;
            if (stop_) return;
            body = body_;
            n = n_;
            nthreads = size();
        }

        // Chunk boundaries are a pure function of (idx, n, nthreads) — no work
        // stealing, no atomic cursor. That is what makes the split reproducible
        // and therefore the arithmetic bit-stable.
        int64_t lo = static_cast<int64_t>(idx + 1) * n / nthreads;
        int64_t hi = static_cast<int64_t>(idx + 2) * n / nthreads;
        if (lo < hi) (*body)(lo, hi);

        {
            std::lock_guard<std::mutex> lk(m_);
            --outstanding_;
        }
        cv_done_.notify_one();
    }
}

void ThreadPool::parallel_for(int64_t n, const std::function<void(int64_t, int64_t)>& body) {
    if (n <= 0) return;
    const unsigned nthreads = size();
    if (nthreads == 1 || n == 1) {
        body(0, n);
        return;
    }

    {
        std::lock_guard<std::mutex> lk(m_);
        body_ = &body;
        n_ = n;
        outstanding_ = nthreads - 1;  // the workers; the caller does its own
        ++epoch_;
    }
    cv_start_.notify_all();

    // The caller takes chunk 0 — free parallelism, and it keeps a size-1 pool
    // from needing any synchronisation at all.
    int64_t hi = n / nthreads;
    if (hi > 0) body(0, hi);

    std::unique_lock<std::mutex> lk(m_);
    cv_done_.wait(lk, [&] { return outstanding_ == 0; });
    body_ = nullptr;
}

bool ThreadPool::pin_to_cores() {
#if defined(__linux__)
    bool ok = true;
    auto pin = [&](pthread_t h, unsigned cpu) {
        cpu_set_t set;
        CPU_ZERO(&set);
        CPU_SET(cpu, &set);
        if (pthread_setaffinity_np(h, sizeof(set), &set) != 0) ok = false;
    };
    pin(pthread_self(), 0);
    for (unsigned i = 0; i < workers_.size(); ++i) pin(workers_[i].native_handle(), i + 1);
    return ok;
#else
    return false;  // Windows/macOS: not needed for correctness, only for jitter.
#endif
}

ThreadPool& global_pool() {
    static ThreadPool pool;
    return pool;
}
