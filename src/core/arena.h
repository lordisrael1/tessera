#pragma once
#include <cstddef>
#include <cstdint>
#include "align.h"

// Bump allocator for the per-token activation working set.
//
// Rule of the whole project: NO malloc/new in any per-token path. Weights are
// mmap'd once, the KV cache is one up-front slab, and every transient
// activation comes from one Arena that is reset() at the top of each token.
//
// high_water() is not decoration: after M1 runs it reports the real activation
// working set per token, which is the number used to size the fictional chip's
// scratchpad in M3. The project feeds itself.
class Arena {
    std::byte* base_;
    std::size_t cap_;
    std::size_t off_ = 0;
    std::size_t high_water_ = 0;

public:
    explicit Arena(std::size_t cap)
        : base_(static_cast<std::byte*>(aligned_alloc64(cap))), cap_(cap) {}
    ~Arena() { aligned_free64(base_); }
    Arena(const Arena&) = delete;
    Arena& operator=(const Arena&) = delete;

    // Returns nullptr on overflow — caller handles it. No throw in the hot path.
    void* alloc(std::size_t bytes, std::size_t align = 64) {
        std::size_t p = (off_ + align - 1) & ~(align - 1);
        if (p + bytes > cap_) return nullptr;
        off_ = p + bytes;
        if (off_ > high_water_) high_water_ = off_;
        return base_ + p;
    }

    template <typename T>
    T* alloc_n(std::size_t count) {
        return static_cast<T*>(alloc(count * sizeof(T), 64));
    }

    void reset() { off_ = 0; }
    std::size_t high_water() const { return high_water_; }
    std::size_t capacity() const { return cap_; }
};
