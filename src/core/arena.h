#pragma once
#include <cstddef>
#include <cstdint>
#include <new>
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
        // `p + bytes` is written as a subtraction so a huge `bytes` cannot wrap
        // the addition and sneak past the bounds check.
        if (p > cap_ || bytes > cap_ - p) return nullptr;
        off_ = p + bytes;
        if (off_ > high_water_) high_water_ = off_;
        return base_ + p;
    }

    // Throwing variant for setup paths. The hot path uses alloc() and checks.
    void* alloc_or_throw(std::size_t bytes, std::size_t align = 64) {
        void* p = alloc(bytes, align);
        if (!p) throw std::bad_alloc{};
        return p;
    }

    template <typename T>
    T* alloc_n(std::size_t count) {
        if (count > SIZE_MAX / sizeof(T)) return nullptr;
        return static_cast<T*>(alloc(count * sizeof(T), 64));
    }

    // One predictable, never-taken branch per allocation, in exchange for a
    // loud failure instead of a nullptr deref when someone grows a buffer past
    // the arena. Worth it: silent activation-arena overflow would present as a
    // logit-parity failure, which is the most expensive kind of bug to bisect.
    template <typename T>
    T* alloc_n_checked(std::size_t count) {
        T* p = alloc_n<T>(count);
        if (!p) throw std::bad_alloc{};
        return p;
    }

    void reset() { off_ = 0; }
    std::size_t high_water() const { return high_water_; }
    std::size_t capacity() const { return cap_; }
};
