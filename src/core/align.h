#pragma once
#include <cstddef>
#include <cstdlib>
#include <new>

// 64-byte aligned allocation. 64 bytes == one cache line on Zen 3 and the
// natural alignment for AVX2 loads/stores. Every hot buffer in the project
// comes from here so we never eat a split-load penalty.
inline void* aligned_alloc64(std::size_t bytes) {
    // std::aligned_alloc requires size to be a multiple of alignment.
    std::size_t rounded = (bytes + 63) & ~std::size_t{63};
#if defined(_WIN32)
    void* p = _aligned_malloc(rounded, 64);
#else
    void* p = std::aligned_alloc(64, rounded);
#endif
    if (!p) throw std::bad_alloc{};
    return p;
}

inline void aligned_free64(void* p) noexcept {
#if defined(_WIN32)
    _aligned_free(p);
#else
    std::free(p);
#endif
}

struct AlignedFree {
    void operator()(void* p) const noexcept { aligned_free64(p); }
};
