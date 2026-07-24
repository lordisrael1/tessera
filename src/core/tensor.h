#pragma once
#include <array>
#include <cstdint>
#include <cassert>

// Non-owning 4-D view. Shapes are derived at runtime from the model config,
// never hardcoded. Pad leading dims with 1 (e.g. a vector is {1,1,1,N}).
template <typename T>
struct View {
    T* data = nullptr;
    std::array<int64_t, 4> shape{1, 1, 1, 1};
    std::array<int64_t, 4> stride{0, 0, 0, 0};  // in elements, not bytes

    int64_t numel() const {
        return shape[0] * shape[1] * shape[2] * shape[3];
    }

    T& at(int64_t a, int64_t b, int64_t c, int64_t d) const {
        return data[a * stride[0] + b * stride[1] + c * stride[2] + d * stride[3]];
    }

    static View contiguous(T* p, std::array<int64_t, 4> s) {
        View v;
        v.data = p;
        v.shape = s;
        v.stride = {s[1] * s[2] * s[3], s[2] * s[3], s[3], 1};
        return v;
    }
};

using F32View = View<float>;
using I8View  = View<int8_t>;
