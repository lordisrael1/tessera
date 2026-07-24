#include "ops.h"
#include <cmath>

namespace ops {

void rmsnorm(const float* x, const float* weight, float* y, int64_t n, float eps) {
    double sumsq = 0.0;  // accumulate in double — do not "optimize" to float
    for (int64_t i = 0; i < n; ++i) sumsq += static_cast<double>(x[i]) * x[i];
    double mean = sumsq / static_cast<double>(n);
    float inv = static_cast<float>(1.0 / std::sqrt(mean + eps));
    for (int64_t i = 0; i < n; ++i) y[i] = x[i] * inv * weight[i];
}

void rope(float* x, int64_t head_dim, int64_t pos, float theta) {
    int64_t half = head_dim / 2;
    for (int64_t i = 0; i < half; ++i) {
        double freq = 1.0 / std::pow(static_cast<double>(theta),
                                     (2.0 * static_cast<double>(i)) / static_cast<double>(head_dim));
        double angle = static_cast<double>(pos) * freq;
        float cos_a = static_cast<float>(std::cos(angle));
        float sin_a = static_cast<float>(std::sin(angle));
        float a = x[i];
        float b = x[i + half];
        x[i]        = a * cos_a - b * sin_a;
        x[i + half] = a * sin_a + b * cos_a;
    }
}

void softmax(float* x, int64_t len) {
    if (len <= 0) return;
    float m = x[0];
    for (int64_t i = 1; i < len; ++i) if (x[i] > m) m = x[i];
    double sum = 0.0;
    for (int64_t i = 0; i < len; ++i) {
        float e = std::exp(x[i] - m);
        x[i] = e;
        sum += e;
    }
    float inv = static_cast<float>(1.0 / sum);
    for (int64_t i = 0; i < len; ++i) x[i] *= inv;
}

void silu_mul(const float* gate, const float* up, float* out, int64_t n) {
    for (int64_t i = 0; i < n; ++i) {
        float g = gate[i];
        float s = g / (1.0f + std::exp(-g));
        out[i] = s * up[i];
    }
}

void gemm_ref(const float* A, const float* B, const float* bias, float* C,
              int64_t M, int64_t K, int64_t N) {
    for (int64_t m = 0; m < M; ++m) {
        for (int64_t n = 0; n < N; ++n) {
            double acc = bias ? static_cast<double>(bias[n]) : 0.0;
            for (int64_t k = 0; k < K; ++k)
                acc += static_cast<double>(A[m * K + k]) * B[k * N + n];
            C[m * N + n] = static_cast<float>(acc);
        }
    }
}

void linear(const float* X, const float* W, const float* bias, float* Y,
            int64_t M, int64_t IN, int64_t OUT) {
    for (int64_t m = 0; m < M; ++m) {
        const float* xrow = X + m * IN;
        for (int64_t o = 0; o < OUT; ++o) {
            const float* wrow = W + o * IN;
            double acc = bias ? static_cast<double>(bias[o]) : 0.0;
            for (int64_t k = 0; k < IN; ++k)
                acc += static_cast<double>(xrow[k]) * wrow[k];
            Y[m * OUT + o] = static_cast<float>(acc);
        }
    }
}

}  // namespace ops
