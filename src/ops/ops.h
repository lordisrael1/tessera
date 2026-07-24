#pragma once
#include <cstdint>

// Scalar reference implementations of the transformer ops. Slow on purpose:
// these are the ORACLE that the vectorized (M2) and simulated (M3) paths must
// match. Correctness lives here; speed lives elsewhere.
namespace ops {

// y = x * rsqrt(mean(x^2) + eps) * weight   (row-wise, length n)
// Mean accumulated in double — the single most common source of a 1e-3 parity
// failure is doing this reduction in float.
void rmsnorm(const float* x, const float* weight, float* y, int64_t n, float eps);

// In-place rotary position embedding on one head vector of length head_dim.
// Qwen uses the SPLIT-HALF convention: rotate pairs (x[i], x[i + head_dim/2]),
// NOT the interleaved (x[2i], x[2i+1]) convention. Getting this wrong produces
// plausible-looking garbage.
void rope(float* x, int64_t head_dim, int64_t pos, float theta);

// Softmax over the first `len` elements, in place. Subtract rowmax for
// numerical stability. Caller passes only the valid (causal) length.
void softmax(float* x, int64_t len);

// SwiGLU activation: out[i] = silu(gate[i]) * up[i], silu(x) = x / (1+exp(-x)).
void silu_mul(const float* gate, const float* up, float* out, int64_t n);

// Naive triple-loop GEMM: C[m,n] = A[m,k] @ B[k,n]  (+ optional bias[n]).
// The oracle for M2's AVX2 kernel. Row-major throughout.
void gemm_ref(const float* A, const float* B, const float* bias, float* C,
              int64_t M, int64_t K, int64_t N);

}  // namespace ops
