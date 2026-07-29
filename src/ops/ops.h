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

// A transformer linear layer with HuggingFace weight layout.
// W is [OUT, IN] row-major (the way HF stores nn.Linear.weight), so this is
// weight-stationary: Y[m,o] = sum_k X[m,k] * W[o,k] + bias[o]. Avoids
// transposing every projection matrix at load time. Accumulate in double.
void linear(const float* X, const float* W, const float* bias, float* Y,
            int64_t M, int64_t IN, int64_t OUT);

// Same maths, AVX2 + threaded (src/ops/linear_f32.cpp). Accumulates in double
// like linear() does, because on this machine the fp32 path is memory-bound by
// ~6:1 and the extra precision is therefore free — see the note at the top of
// that file. Agrees with linear() to ~1e-5 (test_ops asserts it) and is
// bit-stable across thread counts (each output's whole reduction stays on one
// thread), which is what lets the M1 oracle stay green with threading on.
void linear_fast(const float* X, const float* W, const float* bias, float* Y,
                 int64_t M, int64_t IN, int64_t OUT);

}  // namespace ops
