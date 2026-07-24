#include "check.h"
#include "ops/ops.h"

#include <cmath>
#include <vector>

static void test_rmsnorm() {
    float x[4] = {1, 2, 3, 4};
    float w[4] = {1, 1, 1, 1};
    float y[4];
    ops::rmsnorm(x, w, y, 4, 0.0f);
    // mean(x^2) = 30/4 = 7.5 ; inv = 1/sqrt(7.5) = 0.365148...
    float inv = 1.0f / std::sqrt(7.5f);
    for (int i = 0; i < 4; ++i) CHECK_NEAR(y[i], x[i] * inv, 1e-6);
}

static void test_rope_identity() {
    // At position 0 every angle is 0 -> rotation is identity.
    float x[8] = {1, 2, 3, 4, 5, 6, 7, 8};
    float ref[8];
    for (int i = 0; i < 8; ++i) ref[i] = x[i];
    ops::rope(x, 8, /*pos=*/0, /*theta=*/1000000.0f);
    for (int i = 0; i < 8; ++i) CHECK_NEAR(x[i], ref[i], 1e-6);
}

static void test_rope_norm_preserved() {
    // Rotation preserves the norm of each (i, i+half) pair.
    float x[4] = {0.5f, -1.2f, 2.0f, 0.3f};
    double n0 = std::sqrt(x[0] * x[0] + x[2] * x[2]);
    double n1 = std::sqrt(x[1] * x[1] + x[3] * x[3]);
    ops::rope(x, 4, /*pos=*/5, 1000000.0f);
    CHECK_NEAR(std::sqrt(x[0] * x[0] + x[2] * x[2]), n0, 1e-5);
    CHECK_NEAR(std::sqrt(x[1] * x[1] + x[3] * x[3]), n1, 1e-5);
}

static void test_softmax() {
    float x[3] = {1, 2, 3};
    ops::softmax(x, 3);
    double s = x[0] + x[1] + x[2];
    CHECK_NEAR(s, 1.0, 1e-6);
    // ratios: exp(1):exp(2):exp(3); x[2]/x[1] == e
    CHECK_NEAR(x[2] / x[1], std::exp(1.0f), 1e-5);
    // stability with large inputs (no inf/nan)
    float big[2] = {1000.0f, 1001.0f};
    ops::softmax(big, 2);
    CHECK(std::isfinite(big[0]) && std::isfinite(big[1]));
    CHECK_NEAR(big[0] + big[1], 1.0, 1e-6);
}

static void test_silu_mul() {
    float g[3] = {0.0f, 1.0f, -1.0f};
    float u[3] = {2.0f, 3.0f, 4.0f};
    float o[3];
    ops::silu_mul(g, u, o, 3);
    CHECK_NEAR(o[0], 0.0f, 1e-6);                                 // silu(0)=0
    CHECK_NEAR(o[1], (1.0f / (1.0f + std::exp(-1.0f))) * 3.0f, 1e-6);
    CHECK_NEAR(o[2], (-1.0f / (1.0f + std::exp(1.0f))) * 4.0f, 1e-6);
}

static void test_gemm_ref() {
    // A [2x3] @ B [3x2] + bias[2]
    float A[6] = {1, 2, 3, 4, 5, 6};
    float B[6] = {1, 0, 0, 1, 1, 1};   // rows of B
    float bias[2] = {10, 20};
    float C[4];
    ops::gemm_ref(A, B, bias, C, 2, 3, 2);
    // row0: [1*1+2*0+3*1, 1*0+2*1+3*1] = [4,5] + bias = [14,25]
    // row1: [4+6, 5+6] = [10,11] + bias = [20,31]
    CHECK_NEAR(C[0], 14.0f, 1e-5);
    CHECK_NEAR(C[1], 25.0f, 1e-5);
    CHECK_NEAR(C[2], 20.0f, 1e-5);
    CHECK_NEAR(C[3], 31.0f, 1e-5);
}

int main() {
    std::printf("[test_ops]\n");
    test_rmsnorm();
    test_rope_identity();
    test_rope_norm_preserved();
    test_softmax();
    test_silu_mul();
    test_gemm_ref();
    return test_summary();
}
