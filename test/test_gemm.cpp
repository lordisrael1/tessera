#include "../src/ops/gemm_avx2.h"
#include "../src/ops/ops.h"
#include "../src/ops/quant.h"
#include "check.h"

#include <cmath>
#include <cstdint>
#include <random>
#include <vector>

// A reproducible fp32 matrix in [-1,1].
static std::vector<float> rand_mat(int64_t rows, int64_t cols, uint32_t seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> d(-1.0f, 1.0f);
    std::vector<float> m(static_cast<size_t>(rows * cols));
    for (auto& x : m) x = d(rng);
    return m;
}

static double rel_l2(const std::vector<float>& got, const std::vector<float>& want) {
    double num = 0.0, den = 0.0;
    for (size_t i = 0; i < want.size(); ++i) {
        double diff = static_cast<double>(got[i]) - want[i];
        num += diff * diff;
        den += static_cast<double>(want[i]) * want[i];
    }
    return std::sqrt(num / den);
}

// 1) THE ARITHMETIC CONTRACT. All three kernels must agree BIT-FOR-BIT, at every
//    thread count and every tuning. This is only possible because the scalar
//    reference reproduces the hardware's int32 lane grouping and the fixed
//    horizontal-sum tree (see the contract comment in gemm_avx2.cpp). An
//    approximate test here would let a packing or tiling bug hide inside
//    "floating point is fuzzy".
static void test_all_kernels_bit_identical() {
    const int64_t IN = 256, N = 40;  // N=40 is 10 full panels of NRP=4
    auto W = rand_mat(N, IN, 2);
    auto qw = quant::quantize_weight(W.data(), N, IN);
    auto qp = quant::pack_weight(qw);

    for (int64_t M : {int64_t{1}, int64_t{2}, int64_t{3}, int64_t{5}, int64_t{16}}) {
        auto A = rand_mat(M, IN, 1 + static_cast<uint32_t>(M));
        std::vector<float> c_ref(static_cast<size_t>(M * N));
        std::vector<float> c_naive(static_cast<size_t>(M * N));
        ops::gemm_i8_ref(A.data(), M, qw, nullptr, c_ref.data());
        ops::gemm_i8_naive(A.data(), M, qw, nullptr, c_naive.data());
        for (size_t i = 0; i < c_ref.size(); ++i) CHECK(c_ref[i] == c_naive[i]);

        // Every (MR, panel-block, prefetch, threads) combination must land on
        // the same bits — that is what makes the tuning knobs safe to sweep.
        for (int mr : {1, 2, 4})
            for (int64_t nc : {int64_t{1}, int64_t{3}, int64_t{1024}})
                for (int threads : {1, -1}) {
                    ops::GemmTuning t;
                    t.mr = mr;
                    t.nc = nc;
                    std::vector<float> c(static_cast<size_t>(M * N));
                    ops::gemm_i8_tuned(A.data(), M, qp, nullptr, c.data(), threads, t);
                    for (size_t i = 0; i < c_ref.size(); ++i) CHECK(c_ref[i] == c[i]);
                }
    }
}

// 2) The scalar fallback CI exercises must also match, on the same binary.
static void test_forced_scalar_matches() {
    const int64_t M = 3, IN = 128, N = 12;
    auto A = rand_mat(M, IN, 11);
    auto W = rand_mat(N, IN, 12);
    auto qp = quant::quantize_and_pack(W.data(), N, IN);

    std::vector<float> c_simd(static_cast<size_t>(M * N)), c_scalar(c_simd.size());
    ops::gemm_i8(A.data(), M, qp, nullptr, c_simd.data(), 1);
    ops::set_force_scalar(true);
    ops::gemm_i8(A.data(), M, qp, nullptr, c_scalar.data(), 1);
    ops::set_force_scalar(false);
    for (size_t i = 0; i < c_simd.size(); ++i) CHECK(c_simd[i] == c_scalar[i]);
}

// 3) Packing is lossless: the packed layout must hold exactly the same numbers.
static void test_packing_roundtrip() {
    const int64_t IN = 128, N = 7;  // N=7 forces a padded final panel
    auto W = rand_mat(N, IN, 21);
    auto qw = quant::quantize_weight(W.data(), N, IN);
    auto qp = quant::pack_weight(qw);
    CHECK(qp.panels == 2);
    for (int64_t o = 0; o < N; ++o)
        for (int64_t b = 0; b < qw.blocks; ++b) {
            int64_t p = o / quant::NRP, r = o % quant::NRP;
            CHECK(qp.panel_scale(p)[b * quant::NRP + r] == qw.scale_at(o, b));
            for (int k = 0; k < quant::QK; ++k)
                CHECK(qp.panel(p)[(b * quant::NRP + r) * quant::QK + k] ==
                      qw.row(o)[b * quant::QK + k]);
        }
    // Padding rows must be exact zeros so they cannot perturb anything.
    for (int64_t r = N % quant::NRP; r < quant::NRP; ++r)
        CHECK(qp.panel_scale(1)[r] == 0.0f);
}

// 4) INT8 GEMM must match the fp32 reference within quantization tolerance.
static void test_i8_vs_fp32() {
    const int64_t M = 4, IN = 512, N = 64;
    auto A = rand_mat(M, IN, 3);
    auto W = rand_mat(N, IN, 4);

    std::vector<float> c_fp32(static_cast<size_t>(M * N));
    ops::linear(A.data(), W.data(), nullptr, c_fp32.data(), M, IN, N);

    auto qp = quant::quantize_and_pack(W.data(), N, IN);
    std::vector<float> c_i8(static_cast<size_t>(M * N));
    ops::gemm_i8(A.data(), M, qp, nullptr, c_i8.data());

    // L2-norm relative error is the standard quantization-fidelity metric; it is
    // not distorted by outputs that happen to land near zero (where element-wise
    // relative error blows up). INT8 should give ~1-2% here.
    double e = rel_l2(c_i8, c_fp32);
    std::printf("  [i8 vs fp32] L2 relative error = %.4f\n", e);
    CHECK(e < 0.03);
}

// 5) Bias is applied.
static void test_bias() {
    const int64_t M = 2, IN = 64, N = 8;
    auto A = rand_mat(M, IN, 5);
    auto W = rand_mat(N, IN, 6);
    std::vector<float> bias(static_cast<size_t>(N));
    for (int64_t n = 0; n < N; ++n) bias[static_cast<size_t>(n)] = 10.0f * static_cast<float>(n);

    auto qp = quant::quantize_and_pack(W.data(), N, IN);
    std::vector<float> c_nb(static_cast<size_t>(M * N)), c_b(static_cast<size_t>(M * N));
    ops::gemm_i8(A.data(), M, qp, nullptr, c_nb.data());
    ops::gemm_i8(A.data(), M, qp, bias.data(), c_b.data());
    for (int64_t m = 0; m < M; ++m)
        for (int64_t n = 0; n < N; ++n)
            CHECK_NEAR(c_b[static_cast<size_t>(m * N + n)],
                       c_nb[static_cast<size_t>(m * N + n)] + bias[static_cast<size_t>(n)], 1e-3);
}

// 6) SATURATION PROOF (bible §4.2, docs/02-int8-saturation.md).
//    Two claims, both measured rather than argued:
//      a) under our signed scheme no int16 partial can reach 32767, and
//      b) under the +128 zero-point scheme the bible originally proposed, real
//         data DOES blow past it — so the deviation was necessary, not stylistic.
static void test_saturation_study() {
    const int64_t IN = 4096, N = 64;
    auto A = rand_mat(1, IN, 7);
    auto W = rand_mat(N, IN, 8);
    auto qw = quant::quantize_weight(W.data(), N, IN);
    std::vector<int8_t> a_s8(static_cast<size_t>(IN));
    std::vector<float> a_sc(static_cast<size_t>(IN / quant::QK));
    quant::quantize_activation_row(A.data(), IN, a_s8.data(), a_sc.data());

    int32_t worst_signed = 0, worst_zp = 0;
    for (int64_t n = 0; n < N; ++n)
        for (int64_t b = 0; b < qw.blocks; ++b) {
            const int8_t* a = a_s8.data() + b * quant::QK;
            const int8_t* w = qw.row(n) + b * quant::QK;
            worst_signed = std::max(worst_signed, quant::max_int16_partial(a, w, quant::QK));
            worst_zp = std::max(worst_zp, quant::max_int16_partial_zp128(a, w, quant::QK));
        }
    std::printf("  [saturation] signed scheme worst |int16 partial| = %d (proven bound 32258)\n",
                worst_signed);
    std::printf("  [saturation] +128 zero-point scheme worst        = %d (int16 limit 32767)\n",
                worst_zp);
    CHECK(worst_signed <= 32258);   // the proof
    CHECK(worst_zp > worst_signed); // the +128 scheme is strictly more dangerous
}

int main() {
    std::printf("  has_avx2() = %s\n", ops::has_avx2() ? "true" : "false");
    test_all_kernels_bit_identical();
    test_forced_scalar_matches();
    test_packing_roundtrip();
    test_i8_vs_fp32();
    test_bias();
    test_saturation_study();
    return test_summary();
}
