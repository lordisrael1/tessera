#include "check.h"

#include "core/align.h"
#include "core/arena.h"
#include "core/bf16.h"
#include "core/json.h"
#include "core/tensor.h"

#include <cstdint>
#include <memory>

static void test_align() {
    void* p = aligned_alloc64(1000);
    CHECK((reinterpret_cast<uintptr_t>(p) & 63) == 0);
    aligned_free64(p);
}

static void test_arena() {
    Arena a(4096);
    void* p1 = a.alloc(100);
    void* p2 = a.alloc(100);
    CHECK(p1 != nullptr);
    CHECK(p2 != nullptr);
    CHECK((reinterpret_cast<uintptr_t>(p2) & 63) == 0);        // stays aligned
    CHECK(p2 != p1);
    CHECK(a.high_water() >= 164);
    // Overflow returns nullptr, does not throw.
    CHECK(a.alloc(1 << 20) == nullptr);
    a.reset();
    void* p3 = a.alloc(100);
    CHECK(p3 == p1);  // reset rewinds to the start
}

static void test_bf16() {
    // 1.0f: exponent bias 127 -> bf16 bits 0x3F80.
    CHECK_NEAR(bf16_to_f32(0x3F80), 1.0f, 0.0);
    CHECK_NEAR(bf16_to_f32(0x4000), 2.0f, 0.0);
    CHECK_NEAR(bf16_to_f32(0xBF80), -1.0f, 0.0);
    CHECK_NEAR(bf16_to_f32(0x0000), 0.0f, 0.0);
    // f16 1.0 -> 0x3C00
    CHECK_NEAR(f16_to_f32(0x3C00), 1.0f, 0.0);
    CHECK_NEAR(f16_to_f32(0xC000), -2.0f, 0.0);
}

static void test_json() {
    auto j = parse_json(R"({"dtype":"BF16","shape":[2,3],"data_offsets":[0,12],"flag":true})");
    CHECK(j.is_object());
    CHECK(j["dtype"].as_string() == "BF16");
    CHECK(j["shape"].arr.size() == 2);
    CHECK(j["shape"].arr[0].as_int() == 2);
    CHECK(j["shape"].arr[1].as_int() == 3);
    CHECK(j["data_offsets"].arr[1].as_int() == 12);
    CHECK(j["flag"].as_bool() == true);

    auto num = parse_json("  -1.5e-6  ");
    CHECK_NEAR(num.as_double(), -1.5e-6, 1e-18);

    // nested + escaped string
    auto n = parse_json(R"({"a":{"b":"x\ty"}})");
    CHECK(n["a"]["b"].as_string() == std::string("x\ty"));
}

static void test_tensor() {
    float buf[24];
    for (int i = 0; i < 24; ++i) buf[i] = static_cast<float>(i);
    auto v = View<float>::contiguous(buf, {1, 2, 3, 4});
    CHECK(v.numel() == 24);
    CHECK_NEAR(v.at(0, 1, 2, 3), 23.0f, 0.0);  // last element
    CHECK_NEAR(v.at(0, 0, 1, 0), 4.0f, 0.0);   // stride check
}

int main() {
    std::printf("[test_core]\n");
    test_align();
    test_arena();
    test_bf16();
    test_json();
    test_tensor();
    return test_summary();
}
