#include "../src/tokenizer/bpe.h"
#include "check.h"

#include <cstdlib>
#include <string>

// Part 1: synthetic fixture. Proves the BPE MERGE ALGORITHM and byte-level
// roundtrip with NO model download — the algorithm is what matters in an
// interview, and it must be testable offline.
static void test_bpe_algorithm() {
    Tokenizer tk;  // byte tables built; empty vocab

    // Merge order (rank = priority): l+l -> ll, then h+e -> he, then he+ll ->
    // hell, then hell+o -> hello.
    std::vector<std::pair<std::string, std::string>> merges = {
        {"l", "l"}, {"h", "e"}, {"he", "ll"}, {"hell", "o"}};
    std::unordered_map<std::string, int32_t> vocab = {
        {"h", 0}, {"e", 1}, {"l", 2}, {"o", 3},
        {"ll", 4}, {"he", 5}, {"hell", 6}, {"hello", 7}};
    tk.set_vocab_for_test(vocab, merges);

    // bpe on the raw symbols of "hello" must collapse to a single "hello".
    auto out = tk.bpe({"h", "e", "l", "l", "o"});
    CHECK(out.size() == 1);
    CHECK(out.size() == 1 && out[0] == "hello");

    // A word with no applicable merges stays split.
    auto out2 = tk.bpe({"o", "e"});
    CHECK(out2.size() == 2);
}

// Part 2: byte-level mapping + pretokenizer + decode roundtrip, still offline.
static void test_byte_roundtrip() {
    Tokenizer tk;
    // Give every byte-encoded char we need its own id, no merges -> encode
    // yields one id per byte and decode must reconstruct the original text,
    // INCLUDING the leading-space convention (space -> the U+0120 'G-dot' char).
    const std::string text = "hello world";
    // Build a vocab covering the byte-encoded chars of the text.
    std::unordered_map<std::string, int32_t> vocab;
    int32_t next = 0;
    for (const std::string& piece : tk.pretokenize(text))
        for (const std::string& sym : tk.to_symbols(piece))
            if (!vocab.count(sym)) vocab[sym] = next++;
    tk.set_vocab_for_test(vocab, {});

    auto ids = tk.encode(text);
    // "hello" = 5 chars, " world" = 6 chars (leading space encodes to one char).
    CHECK(ids.size() == 11);
    CHECK(tk.decode(ids) == text);

    // Pretokenization must split on the single space with the space attaching to
    // the following word.
    auto pieces = tk.pretokenize("The capital of France is");
    CHECK(pieces.size() == 5);
    CHECK(pieces.size() == 5 && pieces[0] == "The");
    CHECK(pieces.size() == 5 && pieces[1] == " capital");
}

// Part 3: OPTIONAL smoke test against the real tokenizer.json if present.
// Skipped cleanly when the model hasn't been downloaded, so CI stays green.
static void test_real_tokenizer_if_present() {
    const char* env = std::getenv("TESSERA_MODEL_DIR");
    std::string dir = env ? env : "models/qwen2.5-0.5b";
    std::string path = dir + "/tokenizer.json";
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) {
        std::printf("  [skip] real tokenizer.json not found at %s\n", path.c_str());
        return;
    }
    std::fclose(f);

    Tokenizer tk = Tokenizer::from_file(path);
    CHECK(tk.vocab_size() > 150000);  // Qwen2.5 vocab is 151936 (+ specials)

    // Roundtrip: encode -> decode must reproduce the input for plain ASCII.
    const std::string s = "The capital of France is";
    auto ids = tk.encode(s);
    CHECK(!ids.empty());
    CHECK(tk.decode(ids) == s);
    std::printf("  [real] '%s' -> %zu tokens; first id = %d\n", s.c_str(), ids.size(),
                ids.empty() ? -1 : ids[0]);
}

int main() {
    test_bpe_algorithm();
    test_byte_roundtrip();
    test_real_tokenizer_if_present();
    return test_summary();
}
