#include "../src/tokenizer/bpe.h"
#include "check.h"

#include <cstdlib>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

// Undo python repr() for the subset dump_tokenizer_golden.py can emit: a
// single-quoted string with \\, \', \n, \t, \r and \xNN escapes. Non-ASCII
// characters come through as raw UTF-8 (python repr keeps printable unicode
// literal), so they need no handling here.
static std::string unrepr(const std::string& r) {
    if (r.size() < 2) return "";
    std::string body = r.substr(1, r.size() - 2);  // strip the quotes
    std::string out;
    for (size_t i = 0; i < body.size(); ++i) {
        if (body[i] != '\\') { out += body[i]; continue; }
        if (++i >= body.size()) break;
        switch (body[i]) {
            case 'n': out += '\n'; break;
            case 't': out += '\t'; break;
            case 'r': out += '\r'; break;
            case '\\': out += '\\'; break;
            case '\'': out += '\''; break;
            case '"': out += '"'; break;
            case 'x': {
                if (i + 2 < body.size()) {
                    out += static_cast<char>(std::stoi(body.substr(i + 1, 2), nullptr, 16));
                    i += 2;
                }
                break;
            }
            default: out += body[i]; break;
        }
    }
    return out;
}

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

// Part 4: THE GOLDEN DUMP. The bible allows exactly one approximation in the
// tokenizer — the pretokenizer regex (§3.3) — on the grounds that unicode
// \p{L}/\p{N} classes are misery in C++. An allowed approximation still has to
// have a MEASURED boundary, or it is just a bug with an excuse. This compares
// our ids against HuggingFace's over a corpus chosen to probe that boundary
// (ASCII prose, punctuation, contractions, code, digits, CJK, Cyrillic, emoji,
// special tokens) and prints a per-case verdict.
//
// The three oracle prompts MUST be exact — they gate logit_parity. Everything
// else is reported: the pass rate and the exact list of divergent cases are the
// documentation of where the approximation ends. Regenerate the dump with:
//     python3 tools/dump_tokenizer_golden.py --model models/qwen2.5-0.5b
static void test_golden_dump_if_present() {
    const char* genv = std::getenv("TESSERA_GOLDEN_DIR");
    std::string gdir = genv ? genv : "test/golden";
    std::ifstream f(gdir + "/tokenizer.txt");
    if (!f) {
        std::printf("  [skip] %s/tokenizer.txt not found\n", gdir.c_str());
        return;
    }
    const char* env = std::getenv("TESSERA_MODEL_DIR");
    std::string dir = env ? env : "models/qwen2.5-0.5b";
    std::ifstream tf(dir + "/tokenizer.json");
    if (!tf) {
        std::printf("  [skip] real tokenizer.json not found\n");
        return;
    }
    tf.close();
    Tokenizer tk = Tokenizer::from_file(dir + "/tokenizer.json");

    // The prompts logit_parity depends on. Non-negotiable.
    const std::vector<std::string> must_be_exact = {
        "'The capital of France is'",
        "'def fibonacci(n):'",
        "'Once upon a time, in a land far away,'",
    };

    int total = 0, exact = 0;
    std::vector<std::string> divergent;
    std::string line;
    while (std::getline(f, line)) {
        if (line.empty() || line[0] == '#') continue;
        size_t tab = line.find('\t');
        if (tab == std::string::npos) continue;
        std::string repr = line.substr(tab + 1);
        std::istringstream is(line.substr(0, tab));
        size_t n = 0;
        is >> n;
        std::vector<int32_t> ref(n);
        for (size_t i = 0; i < n; ++i) is >> ref[i];

        // The dump stores python repr(); recover the literal for encode().
        std::string text = unrepr(repr);
        ++total;
        std::vector<int32_t> ours;
        bool threw = false;
        try {
            ours = tk.encode(text);
        } catch (const std::exception&) {
            threw = true;
        }
        bool ok = !threw && ours == ref;
        if (ok) {
            ++exact;
        } else {
            divergent.push_back(repr);
        }
        for (const std::string& m : must_be_exact)
            if (repr == m) {
                if (!ok) std::printf("  [golden] ORACLE PROMPT MISMATCH: %s\n", repr.c_str());
                CHECK(ok);
            }
    }

    std::printf("  [golden] %d/%d cases match HuggingFace exactly (%.0f%%)\n", exact, total,
                100.0 * exact / (total ? total : 1));
    if (!divergent.empty()) {
        std::printf("  [golden] pretokenizer approximation shows up on %zu case(s):\n",
                    divergent.size());
        for (const std::string& d : divergent) std::printf("             %s\n", d.c_str());
    }
    // A hard floor rather than an exact expected set: the useful signal is
    // "the approximation is narrow and we know where", not a brittle list.
    CHECK(exact * 100 >= total * 70);
}

int main() {
    test_bpe_algorithm();
    test_byte_roundtrip();
    test_real_tokenizer_if_present();
    test_golden_dump_if_present();
    return test_summary();
}
