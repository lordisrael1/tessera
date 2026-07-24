#pragma once
#include <cstdint>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

// Byte-level BPE tokenizer for Qwen2 (GPT-2 lineage).
//
// What is fully real here (the parts that matter in an interview):
//   * the 256-byte -> unicode byte-level mapping and its inverse (decode),
//   * the BPE merge loop (identical algorithm to HuggingFace's bpe()),
//   * loading vocab + merges straight from tokenizer.json.
//
// The one documented shortcut (§3.3 of the bible): the pretokenizer. The exact
// GPT-2/Qwen regex uses unicode property classes (\p{L}, \p{N}) that are misery
// to implement fully in C++. We implement a hand-written splitter that is exact
// for ASCII (covers all three oracle prompts, which are digit-free) and treats
// UTF-8 multibyte sequences as letters. The boundary is called out in
// docs/step-log.md and validated against a HuggingFace golden dump.
class Tokenizer {
public:
    static Tokenizer from_file(const std::string& tokenizer_json_path);

    std::vector<int32_t> encode(const std::string& text) const;
    std::string decode(const std::vector<int32_t>& ids) const;

    int32_t token_to_id(const std::string& tok) const;
    std::string id_to_token(int32_t id) const;
    size_t vocab_size() const { return id_to_token_.size(); }

    // --- exposed for tests ---
    Tokenizer();  // builds the byte-level tables; empty vocab
    std::vector<std::string> bpe(std::vector<std::string> symbols) const;
    std::vector<std::string> pretokenize(const std::string& text) const;
    // Map one raw-byte piece to its sequence of byte-encoded single-char symbols.
    std::vector<std::string> to_symbols(const std::string& raw_piece) const;
    void set_vocab_for_test(std::unordered_map<std::string, int32_t> vocab,
                            std::vector<std::pair<std::string, std::string>> merges);

private:
    void build_byte_tables();

    std::unordered_map<std::string, int32_t> token_to_id_;
    std::vector<std::string> id_to_token_;                 // id -> token string
    std::unordered_map<std::string, int32_t> merge_rank_;  // "first\x1fsecond" -> rank
    // Special/added tokens matched verbatim in the input before pretokenizing.
    std::vector<std::pair<std::string, int32_t>> specials_;  // sorted longest-first

    uint32_t byte_to_cp_[256];                 // GPT-2 byte -> codepoint
    std::unordered_map<uint32_t, uint8_t> cp_to_byte_;
};
