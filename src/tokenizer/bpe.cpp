#include "bpe.h"
#include "../core/json.h"

#include <algorithm>
#include <climits>
#include <cstring>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace {
constexpr char SEP = '\x1f';  // never appears inside a byte-encoded token

void utf8_append(std::string& out, uint32_t cp) {
    if (cp < 0x80) {
        out += static_cast<char>(cp);
    } else if (cp < 0x800) {
        out += static_cast<char>(0xC0 | (cp >> 6));
        out += static_cast<char>(0x80 | (cp & 0x3F));
    } else {
        out += static_cast<char>(0xE0 | (cp >> 12));
        out += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
        out += static_cast<char>(0x80 | (cp & 0x3F));
    }
}

// Decode one UTF-8 codepoint starting at s[i]; advances i past it.
uint32_t utf8_next(const std::string& s, size_t& i) {
    uint8_t c = static_cast<uint8_t>(s[i]);
    if (c < 0x80) { i += 1; return c; }
    if ((c >> 5) == 0x6) {
        uint32_t cp = ((c & 0x1F) << 6) | (static_cast<uint8_t>(s[i + 1]) & 0x3F);
        i += 2;
        return cp;
    }
    uint32_t cp = ((c & 0x0F) << 12) | ((static_cast<uint8_t>(s[i + 1]) & 0x3F) << 6) |
                  (static_cast<uint8_t>(s[i + 2]) & 0x3F);
    i += 3;
    return cp;
}

bool is_ascii_alpha(uint8_t b) { return (b >= 'A' && b <= 'Z') || (b >= 'a' && b <= 'z'); }
bool is_ascii_digit(uint8_t b) { return b >= '0' && b <= '9'; }
bool is_space(uint8_t b) { return b == ' ' || b == '\t' || b == '\n' || b == '\r' || b == '\f' || b == '\v'; }
// "letter" for our approximate pretokenizer: ASCII letter or any UTF-8 non-ASCII byte.
bool is_letterish(uint8_t b) { return is_ascii_alpha(b) || b >= 0x80; }
}  // namespace

Tokenizer::Tokenizer() { build_byte_tables(); }

void Tokenizer::build_byte_tables() {
    // GPT-2 bytes_to_unicode(): printable byte ranges map to themselves, the
    // rest map to 256+n so every byte becomes a printable, non-space codepoint.
    bool is_printable[256] = {false};
    auto mark = [&](int lo, int hi) { for (int b = lo; b <= hi; ++b) is_printable[b] = true; };
    mark('!', '~');
    mark(0xA1, 0xAC);
    mark(0xAE, 0xFF);
    int n = 0;
    for (int b = 0; b < 256; ++b) {
        uint32_t cp;
        if (is_printable[b]) {
            cp = static_cast<uint32_t>(b);
        } else {
            cp = static_cast<uint32_t>(256 + n);
            ++n;
        }
        byte_to_cp_[b] = cp;
        cp_to_byte_[cp] = static_cast<uint8_t>(b);
    }
}

std::vector<std::string> Tokenizer::to_symbols(const std::string& raw_piece) const {
    std::vector<std::string> syms;
    syms.reserve(raw_piece.size());
    for (unsigned char b : raw_piece) {
        std::string s;
        utf8_append(s, byte_to_cp_[b]);
        syms.push_back(std::move(s));
    }
    return syms;
}

std::vector<std::string> Tokenizer::bpe(std::vector<std::string> word) const {
    if (word.size() < 2) return word;
    while (true) {
        int best_rank = INT_MAX;
        int best_i = -1;
        for (size_t i = 0; i + 1 < word.size(); ++i) {
            std::string key = word[i] + SEP + word[i + 1];
            auto it = merge_rank_.find(key);
            if (it != merge_rank_.end() && it->second < best_rank) {
                best_rank = it->second;
                best_i = static_cast<int>(i);
            }
        }
        if (best_i < 0) break;
        const std::string& first = word[static_cast<size_t>(best_i)];
        const std::string& second = word[static_cast<size_t>(best_i) + 1];
        std::vector<std::string> merged;
        merged.reserve(word.size());
        size_t i = 0;
        while (i < word.size()) {
            if (i + 1 < word.size() && word[i] == first && word[i + 1] == second) {
                merged.push_back(first + second);
                i += 2;
            } else {
                merged.push_back(word[i]);
                i += 1;
            }
        }
        word = std::move(merged);
        if (word.size() == 1) break;
    }
    return word;
}

// Approximate GPT-2/Qwen pretokenizer. Exact for ASCII; UTF-8 multibyte treated
// as letters. Rules tried in order at each position:
//   contraction | ?letters | ?digits | ?others | whitespace
std::vector<std::string> Tokenizer::pretokenize(const std::string& t) const {
    std::vector<std::string> out;
    size_t i = 0, n = t.size();
    auto at = [&](size_t k) -> uint8_t { return static_cast<uint8_t>(t[k]); };

    while (i < n) {
        // contractions: 's 't 're 've 'm 'll 'd
        if (at(i) == '\'' && i + 1 < n) {
            static const char* cons[] = {"'re", "'ve", "'ll", "'s", "'t", "'m", "'d"};
            bool matched = false;
            for (const char* c : cons) {
                size_t len = std::strlen(c);
                if (t.compare(i, len, c) == 0) {
                    out.emplace_back(c);
                    i += len;
                    matched = true;
                    break;
                }
            }
            if (matched) continue;
        }

        size_t start = i;
        bool lead_space = (at(i) == ' ');
        size_t j = lead_space ? i + 1 : i;

        if (j < n && is_letterish(at(j))) {
            while (j < n && is_letterish(at(j))) ++j;
            out.push_back(t.substr(start, j - start));
            i = j;
            continue;
        }
        if (j < n && is_ascii_digit(at(j))) {
            while (j < n && is_ascii_digit(at(j))) ++j;  // GPT-2 style ?\p{N}+ (oracle prompts have no digits)
            out.push_back(t.substr(start, j - start));
            i = j;
            continue;
        }
        if (j < n && !is_space(at(j))) {  // punctuation / symbol run
            while (j < n && !is_space(at(j)) && !is_letterish(at(j)) && !is_ascii_digit(at(j))) ++j;
            out.push_back(t.substr(start, j - start));
            i = j;
            continue;
        }

        // whitespace run. If length >= 2 and a non-space follows, leave the last
        // space for the following token's optional leading space (GPT-2 behavior).
        size_t k = i;
        while (k < n && is_space(at(k))) ++k;
        size_t run_end = k;
        if (run_end - i >= 2 && run_end < n && !is_space(at(run_end))) run_end -= 1;
        if (run_end == i) run_end = i + 1;  // ensure progress
        out.push_back(t.substr(i, run_end - i));
        i = run_end;
    }
    return out;
}

std::vector<int32_t> Tokenizer::encode(const std::string& text) const {
    std::vector<int32_t> ids;

    // Split out any special tokens first (verbatim, longest-first).
    auto encode_plain = [&](const std::string& chunk) {
        for (const std::string& piece : pretokenize(chunk)) {
            std::vector<std::string> syms = to_symbols(piece);
            for (const std::string& tok : bpe(std::move(syms))) {
                auto it = token_to_id_.find(tok);
                if (it == token_to_id_.end())
                    throw std::runtime_error("tokenizer: token not in vocab: '" + tok + "'");
                ids.push_back(it->second);
            }
        }
    };

    size_t pos = 0;
    while (pos < text.size()) {
        size_t best = std::string::npos;
        size_t best_len = 0;
        int32_t best_id = -1;
        for (const auto& [s, id] : specials_) {
            size_t f = text.find(s, pos);
            if (f != std::string::npos && (f < best || (f == best && s.size() > best_len))) {
                best = f;
                best_len = s.size();
                best_id = id;
            }
        }
        if (best == std::string::npos) {
            encode_plain(text.substr(pos));
            break;
        }
        if (best > pos) encode_plain(text.substr(pos, best - pos));
        ids.push_back(best_id);
        pos = best + best_len;
    }
    return ids;
}

std::string Tokenizer::decode(const std::vector<int32_t>& ids) const {
    std::string encoded;  // concatenated byte-encoded token strings
    for (int32_t id : ids) {
        if (id >= 0 && static_cast<size_t>(id) < id_to_token_.size())
            encoded += id_to_token_[static_cast<size_t>(id)];
    }
    // Reverse the byte-level mapping: each codepoint -> original byte.
    std::string out;
    size_t i = 0;
    while (i < encoded.size()) {
        uint32_t cp = utf8_next(encoded, i);
        auto it = cp_to_byte_.find(cp);
        if (it != cp_to_byte_.end())
            out += static_cast<char>(it->second);
        // else: a special token's raw text (e.g. "<|im_start|>") — skip byte-unmap
    }
    return out;
}

int32_t Tokenizer::token_to_id(const std::string& tok) const {
    auto it = token_to_id_.find(tok);
    return it == token_to_id_.end() ? -1 : it->second;
}

std::string Tokenizer::id_to_token(int32_t id) const {
    if (id < 0 || static_cast<size_t>(id) >= id_to_token_.size()) return "";
    return id_to_token_[static_cast<size_t>(id)];
}

void Tokenizer::set_vocab_for_test(std::unordered_map<std::string, int32_t> vocab,
                                   std::vector<std::pair<std::string, std::string>> merges) {
    token_to_id_ = std::move(vocab);
    int32_t maxid = -1;
    for (const auto& [t, id] : token_to_id_) maxid = std::max(maxid, id);
    id_to_token_.assign(static_cast<size_t>(maxid + 1), "");
    for (const auto& [t, id] : token_to_id_) id_to_token_[static_cast<size_t>(id)] = t;
    merge_rank_.clear();
    for (size_t r = 0; r < merges.size(); ++r)
        merge_rank_[merges[r].first + SEP + merges[r].second] = static_cast<int32_t>(r);
}

Tokenizer Tokenizer::from_file(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("tokenizer: cannot open " + path);
    std::stringstream ss;
    ss << f.rdbuf();
    std::string text = ss.str();
    JsonValue root = parse_json(text);

    Tokenizer tk;  // builds byte tables

    const JsonValue& model = root["model"];
    const JsonValue& vocab = model["vocab"];
    int32_t maxid = -1;
    for (const auto& [tok, idv] : vocab.obj) {
        int32_t id = static_cast<int32_t>(idv.as_int());
        tk.token_to_id_.emplace(tok, id);
        maxid = std::max(maxid, id);
    }

    // Added/special tokens live outside model.vocab in the top-level array.
    if (root.contains("added_tokens")) {
        for (const auto& a : root["added_tokens"].arr) {
            int32_t id = static_cast<int32_t>(a["id"].as_int());
            const std::string& content = a["content"].as_string();
            tk.token_to_id_[content] = id;
            tk.specials_.emplace_back(content, id);
            maxid = std::max(maxid, id);
        }
    }
    // Longest-first so overlapping specials match greedily.
    std::sort(tk.specials_.begin(), tk.specials_.end(),
              [](const auto& a, const auto& b) { return a.first.size() > b.first.size(); });

    tk.id_to_token_.assign(static_cast<size_t>(maxid + 1), "");
    for (const auto& [tok, id] : tk.token_to_id_)
        tk.id_to_token_[static_cast<size_t>(id)] = tok;

    // merges: array of "first second" strings (Qwen) or ["first","second"] pairs.
    const JsonValue& merges = model["merges"];
    for (size_t r = 0; r < merges.arr.size(); ++r) {
        const JsonValue& m = merges.arr[r];
        std::string first, second;
        if (m.is_string()) {
            const std::string& s = m.as_string();
            size_t sp = s.find(' ');
            first = s.substr(0, sp);
            second = s.substr(sp + 1);
        } else {
            first = m.arr[0].as_string();
            second = m.arr[1].as_string();
        }
        tk.merge_rank_[first + SEP + second] = static_cast<int32_t>(r);
    }
    return tk;
}
