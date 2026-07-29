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
    } else if (cp < 0x10000) {
        out += static_cast<char>(0xE0 | (cp >> 12));
        out += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
        out += static_cast<char>(0x80 | (cp & 0x3F));
    } else {
        out += static_cast<char>(0xF0 | (cp >> 18));
        out += static_cast<char>(0x80 | ((cp >> 12) & 0x3F));
        out += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
        out += static_cast<char>(0x80 | (cp & 0x3F));
    }
}

// Decode one UTF-8 codepoint starting at s[i]; advances i past it.
// Length-clamped: a truncated sequence at the end of the string advances by one
// byte and returns the lead byte rather than reading past the end. The GPT-2
// byte alphabet only reaches U+01FF, so the 4-byte branch exists for the raw
// text of added/special tokens, not for vocab entries.
uint32_t utf8_next(const std::string& s, size_t& i) {
    auto byte = [&](size_t k) -> uint32_t { return static_cast<uint8_t>(s[k]); };
    uint32_t c = byte(i);
    size_t avail = s.size() - i;
    if (c < 0x80) { i += 1; return c; }
    if ((c >> 5) == 0x6 && avail >= 2) {
        uint32_t cp = ((c & 0x1F) << 6) | (byte(i + 1) & 0x3F);
        i += 2;
        return cp;
    }
    if ((c >> 4) == 0xE && avail >= 3) {
        uint32_t cp = ((c & 0x0F) << 12) | ((byte(i + 1) & 0x3F) << 6) | (byte(i + 2) & 0x3F);
        i += 3;
        return cp;
    }
    if ((c >> 3) == 0x1E && avail >= 4) {
        uint32_t cp = ((c & 0x07) << 18) | ((byte(i + 1) & 0x3F) << 12) |
                      ((byte(i + 2) & 0x3F) << 6) | (byte(i + 3) & 0x3F);
        i += 4;
        return cp;
    }
    i += 1;
    return c;
}

// ---------------------------------------------------------------------------
// Unicode character classes for the pretokenizer.
//
// Qwen2's tokenizer.json specifies this Split regex (cl100k lineage, NOT the
// GPT-2 one):
//
//   (?i:'s|'t|'re|'ve|'m|'ll|'d)
//   | [^\r\n\p{L}\p{N}]?\p{L}+
//   | \p{N}
//   |  ?[^\s\p{L}\p{N}]+[\r\n]*
//   | \s*[\r\n]+
//   | \s+(?!\S)
//   | \s+
//
// with leftmost-first alternation. The control flow below is that regex,
// alternative by alternative. What is APPROXIMATE is only \p{L} / \p{N} / \s:
// implementing the full Unicode property tables would mean shipping ~40 KB of
// range data for a tokenizer, so we cover the scripts below and let
// test/test_tokenizer.cpp measure exactly which inputs fall outside. That is the
// bible's sanctioned §3.3 shortcut with the boundary made empirical.
//
// Covered as \p{L}: ASCII letters, Latin-1/Latin-Extended, IPA, Greek, Cyrillic,
// Armenian, Hebrew, Arabic, Devanagari, Thai, Hiragana, Katakana, CJK unified
// ideographs (+ Ext-A and compatibility), Hangul (Jamo + syllables).
// Covered as \p{N}: ASCII digits, Arabic-Indic, Extended Arabic-Indic,
// Devanagari digits, fullwidth digits.
// Deliberately NOT letters: emoji and symbol blocks (U+2000-U+2BFF, U+1F000+),
// CJK punctuation (U+3000-U+303F) — these must fall to the punctuation
// alternative or the whole split goes wrong.
// ---------------------------------------------------------------------------
struct Range {
    uint32_t lo, hi;
};

bool in_ranges(uint32_t cp, const Range* r, size_t n) {
    for (size_t i = 0; i < n; ++i)
        if (cp >= r[i].lo && cp <= r[i].hi) return true;
    return false;
}

const Range kLetterRanges[] = {
    {0x0041, 0x005A}, {0x0061, 0x007A},              // ASCII
    {0x00AA, 0x00AA}, {0x00B5, 0x00B5}, {0x00BA, 0x00BA},
    {0x00C0, 0x00D6}, {0x00D8, 0x00F6}, {0x00F8, 0x02AF},  // Latin-1 sup + ext
    {0x0370, 0x0373}, {0x0376, 0x0377}, {0x037A, 0x037D},  // Greek (skips
    {0x037F, 0x037F}, {0x0386, 0x0386}, {0x0388, 0x03FF},  //  accents/punct)
    {0x0400, 0x0481}, {0x048A, 0x052F},              // Cyrillic
    {0x0531, 0x0556}, {0x0561, 0x0587},              // Armenian
    {0x05D0, 0x05EA}, {0x05EF, 0x05F2},              // Hebrew
    {0x0620, 0x064A}, {0x066E, 0x06D3},              // Arabic
    {0x0904, 0x0939}, {0x0958, 0x0961},              // Devanagari
    {0x0E01, 0x0E3A}, {0x0E40, 0x0E4E},              // Thai
    {0x1100, 0x11FF},                                // Hangul Jamo
    {0x3041, 0x3096}, {0x309D, 0x309F},              // Hiragana
    {0x30A1, 0x30FA}, {0x30FC, 0x30FF},              // Katakana
    {0x3400, 0x4DBF}, {0x4E00, 0x9FFF},              // CJK ideographs
    {0xAC00, 0xD7A3},                                // Hangul syllables
    {0xF900, 0xFAFF},                                // CJK compatibility
    {0x20000, 0x2FA1F},                              // CJK ext B..F
};

const Range kNumberRanges[] = {
    {0x0030, 0x0039},  // ASCII
    {0x0660, 0x0669},  // Arabic-Indic
    {0x06F0, 0x06F9},  // Extended Arabic-Indic
    {0x0966, 0x096F},  // Devanagari
    {0x0E50, 0x0E59},  // Thai
    {0xFF10, 0xFF19},  // Fullwidth
};

bool is_letter(uint32_t cp) {
    return in_ranges(cp, kLetterRanges, sizeof(kLetterRanges) / sizeof(Range));
}
bool is_number(uint32_t cp) {
    return in_ranges(cp, kNumberRanges, sizeof(kNumberRanges) / sizeof(Range));
}
// \p{White_Space}.
bool is_space(uint32_t cp) {
    return (cp >= 0x09 && cp <= 0x0D) || cp == 0x20 || cp == 0x85 || cp == 0xA0 ||
           cp == 0x1680 || (cp >= 0x2000 && cp <= 0x200A) || cp == 0x2028 || cp == 0x2029 ||
           cp == 0x202F || cp == 0x205F || cp == 0x3000;
}
bool is_nl(uint32_t cp) { return cp == '\r' || cp == '\n'; }
uint32_t lower_ascii(uint32_t cp) {
    return (cp >= 'A' && cp <= 'Z') ? cp + 32 : cp;
}

// One decoded codepoint plus where it started, so pretokenize can slice the
// original bytes without re-encoding.
struct CodePoint {
    uint32_t cp;
    size_t off;
};
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

// Qwen2's pretokenizer, alternative by alternative, leftmost-first — the same
// order the Rust `regex` crate would try them (see the class-table comment
// above for the exact pattern and for what is approximate).
std::vector<std::string> Tokenizer::pretokenize(const std::string& t) const {
    std::vector<CodePoint> cp;
    cp.reserve(t.size());
    for (size_t i = 0; i < t.size();) {
        size_t start = i;
        uint32_t c = utf8_next(t, i);
        cp.push_back({c, start});
    }

    std::vector<std::string> out;
    const size_t n = cp.size();
    auto emit = [&](size_t a, size_t b) {
        size_t begin = cp[a].off;
        size_t end = (b < n) ? cp[b].off : t.size();
        out.push_back(t.substr(begin, end - begin));
    };

    size_t i = 0;
    while (i < n) {
        // (1) (?i:'s|'t|'re|'ve|'m|'ll|'d) — case-insensitive, and note it does
        //     NOT require a preceding letter. "DON'T" -> ... + "'T".
        if (cp[i].cp == '\'') {
            static const char* kCons[] = {"re", "ve", "ll", "s", "t", "m", "d"};
            bool done = false;
            for (const char* c : kCons) {
                size_t len = std::strlen(c);
                if (i + len >= n) continue;
                bool ok = true;
                for (size_t k = 0; k < len; ++k)
                    if (lower_ascii(cp[i + 1 + k].cp) != static_cast<uint32_t>(c[k])) {
                        ok = false;
                        break;
                    }
                if (ok) {
                    emit(i, i + 1 + len);
                    i += 1 + len;
                    done = true;
                    break;
                }
            }
            if (done) continue;
        }

        // (2) [^\r\n\p{L}\p{N}]?\p{L}+
        //     THE ONE THAT BITES: the optional leading character is any non-
        //     letter, non-digit, non-newline — not just a space. That is why HF
        //     tokenizes "def fibonacci(n):" with "(n" as a single piece, and why
        //     a GPT-2-style " ?\p{L}+" rule silently produces different ids.
        {
            size_t j = i;
            if (!is_letter(cp[j].cp) && !is_number(cp[j].cp) && !is_nl(cp[j].cp)) ++j;
            if (j < n && is_letter(cp[j].cp)) {
                while (j < n && is_letter(cp[j].cp)) ++j;
                emit(i, j);
                i = j;
                continue;
            }
        }

        // (3) \p{N} — ONE digit at a time, not a run. "2026" is four pieces.
        if (is_number(cp[i].cp)) {
            emit(i, i + 1);
            ++i;
            continue;
        }

        // (4)  ?[^\s\p{L}\p{N}]+[\r\n]*
        {
            size_t j = i;
            if (cp[j].cp == ' ') ++j;
            size_t k = j;
            while (k < n && !is_space(cp[k].cp) && !is_letter(cp[k].cp) && !is_number(cp[k].cp))
                ++k;
            if (k > j) {
                while (k < n && is_nl(cp[k].cp)) ++k;
                emit(i, k);
                i = k;
                continue;
            }
        }

        // (5) \s*[\r\n]+ — greedy \s* with backtracking means this consumes the
        //     whitespace run up to and including its LAST newline.
        {
            size_t j = i;
            while (j < n && is_space(cp[j].cp)) ++j;
            size_t last_nl = n + 1;
            for (size_t k = i; k < j; ++k)
                if (is_nl(cp[k].cp)) last_nl = k;
            if (last_nl <= j) {
                emit(i, last_nl + 1);
                i = last_nl + 1;
                continue;
            }
        }

        // (6) \s+(?!\S) then (7) \s+.
        //     The lookahead makes the run give back its last character when a
        //     non-space follows, so that character becomes the optional leading
        //     space of the next piece. A run of length 1 cannot give anything
        //     back (\s+ needs at least one), so alternative (7) takes it whole.
        {
            size_t j = i;
            while (j < n && is_space(cp[j].cp)) ++j;
            if (j > i) {
                size_t end = (j < n) ? j - 1 : j;
                if (end == i) end = j;
                emit(i, end);
                i = end;
                continue;
            }
        }

        // Unreachable for well-formed input; guarantees forward progress anyway.
        emit(i, i + 1);
        ++i;
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
