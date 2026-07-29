#pragma once
// Minimal recursive-descent JSON parser for exactly the subset we need to read
// safetensors headers and config.json: objects, arrays, strings, numbers,
// true/false/null. No dependency on nlohmann or anything else (project rule §0.4).
//
// This is deliberately small (~150 lines) and does zero clever error recovery:
// on malformed input it throws. That is fine — our inputs are machine-generated.
#include <cstdint>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

class JsonValue {
public:
    enum class Type { Null, Bool, Number, String, Array, Object };

    Type type = Type::Null;
    bool b = false;
    double num = 0.0;
    std::string str;
    std::vector<JsonValue> arr;
    std::map<std::string, JsonValue> obj;

    bool is_object() const { return type == Type::Object; }
    bool is_array() const { return type == Type::Array; }
    bool is_string() const { return type == Type::String; }
    bool is_number() const { return type == Type::Number; }

    const JsonValue& operator[](const std::string& key) const {
        auto it = obj.find(key);
        if (it == obj.end()) throw std::runtime_error("json: missing key '" + key + "'");
        return it->second;
    }
    bool contains(const std::string& key) const { return obj.count(key) > 0; }

    int64_t as_int() const { return static_cast<int64_t>(num); }
    double as_double() const { return num; }
    const std::string& as_string() const { return str; }
    bool as_bool() const { return b; }
};

class JsonParser {
    std::string_view s_;
    size_t i_ = 0;

    void skip_ws() {
        while (i_ < s_.size() &&
               (s_[i_] == ' ' || s_[i_] == '\t' || s_[i_] == '\n' || s_[i_] == '\r'))
            ++i_;
    }
    char peek() {
        if (i_ >= s_.size()) throw std::runtime_error("json: unexpected end");
        return s_[i_];
    }
    char get() { return s_[i_++]; }
    void expect(char c) {
        if (get() != c) throw std::runtime_error(std::string("json: expected '") + c + "'");
    }

    uint32_t read_hex4() {
        uint32_t cp = 0;
        for (int k = 0; k < 4; ++k) {
            char h = get();
            cp <<= 4;
            if (h >= '0' && h <= '9') cp |= static_cast<uint32_t>(h - '0');
            else if (h >= 'a' && h <= 'f') cp |= static_cast<uint32_t>(h - 'a' + 10);
            else if (h >= 'A' && h <= 'F') cp |= static_cast<uint32_t>(h - 'A' + 10);
            else throw std::runtime_error("json: bad \\u escape");
        }
        return cp;
    }

    static void utf8_encode(std::string& out, uint32_t cp) {
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

    std::string parse_string() {
        expect('"');
        std::string out;
        while (true) {
            char c = get();
            if (c == '"') break;
            if (c == '\\') {
                char e = get();
                switch (e) {
                    case '"': out += '"'; break;
                    case '\\': out += '\\'; break;
                    case '/': out += '/'; break;
                    case 'n': out += '\n'; break;
                    case 't': out += '\t'; break;
                    case 'r': out += '\r'; break;
                    case 'b': out += '\b'; break;
                    case 'f': out += '\f'; break;
                    case 'u': {
                        uint32_t cp = read_hex4();
                        // Surrogate pair: JSON encodes astral codepoints (emoji,
                        // rare CJK) as \uD800-\uDBFF followed by \uDC00-\uDFFF.
                        // Decoding the halves independently yields two garbage
                        // 3-byte sequences, so join them here.
                        if (cp >= 0xD800 && cp <= 0xDBFF && i_ + 1 < s_.size() &&
                            s_[i_] == '\\' && s_[i_ + 1] == 'u') {
                            size_t save = i_;
                            i_ += 2;
                            uint32_t lo = read_hex4();
                            if (lo >= 0xDC00 && lo <= 0xDFFF)
                                cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
                            else
                                i_ = save;  // not a valid pair; leave it alone
                        }
                        utf8_encode(out, cp);
                        break;
                    }
                    default: throw std::runtime_error("json: bad escape");
                }
            } else {
                out += c;
            }
        }
        return out;
    }

    JsonValue parse_number() {
        size_t start = i_;
        while (i_ < s_.size()) {
            char c = s_[i_];
            if ((c >= '0' && c <= '9') || c == '-' || c == '+' || c == '.' ||
                c == 'e' || c == 'E')
                ++i_;
            else
                break;
        }
        JsonValue v;
        v.type = JsonValue::Type::Number;
        v.num = std::stod(std::string(s_.substr(start, i_ - start)));
        return v;
    }

    JsonValue parse_value() {
        skip_ws();
        char c = peek();
        if (c == '{') return parse_object();
        if (c == '[') return parse_array();
        if (c == '"') {
            JsonValue v;
            v.type = JsonValue::Type::String;
            v.str = parse_string();
            return v;
        }
        if (c == 't' || c == 'f') {
            JsonValue v;
            v.type = JsonValue::Type::Bool;
            if (s_.substr(i_, 4) == "true") { v.b = true; i_ += 4; }
            else if (s_.substr(i_, 5) == "false") { v.b = false; i_ += 5; }
            else throw std::runtime_error("json: bad literal");
            return v;
        }
        if (c == 'n') {
            if (s_.substr(i_, 4) == "null") { i_ += 4; return JsonValue{}; }
            throw std::runtime_error("json: bad literal");
        }
        return parse_number();
    }

    JsonValue parse_array() {
        expect('[');
        JsonValue v;
        v.type = JsonValue::Type::Array;
        skip_ws();
        if (peek() == ']') { get(); return v; }
        while (true) {
            v.arr.push_back(parse_value());
            skip_ws();
            char c = get();
            if (c == ']') break;
            if (c != ',') throw std::runtime_error("json: expected ',' or ']'");
        }
        return v;
    }

    JsonValue parse_object() {
        expect('{');
        JsonValue v;
        v.type = JsonValue::Type::Object;
        skip_ws();
        if (peek() == '}') { get(); return v; }
        while (true) {
            skip_ws();
            std::string key = parse_string();
            skip_ws();
            expect(':');
            v.obj.emplace(std::move(key), parse_value());
            skip_ws();
            char c = get();
            if (c == '}') break;
            if (c != ',') throw std::runtime_error("json: expected ',' or '}'");
        }
        return v;
    }

public:
    explicit JsonParser(std::string_view s) : s_(s) {}
    JsonValue parse() {
        JsonValue v = parse_value();
        // safetensors pads its header to an 8-byte boundary with spaces (and
        // some writers use NULs), so those are legal trailers. Anything else
        // means we parsed a prefix of something we did not understand, and
        // silently accepting that is how you end up debugging "half the tensors
        // are missing".
        while (i_ < s_.size() && (s_[i_] == ' ' || s_[i_] == '\t' || s_[i_] == '\n' ||
                                  s_[i_] == '\r' || s_[i_] == '\0'))
            ++i_;
        if (i_ != s_.size())
            throw std::runtime_error("json: trailing data after root value");
        return v;
    }
};

inline JsonValue parse_json(std::string_view s) { return JsonParser(s).parse(); }
