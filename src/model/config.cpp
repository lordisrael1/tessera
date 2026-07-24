#include "config.h"
#include "../core/json.h"

#include <fstream>
#include <sstream>
#include <stdexcept>

ModelConfig ModelConfig::from_json_file(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("config: cannot open " + path);
    std::stringstream ss;
    ss << f.rdbuf();
    std::string text = ss.str();

    JsonValue j = parse_json(text);
    ModelConfig c;

    auto req_int = [&](const char* k) -> int64_t {
        if (!j.contains(k)) throw std::runtime_error(std::string("config: missing ") + k);
        return j[k].as_int();
    };

    c.hidden_size = req_int("hidden_size");
    c.intermediate_size = req_int("intermediate_size");
    c.num_hidden_layers = req_int("num_hidden_layers");
    c.num_attention_heads = req_int("num_attention_heads");
    c.num_key_value_heads = j.contains("num_key_value_heads")
                                ? j["num_key_value_heads"].as_int()
                                : c.num_attention_heads;
    c.vocab_size = req_int("vocab_size");
    c.max_position_embeddings =
        j.contains("max_position_embeddings") ? j["max_position_embeddings"].as_int() : 32768;
    if (j.contains("rms_norm_eps")) c.rms_norm_eps = static_cast<float>(j["rms_norm_eps"].as_double());
    if (j.contains("rope_theta")) c.rope_theta = static_cast<float>(j["rope_theta"].as_double());
    if (j.contains("tie_word_embeddings")) c.tie_word_embeddings = j["tie_word_embeddings"].as_bool();

    return c;
}

std::string ModelConfig::summary() const {
    std::ostringstream os;
    os << "ModelConfig{\n"
       << "  hidden_size=" << hidden_size << "\n"
       << "  intermediate_size=" << intermediate_size << "\n"
       << "  num_hidden_layers=" << num_hidden_layers << "\n"
       << "  num_attention_heads=" << num_attention_heads
       << " (head_dim=" << head_dim() << ")\n"
       << "  num_key_value_heads=" << num_key_value_heads
       << " (q_per_kv=" << q_per_kv() << ")\n"
       << "  vocab_size=" << vocab_size << "\n"
       << "  rms_norm_eps=" << rms_norm_eps << "\n"
       << "  rope_theta=" << rope_theta << "\n"
       << "  tie_word_embeddings=" << (tie_word_embeddings ? "true" : "false") << "\n"
       << "}";
    return os.str();
}
