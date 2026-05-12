#include "horizon_gguf_converter.h"
#include "horizon_gguf_writer.h"

#include <dirent.h>
#include <sys/stat.h>

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <unordered_set>
#include <vector>

namespace {

struct WorkspaceFile {
    std::string name;
    std::string path;
    uint64_t size;
};

struct SafetensorsHeader {
    bool ok;
    uint64_t header_bytes;
    int tensor_count;
    int mapped_tensor_count;
    std::string error;
};

struct SafetensorsTensor {
    std::string source_name;
    std::string gguf_name;
    std::string dtype;
    std::vector<uint64_t> shape;
    std::string source_path;
    uint64_t source_offset;
    uint64_t data_begin;
    uint64_t data_end;
};

bool ends_with(const std::string &value, const std::string &suffix) {
    if (suffix.size() > value.size()) {
        return false;
    }
    return std::equal(suffix.rbegin(), suffix.rend(), value.rbegin(), [](char left, char right) {
        return std::tolower(static_cast<unsigned char>(left)) ==
               std::tolower(static_cast<unsigned char>(right));
    });
}

bool equals_ignore_case(std::string left, std::string right) {
    std::transform(left.begin(), left.end(), left.begin(), [](char value) {
        return std::tolower(static_cast<unsigned char>(value));
    });
    std::transform(right.begin(), right.end(), right.begin(), [](char value) {
        return std::tolower(static_cast<unsigned char>(value));
    });
    return left == right;
}

bool contains_ignore_case(std::string value, std::string needle) {
    std::transform(value.begin(), value.end(), value.begin(), [](char item) {
        return std::tolower(static_cast<unsigned char>(item));
    });
    std::transform(needle.begin(), needle.end(), needle.begin(), [](char item) {
        return std::tolower(static_cast<unsigned char>(item));
    });
    return value.find(needle) != std::string::npos;
}

std::string read_text_file(const std::string &path, size_t max_bytes) {
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        return "";
    }
    std::string output;
    output.resize(max_bytes);
    input.read(output.data(), static_cast<std::streamsize>(output.size()));
    output.resize(static_cast<size_t>(input.gcount()));
    return output;
}

std::string extract_json_string(const std::string &json, const std::string &key) {
    const std::string needle = "\"" + key + "\"";
    size_t key_pos = json.find(needle);
    if (key_pos == std::string::npos) {
        return "";
    }
    size_t colon = json.find(':', key_pos + needle.size());
    if (colon == std::string::npos) {
        return "";
    }
    size_t quote = json.find('"', colon + 1);
    if (quote == std::string::npos) {
        return "";
    }
    size_t end = json.find('"', quote + 1);
    if (end == std::string::npos) {
        return "";
    }
    return json.substr(quote + 1, end - quote - 1);
}

int hex_value(char value) {
    if (value >= '0' && value <= '9') {
        return value - '0';
    }
    if (value >= 'a' && value <= 'f') {
        return 10 + value - 'a';
    }
    if (value >= 'A' && value <= 'F') {
        return 10 + value - 'A';
    }
    return -1;
}

bool parse_json_hex4(const std::string &value, size_t offset, uint32_t &codepoint) {
    if (offset + 4 > value.size()) {
        return false;
    }
    uint32_t parsed = 0;
    for (size_t index = 0; index < 4; ++index) {
        const int digit = hex_value(value[offset + index]);
        if (digit < 0) {
            return false;
        }
        parsed = (parsed << 4U) | static_cast<uint32_t>(digit);
    }
    codepoint = parsed;
    return true;
}

void append_utf8(std::string &output, uint32_t codepoint) {
    if (codepoint <= 0x7FU) {
        output.push_back(static_cast<char>(codepoint));
    } else if (codepoint <= 0x7FFU) {
        output.push_back(static_cast<char>(0xC0U | (codepoint >> 6U)));
        output.push_back(static_cast<char>(0x80U | (codepoint & 0x3FU)));
    } else if (codepoint <= 0xFFFFU) {
        output.push_back(static_cast<char>(0xE0U | (codepoint >> 12U)));
        output.push_back(static_cast<char>(0x80U | ((codepoint >> 6U) & 0x3FU)));
        output.push_back(static_cast<char>(0x80U | (codepoint & 0x3FU)));
    } else {
        output.push_back(static_cast<char>(0xF0U | (codepoint >> 18U)));
        output.push_back(static_cast<char>(0x80U | ((codepoint >> 12U) & 0x3FU)));
        output.push_back(static_cast<char>(0x80U | ((codepoint >> 6U) & 0x3FU)));
        output.push_back(static_cast<char>(0x80U | (codepoint & 0x3FU)));
    }
}

std::string unescape_json_string(const std::string &value) {
    std::string output;
    output.reserve(value.size());
    for (size_t index = 0; index < value.size(); ++index) {
        const char item = value[index];
        if (item == '\\' && index + 1 < value.size()) {
            const char escaped = value[++index];
            switch (escaped) {
                case 'n':
                    output.push_back('\n');
                    break;
                case 'r':
                    output.push_back('\r');
                    break;
                case 't':
                    output.push_back('\t');
                    break;
                case '\\':
                case '"':
                case '/':
                    output.push_back(escaped);
                    break;
                case 'u': {
                    uint32_t codepoint = 0;
                    if (parse_json_hex4(value, index + 1, codepoint)) {
                        index += 4;
                        if (codepoint >= 0xD800U && codepoint <= 0xDBFFU &&
                            index + 6 < value.size() &&
                            value[index + 1] == '\\' &&
                            value[index + 2] == 'u') {
                            uint32_t low_surrogate = 0;
                            if (parse_json_hex4(value, index + 3, low_surrogate) &&
                                low_surrogate >= 0xDC00U &&
                                low_surrogate <= 0xDFFFU) {
                                codepoint = 0x10000U +
                                        ((codepoint - 0xD800U) << 10U) +
                                        (low_surrogate - 0xDC00U);
                                index += 6;
                            }
                        }
                        append_utf8(output, codepoint);
                    } else {
                        output.push_back('u');
                    }
                    break;
                }
                default:
                    output.push_back(escaped);
                    break;
            }
        } else {
            output.push_back(item);
        }
    }
    return output;
}

std::vector<uint32_t> utf8_codepoints(const std::string &value) {
    std::vector<uint32_t> codepoints;
    size_t cursor = 0;
    while (cursor < value.size()) {
        const uint8_t lead = static_cast<uint8_t>(value[cursor]);
        if (lead < 0x80) {
            codepoints.push_back(lead);
            cursor += 1;
        } else if ((lead & 0xE0U) == 0xC0U && cursor + 1 < value.size()) {
            codepoints.push_back(((lead & 0x1FU) << 6U) |
                                 (static_cast<uint8_t>(value[cursor + 1]) & 0x3FU));
            cursor += 2;
        } else if ((lead & 0xF0U) == 0xE0U && cursor + 2 < value.size()) {
            codepoints.push_back(((lead & 0x0FU) << 12U) |
                                 ((static_cast<uint8_t>(value[cursor + 1]) & 0x3FU) << 6U) |
                                 (static_cast<uint8_t>(value[cursor + 2]) & 0x3FU));
            cursor += 3;
        } else if ((lead & 0xF8U) == 0xF0U && cursor + 3 < value.size()) {
            codepoints.push_back(((lead & 0x07U) << 18U) |
                                 ((static_cast<uint8_t>(value[cursor + 1]) & 0x3FU) << 12U) |
                                 ((static_cast<uint8_t>(value[cursor + 2]) & 0x3FU) << 6U) |
                                 (static_cast<uint8_t>(value[cursor + 3]) & 0x3FU));
            cursor += 4;
        } else {
            codepoints.push_back(lead);
            cursor += 1;
        }
    }
    return codepoints;
}

std::map<uint32_t, uint8_t> gpt2_byte_decoder() {
    std::map<uint32_t, uint8_t> decoder;
    std::vector<uint32_t> bytes;
    for (uint32_t value = static_cast<uint32_t>('!'); value <= static_cast<uint32_t>('~'); ++value) {
        bytes.push_back(value);
    }
    for (uint32_t value = 0xA1U; value <= 0xACU; ++value) {
        bytes.push_back(value);
    }
    for (uint32_t value = 0xAEU; value <= 0xFFU; ++value) {
        bytes.push_back(value);
    }

    std::vector<uint32_t> chars = bytes;
    uint32_t extra = 0;
    for (uint32_t value = 0; value < 256U; ++value) {
        if (std::find(bytes.begin(), bytes.end(), value) == bytes.end()) {
            bytes.push_back(value);
            chars.push_back(256U + extra);
            extra += 1;
        }
    }

    for (size_t index = 0; index < bytes.size(); ++index) {
        decoder[chars[index]] = static_cast<uint8_t>(bytes[index]);
    }
    return decoder;
}

std::string decode_gpt2_token_bytes(const std::string &token) {
    static const std::map<uint32_t, uint8_t> decoder = gpt2_byte_decoder();
    std::string output;
    for (uint32_t codepoint : utf8_codepoints(token)) {
        const auto mapped = decoder.find(codepoint);
        if (mapped != decoder.end()) {
            output.push_back(static_cast<char>(mapped->second));
        } else if (codepoint < 0x80U) {
            output.push_back(static_cast<char>(codepoint));
        } else {
            return token;
        }
    }
    return output;
}

std::string extract_object_string(const std::string &object_json, const std::string &key) {
    return extract_json_string(object_json, key);
}

std::vector<uint64_t> extract_uint_array(const std::string &object_json, const std::string &key) {
    std::vector<uint64_t> values;
    const std::string needle = "\"" + key + "\"";
    size_t key_pos = object_json.find(needle);
    if (key_pos == std::string::npos) {
        return values;
    }
    size_t start = object_json.find('[', key_pos + needle.size());
    size_t end = object_json.find(']', start + 1);
    if (start == std::string::npos || end == std::string::npos) {
        return values;
    }

    size_t cursor = start + 1;
    while (cursor < end) {
        while (cursor < end && !std::isdigit(static_cast<unsigned char>(object_json[cursor]))) {
            cursor += 1;
        }
        if (cursor >= end) {
            break;
        }
        size_t number_end = cursor;
        while (number_end < end && std::isdigit(static_cast<unsigned char>(object_json[number_end]))) {
            number_end += 1;
        }
        values.push_back(std::stoull(object_json.substr(cursor, number_end - cursor)));
        cursor = number_end;
    }
    return values;
}

size_t find_matching_brace(const std::string &text, size_t object_start) {
    int depth = 0;
    bool in_string = false;
    bool escaped = false;
    for (size_t index = object_start; index < text.size(); ++index) {
        const char value = text[index];
        if (escaped) {
            escaped = false;
            continue;
        }
        if (value == '\\' && in_string) {
            escaped = true;
            continue;
        }
        if (value == '"') {
            in_string = !in_string;
            continue;
        }
        if (in_string) {
            continue;
        }
        if (value == '{') {
            depth += 1;
        } else if (value == '}') {
            depth -= 1;
            if (depth == 0) {
                return index;
            }
        }
    }
    return std::string::npos;
}

size_t find_matching_bracket(const std::string &text, size_t array_start) {
    int depth = 0;
    bool in_string = false;
    bool escaped = false;
    for (size_t cursor = array_start; cursor < text.size(); ++cursor) {
        const char value = text[cursor];
        if (in_string) {
            if (escaped) {
                escaped = false;
            } else if (value == '\\') {
                escaped = true;
            } else if (value == '"') {
                in_string = false;
            }
            continue;
        }
        if (value == '"') {
            in_string = true;
        } else if (value == '[') {
            depth += 1;
        } else if (value == ']') {
            depth -= 1;
            if (depth == 0) {
                return cursor;
            }
        }
    }
    return std::string::npos;
}

bool read_json_string_at(
        const std::string &json,
        size_t quote_start,
        std::string &value,
        size_t &next_cursor) {
    if (quote_start >= json.size() || json[quote_start] != '"') {
        return false;
    }
    size_t quote_end = quote_start + 1;
    bool escaped = false;
    while (quote_end < json.size()) {
        const char item = json[quote_end];
        if (escaped) {
            escaped = false;
        } else if (item == '\\') {
            escaped = true;
        } else if (item == '"') {
            value = unescape_json_string(json.substr(quote_start + 1, quote_end - quote_start - 1));
            next_cursor = quote_end + 1;
            return true;
        }
        quote_end += 1;
    }
    return false;
}

std::string map_llama_tensor_name(const std::string &source_name) {
    const std::string prefix = "model.layers.";
    if (source_name == "model.embed_tokens.weight") {
        return "token_embd.weight";
    }
    if (source_name == "lm_head.weight") {
        return "output.weight";
    }
    if (source_name == "model.norm.weight") {
        return "output_norm.weight";
    }
    if (source_name.rfind(prefix, 0) != 0) {
        return "";
    }

    const size_t layer_start = prefix.size();
    const size_t layer_end = source_name.find('.', layer_start);
    if (layer_end == std::string::npos) {
        return "";
    }
    const std::string layer = source_name.substr(layer_start, layer_end - layer_start);
    const std::string rest = source_name.substr(layer_end + 1);

    if (rest == "input_layernorm.weight") {
        return "blk." + layer + ".attn_norm.weight";
    }
    if (rest == "post_attention_layernorm.weight") {
        return "blk." + layer + ".ffn_norm.weight";
    }
    if (rest == "self_attn.q_proj.weight") {
        return "blk." + layer + ".attn_q.weight";
    }
    if (rest == "self_attn.k_proj.weight") {
        return "blk." + layer + ".attn_k.weight";
    }
    if (rest == "self_attn.v_proj.weight") {
        return "blk." + layer + ".attn_v.weight";
    }
    if (rest == "self_attn.o_proj.weight") {
        return "blk." + layer + ".attn_output.weight";
    }
    if (rest == "mlp.gate_proj.weight") {
        return "blk." + layer + ".ffn_gate.weight";
    }
    if (rest == "mlp.down_proj.weight") {
        return "blk." + layer + ".ffn_down.weight";
    }
    if (rest == "mlp.up_proj.weight") {
        return "blk." + layer + ".ffn_up.weight";
    }

    return "";
}

std::string map_qwen_tensor_name(const std::string &source_name) {
    const std::string mapped = map_llama_tensor_name(source_name);
    if (!mapped.empty()) {
        return mapped;
    }

    const std::string prefix = "model.layers.";
    if (source_name.rfind(prefix, 0) != 0) {
        return "";
    }

    const size_t layer_start = prefix.size();
    const size_t layer_end = source_name.find('.', layer_start);
    if (layer_end == std::string::npos) {
        return "";
    }
    const std::string layer = source_name.substr(layer_start, layer_end - layer_start);
    const std::string rest = source_name.substr(layer_end + 1);

    if (rest == "self_attn.q_norm.weight") {
        return "blk." + layer + ".attn_q_norm.weight";
    }
    if (rest == "self_attn.k_norm.weight") {
        return "blk." + layer + ".attn_k_norm.weight";
    }
    if (rest == "self_attn.q_proj.bias") {
        return "blk." + layer + ".attn_q.bias";
    }
    if (rest == "self_attn.k_proj.bias") {
        return "blk." + layer + ".attn_k.bias";
    }
    if (rest == "self_attn.v_proj.bias") {
        return "blk." + layer + ".attn_v.bias";
    }
    if (rest == "self_attn.o_proj.bias") {
        return "blk." + layer + ".attn_output.bias";
    }

    return "";
}

std::string map_gemma_tensor_name(
        const std::string &source_name,
        const std::string &gguf_arch) {
    const std::string language_prefix = "language_model.";
    if (source_name.rfind(language_prefix, 0) == 0) {
        return map_gemma_tensor_name(source_name.substr(language_prefix.size()), gguf_arch);
    }

    const std::string prefix = "model.layers.";
    if (source_name == "model.embed_tokens.weight") {
        return "token_embd.weight";
    }
    if (source_name == "model.norm.weight") {
        return "output_norm.weight";
    }
    if (source_name == "lm_head.weight") {
        return "";
    }
    if (source_name.rfind(prefix, 0) != 0) {
        return "";
    }

    const size_t layer_start = prefix.size();
    const size_t layer_end = source_name.find('.', layer_start);
    if (layer_end == std::string::npos) {
        return "";
    }
    const std::string layer = source_name.substr(layer_start, layer_end - layer_start);
    const std::string rest = source_name.substr(layer_end + 1);

    if (rest == "input_layernorm.weight") {
        return "blk." + layer + ".attn_norm.weight";
    }
    if (rest == "post_attention_layernorm.weight" && gguf_arch == "gemma") {
        return "blk." + layer + ".ffn_norm.weight";
    }
    if (rest == "post_attention_layernorm.weight") {
        return "blk." + layer + ".post_attention_norm.weight";
    }
    if (rest == "pre_feedforward_layernorm.weight" && gguf_arch == "gemma3") {
        return "blk." + layer + ".ffn_norm.weight";
    }
    if (rest == "pre_feedforward_layernorm.weight") {
        return "blk." + layer + ".ffn_pre_norm.weight";
    }
    if (rest == "post_feedforward_layernorm.weight") {
        return "blk." + layer + ".post_ffw_norm.weight";
    }
    if (rest == "self_attn.q_proj.weight") {
        return "blk." + layer + ".attn_q.weight";
    }
    if (rest == "self_attn.k_proj.weight") {
        return "blk." + layer + ".attn_k.weight";
    }
    if (rest == "self_attn.v_proj.weight") {
        return "blk." + layer + ".attn_v.weight";
    }
    if (rest == "self_attn.o_proj.weight") {
        return "blk." + layer + ".attn_output.weight";
    }
    if (rest == "self_attn.q_norm.weight") {
        return "blk." + layer + ".attn_q_norm.weight";
    }
    if (rest == "self_attn.k_norm.weight") {
        return "blk." + layer + ".attn_k_norm.weight";
    }
    if (rest == "mlp.gate_proj.weight") {
        return "blk." + layer + ".ffn_gate.weight";
    }
    if (rest == "mlp.down_proj.weight") {
        return "blk." + layer + ".ffn_down.weight";
    }
    if (rest == "mlp.up_proj.weight") {
        return "blk." + layer + ".ffn_up.weight";
    }

    return "";
}

std::string map_phi_tensor_name(
        const std::string &source_name,
        const std::string &gguf_arch) {
    const std::string model_layers_prefix = "model.layers.";
    if (source_name == "model.embed_tokens.weight") {
        return "token_embd.weight";
    }
    if (source_name == "model.final_layernorm.weight") {
        return "output_norm.weight";
    }
    if (source_name == "model.final_layernorm.bias") {
        return "output_norm.bias";
    }
    if (source_name == "model.norm.weight") {
        return "output_norm.weight";
    }
    if (source_name == "model.norm.bias") {
        return "output_norm.bias";
    }
    if (source_name == "lm_head.weight") {
        return "output.weight";
    }
    if (source_name == "lm_head.bias") {
        return "output.bias";
    }
    if (source_name.rfind(model_layers_prefix, 0) == 0) {
        const size_t layer_start = model_layers_prefix.size();
        const size_t layer_end = source_name.find('.', layer_start);
        if (layer_end == std::string::npos) {
            return "";
        }
        const std::string layer = source_name.substr(layer_start, layer_end - layer_start);
        const std::string rest = source_name.substr(layer_end + 1);

        if (rest == "input_layernorm.weight") {
            return "blk." + layer + ".attn_norm.weight";
        }
        if (rest == "input_layernorm.bias") {
            return "blk." + layer + ".attn_norm.bias";
        }
        if (rest == "post_attention_layernorm.weight") {
            return "blk." + layer + ".ffn_norm.weight";
        }
        if (rest == "post_attention_layernorm.bias") {
            return "blk." + layer + ".ffn_norm.bias";
        }
        if (rest == "self_attn.qkv_proj.weight") {
            return "blk." + layer + ".attn_qkv.weight";
        }
        if (rest == "self_attn.qkv_proj.bias") {
            return "blk." + layer + ".attn_qkv.bias";
        }
        if (rest == "self_attn.q_proj.weight") {
            return "blk." + layer + ".attn_q.weight";
        }
        if (rest == "self_attn.q_proj.bias") {
            return "blk." + layer + ".attn_q.bias";
        }
        if (rest == "self_attn.k_proj.weight") {
            return "blk." + layer + ".attn_k.weight";
        }
        if (rest == "self_attn.k_proj.bias") {
            return "blk." + layer + ".attn_k.bias";
        }
        if (rest == "self_attn.v_proj.weight") {
            return "blk." + layer + ".attn_v.weight";
        }
        if (rest == "self_attn.v_proj.bias") {
            return "blk." + layer + ".attn_v.bias";
        }
        if (rest == "self_attn.o_proj.weight") {
            return "blk." + layer + ".attn_output.weight";
        }
        if (rest == "self_attn.o_proj.bias") {
            return "blk." + layer + ".attn_output.bias";
        }
        if (rest == "self_attn.dense.weight") {
            return "blk." + layer + ".attn_output.weight";
        }
        if (rest == "self_attn.dense.bias") {
            return "blk." + layer + ".attn_output.bias";
        }
        if (rest == "mlp.gate_up_proj.weight") {
            return "blk." + layer + ".ffn_up.weight";
        }
        if (rest == "mlp.fc1.weight") {
            return "blk." + layer + ".ffn_up.weight";
        }
        if (rest == "mlp.fc1.bias") {
            return "blk." + layer + ".ffn_up.bias";
        }
        if (rest == "mlp.down_proj.weight") {
            return "blk." + layer + ".ffn_down.weight";
        }
        if (rest == "mlp.fc2.weight") {
            return "blk." + layer + ".ffn_down.weight";
        }
        if (rest == "mlp.fc2.bias") {
            return "blk." + layer + ".ffn_down.bias";
        }
        return "";
    }

    if (gguf_arch == "phi2") {
        const std::string prefix = "transformer.h.";
        if (source_name == "transformer.embd.wte.weight") {
            return "token_embd.weight";
        }
        if (source_name == "embd.wte.weight") {
            return "token_embd.weight";
        }
        if (source_name == "transformer.wte.weight") {
            return "token_embd.weight";
        }
        if (source_name == "lm_head.ln.weight") {
            return "output_norm.weight";
        }
        if (source_name == "lm_head.ln.bias") {
            return "output_norm.bias";
        }
        if (source_name == "lm_head.linear.weight") {
            return "output.weight";
        }
        if (source_name == "lm_head.linear.bias") {
            return "output.bias";
        }
        if (source_name == "lm_head.weight") {
            return "output.weight";
        }
        if (source_name == "lm_head.bias") {
            return "output.bias";
        }
        if (source_name == "ln_f.weight" || source_name == "transformer.ln_f.weight") {
            return "output_norm.weight";
        }
        if (source_name == "ln_f.bias" || source_name == "transformer.ln_f.bias") {
            return "output_norm.bias";
        }
        if (source_name.rfind(prefix, 0) != 0) {
            const std::string layers_prefix = "layers.";
            if (source_name.rfind(layers_prefix, 0) != 0) {
                return "";
            }

            const size_t layer_start = layers_prefix.size();
            const size_t layer_end = source_name.find('.', layer_start);
            if (layer_end == std::string::npos) {
                return "";
            }
            const std::string layer = source_name.substr(layer_start, layer_end - layer_start);
            const std::string rest = source_name.substr(layer_end + 1);

            if (rest == "ln.weight") {
                return "blk." + layer + ".attn_norm.weight";
            }
            if (rest == "ln.bias") {
                return "blk." + layer + ".attn_norm.bias";
            }
            if (rest == "mixer.Wqkv.weight") {
                return "blk." + layer + ".attn_qkv.weight";
            }
            if (rest == "mixer.Wqkv.bias") {
                return "blk." + layer + ".attn_qkv.bias";
            }
            if (rest == "mixer.out_proj.weight") {
                return "blk." + layer + ".attn_output.weight";
            }
            if (rest == "mixer.out_proj.bias") {
                return "blk." + layer + ".attn_output.bias";
            }
            if (rest == "mlp.fc1.weight") {
                return "blk." + layer + ".ffn_up.weight";
            }
            if (rest == "mlp.fc1.bias") {
                return "blk." + layer + ".ffn_up.bias";
            }
            if (rest == "mlp.fc2.weight") {
                return "blk." + layer + ".ffn_down.weight";
            }
            if (rest == "mlp.fc2.bias") {
                return "blk." + layer + ".ffn_down.bias";
            }
            return "";
        }

        const size_t layer_start = prefix.size();
        const size_t layer_end = source_name.find('.', layer_start);
        if (layer_end == std::string::npos) {
            return "";
        }
        const std::string layer = source_name.substr(layer_start, layer_end - layer_start);
        const std::string rest = source_name.substr(layer_end + 1);

        if (rest == "ln.weight") {
            return "blk." + layer + ".attn_norm.weight";
        }
        if (rest == "ln.bias") {
            return "blk." + layer + ".attn_norm.bias";
        }
        if (rest == "mixer.Wqkv.weight") {
            return "blk." + layer + ".attn_qkv.weight";
        }
        if (rest == "mixer.Wqkv.bias") {
            return "blk." + layer + ".attn_qkv.bias";
        }
        if (rest == "mixer.out_proj.weight") {
            return "blk." + layer + ".attn_output.weight";
        }
        if (rest == "mixer.out_proj.bias") {
            return "blk." + layer + ".attn_output.bias";
        }
        if (rest == "mlp.fc1.weight") {
            return "blk." + layer + ".ffn_up.weight";
        }
        if (rest == "mlp.fc1.bias") {
            return "blk." + layer + ".ffn_up.bias";
        }
        if (rest == "mlp.fc2.weight") {
            return "blk." + layer + ".ffn_down.weight";
        }
        if (rest == "mlp.fc2.bias") {
            return "blk." + layer + ".ffn_down.bias";
        }
    }

    const std::string prefix = model_layers_prefix;
    if (source_name.rfind(prefix, 0) != 0) {
        return "";
    }

    const size_t layer_start = prefix.size();
    const size_t layer_end = source_name.find('.', layer_start);
    if (layer_end == std::string::npos) {
        return "";
    }
    const std::string layer = source_name.substr(layer_start, layer_end - layer_start);
    const std::string rest = source_name.substr(layer_end + 1);

    if (rest == "input_layernorm.weight") {
        return "blk." + layer + ".attn_norm.weight";
    }
    if (rest == "input_layernorm.bias") {
        return "blk." + layer + ".attn_norm.bias";
    }
    if (rest == "post_attention_layernorm.weight") {
        return "blk." + layer + ".ffn_norm.weight";
    }
    if (rest == "post_attention_layernorm.bias") {
        return "blk." + layer + ".ffn_norm.bias";
    }
    if (rest == "self_attn.qkv_proj.weight") {
        return "blk." + layer + ".attn_qkv.weight";
    }
    if (rest == "self_attn.qkv_proj.bias") {
        return "blk." + layer + ".attn_qkv.bias";
    }
    if (rest == "self_attn.q_proj.weight") {
        return "blk." + layer + ".attn_q.weight";
    }
    if (rest == "self_attn.q_proj.bias") {
        return "blk." + layer + ".attn_q.bias";
    }
    if (rest == "self_attn.k_proj.weight") {
        return "blk." + layer + ".attn_k.weight";
    }
    if (rest == "self_attn.k_proj.bias") {
        return "blk." + layer + ".attn_k.bias";
    }
    if (rest == "self_attn.v_proj.weight") {
        return "blk." + layer + ".attn_v.weight";
    }
    if (rest == "self_attn.v_proj.bias") {
        return "blk." + layer + ".attn_v.bias";
    }
    if (rest == "self_attn.o_proj.weight") {
        return "blk." + layer + ".attn_output.weight";
    }
    if (rest == "self_attn.o_proj.bias") {
        return "blk." + layer + ".attn_output.bias";
    }
    if (rest == "self_attn.dense.weight") {
        return "blk." + layer + ".attn_output.weight";
    }
    if (rest == "self_attn.dense.bias") {
        return "blk." + layer + ".attn_output.bias";
    }
    if (rest == "mlp.gate_up_proj.weight") {
        return "blk." + layer + ".ffn_up.weight";
    }
    if (rest == "mlp.fc1.weight") {
        return "blk." + layer + ".ffn_up.weight";
    }
    if (rest == "mlp.fc1.bias") {
        return "blk." + layer + ".ffn_up.bias";
    }
    if (rest == "mlp.down_proj.weight") {
        return "blk." + layer + ".ffn_down.weight";
    }
    if (rest == "mlp.fc2.weight") {
        return "blk." + layer + ".ffn_down.weight";
    }
    if (rest == "mlp.fc2.bias") {
        return "blk." + layer + ".ffn_down.bias";
    }

    return "";
}

std::string map_tensor_name(
        const std::string &source_name,
        const std::string &model_family,
        const std::string &gguf_arch) {
    if (model_family == "QWEN") {
        return map_qwen_tensor_name(source_name);
    }
    if (model_family == "GEMMA") {
        return map_gemma_tensor_name(source_name, gguf_arch);
    }
    if (model_family == "PHI") {
        return map_phi_tensor_name(source_name, gguf_arch);
    }
    return map_llama_tensor_name(source_name);
}

HorizonTensorEncoding tensor_encoding_for_dtype(const std::string &dtype) {
    if (dtype == "F16") {
        return HorizonTensorEncoding::F16;
    }
    if (dtype == "F32") {
        return HorizonTensorEncoding::F32;
    }
    if (dtype == "BF16") {
        return HorizonTensorEncoding::BF16;
    }
    return HorizonTensorEncoding::F32;
}

bool is_supported_f16_output_dtype(const std::string &dtype) {
    return dtype == "F16" || dtype == "F32" || dtype == "BF16";
}

uint64_t tensor_element_count(const std::vector<uint64_t> &shape) {
    uint64_t count = 1;
    for (uint64_t dimension : shape) {
        if (dimension == 0) {
            return 0;
        }
        if (count > UINT64_MAX / dimension) {
            return 0;
        }
        count *= dimension;
    }
    return count;
}

uint64_t safetensors_dtype_bytes(const std::string &dtype) {
    if (dtype == "F16" || dtype == "BF16") {
        return 2;
    }
    if (dtype == "F32") {
        return 4;
    }
    return 0;
}

bool is_supported_native_output_quantization(const std::string &quantization) {
    return quantization == "F16" || quantization == "Q8_0" || quantization == "Q4_0" ||
           quantization == "Q5_0" || quantization == "Q6_K" ||
           quantization == "Q5_K_M" || quantization == "Q4_K_M" || quantization == "Q4_K_S" ||
           quantization == "Q3_K_M";
}

bool is_supported_family_quantization(
        const std::string &model_family,
        const std::string &quantization) {
    if (model_family == "PHI" &&
            quantization != "F16" &&
            quantization != "Q8_0" &&
            quantization != "Q6_K" &&
            quantization != "Q5_K_M" &&
            quantization != "Q4_0" &&
            quantization != "Q4_K_M" &&
            quantization != "Q3_K_M") {
        return false;
    }
    if (quantization == "Q5_0" && model_family != "QWEN") {
        return false;
    }
    if (quantization == "Q4_0" && model_family != "QWEN" && model_family != "GEMMA" &&
            model_family != "PHI") {
        return false;
    }
    if (model_family == "GEMMA" &&
            quantization != "F16" &&
            quantization != "Q8_0" &&
            quantization != "Q6_K" &&
            quantization != "Q5_K_M" &&
            quantization != "Q4_0" &&
            quantization != "Q4_K_M" &&
            quantization != "Q4_K_S" &&
            quantization != "Q3_K_M") {
        return false;
    }
    return true;
}

bool should_quantize_tensor_q8_0(
        const SafetensorsTensor &tensor,
        uint64_t element_count,
        const std::string &model_family) {
    const bool is_llama_permuted_attention =
            model_family == "LLAMA" &&
            (tensor.gguf_name.find(".attn_q.weight") != std::string::npos ||
             tensor.gguf_name.find(".attn_k.weight") != std::string::npos);
    return tensor.shape.size() >= 2 &&
           !tensor.shape.empty() &&
           tensor.gguf_name != "output.weight" &&
           !is_llama_permuted_attention &&
           tensor.shape.back() % 32 == 0 &&
           element_count % 32 == 0;
}

bool should_quantize_tensor_q5_0(
        const SafetensorsTensor &tensor,
        uint64_t element_count,
        const std::string &model_family) {
    const bool is_llama_permuted_attention =
            model_family == "LLAMA" &&
            (tensor.gguf_name.find(".attn_q.weight") != std::string::npos ||
             tensor.gguf_name.find(".attn_k.weight") != std::string::npos);
    return tensor.shape.size() >= 2 &&
           !tensor.shape.empty() &&
           tensor.gguf_name != "output.weight" &&
           !is_llama_permuted_attention &&
           tensor.shape.back() % 32 == 0 &&
           element_count % 32 == 0;
}

bool should_quantize_tensor_q4_0(
        const SafetensorsTensor &tensor,
        uint64_t element_count,
        const std::string &model_family) {
    return should_quantize_tensor_q5_0(tensor, element_count, model_family);
}

bool should_quantize_tensor_q6_k(
        const SafetensorsTensor &tensor,
        uint64_t element_count,
        const std::string &model_family) {
    const bool is_llama_permuted_attention =
            model_family == "LLAMA" &&
            (tensor.gguf_name.find(".attn_q.weight") != std::string::npos ||
             tensor.gguf_name.find(".attn_k.weight") != std::string::npos);
    return tensor.shape.size() >= 2 &&
           !tensor.shape.empty() &&
           tensor.gguf_name != "output.weight" &&
           !is_llama_permuted_attention &&
           tensor.shape.back() % 256 == 0 &&
           element_count % 256 == 0;
}

bool should_quantize_tensor_q5_k(
        const SafetensorsTensor &tensor,
        uint64_t element_count,
        const std::string &model_family) {
    const bool is_llama_permuted_attention =
            model_family == "LLAMA" &&
            (tensor.gguf_name.find(".attn_q.weight") != std::string::npos ||
             tensor.gguf_name.find(".attn_k.weight") != std::string::npos);
    return tensor.shape.size() >= 2 &&
           !tensor.shape.empty() &&
           tensor.gguf_name != "output.weight" &&
           !is_llama_permuted_attention &&
           tensor.shape.back() % 256 == 0 &&
           element_count % 256 == 0;
}

bool should_quantize_tensor_q4_k(
        const SafetensorsTensor &tensor,
        uint64_t element_count,
        const std::string &model_family) {
    const bool is_llama_permuted_attention =
            model_family == "LLAMA" &&
            (tensor.gguf_name.find(".attn_q.weight") != std::string::npos ||
             tensor.gguf_name.find(".attn_k.weight") != std::string::npos);
    return tensor.shape.size() >= 2 &&
           !tensor.shape.empty() &&
           tensor.gguf_name != "output.weight" &&
           !is_llama_permuted_attention &&
           tensor.shape.back() % 256 == 0 &&
           element_count % 256 == 0;
}

bool should_quantize_tensor_q3_k(
        const SafetensorsTensor &tensor,
        uint64_t element_count,
        const std::string &model_family) {
    const bool is_llama_permuted_attention =
            model_family == "LLAMA" &&
            (tensor.gguf_name.find(".attn_q.weight") != std::string::npos ||
             tensor.gguf_name.find(".attn_k.weight") != std::string::npos);
    return tensor.shape.size() >= 2 &&
           !tensor.shape.empty() &&
           tensor.gguf_name != "output.weight" &&
           !is_llama_permuted_attention &&
           tensor.shape.back() % 256 == 0 &&
           element_count % 256 == 0;
}

bool should_keep_tensor_f32(const SafetensorsTensor &tensor) {
    return tensor.shape.size() <= 1 || ends_with(tensor.gguf_name, "_norm.weight");
}

float source_float_add_for_tensor(
        const SafetensorsTensor &tensor,
        const std::string &model_family) {
    return model_family == "GEMMA" && ends_with(tensor.source_name, "norm.weight") ? 1.0f : 0.0f;
}

uint32_t row_permutation_heads_for_tensor(
        const SafetensorsTensor &tensor,
        uint32_t attention_head_count,
        uint32_t key_value_head_count,
        const std::string &model_family) {
    if (model_family != "LLAMA") {
        return 0;
    }
    if (tensor.shape.size() != 2) {
        return 0;
    }
    if (tensor.gguf_name.find(".attn_q.weight") != std::string::npos) {
        return attention_head_count;
    }
    if (tensor.gguf_name.find(".attn_k.weight") != std::string::npos) {
        return key_value_head_count == 0 ? attention_head_count : key_value_head_count;
    }
    return 0;
}

uint32_t output_ggml_type_for_tensor(
        const std::string &quantization,
        const SafetensorsTensor &tensor,
        uint64_t element_count,
        const std::string &model_family) {
    if (should_keep_tensor_f32(tensor)) {
        return 0;
    }
    if (quantization == "Q8_0" && should_quantize_tensor_q8_0(tensor, element_count, model_family)) {
        return 8;
    }
    if (quantization == "Q5_0" && should_quantize_tensor_q5_0(tensor, element_count, model_family)) {
        return 6;
    }
    if (quantization == "Q4_0" && should_quantize_tensor_q4_0(tensor, element_count, model_family)) {
        return 2;
    }
    if (quantization == "Q6_K" && should_quantize_tensor_q6_k(tensor, element_count, model_family)) {
        return 14;
    }
    if (quantization == "Q6_K" && should_quantize_tensor_q8_0(tensor, element_count, model_family)) {
        return 8;
    }
    if ((quantization == "Q5_K_M" || quantization == "Q5_K_S") &&
            should_quantize_tensor_q5_k(tensor, element_count, model_family)) {
        return 13;
    }
    if (quantization == "Q5_K_M" && should_quantize_tensor_q8_0(tensor, element_count, model_family)) {
        return 8;
    }
    if ((quantization == "Q4_K_M" || quantization == "Q4_K_S") &&
            should_quantize_tensor_q4_k(tensor, element_count, model_family)) {
        return 12;
    }
    if ((quantization == "Q4_K_M" || quantization == "Q4_K_S") &&
            should_quantize_tensor_q5_0(tensor, element_count, model_family)) {
        return 6;
    }
    if (quantization == "Q3_K_M" && should_quantize_tensor_q3_k(tensor, element_count, model_family)) {
        return 11;
    }
    if (quantization == "Q3_K_M" &&
            (model_family == "QWEN" || model_family == "GEMMA" || model_family == "PHI") &&
            should_quantize_tensor_q4_0(tensor, element_count, model_family)) {
        return 2;
    }
    return 1;
}

HorizonTensorOutputEncoding output_encoding_for_tensor(
        const std::string &quantization,
        const SafetensorsTensor &tensor,
        uint64_t element_count,
        const std::string &model_family) {
    if (should_keep_tensor_f32(tensor)) {
        return HorizonTensorOutputEncoding::F32;
    }
    if (quantization == "Q8_0" && should_quantize_tensor_q8_0(tensor, element_count, model_family)) {
        return HorizonTensorOutputEncoding::Q8_0;
    }
    if (quantization == "Q5_0" && should_quantize_tensor_q5_0(tensor, element_count, model_family)) {
        return HorizonTensorOutputEncoding::Q5_0;
    }
    if (quantization == "Q4_0" && should_quantize_tensor_q4_0(tensor, element_count, model_family)) {
        return HorizonTensorOutputEncoding::Q4_0;
    }
    if (quantization == "Q6_K" && should_quantize_tensor_q6_k(tensor, element_count, model_family)) {
        return HorizonTensorOutputEncoding::Q6_K;
    }
    if (quantization == "Q6_K" && should_quantize_tensor_q8_0(tensor, element_count, model_family)) {
        return HorizonTensorOutputEncoding::Q8_0;
    }
    if ((quantization == "Q5_K_M" || quantization == "Q5_K_S") &&
            should_quantize_tensor_q5_k(tensor, element_count, model_family)) {
        return HorizonTensorOutputEncoding::Q5_K;
    }
    if (quantization == "Q5_K_M" && should_quantize_tensor_q8_0(tensor, element_count, model_family)) {
        return HorizonTensorOutputEncoding::Q8_0;
    }
    if ((quantization == "Q4_K_M" || quantization == "Q4_K_S") &&
            should_quantize_tensor_q4_k(tensor, element_count, model_family)) {
        return HorizonTensorOutputEncoding::Q4_K;
    }
    if ((quantization == "Q4_K_M" || quantization == "Q4_K_S") &&
            should_quantize_tensor_q5_0(tensor, element_count, model_family)) {
        return HorizonTensorOutputEncoding::Q5_0;
    }
    if (quantization == "Q3_K_M" && should_quantize_tensor_q3_k(tensor, element_count, model_family)) {
        return HorizonTensorOutputEncoding::Q3_K;
    }
    if (quantization == "Q3_K_M" &&
            (model_family == "QWEN" || model_family == "GEMMA" || model_family == "PHI") &&
            should_quantize_tensor_q4_0(tensor, element_count, model_family)) {
        return HorizonTensorOutputEncoding::Q4_0;
    }
    return HorizonTensorOutputEncoding::F16;
}

uint64_t output_data_size_for_tensor(
        const std::string &quantization,
        const SafetensorsTensor &tensor,
        uint64_t element_count,
        const std::string &model_family) {
    if (should_keep_tensor_f32(tensor)) {
        return element_count * 4;
    }
    if (quantization == "Q8_0" && should_quantize_tensor_q8_0(tensor, element_count, model_family)) {
        return (element_count / 32) * 34;
    }
    if (quantization == "Q5_0" && should_quantize_tensor_q5_0(tensor, element_count, model_family)) {
        return (element_count / 32) * 22;
    }
    if (quantization == "Q4_0" && should_quantize_tensor_q4_0(tensor, element_count, model_family)) {
        return (element_count / 32) * 18;
    }
    if (quantization == "Q6_K" && should_quantize_tensor_q6_k(tensor, element_count, model_family)) {
        return (element_count / 256) * 210;
    }
    if (quantization == "Q6_K" && should_quantize_tensor_q8_0(tensor, element_count, model_family)) {
        return (element_count / 32) * 34;
    }
    if ((quantization == "Q5_K_M" || quantization == "Q5_K_S") &&
            should_quantize_tensor_q5_k(tensor, element_count, model_family)) {
        return (element_count / 256) * 176;
    }
    if (quantization == "Q5_K_M" && should_quantize_tensor_q8_0(tensor, element_count, model_family)) {
        return (element_count / 32) * 34;
    }
    if ((quantization == "Q4_K_M" || quantization == "Q4_K_S") &&
            should_quantize_tensor_q4_k(tensor, element_count, model_family)) {
        return (element_count / 256) * 144;
    }
    if ((quantization == "Q4_K_M" || quantization == "Q4_K_S") &&
            should_quantize_tensor_q5_0(tensor, element_count, model_family)) {
        return (element_count / 32) * 22;
    }
    if (quantization == "Q3_K_M" && should_quantize_tensor_q3_k(tensor, element_count, model_family)) {
        return (element_count / 256) * 110;
    }
    if (quantization == "Q3_K_M" &&
            (model_family == "QWEN" || model_family == "GEMMA" || model_family == "PHI") &&
            should_quantize_tensor_q4_0(tensor, element_count, model_family)) {
        return (element_count / 32) * 18;
    }
    return element_count * 2;
}

SafetensorsTensor tensor_with_shape(
        const SafetensorsTensor &tensor,
        const std::vector<uint64_t> &shape) {
    SafetensorsTensor output = tensor;
    output.shape = shape;
    return output;
}

std::vector<SafetensorsTensor> parse_safetensors_tensors(
        const std::string &header,
        const WorkspaceFile &source_file,
        uint64_t header_length,
        const std::string &model_family,
        const std::string &gguf_arch) {
    std::vector<SafetensorsTensor> tensors;
    size_t cursor = 0;
    while (cursor < header.size()) {
        size_t name_start = header.find('"', cursor);
        if (name_start == std::string::npos) {
            break;
        }
        size_t name_end = header.find('"', name_start + 1);
        if (name_end == std::string::npos) {
            break;
        }
        std::string name = header.substr(name_start + 1, name_end - name_start - 1);
        cursor = name_end + 1;

        if (name == "__metadata__") {
            size_t metadata_start = header.find('{', cursor);
            size_t metadata_end = metadata_start == std::string::npos
                    ? std::string::npos
                    : find_matching_brace(header, metadata_start);
            cursor = metadata_end == std::string::npos ? cursor : metadata_end + 1;
            continue;
        }

        size_t colon = header.find(':', cursor);
        size_t object_start = header.find('{', colon == std::string::npos ? cursor : colon + 1);
        if (colon == std::string::npos || object_start == std::string::npos) {
            continue;
        }
        size_t object_end = find_matching_brace(header, object_start);
        if (object_end == std::string::npos) {
            break;
        }

        const std::string object_json = header.substr(object_start, object_end - object_start + 1);
        const std::vector<uint64_t> offsets = extract_uint_array(object_json, "data_offsets");
        if (object_json.find("\"data_offsets\"") != std::string::npos && offsets.size() == 2) {
            SafetensorsTensor tensor {
                    name,
                    map_tensor_name(name, model_family, gguf_arch),
                    extract_object_string(object_json, "dtype"),
                    extract_uint_array(object_json, "shape"),
                    source_file.path,
                    8 + header_length + offsets[0],
                    offsets[0],
                    offsets[1],
            };
            tensors.push_back(tensor);
        }
        cursor = object_end + 1;
    }
    return tensors;
}

std::string extract_first_architecture(const std::string &config_json) {
    const std::string key = "\"architectures\"";
    size_t key_pos = config_json.find(key);
    if (key_pos == std::string::npos) {
        return "";
    }
    size_t array_start = config_json.find('[', key_pos + key.size());
    if (array_start == std::string::npos) {
        return "";
    }
    size_t quote = config_json.find('"', array_start + 1);
    if (quote == std::string::npos) {
        return "";
    }
    size_t end = config_json.find('"', quote + 1);
    if (end == std::string::npos) {
        return "";
    }
    return config_json.substr(quote + 1, end - quote - 1);
}

uint32_t extract_json_uint32(const std::string &json, const std::string &key, uint32_t fallback) {
    const std::string needle = "\"" + key + "\"";
    size_t key_pos = json.find(needle);
    if (key_pos == std::string::npos) {
        return fallback;
    }
    size_t colon = json.find(':', key_pos + needle.size());
    if (colon == std::string::npos) {
        return fallback;
    }
    size_t cursor = colon + 1;
    while (cursor < json.size() && !std::isdigit(static_cast<unsigned char>(json[cursor]))) {
        cursor += 1;
    }
    if (cursor >= json.size()) {
        return fallback;
    }
    size_t end = cursor;
    while (end < json.size() && std::isdigit(static_cast<unsigned char>(json[end]))) {
        end += 1;
    }
    return static_cast<uint32_t>(std::stoul(json.substr(cursor, end - cursor)));
}

float extract_json_float(const std::string &json, const std::string &key, float fallback) {
    const std::string needle = "\"" + key + "\"";
    size_t key_pos = json.find(needle);
    if (key_pos == std::string::npos) {
        return fallback;
    }
    size_t colon = json.find(':', key_pos + needle.size());
    if (colon == std::string::npos) {
        return fallback;
    }
    size_t cursor = colon + 1;
    while (cursor < json.size() &&
           !(std::isdigit(static_cast<unsigned char>(json[cursor])) || json[cursor] == '-' || json[cursor] == '.')) {
        cursor += 1;
    }
    if (cursor >= json.size()) {
        return fallback;
    }
    size_t end = cursor;
    while (end < json.size() &&
           (std::isdigit(static_cast<unsigned char>(json[end])) ||
            json[end] == '-' ||
            json[end] == '+' ||
            json[end] == '.' ||
            json[end] == 'e' ||
            json[end] == 'E')) {
        end += 1;
    }
    return std::stof(json.substr(cursor, end - cursor));
}

bool has_json_number(const std::string &json, const std::string &key) {
    const std::string needle = "\"" + key + "\"";
    size_t key_pos = json.find(needle);
    if (key_pos == std::string::npos) {
        return false;
    }
    size_t colon = json.find(':', key_pos + needle.size());
    if (colon == std::string::npos) {
        return false;
    }
    size_t cursor = colon + 1;
    while (cursor < json.size() && std::isspace(static_cast<unsigned char>(json[cursor]))) {
        cursor += 1;
    }
    return cursor < json.size() &&
           (std::isdigit(static_cast<unsigned char>(json[cursor])) ||
            json[cursor] == '-' ||
            json[cursor] == '.');
}

std::vector<std::string> extract_tokenizer_json_vocab(
        const std::string &tokenizer_json,
        std::vector<float> &scores) {
    const std::string vocab_key = "\"vocab\"";
    size_t key_pos = tokenizer_json.find(vocab_key);
    if (key_pos == std::string::npos) {
        return {};
    }
    size_t colon = tokenizer_json.find(':', key_pos + vocab_key.size());
    if (colon == std::string::npos) {
        return {};
    }
    size_t value_start = colon + 1;
    while (value_start < tokenizer_json.size() &&
           std::isspace(static_cast<unsigned char>(tokenizer_json[value_start]))) {
        value_start += 1;
    }
    if (value_start >= tokenizer_json.size()) {
        return {};
    }

    std::map<uint32_t, std::string> id_to_token;
    std::map<uint32_t, float> id_to_score;
    if (tokenizer_json[value_start] == '{') {
        const size_t object_start = value_start;
        const size_t object_end = find_matching_brace(tokenizer_json, object_start);
        if (object_end == std::string::npos) {
            return {};
        }

        size_t cursor = object_start + 1;
        while (cursor < object_end) {
            size_t token_start = tokenizer_json.find('"', cursor);
            if (token_start == std::string::npos || token_start >= object_end) {
                break;
            }
            size_t token_end = token_start + 1;
            bool escaped = false;
            while (token_end < object_end) {
                const char value = tokenizer_json[token_end];
                if (escaped) {
                    escaped = false;
                } else if (value == '\\') {
                    escaped = true;
                } else if (value == '"') {
                    break;
                }
                token_end += 1;
            }
            if (token_end >= object_end) {
                break;
            }

            size_t id_colon = tokenizer_json.find(':', token_end + 1);
            if (id_colon == std::string::npos || id_colon >= object_end) {
                break;
            }
            size_t id_start = id_colon + 1;
            while (id_start < object_end && !std::isdigit(static_cast<unsigned char>(tokenizer_json[id_start]))) {
                id_start += 1;
            }
            if (id_start >= object_end) {
                break;
            }
            size_t id_end = id_start;
            while (id_end < object_end && std::isdigit(static_cast<unsigned char>(tokenizer_json[id_end]))) {
                id_end += 1;
            }

            const std::string raw_token = tokenizer_json.substr(token_start + 1, token_end - token_start - 1);
            const uint32_t token_id =
                    static_cast<uint32_t>(std::stoul(tokenizer_json.substr(id_start, id_end - id_start)));
            id_to_token[token_id] = unescape_json_string(raw_token);
            id_to_score[token_id] = 0.0f;
            cursor = id_end + 1;
        }
    } else if (tokenizer_json[value_start] == '[') {
        uint32_t token_id = 0;
        size_t cursor = value_start + 1;
        while (cursor < tokenizer_json.size()) {
            while (cursor < tokenizer_json.size() &&
                   std::isspace(static_cast<unsigned char>(tokenizer_json[cursor]))) {
                cursor += 1;
            }
            if (cursor >= tokenizer_json.size() || tokenizer_json[cursor] == ']') {
                break;
            }
            if (tokenizer_json[cursor] != '[') {
                cursor += 1;
                continue;
            }
            size_t token_start = tokenizer_json.find('"', cursor + 1);
            if (token_start == std::string::npos) {
                break;
            }
            size_t token_end = token_start + 1;
            bool escaped = false;
            while (token_end < tokenizer_json.size()) {
                const char value = tokenizer_json[token_end];
                if (escaped) {
                    escaped = false;
                } else if (value == '\\') {
                    escaped = true;
                } else if (value == '"') {
                    break;
                }
                token_end += 1;
            }
            if (token_end >= tokenizer_json.size()) {
                break;
            }
            size_t comma = tokenizer_json.find(',', token_end + 1);
            size_t entry_end = tokenizer_json.find(']', token_end + 1);
            if (comma == std::string::npos || entry_end == std::string::npos || comma > entry_end) {
                break;
            }
            size_t score_start = comma + 1;
            while (score_start < entry_end &&
                   std::isspace(static_cast<unsigned char>(tokenizer_json[score_start]))) {
                score_start += 1;
            }
            size_t score_end = score_start;
            while (score_end < entry_end &&
                   (std::isdigit(static_cast<unsigned char>(tokenizer_json[score_end])) ||
                    tokenizer_json[score_end] == '-' ||
                    tokenizer_json[score_end] == '+' ||
                    tokenizer_json[score_end] == '.' ||
                    tokenizer_json[score_end] == 'e' ||
                    tokenizer_json[score_end] == 'E')) {
                score_end += 1;
            }
            const std::string raw_token = tokenizer_json.substr(token_start + 1, token_end - token_start - 1);
            id_to_token[token_id] = unescape_json_string(raw_token);
            id_to_score[token_id] = score_end > score_start
                    ? std::stof(tokenizer_json.substr(score_start, score_end - score_start))
                    : 0.0f;
            token_id += 1;
            cursor = entry_end + 1;
        }
    }

    const std::string added_key = "\"added_tokens\"";
    key_pos = tokenizer_json.find(added_key);
    if (key_pos != std::string::npos) {
        size_t array_start = tokenizer_json.find('[', key_pos + added_key.size());
        size_t added_cursor = array_start == std::string::npos ? std::string::npos : array_start + 1;
        while (added_cursor != std::string::npos && added_cursor < tokenizer_json.size()) {
            size_t array_end = tokenizer_json.find(']', added_cursor);
            size_t added_object_start = tokenizer_json.find('{', added_cursor);
            if (array_end != std::string::npos &&
                (added_object_start == std::string::npos || array_end < added_object_start)) {
                break;
            }
            if (added_object_start == std::string::npos) {
                break;
            }
            size_t added_object_end = find_matching_brace(tokenizer_json, added_object_start);
            if (added_object_end == std::string::npos) {
                break;
            }

            const std::string added_object =
                    tokenizer_json.substr(added_object_start, added_object_end - added_object_start + 1);
            const uint32_t token_id = extract_json_uint32(added_object, "id", UINT32_MAX);
            const std::string token_content = extract_json_string(added_object, "content");
            if (token_id != UINT32_MAX && !token_content.empty()) {
                id_to_token[token_id] = token_content;
                id_to_score[token_id] = 0.0f;
            }
            added_cursor = added_object_end + 1;
        }
    }

    if (id_to_token.empty()) {
        return {};
    }
    const uint32_t max_id = id_to_token.rbegin()->first;
    std::vector<std::string> tokens(static_cast<size_t>(max_id) + 1);
    scores.assign(tokens.size(), 0.0f);
    for (const auto &entry : id_to_token) {
        tokens[entry.first] = entry.second;
    }
    for (const auto &entry : id_to_score) {
        if (entry.first < scores.size()) {
            scores[entry.first] = entry.second;
        }
    }
    return tokens;
}

void normalize_qwen_tokens(
        std::vector<std::string> &tokens,
        uint32_t config_vocab_size) {
    if (config_vocab_size > tokens.size()) {
        tokens.resize(config_vocab_size);
    }
    for (size_t index = 0; index < tokens.size(); ++index) {
        if (tokens[index].empty()) {
            tokens[index] = "[PAD" + std::to_string(index) + "]";
        }
    }
}

void normalize_gemma_tokens(
        std::vector<std::string> &tokens,
        std::vector<float> &scores,
        uint32_t config_vocab_size) {
    if (config_vocab_size > 0 && tokens.size() > config_vocab_size) {
        tokens.resize(config_vocab_size);
        if (scores.size() > config_vocab_size) {
            scores.resize(config_vocab_size);
        }
    }

    std::unordered_set<std::string> seen;
    seen.reserve(tokens.size());
    for (size_t index = 0; index < tokens.size(); ++index) {
        std::string token = tokens[index];
        if (!token.empty() && seen.insert(token).second) {
            continue;
        }

        std::string replacement;
        size_t candidate = index;
        do {
            replacement = "[PAD" + std::to_string(candidate) + "]";
            ++candidate;
        } while (seen.find(replacement) != seen.end());

        tokens[index] = replacement;
        seen.insert(replacement);
    }
}

void normalize_phi_tokens(
        std::vector<std::string> &tokens,
        std::vector<float> &scores,
        uint32_t config_vocab_size) {
    if (config_vocab_size > tokens.size()) {
        tokens.resize(config_vocab_size);
    }
    if (scores.size() < tokens.size()) {
        scores.resize(tokens.size(), 0.0f);
    }

    std::unordered_set<std::string> seen;
    seen.reserve(tokens.size());
    for (size_t index = 0; index < tokens.size(); ++index) {
        std::string token = tokens[index];
        if (!token.empty() && seen.insert(token).second) {
            continue;
        }

        std::string replacement;
        size_t candidate = index;
        do {
            replacement = "[PAD" + std::to_string(candidate) + "]";
            ++candidate;
        } while (seen.find(replacement) != seen.end());

        tokens[index] = replacement;
        seen.insert(replacement);
    }
}

std::string extract_tokenizer_json_model_type(const std::string &tokenizer_json) {
    const std::string model_key = "\"model\"";
    size_t key_pos = tokenizer_json.find(model_key);
    if (key_pos == std::string::npos) {
        return "";
    }
    size_t object_start = tokenizer_json.find('{', key_pos + model_key.size());
    if (object_start == std::string::npos) {
        return "";
    }
    size_t object_end = find_matching_brace(tokenizer_json, object_start);
    if (object_end == std::string::npos) {
        return "";
    }
    return extract_json_string(tokenizer_json.substr(object_start, object_end - object_start + 1), "type");
}

std::vector<std::string> extract_tokenizer_json_merges(const std::string &tokenizer_json) {
    const std::string merges_key = "\"merges\"";
    size_t key_pos = tokenizer_json.find(merges_key);
    if (key_pos == std::string::npos) {
        return {};
    }
    size_t array_start = tokenizer_json.find('[', key_pos + merges_key.size());
    if (array_start == std::string::npos) {
        return {};
    }
    const size_t array_end = find_matching_bracket(tokenizer_json, array_start);
    if (array_end == std::string::npos) {
        return {};
    }

    std::vector<std::string> merges;
    size_t cursor = array_start + 1;
    while (cursor < array_end) {
        while (cursor < array_end &&
               (std::isspace(static_cast<unsigned char>(tokenizer_json[cursor])) ||
                tokenizer_json[cursor] == ',')) {
            cursor += 1;
        }
        if (cursor >= array_end) {
            break;
        }
        if (tokenizer_json[cursor] == '[') {
            const size_t pair_end = find_matching_bracket(tokenizer_json, cursor);
            if (pair_end == std::string::npos || pair_end > array_end) {
                break;
            }
            size_t left_start = tokenizer_json.find('"', cursor + 1);
            if (left_start == std::string::npos || left_start >= pair_end) {
                cursor = pair_end + 1;
                continue;
            }
            std::string left;
            size_t after_left = 0;
            if (!read_json_string_at(tokenizer_json, left_start, left, after_left)) {
                cursor = pair_end + 1;
                continue;
            }
            size_t right_start = tokenizer_json.find('"', after_left);
            if (right_start == std::string::npos || right_start >= pair_end) {
                cursor = pair_end + 1;
                continue;
            }
            std::string right;
            size_t after_right = 0;
            if (read_json_string_at(tokenizer_json, right_start, right, after_right)) {
                merges.push_back(left + " " + right);
            }
            cursor = pair_end + 1;
        } else if (tokenizer_json[cursor] == '"') {
            std::string item;
            size_t next = 0;
            if (read_json_string_at(tokenizer_json, cursor, item, next)) {
                merges.push_back(item);
                cursor = next;
            } else {
                break;
            }
        } else {
            cursor += 1;
        }
    }
    return merges;
}

bool object_json_has_true(const std::string &object_json, const std::string &key) {
    const std::string needle = "\"" + key + "\"";
    size_t key_pos = object_json.find(needle);
    if (key_pos == std::string::npos) {
        return false;
    }
    size_t colon = object_json.find(':', key_pos + needle.size());
    if (colon == std::string::npos) {
        return false;
    }
    size_t cursor = colon + 1;
    while (cursor < object_json.size() && std::isspace(static_cast<unsigned char>(object_json[cursor]))) {
        cursor += 1;
    }
    return object_json.compare(cursor, 4, "true") == 0;
}

std::map<uint32_t, bool> extract_added_token_special_map(const std::string &tokenizer_json) {
    std::map<uint32_t, bool> added_tokens;
    const std::string added_key = "\"added_tokens\"";
    size_t key_pos = tokenizer_json.find(added_key);
    if (key_pos == std::string::npos) {
        return added_tokens;
    }
    size_t array_start = tokenizer_json.find('[', key_pos + added_key.size());
    if (array_start == std::string::npos) {
        return added_tokens;
    }

    size_t cursor = array_start + 1;
    while (cursor < tokenizer_json.size()) {
        size_t array_end = tokenizer_json.find(']', cursor);
        size_t object_start = tokenizer_json.find('{', cursor);
        if (array_end != std::string::npos && (object_start == std::string::npos || array_end < object_start)) {
            break;
        }
        if (object_start == std::string::npos) {
            break;
        }
        size_t object_end = find_matching_brace(tokenizer_json, object_start);
        if (object_end == std::string::npos) {
            break;
        }

        const std::string object_json = tokenizer_json.substr(object_start, object_end - object_start + 1);
        const uint32_t token_id = extract_json_uint32(object_json, "id", UINT32_MAX);
        if (token_id != UINT32_MAX) {
            added_tokens[token_id] = object_json_has_true(object_json, "special");
        }
        cursor = object_end + 1;
    }

    return added_tokens;
}

uint32_t token_id_for_content(
        const std::string &tokenizer_json,
        const std::string &content,
        uint32_t fallback) {
    const std::string added_key = "\"added_tokens\"";
    size_t key_pos = tokenizer_json.find(added_key);
    if (key_pos == std::string::npos) {
        return fallback;
    }
    size_t array_start = tokenizer_json.find('[', key_pos + added_key.size());
    if (array_start == std::string::npos) {
        return fallback;
    }

    size_t cursor = array_start + 1;
    while (cursor < tokenizer_json.size()) {
        size_t array_end = tokenizer_json.find(']', cursor);
        size_t object_start = tokenizer_json.find('{', cursor);
        if (array_end != std::string::npos && (object_start == std::string::npos || array_end < object_start)) {
            break;
        }
        if (object_start == std::string::npos) {
            break;
        }
        size_t object_end = find_matching_brace(tokenizer_json, object_start);
        if (object_end == std::string::npos) {
            break;
        }

        const std::string object_json = tokenizer_json.substr(object_start, object_end - object_start + 1);
        if (extract_json_string(object_json, "content") == content) {
            return extract_json_uint32(object_json, "id", fallback);
        }
        cursor = object_end + 1;
    }

    return fallback;
}

uint32_t token_id_for_token(
        const std::vector<std::string> &tokens,
        const std::string &content,
        uint32_t fallback) {
    for (size_t index = 0; index < tokens.size(); ++index) {
        if (tokens[index] == content) {
            return static_cast<uint32_t>(index);
        }
    }
    return fallback;
}

std::vector<int32_t> build_token_types(const std::string &tokenizer_json, size_t token_count) {
    constexpr int32_t kNormalToken = 1;
    constexpr int32_t kControlToken = 3;
    std::vector<int32_t> token_types(token_count, kNormalToken);

    const std::string added_key = "\"added_tokens\"";
    size_t key_pos = tokenizer_json.find(added_key);
    if (key_pos == std::string::npos) {
        return token_types;
    }
    size_t array_start = tokenizer_json.find('[', key_pos + added_key.size());
    if (array_start == std::string::npos) {
        return token_types;
    }

    size_t cursor = array_start + 1;
    while (cursor < tokenizer_json.size()) {
        size_t array_end = tokenizer_json.find(']', cursor);
        size_t object_start = tokenizer_json.find('{', cursor);
        if (array_end != std::string::npos && (object_start == std::string::npos || array_end < object_start)) {
            break;
        }
        if (object_start == std::string::npos) {
            break;
        }
        size_t object_end = find_matching_brace(tokenizer_json, object_start);
        if (object_end == std::string::npos) {
            break;
        }

        const std::string object_json = tokenizer_json.substr(object_start, object_end - object_start + 1);
        const uint32_t token_id = extract_json_uint32(object_json, "id", UINT32_MAX);
        if (token_id != UINT32_MAX && token_id < token_types.size() && object_json_has_true(object_json, "special")) {
            token_types[token_id] = kControlToken;
        }
        cursor = object_end + 1;
    }

    return token_types;
}

std::vector<int32_t> build_token_types(
        const std::string &tokenizer_json,
        const std::vector<std::string> &tokens,
        const std::string &model_family) {
    std::vector<int32_t> token_types = build_token_types(tokenizer_json, tokens.size());
    constexpr int32_t kUnknownToken = 2;
    constexpr int32_t kByteToken = 6;
    constexpr int32_t kControlToken = 3;
    constexpr int32_t kUserDefinedToken = 4;
    constexpr int32_t kUnusedToken = 5;

    if (model_family == "QWEN") {
        const std::map<uint32_t, bool> added_tokens = extract_added_token_special_map(tokenizer_json);
        for (size_t index = 0; index < tokens.size(); ++index) {
            const auto added = added_tokens.find(static_cast<uint32_t>(index));
            if (added != added_tokens.end()) {
                token_types[index] = added->second ? kControlToken : kUserDefinedToken;
            } else if (tokens[index].rfind("[PAD", 0) == 0 || tokens[index].rfind("[unused", 0) == 0) {
                token_types[index] = kUnusedToken;
            }
        }
    }
    if (model_family == "GEMMA") {
        for (size_t index = 0; index < tokens.size(); ++index) {
            const std::string &token = tokens[index];
            if (token == "<unk>") {
                token_types[index] = kUnknownToken;
            } else if (token == "<pad>" ||
                       token == "<eos>" ||
                       token == "<bos>" ||
                       token == "<mask>" ||
                       token.rfind("<start_of_", 0) == 0 ||
                       token.rfind("<end_of_", 0) == 0 ||
                       token.rfind("<unused", 0) == 0) {
                token_types[index] = kControlToken;
            }
        }
    }

    for (size_t index = 0; index < tokens.size(); ++index) {
        const std::string &token = tokens[index];
        if (token.size() == 6 &&
            token[0] == '<' &&
            token[1] == '0' &&
            token[2] == 'x' &&
            std::isxdigit(static_cast<unsigned char>(token[3])) &&
            std::isxdigit(static_cast<unsigned char>(token[4])) &&
            token[5] == '>') {
            token_types[index] = kByteToken;
        }
    }

    return token_types;
}

std::string tokenizer_pre_for_model(
        const std::string &model_family,
        const std::string &tokenizer_model,
        uint32_t vocab_size) {
    if (model_family == "GEMMA" && tokenizer_model == "gemma4") {
        return "gemma4";
    }
    if (tokenizer_model != "gpt2") {
        return "default";
    }
    if (model_family == "QWEN") {
        return "qwen2";
    }
    if (model_family == "PHI" && vocab_size == 51200) {
        return "phi-2";
    }
    if (vocab_size == 49152) {
        return "smollm";
    }
    return "default";
}

bool should_add_bos_token(
        const std::string &model_family,
        uint32_t vocab_size) {
    if (model_family == "QWEN") {
        return false;
    }
    return vocab_size != 49152;
}

std::string gguf_architecture_name(
        const std::string &architecture,
        const std::string &model_type,
        const std::string &model_family) {
    std::string source = !model_type.empty() ? model_type : architecture;
    std::transform(source.begin(), source.end(), source.begin(), [](char value) {
        return std::tolower(static_cast<unsigned char>(value));
    });
    if (model_family == "QWEN") {
        if (source.find("qwen3") != std::string::npos) {
            return "qwen3";
        }
        return "qwen2";
    }
    if (model_family == "GEMMA") {
        if (source.find("gemma3n") != std::string::npos || source.find("gemma-3n") != std::string::npos) {
            return "gemma3n";
        }
        if (source.find("gemma3") != std::string::npos) {
            return "gemma3";
        }
        if (source.find("gemma2") != std::string::npos) {
            return "gemma2";
        }
        return "gemma";
    }
    if (model_family == "PHI") {
        if (source.find("phimoe") != std::string::npos || source.find("phi_moe") != std::string::npos) {
            return "phimoe";
        }
        if (source.find("phi3") != std::string::npos || source.find("phi-3") != std::string::npos ||
                source.find("phi4") != std::string::npos || source.find("phi-4") != std::string::npos) {
            return "phi3";
        }
        return "phi2";
    }
    if (source.find("mistral") != std::string::npos) {
        return "llama";
    }
    if (source.find("llama") != std::string::npos) {
        return "llama";
    }
    return source;
}

uint32_t gguf_file_type(const std::string &quantization) {
    if (quantization == "F16") {
        return 1;
    }
    if (quantization == "Q8_0") {
        return 7;
    }
    if (quantization == "Q4_0") {
        return 2;
    }
    if (quantization == "Q5_0") {
        return 8;
    }
    if (quantization == "Q4_K_S") {
        return 14;
    }
    if (quantization == "Q4_K_M") {
        return 15;
    }
    if (quantization == "Q5_K_S") {
        return 16;
    }
    if (quantization == "Q5_K_M") {
        return 17;
    }
    if (quantization == "Q6_K") {
        return 18;
    }
    if (quantization == "Q3_K_M") {
        return 12;
    }
    return 0;
}

HorizonGgufMetadataWriter build_metadata_writer(
        const std::string &config_json,
        const std::string &tokenizer_json,
        const std::vector<std::string> &tokens,
        const std::vector<float> &scores,
        const std::string &tokenizer_model,
        const std::vector<std::string> &merges,
        const std::vector<int32_t> &token_types,
        const std::string &architecture,
        const std::string &model_type,
        const std::string &quantization,
        const std::string &model_family) {
    HorizonGgufMetadataWriter writer;
    const std::string gguf_arch = gguf_architecture_name(architecture, model_type, model_family);

    writer.add_string("general.architecture", gguf_arch.empty() ? "llama" : gguf_arch);
    writer.add_string("general.name", "HorizonAssimilator converted model");
    writer.add_uint32("general.alignment", 32);
    writer.add_uint32("general.file_type", gguf_file_type(quantization));
    writer.add_uint32("general.quantization_version", 2);

    const std::string prefix = gguf_arch.empty() ? "llama" : gguf_arch;
    const uint32_t vocab_size = static_cast<uint32_t>(tokens.size());
    writer.add_uint32(
            prefix + ".context_length",
            extract_json_uint32(
                    config_json,
                    "max_position_embeddings",
                    extract_json_uint32(config_json, "n_positions", 0)));
    writer.add_uint32(prefix + ".vocab_size", vocab_size);
    const uint32_t embedding_length = extract_json_uint32(
            config_json,
            "hidden_size",
            extract_json_uint32(config_json, "n_embd", 0));
    writer.add_uint32(prefix + ".embedding_length", embedding_length);
    writer.add_uint32(
            prefix + ".block_count",
            extract_json_uint32(
                    config_json,
                    "num_hidden_layers",
                    extract_json_uint32(config_json, "n_layer", 0)));
    const uint32_t feed_forward_length = model_family == "PHI" && gguf_arch == "phi2"
            ? 4 * embedding_length
            : extract_json_uint32(config_json, "intermediate_size", 0);
    writer.add_uint32(prefix + ".feed_forward_length", feed_forward_length);
    const uint32_t head_count = extract_json_uint32(
            config_json,
            "num_attention_heads",
            extract_json_uint32(config_json, "n_head", 0));
    writer.add_uint32(prefix + ".attention.head_count", head_count);
    uint32_t head_count_kv = extract_json_uint32(
            config_json,
            "num_key_value_heads",
            extract_json_uint32(config_json, "n_head_kv", head_count));
    if (model_family == "PHI" && head_count_kv == 0) {
        head_count_kv = head_count;
    }
    writer.add_uint32(prefix + ".attention.head_count_kv", head_count_kv);
    uint32_t head_dim = extract_json_uint32(
            config_json,
            "head_dim",
            embedding_length / (head_count == 0 ? 1 : head_count));
    if (model_family == "PHI") {
        const float partial_rotary_factor = extract_json_float(config_json, "partial_rotary_factor", 1.0f);
        head_dim = static_cast<uint32_t>(partial_rotary_factor * static_cast<float>(embedding_length)) /
                (head_count == 0 ? 1 : head_count);
    }
    writer.add_uint32(prefix + ".rope.dimension_count", head_dim);
    if (model_family == "GEMMA" && head_dim != 0) {
        writer.add_uint32(prefix + ".attention.key_length", head_dim);
        writer.add_uint32(prefix + ".attention.value_length", head_dim);
    }
    writer.add_float32(prefix + ".rope.freq_base", extract_json_float(config_json, "rope_theta", 10000.0f));
    if (model_family == "PHI" && gguf_arch == "phi2") {
        writer.add_float32(
                prefix + ".attention.layer_norm_epsilon",
                extract_json_float(
                        config_json,
                        "layer_norm_epsilon",
                        extract_json_float(config_json, "layer_norm_eps", 0.00001f)));
    } else {
        writer.add_float32(
                prefix + ".attention.layer_norm_rms_epsilon",
                extract_json_float(config_json, "rms_norm_eps", 0.00001f));
    }
    if (model_family == "GEMMA") {
        if (has_json_number(config_json, "attn_logit_softcapping")) {
            writer.add_float32(
                    prefix + ".attn_logit_softcapping",
                    extract_json_float(config_json, "attn_logit_softcapping", 0.0f));
        }
        if (has_json_number(config_json, "final_logit_softcapping")) {
            writer.add_float32(
                    prefix + ".final_logit_softcapping",
                    extract_json_float(config_json, "final_logit_softcapping", 0.0f));
        }
        const uint32_t sliding_window = extract_json_uint32(config_json, "sliding_window", 0);
        if (sliding_window != 0) {
            writer.add_uint32(prefix + ".attention.sliding_window", sliding_window);
            writer.add_float32(
                    prefix + ".rope.freq_base_swa",
                    extract_json_float(config_json, "rope_local_base_freq", 10000.0f));
            const uint32_t sliding_window_pattern = extract_json_uint32(
                    config_json,
                    "sliding_window_pattern",
                    extract_json_uint32(config_json, "_sliding_window_pattern", 0));
            if (sliding_window_pattern != 0) {
                writer.add_uint32(prefix + ".attention.sliding_window_pattern", sliding_window_pattern);
            }
        }
    }
    writer.add_string("tokenizer.ggml.model", tokenizer_model);
    writer.add_string("tokenizer.ggml.pre", tokenizer_pre_for_model(model_family, tokenizer_model, vocab_size));
    writer.add_string_array("tokenizer.ggml.tokens", tokens);
    writer.add_float32_array(
            "tokenizer.ggml.scores",
            scores.size() == tokens.size() ? scores : std::vector<float>(tokens.size(), 0.0f));
    writer.add_int32_array("tokenizer.ggml.token_type", token_types);
    if (!merges.empty()) {
        writer.add_string_array("tokenizer.ggml.merges", merges);
    }
    if (model_family == "QWEN") {
        const uint32_t eos_token_id = extract_json_uint32(config_json, "eos_token_id", 151643);
        writer.add_uint32("tokenizer.ggml.bos_token_id", eos_token_id);
        writer.add_uint32("tokenizer.ggml.eos_token_id", eos_token_id);
        writer.add_uint32("tokenizer.ggml.unknown_token_id", eos_token_id);
        writer.add_uint32(
                "tokenizer.ggml.eot_token_id",
                token_id_for_content(tokenizer_json, "<|im_end|>", eos_token_id));
    } else if (model_family == "PHI" && gguf_arch == "phi2") {
        const uint32_t end_of_text_id = token_id_for_token(tokens, "<|endoftext|>", 50256);
        writer.add_uint32("tokenizer.ggml.bos_token_id", end_of_text_id);
        writer.add_uint32("tokenizer.ggml.eos_token_id", end_of_text_id);
        writer.add_uint32("tokenizer.ggml.eot_token_id", end_of_text_id);
        writer.add_uint32("tokenizer.ggml.unknown_token_id", end_of_text_id);
    } else {
        writer.add_uint32("tokenizer.ggml.bos_token_id", extract_json_uint32(config_json, "bos_token_id", 1));
        writer.add_uint32("tokenizer.ggml.eos_token_id", extract_json_uint32(config_json, "eos_token_id", 2));
        writer.add_uint32(
                "tokenizer.ggml.unknown_token_id",
                extract_json_uint32(config_json, "unk_token_id", model_family == "GEMMA" ? 3 : 0));
        if (model_family == "GEMMA" && has_json_number(config_json, "pad_token_id")) {
            writer.add_uint32(
                    "tokenizer.ggml.padding_token_id",
                    extract_json_uint32(config_json, "pad_token_id", 0));
        }
    }
    writer.add_bool(
            "tokenizer.ggml.add_bos_token",
            model_family == "PHI" && gguf_arch == "phi2" ? false : should_add_bos_token(model_family, vocab_size));
    writer.add_bool("tokenizer.ggml.add_eos_token", false);
    if (model_family == "GEMMA") {
        writer.add_bool("tokenizer.ggml.add_sep_token", false);
        writer.add_bool("tokenizer.ggml.add_space_prefix", false);
    }

    return writer;
}

int count_weight_map_entries(const std::string &index_json) {
    const std::string key = "\"weight_map\"";
    size_t key_pos = index_json.find(key);
    if (key_pos == std::string::npos) {
        return 0;
    }
    size_t object_start = index_json.find('{', key_pos + key.size());
    size_t object_end = index_json.find('}', object_start + 1);
    if (object_start == std::string::npos || object_end == std::string::npos) {
        return 0;
    }
    int entries = 0;
    size_t cursor = object_start + 1;
    while (cursor < object_end) {
        size_t quote = index_json.find('"', cursor);
        if (quote == std::string::npos || quote >= object_end) {
            break;
        }
        size_t end = index_json.find('"', quote + 1);
        if (end == std::string::npos || end >= object_end) {
            break;
        }
        size_t colon = index_json.find(':', end + 1);
        if (colon == std::string::npos || colon >= object_end) {
            break;
        }
        entries += 1;
        cursor = colon + 1;
    }
    return entries;
}

SafetensorsHeader read_safetensors_header(
        const WorkspaceFile &file,
        const std::string &model_family,
        const std::string &gguf_arch) {
    if (file.size < 8) {
        return {false, 0, 0, 0, "file is too small for a safetensors header"};
    }

    std::ifstream input(file.path, std::ios::binary);
    if (!input) {
        return {false, 0, 0, 0, "cannot open"};
    }

    uint8_t length_bytes[8] = {};
    input.read(reinterpret_cast<char *>(length_bytes), sizeof(length_bytes));
    if (input.gcount() != sizeof(length_bytes)) {
        return {false, 0, 0, 0, "missing 8-byte header length"};
    }

    uint64_t header_length = 0;
    for (int index = 0; index < 8; ++index) {
        header_length |= static_cast<uint64_t>(length_bytes[index]) << (index * 8);
    }

    if (header_length == 0) {
        return {false, 0, 0, 0, "empty safetensors header"};
    }
    if (header_length > file.size - 8) {
        return {false, header_length, 0, 0, "header exceeds file size"};
    }
    if (header_length > 128ULL * 1024ULL * 1024ULL) {
        return {false, header_length, 0, 0, "header is unexpectedly large"};
    }

    std::string header;
    header.resize(static_cast<size_t>(header_length));
    input.read(header.data(), static_cast<std::streamsize>(header.size()));
    if (static_cast<uint64_t>(input.gcount()) != header_length) {
        return {false, header_length, 0, 0, "could not read full header"};
    }
    if (header.find('{') == std::string::npos || header.find("data_offsets") == std::string::npos) {
        return {false, header_length, 0, 0, "header does not look like safetensors JSON"};
    }

    const std::vector<SafetensorsTensor> tensors =
            parse_safetensors_tensors(header, file, header_length, model_family, gguf_arch);
    int mapped_tensor_count = 0;
    for (const SafetensorsTensor &tensor : tensors) {
        if (!tensor.gguf_name.empty()) {
            mapped_tensor_count += 1;
        }
    }

    return {true, header_length, static_cast<int>(tensors.size()), mapped_tensor_count, ""};
}

bool read_safetensors_tensors(
        const WorkspaceFile &file,
        std::vector<SafetensorsTensor> &tensors,
        std::string &error,
        const std::string &model_family,
        const std::string &gguf_arch) {
    if (file.size < 8) {
        error = file.name + " is too small for a safetensors header.";
        return false;
    }

    std::ifstream input(file.path, std::ios::binary);
    if (!input) {
        error = "Cannot open " + file.name + ".";
        return false;
    }

    uint8_t length_bytes[8] = {};
    input.read(reinterpret_cast<char *>(length_bytes), sizeof(length_bytes));
    if (input.gcount() != sizeof(length_bytes)) {
        error = file.name + " is missing its safetensors header length.";
        return false;
    }

    uint64_t header_length = 0;
    for (int index = 0; index < 8; ++index) {
        header_length |= static_cast<uint64_t>(length_bytes[index]) << (index * 8);
    }
    if (header_length == 0 || header_length > file.size - 8) {
        error = file.name + " has an invalid safetensors header length.";
        return false;
    }
    if (header_length > 128ULL * 1024ULL * 1024ULL) {
        error = file.name + " has an unexpectedly large safetensors header.";
        return false;
    }

    std::string header;
    header.resize(static_cast<size_t>(header_length));
    input.read(header.data(), static_cast<std::streamsize>(header.size()));
    if (static_cast<uint64_t>(input.gcount()) != header_length) {
        error = "Could not read the full safetensors header for " + file.name + ".";
        return false;
    }

    std::vector<SafetensorsTensor> parsed =
            parse_safetensors_tensors(header, file, header_length, model_family, gguf_arch);
    tensors.insert(tensors.end(), parsed.begin(), parsed.end());
    return true;
}

std::vector<WorkspaceFile> list_workspace_files(const std::string &model_directory) {
    std::vector<WorkspaceFile> files;
    DIR *dir = opendir(model_directory.c_str());
    if (dir == nullptr) {
        return files;
    }

    while (dirent *entry = readdir(dir)) {
        std::string name = entry->d_name;
        if (name == "." || name == "..") {
            continue;
        }
        std::string path = model_directory + "/" + name;
        struct stat info {};
        if (stat(path.c_str(), &info) != 0 || !S_ISREG(info.st_mode)) {
            continue;
        }
        files.push_back({name, path, static_cast<uint64_t>(info.st_size)});
    }

    closedir(dir);
    std::sort(files.begin(), files.end(), [](const WorkspaceFile &left, const WorkspaceFile &right) {
        return left.name < right.name;
    });
    return files;
}

bool is_supported_model_family(const std::string &model_family) {
    return model_family == "LLAMA" || model_family == "QWEN" || model_family == "GEMMA" ||
            model_family == "PHI";
}

std::string model_family_label(const std::string &model_family) {
    if (model_family == "QWEN") {
        return "Qwen";
    }
    if (model_family == "GEMMA") {
        return "Gemma";
    }
    if (model_family == "PHI") {
        return "Phi";
    }
    return "LLaMA/Mistral";
}

}  // namespace

HorizonConversionSummary inspect_hf_safetensors_model(
        const std::string &model_directory,
        const std::string &output_file,
        const std::string &quantization,
        const std::string &model_family,
        const std::function<void(float, const std::string &)> &on_progress) {
    if (!is_supported_model_family(model_family)) {
        return {false, "Native GGUF writing currently supports only the LLaMA / Mistral, Qwen, Gemma, and Phi model families."};
    }

    const std::vector<WorkspaceFile> files = list_workspace_files(model_directory);
    if (files.empty()) {
        return {false, "Model workspace is empty or cannot be read."};
    }

    std::vector<WorkspaceFile> safetensors_files;
    std::string config_path;
    std::string tokenizer_json_path;
    std::string index_path;
    bool has_tokenizer = false;

    for (const WorkspaceFile &file : files) {
        if (ends_with(file.name, ".safetensors")) {
            safetensors_files.push_back(file);
        } else if (ends_with(file.name, ".safetensors.index.json")) {
            index_path = file.path;
        } else if (ends_with(file.name, ".json") &&
                   (equals_ignore_case(file.name, "tokenizer.json") ||
                    contains_ignore_case(file.name, "tokenizer"))) {
            if (tokenizer_json_path.empty() || equals_ignore_case(file.name, "tokenizer.json")) {
                tokenizer_json_path = file.path;
            }
            has_tokenizer = true;
        } else if (ends_with(file.name, ".model") &&
                   (equals_ignore_case(file.name, "tokenizer.model") ||
                    contains_ignore_case(file.name, "tokenizer"))) {
            has_tokenizer = true;
        } else if (ends_with(file.name, ".json") &&
                   (equals_ignore_case(file.name, "config.json") ||
                    contains_ignore_case(file.name, "config") ||
                    (!contains_ignore_case(file.name, "tokenizer") &&
                     contains_ignore_case(file.name, "model")))) {
            if (config_path.empty() || equals_ignore_case(file.name, "config.json")) {
                config_path = file.path;
            }
        }
    }

    if (safetensors_files.empty()) {
        return {false, "No .safetensors files were found in the staged workspace."};
    }
    if (config_path.empty()) {
        return {false, "config.json is required before native conversion can map tensors."};
    }
    if (!has_tokenizer) {
        return {false, "tokenizer.json or tokenizer.model is required before writing GGUF metadata."};
    }
    if (safetensors_files.size() > 1 && index_path.empty()) {
        return {false, "Sharded safetensors conversion requires the .safetensors.index.json file."};
    }

    const std::string config_json = read_text_file(config_path, 1024 * 1024);
    const std::string architecture = extract_first_architecture(config_json);
    const std::string model_type = extract_json_string(config_json, "model_type");
    const std::string gguf_arch = gguf_architecture_name(architecture, model_type, model_family);
    if (gguf_arch == "gemma3n") {
        return {false, "Gemma 3n is not supported by this native converter yet; it has extra per-layer tensors that are not mapped."};
    }

    uint64_t source_bytes = 0;
    int tensor_headers = 0;
    int mapped_tensor_headers = 0;
    for (const WorkspaceFile &file : safetensors_files) {
        source_bytes += file.size;
        SafetensorsHeader header = read_safetensors_header(file, model_family, gguf_arch);
        if (!header.ok) {
            return {false, file.name + " is not a readable safetensors file: " + header.error + "."};
        }
        tensor_headers += header.tensor_count;
        mapped_tensor_headers += header.mapped_tensor_count;
    }

    const std::string tokenizer_json = tokenizer_json_path.empty()
            ? std::string()
            : read_text_file(tokenizer_json_path, 128 * 1024 * 1024);
    std::vector<float> token_scores;
    std::vector<std::string> tokens = tokenizer_json_path.empty()
            ? std::vector<std::string>()
            : extract_tokenizer_json_vocab(tokenizer_json, token_scores);
    if (tokens.empty()) {
        return {false, "Native GGUF writing currently requires a tokenizer.json with a readable vocab object."};
    }
    if (model_family == "QWEN") {
        normalize_qwen_tokens(tokens, extract_json_uint32(config_json, "vocab_size", 0));
        token_scores.resize(tokens.size(), 0.0f);
    } else if (model_family == "GEMMA") {
        normalize_gemma_tokens(tokens, token_scores, extract_json_uint32(config_json, "vocab_size", 0));
    } else if (model_family == "PHI") {
        normalize_phi_tokens(tokens, token_scores, extract_json_uint32(config_json, "vocab_size", 0));
    }

    const std::string tokenizer_json_model_type = extract_tokenizer_json_model_type(tokenizer_json);
    const bool tokenizer_is_bpe = tokenizer_json_model_type == "BPE";
    const std::string gguf_tokenizer_model = tokenizer_is_bpe
            ? (model_family == "GEMMA" ? "gemma4" : "gpt2")
            : "llama";
    const std::vector<std::string> merges = tokenizer_is_bpe
            ? extract_tokenizer_json_merges(tokenizer_json)
            : std::vector<std::string>();
    if (tokenizer_is_bpe && merges.empty()) {
        return {false, "BPE tokenizer.json requires readable merges for llama.cpp GGUF loading."};
    }

    if (!is_supported_native_output_quantization(quantization)) {
        std::ostringstream blocked;
        blocked << "Native GGUF writing currently supports F16, Q8_0, Q4_0, Q5_0, Q6_K, Q5_K_M, Q4_K_M, Q4_K_S, and Q3_K_M output. Requested " << quantization
                << " still needs native quantization kernels.";
        return {false, blocked.str()};
    }
    if (!is_supported_family_quantization(model_family, quantization)) {
        return {false, quantization + " is not currently enabled for the " + model_family_label(model_family) + " model family."};
    }

    const uint32_t attention_head_count = extract_json_uint32(
            config_json,
            "num_attention_heads",
            extract_json_uint32(config_json, "n_head", 0));
    uint32_t key_value_head_count =
            extract_json_uint32(config_json, "num_key_value_heads", attention_head_count);
    if (model_family == "PHI" && key_value_head_count == 0) {
        key_value_head_count = attention_head_count;
    }
    const std::vector<int32_t> token_types = build_token_types(tokenizer_json, tokens, model_family);
    HorizonGgufMetadataWriter metadata_writer = build_metadata_writer(
            config_json,
            tokenizer_json,
            tokens,
            token_scores,
            gguf_tokenizer_model,
            merges,
            token_types,
            architecture,
            model_type,
            quantization,
            model_family);
    const std::vector<uint8_t> metadata_preview = metadata_writer.build(
            static_cast<uint64_t>(mapped_tensor_headers));

    int indexed_tensors = 0;
    if (!index_path.empty()) {
        indexed_tensors = count_weight_map_entries(read_text_file(index_path, 16 * 1024 * 1024));
    }

    std::vector<SafetensorsTensor> parsed_tensors;
    for (const WorkspaceFile &file : safetensors_files) {
        std::string parse_error;
        if (!read_safetensors_tensors(file, parsed_tensors, parse_error, model_family, gguf_arch)) {
            return {false, parse_error};
        }
    }

    std::vector<HorizonGgufTensorSource> gguf_tensors;
    int q6_k_tensor_count = 0;
    int q5_k_tensor_count = 0;
    int q4_k_tensor_count = 0;
    int q3_k_tensor_count = 0;
    int q8_0_tensor_count = 0;
    int q4_0_tensor_count = 0;
    int q5_0_tensor_count = 0;
    int f16_tensor_count = 0;
    int f32_tensor_count = 0;
    for (const SafetensorsTensor &tensor : parsed_tensors) {
        if (tensor.gguf_name.empty()) {
            continue;
        }
        if (tensor.shape.empty()) {
            return {false, tensor.source_name + " does not include a tensor shape."};
        }
        if (tensor.data_end <= tensor.data_begin) {
            return {false, tensor.source_name + " has invalid safetensors data offsets."};
        }

        if (!is_supported_f16_output_dtype(tensor.dtype)) {
            return {false, tensor.source_name + " uses unsupported dtype " + tensor.dtype + "."};
        }
        const uint64_t source_bytes = tensor.data_end - tensor.data_begin;
        const uint64_t element_count = tensor_element_count(tensor.shape);
        const uint64_t expected_source_bytes = element_count * safetensors_dtype_bytes(tensor.dtype);
        if (element_count == 0 || expected_source_bytes != source_bytes) {
            return {false, tensor.source_name + " shape and safetensors byte range do not agree."};
        }

        std::vector<uint64_t> output_shape = tensor.shape;
        uint64_t output_element_count = element_count;
        uint64_t output_source_bytes = source_bytes;
        if (model_family == "GEMMA" &&
                tensor.gguf_name == "token_embd.weight" &&
                output_shape.size() == 2 &&
                !tokens.empty() &&
                output_shape[0] > tokens.size()) {
            output_shape[0] = static_cast<uint64_t>(tokens.size());
            output_element_count = tensor_element_count(output_shape);
            output_source_bytes = output_element_count * safetensors_dtype_bytes(tensor.dtype);
        }
        const SafetensorsTensor output_tensor = tensor_with_shape(tensor, output_shape);

        const uint64_t output_data_size =
                output_data_size_for_tensor(quantization, output_tensor, output_element_count, model_family);

        const uint32_t output_ggml_type =
                output_ggml_type_for_tensor(quantization, output_tensor, output_element_count, model_family);
        const HorizonTensorOutputEncoding output_encoding =
                output_encoding_for_tensor(quantization, output_tensor, output_element_count, model_family);
        if (output_encoding == HorizonTensorOutputEncoding::Q6_K) {
            q6_k_tensor_count += 1;
        } else if (output_encoding == HorizonTensorOutputEncoding::Q5_K) {
            q5_k_tensor_count += 1;
        } else if (output_encoding == HorizonTensorOutputEncoding::Q4_K) {
            q4_k_tensor_count += 1;
        } else if (output_encoding == HorizonTensorOutputEncoding::Q3_K) {
            q3_k_tensor_count += 1;
        } else if (output_encoding == HorizonTensorOutputEncoding::Q8_0) {
            q8_0_tensor_count += 1;
        } else if (output_encoding == HorizonTensorOutputEncoding::Q4_0) {
            q4_0_tensor_count += 1;
        } else if (output_encoding == HorizonTensorOutputEncoding::Q5_0) {
            q5_0_tensor_count += 1;
        } else if (output_encoding == HorizonTensorOutputEncoding::F32) {
            f32_tensor_count += 1;
        } else {
            f16_tensor_count += 1;
        }

        gguf_tensors.push_back({
                tensor.gguf_name,
                output_shape,
                output_ggml_type,
                tensor.source_path,
                tensor.source_offset,
                output_source_bytes,
                output_data_size,
                tensor_encoding_for_dtype(tensor.dtype),
                output_encoding,
                row_permutation_heads_for_tensor(output_tensor, attention_head_count, key_value_head_count, model_family),
                source_float_add_for_tensor(tensor, model_family),
        });
    }

    if (gguf_tensors.empty()) {
        std::ostringstream unmapped;
        unmapped << "No mapped " << model_family_label(model_family)
                 << " tensors are available for GGUF writing. Architecture "
                 << (gguf_arch.empty() ? "unknown" : gguf_arch) << ".";
        int examples = 0;
        for (const SafetensorsTensor &tensor : parsed_tensors) {
            if (!tensor.source_name.empty()) {
                if (examples == 0) {
                    unmapped << " First safetensors tensor names:";
                }
                unmapped << " " << tensor.source_name;
                examples += 1;
                if (examples >= 8) {
                    break;
                }
            }
        }
        return {false, unmapped.str()};
    }

    if (is_supported_native_output_quantization(quantization)) {
        std::string write_error;
        if (on_progress) {
            on_progress(0.32f, "Writing GGUF metadata");
        }
        const auto tensor_progress = [&on_progress](size_t index, size_t total, const std::string &name) {
            if (!on_progress || total == 0) {
                return;
            }
            const float tensor_fraction = static_cast<float>(index) / static_cast<float>(total);
            const float progress = 0.35f + tensor_fraction * 0.60f;
            std::ostringstream message;
            message << "Writing tensor " << (index + 1) << "/" << total << ": " << name;
            on_progress(progress, message.str());
        };
        if (!metadata_writer.write_file(output_file, gguf_tensors, write_error, tensor_progress)) {
            return {false, write_error};
        }

        std::ostringstream success;
        success << "Native " << quantization << " GGUF writer emitted " << gguf_tensors.size()
                << " mapped tensor(s), " << metadata_writer.kv_count()
                << " metadata kv pair(s), " << tokens.size()
                << " tokenizer token(s), architecture "
                << (architecture.empty() ? model_type : architecture)
                << ", family " << model_family_label(model_family)
                << ", tensor encodings Q6_K=" << q6_k_tensor_count
                << ", Q5_K=" << q5_k_tensor_count
                << ", Q4_K=" << q4_k_tensor_count
                << ", Q3_K=" << q3_k_tensor_count
                << ", Q8_0=" << q8_0_tensor_count
                << ", Q4_0=" << q4_0_tensor_count
                << ", Q5_0=" << q5_0_tensor_count
                << ", F16=" << f16_tensor_count
                << ", F32=" << f32_tensor_count
                << ", output " << output_file << ".";
        return {true, success.str()};
    }

    std::ostringstream message;
    message << "Native pre-converter inspected " << safetensors_files.size()
            << " safetensors file(s), " << tensor_headers << " tensor header(s), "
            << mapped_tensor_headers << " " << model_family_label(model_family)
            << " GGUF tensor name mapping(s)";
    if (indexed_tensors > 0) {
        message << ", " << indexed_tensors << " indexed tensor(s)";
    }
    if (!architecture.empty()) {
        message << ", architecture " << architecture;
    } else if (!model_type.empty()) {
        message << ", model_type " << model_type;
    }
    message << ". Requested quantization: " << quantization << ". Output: " << output_file
            << ". GGUF metadata scaffold: " << metadata_writer.kv_count()
            << " kv pair(s), " << metadata_preview.size() << " byte preview"
            << ". F16 GGUF writing is implemented for already-F16 safetensors. "
            << "Quantized output still needs native quantization kernels.";

    return {false, message.str()};
}
