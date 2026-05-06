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

std::string unescape_json_string(const std::string &value) {
    std::string output;
    output.reserve(value.size());
    bool escaped = false;
    for (char item : value) {
        if (escaped) {
            switch (item) {
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
                    output.push_back(item);
                    break;
                default:
                    output.push_back(item);
                    break;
            }
            escaped = false;
        } else if (item == '\\') {
            escaped = true;
        } else {
            output.push_back(item);
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
    return quantization == "F16" || quantization == "Q8_0";
}

bool should_quantize_tensor_q8_0(
        const SafetensorsTensor &tensor,
        uint64_t element_count) {
    return tensor.shape.size() >= 2 &&
           !tensor.shape.empty() &&
           tensor.gguf_name != "output.weight" &&
           tensor.gguf_name.find(".attn_q.weight") == std::string::npos &&
           tensor.gguf_name.find(".attn_k.weight") == std::string::npos &&
           tensor.shape.back() % 32 == 0 &&
           element_count % 32 == 0;
}

bool should_keep_tensor_f32(const SafetensorsTensor &tensor) {
    return tensor.shape.size() <= 1 || ends_with(tensor.gguf_name, "_norm.weight");
}

uint32_t row_permutation_heads_for_tensor(
        const SafetensorsTensor &tensor,
        uint32_t attention_head_count,
        uint32_t key_value_head_count) {
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
        uint64_t element_count) {
    if (should_keep_tensor_f32(tensor)) {
        return 0;
    }
    if (quantization == "Q8_0" && should_quantize_tensor_q8_0(tensor, element_count)) {
        return 8;
    }
    return 1;
}

HorizonTensorOutputEncoding output_encoding_for_tensor(
        const std::string &quantization,
        const SafetensorsTensor &tensor,
        uint64_t element_count) {
    if (should_keep_tensor_f32(tensor)) {
        return HorizonTensorOutputEncoding::F32;
    }
    if (quantization == "Q8_0" && should_quantize_tensor_q8_0(tensor, element_count)) {
        return HorizonTensorOutputEncoding::Q8_0;
    }
    return HorizonTensorOutputEncoding::F16;
}

uint64_t output_data_size_for_tensor(
        const std::string &quantization,
        const SafetensorsTensor &tensor,
        uint64_t element_count) {
    if (should_keep_tensor_f32(tensor)) {
        return element_count * 4;
    }
    if (quantization == "Q8_0" && should_quantize_tensor_q8_0(tensor, element_count)) {
        return (element_count / 32) * 34;
    }
    return element_count * 2;
}

std::vector<SafetensorsTensor> parse_safetensors_tensors(
        const std::string &header,
        const WorkspaceFile &source_file,
        uint64_t header_length) {
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
                    map_llama_tensor_name(name),
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

std::vector<std::string> extract_tokenizer_json_vocab(const std::string &tokenizer_json) {
    const std::string vocab_key = "\"vocab\"";
    size_t key_pos = tokenizer_json.find(vocab_key);
    if (key_pos == std::string::npos) {
        return {};
    }
    size_t object_start = tokenizer_json.find('{', key_pos + vocab_key.size());
    if (object_start == std::string::npos) {
        return {};
    }
    size_t object_end = find_matching_brace(tokenizer_json, object_start);
    if (object_end == std::string::npos) {
        return {};
    }

    std::map<uint32_t, std::string> id_to_token;
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

        size_t colon = tokenizer_json.find(':', token_end + 1);
        if (colon == std::string::npos || colon >= object_end) {
            break;
        }
        size_t id_start = colon + 1;
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
        const uint32_t token_id = static_cast<uint32_t>(std::stoul(tokenizer_json.substr(id_start, id_end - id_start)));
        id_to_token[token_id] = unescape_json_string(raw_token);
        cursor = id_end + 1;
    }

    if (id_to_token.empty()) {
        return {};
    }
    const uint32_t max_id = id_to_token.rbegin()->first;
    std::vector<std::string> tokens(static_cast<size_t>(max_id) + 1);
    for (const auto &entry : id_to_token) {
        tokens[entry.first] = entry.second;
    }
    return tokens;
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

    std::vector<std::string> merges;
    size_t cursor = array_start + 1;
    while (cursor < tokenizer_json.size()) {
        size_t item_start = tokenizer_json.find('"', cursor);
        size_t array_end = tokenizer_json.find(']', cursor);
        if (array_end != std::string::npos && (item_start == std::string::npos || array_end < item_start)) {
            break;
        }
        if (item_start == std::string::npos) {
            break;
        }

        size_t item_end = item_start + 1;
        bool escaped = false;
        while (item_end < tokenizer_json.size()) {
            const char value = tokenizer_json[item_end];
            if (escaped) {
                escaped = false;
            } else if (value == '\\') {
                escaped = true;
            } else if (value == '"') {
                break;
            }
            item_end += 1;
        }
        if (item_end >= tokenizer_json.size()) {
            break;
        }

        merges.push_back(unescape_json_string(tokenizer_json.substr(item_start + 1, item_end - item_start - 1)));
        cursor = item_end + 1;
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
        const std::vector<std::string> &tokens) {
    std::vector<int32_t> token_types = build_token_types(tokenizer_json, tokens.size());
    constexpr int32_t kByteToken = 6;

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
        const std::string &tokenizer_model,
        uint32_t vocab_size) {
    if (tokenizer_model != "gpt2") {
        return "default";
    }
    if (vocab_size == 49152) {
        return "smollm";
    }
    return "default";
}

bool should_add_bos_token(uint32_t vocab_size) {
    return vocab_size != 49152;
}

std::string gguf_architecture_name(const std::string &architecture, const std::string &model_type) {
    std::string source = !model_type.empty() ? model_type : architecture;
    std::transform(source.begin(), source.end(), source.begin(), [](char value) {
        return std::tolower(static_cast<unsigned char>(value));
    });
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

HorizonGgufMetadataWriter build_llama_metadata_writer(
        const std::string &config_json,
        const std::vector<std::string> &tokens,
        const std::string &tokenizer_model,
        const std::vector<std::string> &merges,
        const std::vector<int32_t> &token_types,
        const std::string &architecture,
        const std::string &model_type,
        const std::string &quantization) {
    HorizonGgufMetadataWriter writer;
    const std::string gguf_arch = gguf_architecture_name(architecture, model_type);

    writer.add_string("general.architecture", gguf_arch.empty() ? "llama" : gguf_arch);
    writer.add_string("general.name", "HorizonAssimilator converted model");
    writer.add_uint32("general.alignment", 32);
    writer.add_uint32("general.file_type", gguf_file_type(quantization));
    writer.add_uint32("general.quantization_version", 2);

    const std::string prefix = gguf_arch.empty() ? "llama" : gguf_arch;
    const uint32_t vocab_size = static_cast<uint32_t>(tokens.size());
    writer.add_uint32(prefix + ".context_length", extract_json_uint32(config_json, "max_position_embeddings", 0));
    writer.add_uint32(prefix + ".vocab_size", vocab_size);
    writer.add_uint32(prefix + ".embedding_length", extract_json_uint32(config_json, "hidden_size", 0));
    writer.add_uint32(prefix + ".block_count", extract_json_uint32(config_json, "num_hidden_layers", 0));
    writer.add_uint32(prefix + ".feed_forward_length", extract_json_uint32(config_json, "intermediate_size", 0));
    const uint32_t head_count = extract_json_uint32(config_json, "num_attention_heads", 0);
    writer.add_uint32(prefix + ".attention.head_count", head_count);
    writer.add_uint32(prefix + ".attention.head_count_kv", extract_json_uint32(config_json, "num_key_value_heads", head_count));
    writer.add_uint32(prefix + ".rope.dimension_count", extract_json_uint32(config_json, "hidden_size", 0) / (head_count == 0 ? 1 : head_count));
    writer.add_float32(prefix + ".rope.freq_base", extract_json_float(config_json, "rope_theta", 10000.0f));
    writer.add_float32(
            prefix + ".attention.layer_norm_rms_epsilon",
            extract_json_float(config_json, "rms_norm_eps", 0.00001f));
    writer.add_string("tokenizer.ggml.model", tokenizer_model);
    writer.add_string("tokenizer.ggml.pre", tokenizer_pre_for_model(tokenizer_model, vocab_size));
    writer.add_string_array("tokenizer.ggml.tokens", tokens);
    writer.add_float32_array("tokenizer.ggml.scores", std::vector<float>(tokens.size(), 0.0f));
    writer.add_int32_array("tokenizer.ggml.token_type", token_types);
    if (!merges.empty()) {
        writer.add_string_array("tokenizer.ggml.merges", merges);
    }
    writer.add_uint32("tokenizer.ggml.bos_token_id", extract_json_uint32(config_json, "bos_token_id", 1));
    writer.add_uint32("tokenizer.ggml.eos_token_id", extract_json_uint32(config_json, "eos_token_id", 2));
    writer.add_uint32("tokenizer.ggml.unknown_token_id", extract_json_uint32(config_json, "unk_token_id", 0));
    writer.add_bool("tokenizer.ggml.add_bos_token", should_add_bos_token(vocab_size));
    writer.add_bool("tokenizer.ggml.add_eos_token", false);

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

SafetensorsHeader read_safetensors_header(const WorkspaceFile &file) {
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

    const std::vector<SafetensorsTensor> tensors = parse_safetensors_tensors(header, file, header_length);
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
        std::string &error) {
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

    std::vector<SafetensorsTensor> parsed = parse_safetensors_tensors(header, file, header_length);
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

}  // namespace

HorizonConversionSummary inspect_hf_safetensors_model(
        const std::string &model_directory,
        const std::string &output_file,
        const std::string &quantization) {
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
        } else if (equals_ignore_case(file.name, "config.json")) {
            config_path = file.path;
        } else if (equals_ignore_case(file.name, "tokenizer.json")) {
            tokenizer_json_path = file.path;
            has_tokenizer = true;
        } else if (equals_ignore_case(file.name, "tokenizer.model")) {
            has_tokenizer = true;
        } else if (ends_with(file.name, ".safetensors.index.json")) {
            index_path = file.path;
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

    uint64_t source_bytes = 0;
    int tensor_headers = 0;
    int mapped_tensor_headers = 0;
    for (const WorkspaceFile &file : safetensors_files) {
        source_bytes += file.size;
        SafetensorsHeader header = read_safetensors_header(file);
        if (!header.ok) {
            return {false, file.name + " is not a readable safetensors file: " + header.error + "."};
        }
        tensor_headers += header.tensor_count;
        mapped_tensor_headers += header.mapped_tensor_count;
    }

    const std::string config_json = read_text_file(config_path, 1024 * 1024);
    const std::string tokenizer_json = tokenizer_json_path.empty()
            ? std::string()
            : read_text_file(tokenizer_json_path, 128 * 1024 * 1024);
    const std::vector<std::string> tokens = tokenizer_json_path.empty()
            ? std::vector<std::string>()
            : extract_tokenizer_json_vocab(tokenizer_json);
    if (tokens.empty()) {
        return {false, "Native GGUF writing currently requires a tokenizer.json with a readable vocab object."};
    }

    const std::string tokenizer_json_model_type = extract_tokenizer_json_model_type(tokenizer_json);
    const bool tokenizer_is_bpe = tokenizer_json_model_type == "BPE";
    const std::string gguf_tokenizer_model = tokenizer_is_bpe ? "gpt2" : "llama";
    const std::vector<std::string> merges = tokenizer_is_bpe
            ? extract_tokenizer_json_merges(tokenizer_json)
            : std::vector<std::string>();
    if (tokenizer_is_bpe && merges.empty()) {
        return {false, "BPE tokenizer.json requires readable merges for llama.cpp GGUF loading."};
    }

    if (!is_supported_native_output_quantization(quantization)) {
        std::ostringstream blocked;
        blocked << "Native GGUF writing currently supports F16 and Q8_0 output. Requested " << quantization
                << " still needs native quantization kernels.";
        return {false, blocked.str()};
    }

    const std::string architecture = extract_first_architecture(config_json);
    const std::string model_type = extract_json_string(config_json, "model_type");
    const uint32_t attention_head_count = extract_json_uint32(config_json, "num_attention_heads", 0);
    const uint32_t key_value_head_count =
            extract_json_uint32(config_json, "num_key_value_heads", attention_head_count);
    const std::vector<int32_t> token_types = build_token_types(tokenizer_json, tokens);
    HorizonGgufMetadataWriter metadata_writer = build_llama_metadata_writer(
            config_json,
            tokens,
            gguf_tokenizer_model,
            merges,
            token_types,
            architecture,
            model_type,
            quantization);
    const std::vector<uint8_t> metadata_preview = metadata_writer.build(
            static_cast<uint64_t>(mapped_tensor_headers));

    int indexed_tensors = 0;
    if (!index_path.empty()) {
        indexed_tensors = count_weight_map_entries(read_text_file(index_path, 16 * 1024 * 1024));
    }

    std::vector<SafetensorsTensor> parsed_tensors;
    for (const WorkspaceFile &file : safetensors_files) {
        std::string parse_error;
        if (!read_safetensors_tensors(file, parsed_tensors, parse_error)) {
            return {false, parse_error};
        }
    }

    std::vector<HorizonGgufTensorSource> gguf_tensors;
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

        const uint64_t output_data_size = output_data_size_for_tensor(quantization, tensor, element_count);

        gguf_tensors.push_back({
                tensor.gguf_name,
                tensor.shape,
                output_ggml_type_for_tensor(quantization, tensor, element_count),
                tensor.source_path,
                tensor.source_offset,
                source_bytes,
                output_data_size,
                tensor_encoding_for_dtype(tensor.dtype),
                output_encoding_for_tensor(quantization, tensor, element_count),
                row_permutation_heads_for_tensor(tensor, attention_head_count, key_value_head_count),
        });
    }

    if (gguf_tensors.empty()) {
        return {false, "No mapped LLaMA/Mistral tensors are available for GGUF writing."};
    }

    if (is_supported_native_output_quantization(quantization)) {
        std::string write_error;
        if (!metadata_writer.write_file(output_file, gguf_tensors, write_error)) {
            return {false, write_error};
        }

        std::ostringstream success;
        success << "Native " << quantization << " GGUF writer emitted " << gguf_tensors.size()
                << " mapped tensor(s), " << metadata_writer.kv_count()
                << " metadata kv pair(s), " << tokens.size()
                << " tokenizer token(s), architecture "
                << (architecture.empty() ? model_type : architecture)
                << ", output " << output_file << ".";
        return {true, success.str()};
    }

    std::ostringstream message;
    message << "Native pre-converter inspected " << safetensors_files.size()
            << " safetensors file(s), " << tensor_headers << " tensor header(s), "
            << mapped_tensor_headers << " LLaMA/Mistral GGUF tensor name mapping(s)";
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
