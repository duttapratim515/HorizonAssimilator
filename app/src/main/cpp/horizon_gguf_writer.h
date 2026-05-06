#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

enum class HorizonTensorEncoding {
    F16,
    F32,
    BF16,
};

enum class HorizonTensorOutputEncoding {
    F16,
    F32,
    Q8_0,
    Q6_K,
    Q5_K,
};

struct HorizonGgufTensorSource {
    std::string name;
    std::vector<uint64_t> shape;
    uint32_t ggml_type;
    std::string source_path;
    uint64_t source_offset;
    uint64_t source_data_size;
    uint64_t output_data_size;
    HorizonTensorEncoding source_encoding;
    HorizonTensorOutputEncoding output_encoding;
    uint32_t row_permutation_heads;
};

class HorizonGgufMetadataWriter {
public:
    void add_string(const std::string &key, const std::string &value);
    void add_string_array(const std::string &key, const std::vector<std::string> &values);
    void add_uint32(const std::string &key, uint32_t value);
    void add_bool(const std::string &key, bool value);
    void add_float32(const std::string &key, float value);
    void add_float32_array(const std::string &key, const std::vector<float> &values);
    void add_int32_array(const std::string &key, const std::vector<int32_t> &values);

    std::vector<uint8_t> build(uint64_t tensor_count) const;
    bool write_file(
            const std::string &output_path,
            const std::vector<HorizonGgufTensorSource> &tensors,
            std::string &error) const;
    size_t kv_count() const;

private:
    enum class ValueType : uint32_t {
        Uint32 = 4,
        Int32 = 5,
        Float32 = 6,
        Bool = 7,
        String = 8,
        Array = 9,
    };

    struct Entry {
        std::string key;
        ValueType type;
        std::string string_value;
        std::vector<std::string> string_array_value;
        std::vector<float> float32_array_value;
        std::vector<int32_t> int32_array_value;
        uint32_t uint32_value = 0;
        int32_t int32_value = 0;
        float float32_value = 0.0f;
        bool bool_value = false;
    };

    std::vector<Entry> entries_;
};
