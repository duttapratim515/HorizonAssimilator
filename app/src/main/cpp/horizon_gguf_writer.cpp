#include "horizon_gguf_writer.h"

#include <cmath>
#include <cstring>
#include <fstream>
#include <limits>

namespace {

constexpr uint32_t kGgufMagic = 0x46554747;
constexpr uint32_t kGgufVersion = 3;
constexpr uint64_t kDefaultAlignment = 32;
constexpr int kQkK = 256;
constexpr float kGroupMaxEps = 1e-15f;

template <typename T>
void append_plain(std::vector<uint8_t> &buffer, T value) {
    const auto *bytes = reinterpret_cast<const uint8_t *>(&value);
    buffer.insert(buffer.end(), bytes, bytes + sizeof(T));
}

void append_string(std::vector<uint8_t> &buffer, const std::string &value) {
    append_plain<uint64_t>(buffer, static_cast<uint64_t>(value.size()));
    buffer.insert(buffer.end(), value.begin(), value.end());
}

uint64_t align_to(uint64_t value, uint64_t alignment) {
    const uint64_t remainder = value % alignment;
    return remainder == 0 ? value : value + alignment - remainder;
}

bool write_zero_padding(std::ofstream &output, uint64_t count) {
    static const char zeros[32] = {};
    while (count > 0) {
        const uint64_t chunk = count > sizeof(zeros) ? sizeof(zeros) : count;
        output.write(zeros, static_cast<std::streamsize>(chunk));
        if (!output) {
            return false;
        }
        count -= chunk;
    }
    return true;
}

bool copy_range(
        const HorizonGgufTensorSource &tensor,
        std::ofstream &output,
        std::string &error) {
    std::ifstream input(tensor.source_path, std::ios::binary);
    if (!input) {
        error = "Unable to open tensor source " + tensor.source_path + ".";
        return false;
    }
    input.seekg(static_cast<std::streamoff>(tensor.source_offset), std::ios::beg);
    if (!input) {
        error = "Unable to seek tensor source " + tensor.source_path + ".";
        return false;
    }

    std::vector<char> buffer(1024 * 1024);
    uint64_t remaining = tensor.source_data_size;
    while (remaining > 0) {
        const uint64_t chunk = remaining > buffer.size() ? buffer.size() : remaining;
        input.read(buffer.data(), static_cast<std::streamsize>(chunk));
        if (static_cast<uint64_t>(input.gcount()) != chunk) {
            error = "Unable to read tensor data for " + tensor.name + ".";
            return false;
        }
        output.write(buffer.data(), static_cast<std::streamsize>(chunk));
        if (!output) {
            error = "Unable to write tensor data for " + tensor.name + ".";
            return false;
        }
        remaining -= chunk;
    }

    return true;
}

uint16_t float32_bits_to_float16(uint32_t value) {
    const uint32_t sign = (value >> 16) & 0x8000U;
    const uint32_t exponent = (value >> 23) & 0xFFU;
    const uint32_t mantissa = value & 0x7FFFFFU;

    if (exponent == 0xFFU) {
        if (mantissa == 0) {
            return static_cast<uint16_t>(sign | 0x7C00U);
        }
        return static_cast<uint16_t>(sign | 0x7E00U);
    }

    const int32_t half_exponent = static_cast<int32_t>(exponent) - 127 + 15;
    if (half_exponent >= 0x1F) {
        return static_cast<uint16_t>(sign | 0x7C00U);
    }
    if (half_exponent <= 0) {
        if (half_exponent < -10) {
            return static_cast<uint16_t>(sign);
        }
        uint32_t shifted_mantissa = mantissa | 0x800000U;
        const uint32_t shift = static_cast<uint32_t>(14 - half_exponent);
        uint32_t half_mantissa = shifted_mantissa >> shift;
        if ((shifted_mantissa >> (shift - 1)) & 1U) {
            half_mantissa += 1;
        }
        return static_cast<uint16_t>(sign | half_mantissa);
    }

    uint32_t half_mantissa = mantissa >> 13;
    if (mantissa & 0x1000U) {
        half_mantissa += 1;
        if (half_mantissa & 0x0400U) {
            half_mantissa = 0;
            return static_cast<uint16_t>(sign | ((half_exponent + 1) << 10));
        }
    }
    return static_cast<uint16_t>(sign | (static_cast<uint32_t>(half_exponent) << 10) | half_mantissa);
}

uint32_t read_little_u32(const char *bytes) {
    return static_cast<uint32_t>(static_cast<unsigned char>(bytes[0])) |
           (static_cast<uint32_t>(static_cast<unsigned char>(bytes[1])) << 8) |
           (static_cast<uint32_t>(static_cast<unsigned char>(bytes[2])) << 16) |
           (static_cast<uint32_t>(static_cast<unsigned char>(bytes[3])) << 24);
}

uint16_t read_little_u16(const char *bytes) {
    return static_cast<uint16_t>(static_cast<unsigned char>(bytes[0]) |
                                 (static_cast<uint16_t>(static_cast<unsigned char>(bytes[1])) << 8));
}

void append_little_u16(std::vector<char> &buffer, uint16_t value) {
    buffer.push_back(static_cast<char>(value & 0xFFU));
    buffer.push_back(static_cast<char>((value >> 8) & 0xFFU));
}

bool convert_range_to_f16(
        const HorizonGgufTensorSource &tensor,
        std::ofstream &output,
        std::string &error) {
    std::ifstream input(tensor.source_path, std::ios::binary);
    if (!input) {
        error = "Unable to open tensor source " + tensor.source_path + ".";
        return false;
    }
    input.seekg(static_cast<std::streamoff>(tensor.source_offset), std::ios::beg);
    if (!input) {
        error = "Unable to seek tensor source " + tensor.source_path + ".";
        return false;
    }

    const uint64_t source_stride = tensor.source_encoding == HorizonTensorEncoding::F32 ? 4 : 2;
    if (tensor.source_data_size % source_stride != 0) {
        error = "Tensor " + tensor.name + " has a byte size that does not match its source dtype.";
        return false;
    }

    std::vector<char> input_buffer(1024 * 1024);
    std::vector<char> output_buffer;
    output_buffer.reserve(input_buffer.size() / source_stride * 2);
    uint64_t remaining = tensor.source_data_size;

    while (remaining > 0) {
        uint64_t chunk = remaining > input_buffer.size() ? input_buffer.size() : remaining;
        chunk -= chunk % source_stride;
        if (chunk == 0) {
            chunk = source_stride;
        }

        input.read(input_buffer.data(), static_cast<std::streamsize>(chunk));
        if (static_cast<uint64_t>(input.gcount()) != chunk) {
            error = "Unable to read tensor data for " + tensor.name + ".";
            return false;
        }

        output_buffer.clear();
        for (uint64_t offset = 0; offset < chunk; offset += source_stride) {
            uint16_t half = 0;
            if (tensor.source_encoding == HorizonTensorEncoding::F32) {
                half = float32_bits_to_float16(read_little_u32(input_buffer.data() + offset));
            } else {
                const uint16_t bf16 = read_little_u16(input_buffer.data() + offset);
                half = float32_bits_to_float16(static_cast<uint32_t>(bf16) << 16);
            }
            append_little_u16(output_buffer, half);
        }

        output.write(output_buffer.data(), static_cast<std::streamsize>(output_buffer.size()));
        if (!output) {
            error = "Unable to write converted tensor data for " + tensor.name + ".";
            return false;
        }
        remaining -= chunk;
    }

    return true;
}

bool write_row_as_f16(
        std::ifstream &input,
        std::ofstream &output,
        const HorizonGgufTensorSource &tensor,
        uint64_t row_offset,
        uint64_t row_source_bytes,
        std::vector<char> &input_buffer,
        std::vector<char> &output_buffer,
        std::string &error) {
    input.seekg(static_cast<std::streamoff>(tensor.source_offset + row_offset), std::ios::beg);
    if (!input) {
        error = "Unable to seek permuted row data for " + tensor.name + ".";
        return false;
    }

    input.read(input_buffer.data(), static_cast<std::streamsize>(row_source_bytes));
    if (static_cast<uint64_t>(input.gcount()) != row_source_bytes) {
        error = "Unable to read permuted row data for " + tensor.name + ".";
        return false;
    }

    if (tensor.source_encoding == HorizonTensorEncoding::F16) {
        output.write(input_buffer.data(), static_cast<std::streamsize>(row_source_bytes));
    } else {
        const uint64_t source_stride = tensor.source_encoding == HorizonTensorEncoding::F32 ? 4 : 2;
        output_buffer.clear();
        for (uint64_t offset = 0; offset < row_source_bytes; offset += source_stride) {
            uint16_t half = 0;
            if (tensor.source_encoding == HorizonTensorEncoding::F32) {
                half = float32_bits_to_float16(read_little_u32(input_buffer.data() + offset));
            } else {
                const uint16_t bf16 = read_little_u16(input_buffer.data() + offset);
                half = float32_bits_to_float16(static_cast<uint32_t>(bf16) << 16);
            }
            append_little_u16(output_buffer, half);
        }
        output.write(output_buffer.data(), static_cast<std::streamsize>(output_buffer.size()));
    }

    if (!output) {
        error = "Unable to write permuted row data for " + tensor.name + ".";
        return false;
    }
    return true;
}

bool permute_rows_to_f16(
        const HorizonGgufTensorSource &tensor,
        std::ofstream &output,
        std::string &error) {
    if (tensor.shape.size() != 2 || tensor.row_permutation_heads == 0) {
        error = "Tensor " + tensor.name + " cannot be row-permuted because its shape is unsupported.";
        return false;
    }

    const uint64_t rows = tensor.shape[0];
    const uint64_t columns = tensor.shape[1];
    const uint64_t heads = tensor.row_permutation_heads;
    if (heads == 0 || rows % heads != 0 || (rows / heads) % 2 != 0) {
        error = "Tensor " + tensor.name + " cannot be row-permuted because its head dimensions are unsupported.";
        return false;
    }

    const uint64_t source_stride = tensor.source_encoding == HorizonTensorEncoding::F32 ? 4 : 2;
    const uint64_t row_source_bytes = columns * source_stride;
    const uint64_t rows_per_head_half = rows / heads / 2;

    std::ifstream input(tensor.source_path, std::ios::binary);
    if (!input) {
        error = "Unable to open tensor source " + tensor.source_path + ".";
        return false;
    }

    std::vector<char> input_buffer(static_cast<size_t>(row_source_bytes));
    std::vector<char> output_buffer;
    output_buffer.reserve(static_cast<size_t>(columns * 2));

    for (uint64_t output_row = 0; output_row < rows; ++output_row) {
        const uint64_t head = output_row / (rows_per_head_half * 2);
        const uint64_t rem = output_row % (rows_per_head_half * 2);
        const uint64_t row_in_half = rem / 2;
        const uint64_t pair_index = rem % 2;
        const uint64_t source_row = (head * 2 + pair_index) * rows_per_head_half + row_in_half;
        const uint64_t row_offset = source_row * row_source_bytes;
        if (!write_row_as_f16(
                    input,
                    output,
                    tensor,
                    row_offset,
                    row_source_bytes,
                    input_buffer,
                    output_buffer,
                    error)) {
            return false;
        }
    }

    return true;
}

float float32_from_bits(uint32_t value) {
    float output = 0.0f;
    std::memcpy(&output, &value, sizeof(float));
    return output;
}

uint32_t float32_to_bits(float value) {
    uint32_t output = 0;
    std::memcpy(&output, &value, sizeof(uint32_t));
    return output;
}

void append_little_u32(std::vector<char> &buffer, uint32_t value) {
    buffer.push_back(static_cast<char>(value & 0xFFU));
    buffer.push_back(static_cast<char>((value >> 8) & 0xFFU));
    buffer.push_back(static_cast<char>((value >> 16) & 0xFFU));
    buffer.push_back(static_cast<char>((value >> 24) & 0xFFU));
}

float float16_bits_to_float32(uint16_t value) {
    const uint32_t sign = static_cast<uint32_t>(value & 0x8000U) << 16;
    uint32_t exponent = (value >> 10) & 0x1FU;
    uint32_t mantissa = value & 0x03FFU;

    if (exponent == 0) {
        if (mantissa == 0) {
            return float32_from_bits(sign);
        }
        exponent = 1;
        while ((mantissa & 0x0400U) == 0) {
            mantissa <<= 1;
            exponent -= 1;
        }
        mantissa &= 0x03FFU;
    } else if (exponent == 0x1FU) {
        return float32_from_bits(sign | 0x7F800000U | (mantissa << 13));
    }

    const uint32_t float_exponent = exponent + (127 - 15);
    return float32_from_bits(sign | (float_exponent << 23) | (mantissa << 13));
}

float read_source_float(
        const char *bytes,
        HorizonTensorEncoding source_encoding) {
    if (source_encoding == HorizonTensorEncoding::F32) {
        return float32_from_bits(read_little_u32(bytes));
    }
    if (source_encoding == HorizonTensorEncoding::BF16) {
        return float32_from_bits(static_cast<uint32_t>(read_little_u16(bytes)) << 16);
    }
    return float16_bits_to_float32(read_little_u16(bytes));
}

int nearest_int(float value) {
    return static_cast<int>(std::nearbyintf(value));
}

int clamp_int(int value, int low, int high) {
    if (value < low) {
        return low;
    }
    if (value > high) {
        return high;
    }
    return value;
}

float make_qx_quants(
        int n,
        int nmax,
        const float *values,
        int8_t *levels) {
    float max = 0.0f;
    float amax = 0.0f;
    for (int index = 0; index < n; ++index) {
        const float absolute = std::fabs(values[index]);
        if (absolute > amax) {
            amax = absolute;
            max = values[index];
        }
    }
    if (amax < kGroupMaxEps) {
        std::memset(levels, 0, static_cast<size_t>(n));
        return 0.0f;
    }

    float iscale = -static_cast<float>(nmax) / max;
    float sumlx = 0.0f;
    float suml2 = 0.0f;
    for (int index = 0; index < n; ++index) {
        const int level = clamp_int(nearest_int(iscale * values[index]), -nmax, nmax - 1);
        levels[index] = static_cast<int8_t>(level + nmax);
        const float weight = values[index] * values[index];
        sumlx += weight * values[index] * static_cast<float>(level);
        suml2 += weight * static_cast<float>(level * level);
    }

    float scale = suml2 != 0.0f ? sumlx / suml2 : 0.0f;
    float best = scale * sumlx;
    for (int step = -9; step <= 9; ++step) {
        if (step == 0) {
            continue;
        }
        iscale = -(static_cast<float>(nmax) + 0.1f * static_cast<float>(step)) / max;
        sumlx = 0.0f;
        suml2 = 0.0f;
        for (int index = 0; index < n; ++index) {
            const int level = clamp_int(nearest_int(iscale * values[index]), -nmax, nmax - 1);
            const float weight = values[index] * values[index];
            sumlx += weight * values[index] * static_cast<float>(level);
            suml2 += weight * static_cast<float>(level * level);
        }
        if (suml2 > 0.0f && sumlx * sumlx > best * suml2) {
            for (int index = 0; index < n; ++index) {
                const int level = clamp_int(nearest_int(iscale * values[index]), -nmax, nmax - 1);
                levels[index] = static_cast<int8_t>(level + nmax);
            }
            scale = sumlx / suml2;
            best = scale * sumlx;
        }
    }
    return scale;
}

float make_q3_quants(
        int n,
        int nmax,
        const float *values,
        int8_t *levels) {
    float max = 0.0f;
    float amax = 0.0f;
    for (int index = 0; index < n; ++index) {
        const float absolute = std::fabs(values[index]);
        if (absolute > amax) {
            amax = absolute;
            max = values[index];
        }
    }
    if (amax < kGroupMaxEps) {
        std::memset(levels, 0, static_cast<size_t>(n));
        return 0.0f;
    }

    float iscale = -static_cast<float>(nmax) / max;
    float sumlx = 0.0f;
    float suml2 = 0.0f;
    for (int index = 0; index < n; ++index) {
        const int level = clamp_int(nearest_int(iscale * values[index]), -nmax, nmax - 1);
        levels[index] = static_cast<int8_t>(level);
        const float weight = values[index] * values[index];
        sumlx += weight * values[index] * static_cast<float>(level);
        suml2 += weight * static_cast<float>(level * level);
    }

    for (int attempt = 0; attempt < 5; ++attempt) {
        int changed = 0;
        for (int index = 0; index < n; ++index) {
            const float weight = values[index] * values[index];
            float slx = sumlx - weight * values[index] * static_cast<float>(levels[index]);
            if (slx > 0.0f) {
                float sl2 = suml2 - weight * static_cast<float>(levels[index] * levels[index]);
                const int new_level = clamp_int(nearest_int(values[index] * sl2 / slx), -nmax, nmax - 1);
                if (new_level != levels[index]) {
                    slx += weight * values[index] * static_cast<float>(new_level);
                    sl2 += weight * static_cast<float>(new_level * new_level);
                    if (sl2 > 0.0f && slx * slx * suml2 > sumlx * sumlx * sl2) {
                        levels[index] = static_cast<int8_t>(new_level);
                        sumlx = slx;
                        suml2 = sl2;
                        changed += 1;
                    }
                }
            }
        }
        if (changed == 0) {
            break;
        }
    }

    for (int index = 0; index < n; ++index) {
        levels[index] = static_cast<int8_t>(levels[index] + nmax);
    }
    return suml2 > 0.0f ? sumlx / suml2 : 0.0f;
}

float make_qkx2_quants(
        int n,
        int nmax,
        const float *values,
        const float *weights,
        uint8_t *levels,
        float *the_min,
        uint8_t *scratch,
        float rmin,
        float rdelta,
        int nstep,
        bool use_mad) {
    float min = values[0];
    float max = values[0];
    float sum_w = weights[0];
    float sum_x = sum_w * values[0];
    for (int index = 1; index < n; ++index) {
        if (values[index] < min) {
            min = values[index];
        }
        if (values[index] > max) {
            max = values[index];
        }
        const float weight = weights[index];
        sum_w += weight;
        sum_x += weight * values[index];
    }
    if (min > 0.0f) {
        min = 0.0f;
    }
    if (max == min) {
        std::memset(levels, 0, static_cast<size_t>(n));
        *the_min = -min;
        return 0.0f;
    }

    float iscale = static_cast<float>(nmax) / (max - min);
    float scale = 1.0f / iscale;
    float best_error = 0.0f;
    for (int index = 0; index < n; ++index) {
        const int level = clamp_int(nearest_int(iscale * (values[index] - min)), 0, nmax);
        levels[index] = static_cast<uint8_t>(level);
        float diff = scale * static_cast<float>(level) + min - values[index];
        diff = use_mad ? std::fabs(diff) : diff * diff;
        best_error += weights[index] * diff;
    }
    if (nstep < 1) {
        *the_min = -min;
        return scale;
    }

    for (int step = 0; step <= nstep; ++step) {
        iscale = (rmin + rdelta * static_cast<float>(step) + static_cast<float>(nmax)) / (max - min);
        float sum_l = 0.0f;
        float sum_l2 = 0.0f;
        float sum_xl = 0.0f;
        for (int index = 0; index < n; ++index) {
            const int level = clamp_int(nearest_int(iscale * (values[index] - min)), 0, nmax);
            scratch[index] = static_cast<uint8_t>(level);
            const float weight = weights[index];
            sum_l += weight * static_cast<float>(level);
            sum_l2 += weight * static_cast<float>(level * level);
            sum_xl += weight * static_cast<float>(level) * values[index];
        }

        const float denominator = sum_w * sum_l2 - sum_l * sum_l;
        if (denominator > 0.0f) {
            float this_scale = (sum_w * sum_xl - sum_x * sum_l) / denominator;
            float this_min = (sum_l2 * sum_x - sum_l * sum_xl) / denominator;
            if (this_min > 0.0f) {
                this_min = 0.0f;
                this_scale = sum_xl / sum_l2;
            }

            float current_error = 0.0f;
            for (int index = 0; index < n; ++index) {
                float diff = this_scale * static_cast<float>(scratch[index]) + this_min - values[index];
                diff = use_mad ? std::fabs(diff) : diff * diff;
                current_error += weights[index] * diff;
            }
            if (current_error < best_error) {
                std::memcpy(levels, scratch, static_cast<size_t>(n));
                best_error = current_error;
                scale = this_scale;
                min = this_min;
            }
        }
    }

    *the_min = -min;
    return scale;
}

void write_scale_min_k4(
        int index,
        uint8_t *scales,
        uint8_t scale,
        uint8_t min) {
    if (index < 4) {
        scales[index] = scale;
        scales[index + 4] = min;
    } else {
        scales[index + 4] = static_cast<uint8_t>((scale & 0x0FU) | ((min & 0x0FU) << 4));
        scales[index - 4] = static_cast<uint8_t>(scales[index - 4] | ((scale >> 4) << 6));
        scales[index] = static_cast<uint8_t>(scales[index] | ((min >> 4) << 6));
    }
}

void read_scale_min_k4(
        int index,
        const uint8_t *scales,
        uint8_t *scale,
        uint8_t *min) {
    if (index < 4) {
        *scale = scales[index] & 0x3FU;
        *min = scales[index + 4] & 0x3FU;
    } else {
        *scale = static_cast<uint8_t>((scales[index + 4] & 0x0FU) | ((scales[index - 4] >> 6) << 4));
        *min = static_cast<uint8_t>((scales[index + 4] >> 4) | ((scales[index] >> 6) << 4));
    }
}

void quantize_q6_k_block(const float *values, char *encoded) {
    int8_t levels[kQkK] = {};
    float scales[kQkK / 16] = {};

    float max_scale = 0.0f;
    float max_abs_scale = 0.0f;
    for (int block = 0; block < kQkK / 16; ++block) {
        const float scale = make_qx_quants(16, 32, values + 16 * block, levels + 16 * block);
        scales[block] = scale;

        const float abs_scale = std::fabs(scale);
        if (abs_scale > max_abs_scale) {
            max_abs_scale = abs_scale;
            max_scale = scale;
        }
    }

    std::memset(encoded, 0, 210);
    if (max_abs_scale < kGroupMaxEps) {
        return;
    }

    const float iscale = -128.0f / max_scale;
    const uint16_t d = float32_bits_to_float16(float32_to_bits(1.0f / iscale));
    uint8_t *ql = reinterpret_cast<uint8_t *>(encoded);
    uint8_t *qh = reinterpret_cast<uint8_t *>(encoded + 128);
    int8_t *block_scales = reinterpret_cast<int8_t *>(encoded + 192);
    encoded[208] = static_cast<char>(d & 0xFFU);
    encoded[209] = static_cast<char>((d >> 8) & 0xFFU);

    const float super_scale = float16_bits_to_float32(d);
    for (int block = 0; block < kQkK / 16; ++block) {
        const int scale = clamp_int(nearest_int(iscale * scales[block]), -128, 127);
        block_scales[block] = static_cast<int8_t>(scale);
    }

    for (int block = 0; block < kQkK / 16; ++block) {
        const float local_scale = super_scale * static_cast<float>(block_scales[block]);
        if (local_scale == 0.0f) {
            continue;
        }
        for (int index = 0; index < 16; ++index) {
            const int level = clamp_int(nearest_int(values[16 * block + index] / local_scale), -32, 31);
            levels[16 * block + index] = static_cast<int8_t>(level + 32);
        }
    }

    for (int base = 0; base < kQkK; base += 128) {
        for (int offset = 0; offset < 32; ++offset) {
            const uint8_t q1 = static_cast<uint8_t>(levels[base + offset + 0]) & 0x0FU;
            const uint8_t q2 = static_cast<uint8_t>(levels[base + offset + 32]) & 0x0FU;
            const uint8_t q3 = static_cast<uint8_t>(levels[base + offset + 64]) & 0x0FU;
            const uint8_t q4 = static_cast<uint8_t>(levels[base + offset + 96]) & 0x0FU;
            ql[offset + 0] = static_cast<uint8_t>(q1 | (q3 << 4));
            ql[offset + 32] = static_cast<uint8_t>(q2 | (q4 << 4));
            qh[offset] = static_cast<uint8_t>(
                    (levels[base + offset + 0] >> 4) |
                    ((levels[base + offset + 32] >> 4) << 2) |
                    ((levels[base + offset + 64] >> 4) << 4) |
                    ((levels[base + offset + 96] >> 4) << 6));
        }
        ql += 64;
        qh += 32;
    }
}

void quantize_q3_k_block(const float *values, char *encoded) {
    int8_t levels[kQkK] = {};
    float scales[kQkK / 16] = {};

    float max_scale = 0.0f;
    float max_abs_scale = 0.0f;
    for (int block = 0; block < kQkK / 16; ++block) {
        scales[block] = make_q3_quants(16, 4, values + 16 * block, levels + 16 * block);
        const float abs_scale = std::fabs(scales[block]);
        if (abs_scale > max_abs_scale) {
            max_abs_scale = abs_scale;
            max_scale = scales[block];
        }
    }

    std::memset(encoded, 0, 110);
    uint8_t *high_mask = reinterpret_cast<uint8_t *>(encoded);
    uint8_t *packed_levels = reinterpret_cast<uint8_t *>(encoded + 32);
    uint8_t *packed_scales = reinterpret_cast<uint8_t *>(encoded + 96);

    uint16_t d = 0;
    if (max_scale != 0.0f) {
        const float iscale = -32.0f / max_scale;
        for (int block = 0; block < kQkK / 16; ++block) {
            int8_t scale = static_cast<int8_t>(clamp_int(nearest_int(iscale * scales[block]), -32, 31) + 32);
            if (block < 8) {
                packed_scales[block] = static_cast<uint8_t>(scale & 0x0F);
            } else {
                packed_scales[block - 8] = static_cast<uint8_t>(packed_scales[block - 8] | ((scale & 0x0F) << 4));
            }
            scale = static_cast<int8_t>(scale >> 4);
            packed_scales[block % 4 + 8] = static_cast<uint8_t>(
                    packed_scales[block % 4 + 8] |
                    (scale << (2 * (block / 4))));
        }
        d = float32_bits_to_float16(float32_to_bits(1.0f / iscale));
    }
    encoded[108] = static_cast<char>(d & 0xFFU);
    encoded[109] = static_cast<char>((d >> 8) & 0xFFU);

    const float super_scale = float16_bits_to_float32(d);
    for (int block = 0; block < kQkK / 16; ++block) {
        int8_t scale = block < 8
                ? static_cast<int8_t>(packed_scales[block] & 0x0F)
                : static_cast<int8_t>(packed_scales[block - 8] >> 4);
        scale = static_cast<int8_t>(
                (scale |
                 (((packed_scales[8 + block % 4] >> (2 * (block / 4))) & 0x03) << 4)) -
                32);
        const float local_scale = super_scale * static_cast<float>(scale);
        if (local_scale == 0.0f) {
            continue;
        }
        for (int index = 0; index < 16; ++index) {
            const int level = clamp_int(nearest_int(values[16 * block + index] / local_scale), -4, 3);
            levels[16 * block + index] = static_cast<int8_t>(level + 4);
        }
    }

    int mask_index = 0;
    uint8_t mask_bit = 1;
    for (int index = 0; index < kQkK; ++index) {
        if (levels[index] > 3) {
            high_mask[mask_index] = static_cast<uint8_t>(high_mask[mask_index] | mask_bit);
            levels[index] = static_cast<int8_t>(levels[index] - 4);
        }
        mask_index += 1;
        if (mask_index == kQkK / 8) {
            mask_index = 0;
            mask_bit = static_cast<uint8_t>(mask_bit << 1);
        }
    }

    for (int base = 0; base < kQkK; base += 128) {
        for (int offset = 0; offset < 32; ++offset) {
            packed_levels[base / 4 + offset] = static_cast<uint8_t>(
                    levels[base + offset] |
                    (levels[base + offset + 32] << 2) |
                    (levels[base + offset + 64] << 4) |
                    (levels[base + offset + 96] << 6));
        }
    }
}

void quantize_q5_k_block(const float *values, char *encoded) {
    uint8_t levels[kQkK] = {};
    float mins[kQkK / 32] = {};
    float scales[kQkK / 32] = {};
    float weights[32] = {};
    uint8_t scratch[32] = {};

    float max_scale = 0.0f;
    float max_min = 0.0f;
    for (int block = 0; block < kQkK / 32; ++block) {
        float sum_x2 = 0.0f;
        for (int index = 0; index < 32; ++index) {
            const float value = values[32 * block + index];
            sum_x2 += value * value;
        }
        const float average_abs = std::sqrt(sum_x2 / 32.0f);
        for (int index = 0; index < 32; ++index) {
            weights[index] = average_abs + std::fabs(values[32 * block + index]);
        }

        scales[block] = make_qkx2_quants(
                32,
                31,
                values + 32 * block,
                weights,
                levels + 32 * block,
                &mins[block],
                scratch,
                -0.5f,
                0.1f,
                15,
                false);
        if (scales[block] > max_scale) {
            max_scale = scales[block];
        }
        if (mins[block] > max_min) {
            max_min = mins[block];
        }
    }

    std::memset(encoded, 0, 176);
    uint8_t *packed_scales = reinterpret_cast<uint8_t *>(encoded + 4);
    uint8_t *qh = reinterpret_cast<uint8_t *>(encoded + 16);
    uint8_t *ql = reinterpret_cast<uint8_t *>(encoded + 48);

    const float inv_scale = max_scale > 0.0f ? 63.0f / max_scale : 0.0f;
    const float inv_min = max_min > 0.0f ? 63.0f / max_min : 0.0f;
    for (int block = 0; block < kQkK / 32; ++block) {
        const uint8_t scale = static_cast<uint8_t>(clamp_int(nearest_int(inv_scale * scales[block]), 0, 63));
        const uint8_t min = static_cast<uint8_t>(clamp_int(nearest_int(inv_min * mins[block]), 0, 63));
        write_scale_min_k4(block, packed_scales, scale, min);
    }

    const uint16_t d = float32_bits_to_float16(float32_to_bits(max_scale / 63.0f));
    const uint16_t dmin = float32_bits_to_float16(float32_to_bits(max_min / 63.0f));
    encoded[0] = static_cast<char>(d & 0xFFU);
    encoded[1] = static_cast<char>((d >> 8) & 0xFFU);
    encoded[2] = static_cast<char>(dmin & 0xFFU);
    encoded[3] = static_cast<char>((dmin >> 8) & 0xFFU);

    const float super_scale = float16_bits_to_float32(d);
    const float super_min = float16_bits_to_float32(dmin);
    for (int block = 0; block < kQkK / 32; ++block) {
        uint8_t scale = 0;
        uint8_t min = 0;
        read_scale_min_k4(block, packed_scales, &scale, &min);
        const float local_scale = super_scale * static_cast<float>(scale);
        if (local_scale == 0.0f) {
            continue;
        }
        const float local_min = super_min * static_cast<float>(min);
        for (int index = 0; index < 32; ++index) {
            const int level = clamp_int(nearest_int((values[32 * block + index] + local_min) / local_scale), 0, 31);
            levels[32 * block + index] = static_cast<uint8_t>(level);
        }
    }

    uint8_t mask_low = 1;
    uint8_t mask_high = 2;
    for (int base = 0; base < kQkK; base += 64) {
        for (int offset = 0; offset < 32; ++offset) {
            int low = levels[base + offset];
            if (low > 15) {
                low -= 16;
                qh[offset] = static_cast<uint8_t>(qh[offset] | mask_low);
            }
            int high = levels[base + offset + 32];
            if (high > 15) {
                high -= 16;
                qh[offset] = static_cast<uint8_t>(qh[offset] | mask_high);
            }
            ql[offset] = static_cast<uint8_t>(low | (high << 4));
        }
        mask_low = static_cast<uint8_t>(mask_low << 2);
        mask_high = static_cast<uint8_t>(mask_high << 2);
        ql += 32;
    }
}

void quantize_q4_k_block(const float *values, char *encoded) {
    uint8_t levels[kQkK] = {};
    float mins[kQkK / 32] = {};
    float scales[kQkK / 32] = {};
    float weights[32] = {};
    uint8_t scratch[32] = {};

    float max_scale = 0.0f;
    float max_min = 0.0f;
    for (int block = 0; block < kQkK / 32; ++block) {
        float sum_x2 = 0.0f;
        for (int index = 0; index < 32; ++index) {
            const float value = values[32 * block + index];
            sum_x2 += value * value;
        }
        const float average_abs = std::sqrt(sum_x2 / 32.0f);
        for (int index = 0; index < 32; ++index) {
            weights[index] = average_abs + std::fabs(values[32 * block + index]);
        }

        scales[block] = make_qkx2_quants(
                32,
                15,
                values + 32 * block,
                weights,
                levels + 32 * block,
                &mins[block],
                scratch,
                -1.0f,
                0.1f,
                20,
                false);
        if (scales[block] > max_scale) {
            max_scale = scales[block];
        }
        if (mins[block] > max_min) {
            max_min = mins[block];
        }
    }

    std::memset(encoded, 0, 144);
    uint8_t *packed_scales = reinterpret_cast<uint8_t *>(encoded + 4);
    uint8_t *packed_levels = reinterpret_cast<uint8_t *>(encoded + 16);

    const float inv_scale = max_scale > 0.0f ? 63.0f / max_scale : 0.0f;
    const float inv_min = max_min > 0.0f ? 63.0f / max_min : 0.0f;
    for (int block = 0; block < kQkK / 32; ++block) {
        const uint8_t scale = static_cast<uint8_t>(clamp_int(nearest_int(inv_scale * scales[block]), 0, 63));
        const uint8_t min = static_cast<uint8_t>(clamp_int(nearest_int(inv_min * mins[block]), 0, 63));
        write_scale_min_k4(block, packed_scales, scale, min);
    }

    const uint16_t d = float32_bits_to_float16(float32_to_bits(max_scale / 63.0f));
    const uint16_t dmin = float32_bits_to_float16(float32_to_bits(max_min / 63.0f));
    encoded[0] = static_cast<char>(d & 0xFFU);
    encoded[1] = static_cast<char>((d >> 8) & 0xFFU);
    encoded[2] = static_cast<char>(dmin & 0xFFU);
    encoded[3] = static_cast<char>((dmin >> 8) & 0xFFU);

    const float super_scale = float16_bits_to_float32(d);
    const float super_min = float16_bits_to_float32(dmin);
    for (int block = 0; block < kQkK / 32; ++block) {
        uint8_t scale = 0;
        uint8_t min = 0;
        read_scale_min_k4(block, packed_scales, &scale, &min);
        const float local_scale = super_scale * static_cast<float>(scale);
        if (local_scale == 0.0f) {
            continue;
        }
        const float local_min = super_min * static_cast<float>(min);
        for (int index = 0; index < 32; ++index) {
            const int level = clamp_int(nearest_int((values[32 * block + index] + local_min) / local_scale), 0, 15);
            levels[32 * block + index] = static_cast<uint8_t>(level);
        }
    }

    for (int base = 0; base < kQkK; base += 64) {
        for (int offset = 0; offset < 32; ++offset) {
            packed_levels[offset] = static_cast<uint8_t>(
                    levels[base + offset] |
                    (levels[base + offset + 32] << 4));
        }
        packed_levels += 32;
    }
}

bool convert_range_to_f32(
        const HorizonGgufTensorSource &tensor,
        std::ofstream &output,
        std::string &error) {
    if (tensor.source_encoding == HorizonTensorEncoding::F32) {
        return copy_range(tensor, output, error);
    }

    std::ifstream input(tensor.source_path, std::ios::binary);
    if (!input) {
        error = "Unable to open tensor source " + tensor.source_path + ".";
        return false;
    }
    input.seekg(static_cast<std::streamoff>(tensor.source_offset), std::ios::beg);
    if (!input) {
        error = "Unable to seek tensor source " + tensor.source_path + ".";
        return false;
    }

    const uint64_t source_stride = 2;
    if (tensor.source_data_size % source_stride != 0) {
        error = "Tensor " + tensor.name + " has a byte size that does not match its source dtype.";
        return false;
    }

    std::vector<char> input_buffer(1024 * 1024);
    std::vector<char> output_buffer;
    output_buffer.reserve(input_buffer.size() / source_stride * 4);
    uint64_t remaining = tensor.source_data_size;

    while (remaining > 0) {
        uint64_t chunk = remaining > input_buffer.size() ? input_buffer.size() : remaining;
        chunk -= chunk % source_stride;
        if (chunk == 0) {
            chunk = source_stride;
        }

        input.read(input_buffer.data(), static_cast<std::streamsize>(chunk));
        if (static_cast<uint64_t>(input.gcount()) != chunk) {
            error = "Unable to read tensor data for " + tensor.name + ".";
            return false;
        }

        output_buffer.clear();
        for (uint64_t offset = 0; offset < chunk; offset += source_stride) {
            append_little_u32(
                    output_buffer,
                    float32_to_bits(read_source_float(input_buffer.data() + offset, tensor.source_encoding)));
        }

        output.write(output_buffer.data(), static_cast<std::streamsize>(output_buffer.size()));
        if (!output) {
            error = "Unable to write F32 tensor data for " + tensor.name + ".";
            return false;
        }
        remaining -= chunk;
    }

    return true;
}

bool quantize_range_to_q8_0(
        const HorizonGgufTensorSource &tensor,
        std::ofstream &output,
        std::string &error) {
    const uint64_t source_stride = tensor.source_encoding == HorizonTensorEncoding::F32 ? 4 : 2;
    const uint64_t element_count = tensor.source_data_size / source_stride;
    if (tensor.source_data_size % source_stride != 0 || element_count % 32 != 0) {
        error = "Tensor " + tensor.name + " cannot be Q8_0 quantized because its element count is not a multiple of 32.";
        return false;
    }

    std::ifstream input(tensor.source_path, std::ios::binary);
    if (!input) {
        error = "Unable to open tensor source " + tensor.source_path + ".";
        return false;
    }
    input.seekg(static_cast<std::streamoff>(tensor.source_offset), std::ios::beg);
    if (!input) {
        error = "Unable to seek tensor source " + tensor.source_path + ".";
        return false;
    }

    std::vector<char> source_block(static_cast<size_t>(source_stride * 32));
    std::vector<float> values(32);
    for (uint64_t block = 0; block < element_count / 32; ++block) {
        input.read(source_block.data(), static_cast<std::streamsize>(source_block.size()));
        if (static_cast<size_t>(input.gcount()) != source_block.size()) {
            error = "Unable to read tensor data for " + tensor.name + ".";
            return false;
        }

        float abs_max = 0.0f;
        for (size_t index = 0; index < 32; ++index) {
            values[index] = read_source_float(source_block.data() + index * source_stride, tensor.source_encoding);
            const float absolute = values[index] < 0.0f ? -values[index] : values[index];
            if (absolute > abs_max) {
                abs_max = absolute;
            }
        }

        const float scale = abs_max == 0.0f ? 0.0f : abs_max / 127.0f;
        const uint16_t scale_half = float32_bits_to_float16(float32_to_bits(scale));
        char encoded[34] = {};
        encoded[0] = static_cast<char>(scale_half & 0xFFU);
        encoded[1] = static_cast<char>((scale_half >> 8) & 0xFFU);
        for (size_t index = 0; index < 32; ++index) {
            int quantized = 0;
            if (scale != 0.0f) {
                const float scaled = values[index] / scale;
                quantized = static_cast<int>(scaled >= 0.0f ? scaled + 0.5f : scaled - 0.5f);
                if (quantized < -128) {
                    quantized = -128;
                } else if (quantized > 127) {
                    quantized = 127;
                }
            }
            encoded[2 + index] = static_cast<char>(quantized);
        }

        output.write(encoded, sizeof(encoded));
        if (!output) {
            error = "Unable to write Q8_0 tensor data for " + tensor.name + ".";
            return false;
        }
    }

    return true;
}

bool quantize_range_to_q6_k(
        const HorizonGgufTensorSource &tensor,
        std::ofstream &output,
        std::string &error) {
    const uint64_t source_stride = tensor.source_encoding == HorizonTensorEncoding::F32 ? 4 : 2;
    const uint64_t element_count = tensor.source_data_size / source_stride;
    if (tensor.source_data_size % source_stride != 0 || element_count % kQkK != 0) {
        error = "Tensor " + tensor.name + " cannot be Q6_K quantized because its element count is not a multiple of 256.";
        return false;
    }

    std::ifstream input(tensor.source_path, std::ios::binary);
    if (!input) {
        error = "Unable to open tensor source " + tensor.source_path + ".";
        return false;
    }
    input.seekg(static_cast<std::streamoff>(tensor.source_offset), std::ios::beg);
    if (!input) {
        error = "Unable to seek tensor source " + tensor.source_path + ".";
        return false;
    }

    std::vector<char> source_block(static_cast<size_t>(source_stride * kQkK));
    std::vector<float> values(kQkK);
    char encoded[210] = {};
    for (uint64_t block = 0; block < element_count / kQkK; ++block) {
        input.read(source_block.data(), static_cast<std::streamsize>(source_block.size()));
        if (static_cast<size_t>(input.gcount()) != source_block.size()) {
            error = "Unable to read tensor data for " + tensor.name + ".";
            return false;
        }

        for (int index = 0; index < kQkK; ++index) {
            values[index] = read_source_float(source_block.data() + index * source_stride, tensor.source_encoding);
        }
        quantize_q6_k_block(values.data(), encoded);

        output.write(encoded, sizeof(encoded));
        if (!output) {
            error = "Unable to write Q6_K tensor data for " + tensor.name + ".";
            return false;
        }
    }

    return true;
}

bool quantize_range_to_q3_k(
        const HorizonGgufTensorSource &tensor,
        std::ofstream &output,
        std::string &error) {
    const uint64_t source_stride = tensor.source_encoding == HorizonTensorEncoding::F32 ? 4 : 2;
    const uint64_t element_count = tensor.source_data_size / source_stride;
    if (tensor.source_data_size % source_stride != 0 || element_count % kQkK != 0) {
        error = "Tensor " + tensor.name + " cannot be Q3_K quantized because its element count is not a multiple of 256.";
        return false;
    }

    std::ifstream input(tensor.source_path, std::ios::binary);
    if (!input) {
        error = "Unable to open tensor source " + tensor.source_path + ".";
        return false;
    }
    input.seekg(static_cast<std::streamoff>(tensor.source_offset), std::ios::beg);
    if (!input) {
        error = "Unable to seek tensor source " + tensor.source_path + ".";
        return false;
    }

    std::vector<char> source_block(static_cast<size_t>(source_stride * kQkK));
    std::vector<float> values(kQkK);
    char encoded[110] = {};
    for (uint64_t block = 0; block < element_count / kQkK; ++block) {
        input.read(source_block.data(), static_cast<std::streamsize>(source_block.size()));
        if (static_cast<size_t>(input.gcount()) != source_block.size()) {
            error = "Unable to read tensor data for " + tensor.name + ".";
            return false;
        }

        for (int index = 0; index < kQkK; ++index) {
            values[index] = read_source_float(source_block.data() + index * source_stride, tensor.source_encoding);
        }
        quantize_q3_k_block(values.data(), encoded);

        output.write(encoded, sizeof(encoded));
        if (!output) {
            error = "Unable to write Q3_K tensor data for " + tensor.name + ".";
            return false;
        }
    }

    return true;
}

bool quantize_range_to_q5_k(
        const HorizonGgufTensorSource &tensor,
        std::ofstream &output,
        std::string &error) {
    const uint64_t source_stride = tensor.source_encoding == HorizonTensorEncoding::F32 ? 4 : 2;
    const uint64_t element_count = tensor.source_data_size / source_stride;
    if (tensor.source_data_size % source_stride != 0 || element_count % kQkK != 0) {
        error = "Tensor " + tensor.name + " cannot be Q5_K quantized because its element count is not a multiple of 256.";
        return false;
    }

    std::ifstream input(tensor.source_path, std::ios::binary);
    if (!input) {
        error = "Unable to open tensor source " + tensor.source_path + ".";
        return false;
    }
    input.seekg(static_cast<std::streamoff>(tensor.source_offset), std::ios::beg);
    if (!input) {
        error = "Unable to seek tensor source " + tensor.source_path + ".";
        return false;
    }

    std::vector<char> source_block(static_cast<size_t>(source_stride * kQkK));
    std::vector<float> values(kQkK);
    char encoded[176] = {};
    for (uint64_t block = 0; block < element_count / kQkK; ++block) {
        input.read(source_block.data(), static_cast<std::streamsize>(source_block.size()));
        if (static_cast<size_t>(input.gcount()) != source_block.size()) {
            error = "Unable to read tensor data for " + tensor.name + ".";
            return false;
        }

        for (int index = 0; index < kQkK; ++index) {
            values[index] = read_source_float(source_block.data() + index * source_stride, tensor.source_encoding);
        }
        quantize_q5_k_block(values.data(), encoded);

        output.write(encoded, sizeof(encoded));
        if (!output) {
            error = "Unable to write Q5_K tensor data for " + tensor.name + ".";
            return false;
        }
    }

    return true;
}

bool quantize_range_to_q4_k(
        const HorizonGgufTensorSource &tensor,
        std::ofstream &output,
        std::string &error) {
    const uint64_t source_stride = tensor.source_encoding == HorizonTensorEncoding::F32 ? 4 : 2;
    const uint64_t element_count = tensor.source_data_size / source_stride;
    if (tensor.source_data_size % source_stride != 0 || element_count % kQkK != 0) {
        error = "Tensor " + tensor.name + " cannot be Q4_K quantized because its element count is not a multiple of 256.";
        return false;
    }

    std::ifstream input(tensor.source_path, std::ios::binary);
    if (!input) {
        error = "Unable to open tensor source " + tensor.source_path + ".";
        return false;
    }
    input.seekg(static_cast<std::streamoff>(tensor.source_offset), std::ios::beg);
    if (!input) {
        error = "Unable to seek tensor source " + tensor.source_path + ".";
        return false;
    }

    std::vector<char> source_block(static_cast<size_t>(source_stride * kQkK));
    std::vector<float> values(kQkK);
    char encoded[144] = {};
    for (uint64_t block = 0; block < element_count / kQkK; ++block) {
        input.read(source_block.data(), static_cast<std::streamsize>(source_block.size()));
        if (static_cast<size_t>(input.gcount()) != source_block.size()) {
            error = "Unable to read tensor data for " + tensor.name + ".";
            return false;
        }

        for (int index = 0; index < kQkK; ++index) {
            values[index] = read_source_float(source_block.data() + index * source_stride, tensor.source_encoding);
        }
        quantize_q4_k_block(values.data(), encoded);

        output.write(encoded, sizeof(encoded));
        if (!output) {
            error = "Unable to write Q4_K tensor data for " + tensor.name + ".";
            return false;
        }
    }

    return true;
}

bool write_tensor_data(
        const HorizonGgufTensorSource &tensor,
        std::ofstream &output,
        std::string &error) {
    if (tensor.row_permutation_heads != 0) {
        if (tensor.output_encoding != HorizonTensorOutputEncoding::F16) {
            error = "Tensor " + tensor.name + " needs row permutation before quantized output can be written.";
            return false;
        }
        return permute_rows_to_f16(tensor, output, error);
    }
    if (tensor.output_encoding == HorizonTensorOutputEncoding::F32) {
        return convert_range_to_f32(tensor, output, error);
    }
    if (tensor.output_encoding == HorizonTensorOutputEncoding::Q8_0) {
        return quantize_range_to_q8_0(tensor, output, error);
    }
    if (tensor.output_encoding == HorizonTensorOutputEncoding::Q6_K) {
        return quantize_range_to_q6_k(tensor, output, error);
    }
    if (tensor.output_encoding == HorizonTensorOutputEncoding::Q3_K) {
        return quantize_range_to_q3_k(tensor, output, error);
    }
    if (tensor.output_encoding == HorizonTensorOutputEncoding::Q5_K) {
        return quantize_range_to_q5_k(tensor, output, error);
    }
    if (tensor.output_encoding == HorizonTensorOutputEncoding::Q4_K) {
        return quantize_range_to_q4_k(tensor, output, error);
    }
    if (tensor.source_encoding == HorizonTensorEncoding::F16) {
        return copy_range(tensor, output, error);
    }
    return convert_range_to_f16(tensor, output, error);
}

}  // namespace

void HorizonGgufMetadataWriter::add_string(const std::string &key, const std::string &value) {
    entries_.push_back({key, ValueType::String, value});
}

void HorizonGgufMetadataWriter::add_string_array(
        const std::string &key,
        const std::vector<std::string> &values) {
    Entry entry {key, ValueType::Array};
    entry.string_array_value = values;
    entries_.push_back(entry);
}

void HorizonGgufMetadataWriter::add_uint32(const std::string &key, uint32_t value) {
    Entry entry {key, ValueType::Uint32};
    entry.uint32_value = value;
    entries_.push_back(entry);
}

void HorizonGgufMetadataWriter::add_bool(const std::string &key, bool value) {
    Entry entry {key, ValueType::Bool};
    entry.bool_value = value;
    entries_.push_back(entry);
}

void HorizonGgufMetadataWriter::add_float32(const std::string &key, float value) {
    Entry entry {key, ValueType::Float32};
    entry.float32_value = value;
    entries_.push_back(entry);
}

void HorizonGgufMetadataWriter::add_float32_array(
        const std::string &key,
        const std::vector<float> &values) {
    Entry entry {key, ValueType::Array};
    entry.float32_array_value = values;
    entries_.push_back(entry);
}

void HorizonGgufMetadataWriter::add_int32_array(
        const std::string &key,
        const std::vector<int32_t> &values) {
    Entry entry {key, ValueType::Array};
    entry.int32_array_value = values;
    entries_.push_back(entry);
}

std::vector<uint8_t> HorizonGgufMetadataWriter::build(uint64_t tensor_count) const {
    std::vector<uint8_t> buffer;
    append_plain<uint32_t>(buffer, kGgufMagic);
    append_plain<uint32_t>(buffer, kGgufVersion);
    append_plain<uint64_t>(buffer, tensor_count);
    append_plain<uint64_t>(buffer, static_cast<uint64_t>(entries_.size()));

    for (const Entry &entry : entries_) {
        append_string(buffer, entry.key);
        append_plain<uint32_t>(buffer, static_cast<uint32_t>(entry.type));
        switch (entry.type) {
            case ValueType::String:
                append_string(buffer, entry.string_value);
                break;
            case ValueType::Array:
                if (!entry.string_array_value.empty()) {
                    append_plain<uint32_t>(buffer, static_cast<uint32_t>(ValueType::String));
                    append_plain<uint64_t>(buffer, static_cast<uint64_t>(entry.string_array_value.size()));
                    for (const std::string &value : entry.string_array_value) {
                        append_string(buffer, value);
                    }
                } else if (!entry.float32_array_value.empty()) {
                    append_plain<uint32_t>(buffer, static_cast<uint32_t>(ValueType::Float32));
                    append_plain<uint64_t>(buffer, static_cast<uint64_t>(entry.float32_array_value.size()));
                    for (float value : entry.float32_array_value) {
                        append_plain<float>(buffer, value);
                    }
                } else {
                    append_plain<uint32_t>(buffer, static_cast<uint32_t>(ValueType::Int32));
                    append_plain<uint64_t>(buffer, static_cast<uint64_t>(entry.int32_array_value.size()));
                    for (int32_t value : entry.int32_array_value) {
                        append_plain<int32_t>(buffer, value);
                    }
                }
                break;
            case ValueType::Uint32:
                append_plain<uint32_t>(buffer, entry.uint32_value);
                break;
            case ValueType::Int32:
                append_plain<int32_t>(buffer, entry.int32_value);
                break;
            case ValueType::Float32:
                append_plain<float>(buffer, entry.float32_value);
                break;
            case ValueType::Bool:
                append_plain<uint8_t>(buffer, entry.bool_value ? 1 : 0);
                break;
        }
    }

    return buffer;
}

bool HorizonGgufMetadataWriter::write_file(
        const std::string &output_path,
        const std::vector<HorizonGgufTensorSource> &tensors,
        std::string &error) const {
    std::vector<uint64_t> offsets;
    offsets.reserve(tensors.size());
    uint64_t cursor = 0;
    for (const HorizonGgufTensorSource &tensor : tensors) {
        cursor = align_to(cursor, kDefaultAlignment);
        offsets.push_back(cursor);
        cursor += tensor.output_data_size;
    }

    std::vector<uint8_t> header = build(static_cast<uint64_t>(tensors.size()));
    for (size_t index = 0; index < tensors.size(); ++index) {
        const HorizonGgufTensorSource &tensor = tensors[index];
        append_string(header, tensor.name);
        append_plain<uint32_t>(header, static_cast<uint32_t>(tensor.shape.size()));
        for (auto dimension = tensor.shape.rbegin(); dimension != tensor.shape.rend(); ++dimension) {
            append_plain<uint64_t>(header, *dimension);
        }
        append_plain<uint32_t>(header, tensor.ggml_type);
        append_plain<uint64_t>(header, offsets[index]);
    }

    const uint64_t tensor_data_start = align_to(static_cast<uint64_t>(header.size()), kDefaultAlignment);

    std::ofstream output(output_path, std::ios::binary | std::ios::trunc);
    if (!output) {
        error = "Unable to open GGUF output path " + output_path + ".";
        return false;
    }
    output.write(reinterpret_cast<const char *>(header.data()), static_cast<std::streamsize>(header.size()));
    if (!output) {
        error = "Unable to write GGUF metadata.";
        return false;
    }
    if (!write_zero_padding(output, tensor_data_start - header.size())) {
        error = "Unable to write GGUF metadata padding.";
        return false;
    }

    uint64_t written_tensor_bytes = 0;
    for (const HorizonGgufTensorSource &tensor : tensors) {
        const uint64_t aligned = align_to(written_tensor_bytes, kDefaultAlignment);
        if (!write_zero_padding(output, aligned - written_tensor_bytes)) {
            error = "Unable to write tensor alignment padding.";
            return false;
        }
        written_tensor_bytes = aligned;

        if (!write_tensor_data(tensor, output, error)) {
            return false;
        }
        written_tensor_bytes += tensor.output_data_size;
    }

    return true;
}

size_t HorizonGgufMetadataWriter::kv_count() const {
    return entries_.size();
}
