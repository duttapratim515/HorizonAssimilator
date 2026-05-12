#pragma once

#include <functional>
#include <string>

struct HorizonConversionSummary {
    bool ok;
    std::string message;
};

HorizonConversionSummary inspect_hf_safetensors_model(
        const std::string &model_directory,
        const std::string &output_file,
        const std::string &quantization,
        const std::string &model_family,
        const std::function<void(float, const std::string &)> &on_progress = nullptr);
