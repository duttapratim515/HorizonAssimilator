#include <jni.h>

#include "horizon_gguf_converter.h"

#include <string>

static std::string g_last_error;

extern "C"
JNIEXPORT jint JNICALL
Java_com_example_horizonassimilator_conversion_NativeGgufEngine_nativeConvertToGguf(
        JNIEnv *env,
        jobject,
        jstring model_directory_path,
        jstring output_file_path,
        jstring quantization_value) {
    const char *model_directory = env->GetStringUTFChars(model_directory_path, nullptr);
    const char *output_file = env->GetStringUTFChars(output_file_path, nullptr);
    const char *quantization = env->GetStringUTFChars(quantization_value, nullptr);

    HorizonConversionSummary result = inspect_hf_safetensors_model(
            model_directory == nullptr ? "" : model_directory,
            output_file == nullptr ? "" : output_file,
            quantization == nullptr ? "" : quantization);
    g_last_error = result.message;

    if (model_directory != nullptr) {
        env->ReleaseStringUTFChars(model_directory_path, model_directory);
    }
    if (output_file != nullptr) {
        env->ReleaseStringUTFChars(output_file_path, output_file);
    }
    if (quantization != nullptr) {
        env->ReleaseStringUTFChars(quantization_value, quantization);
    }

    return result.ok ? 0 : 1;
}

extern "C"
JNIEXPORT jstring JNICALL
Java_com_example_horizonassimilator_conversion_NativeGgufEngine_nativeLastError(
        JNIEnv *env,
        jobject) {
    return env->NewStringUTF(g_last_error.c_str());
}
