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
        jstring quantization_value,
        jstring model_family_value,
        jobject progress_callback) {
    const char *model_directory = env->GetStringUTFChars(model_directory_path, nullptr);
    const char *output_file = env->GetStringUTFChars(output_file_path, nullptr);
    const char *quantization = env->GetStringUTFChars(quantization_value, nullptr);
    const char *model_family = env->GetStringUTFChars(model_family_value, nullptr);
    jmethodID on_progress_method = nullptr;
    if (progress_callback != nullptr) {
        jclass callback_class = env->GetObjectClass(progress_callback);
        on_progress_method = env->GetMethodID(callback_class, "onProgress", "(FLjava/lang/String;)V");
        env->DeleteLocalRef(callback_class);
    }

    HorizonConversionSummary result = inspect_hf_safetensors_model(
            model_directory == nullptr ? "" : model_directory,
            output_file == nullptr ? "" : output_file,
            quantization == nullptr ? "" : quantization,
            model_family == nullptr ? "" : model_family,
            [env, progress_callback, on_progress_method](float progress, const std::string &message) {
                if (progress_callback == nullptr || on_progress_method == nullptr) {
                    return;
                }
                jstring java_message = env->NewStringUTF(message.c_str());
                env->CallVoidMethod(progress_callback, on_progress_method, progress, java_message);
                env->DeleteLocalRef(java_message);
            });
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
    if (model_family != nullptr) {
        env->ReleaseStringUTFChars(model_family_value, model_family);
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
