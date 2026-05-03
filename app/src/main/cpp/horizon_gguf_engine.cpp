#include <jni.h>

#include <string>

static std::string g_last_error;

extern "C"
JNIEXPORT jint JNICALL
Java_com_example_horizonassimilator_conversion_NativeGgufEngine_nativeConvertToGguf(
        JNIEnv *env,
        jobject,
        jstring model_directory_path,
        jstring output_file_path) {
    const char *model_directory = env->GetStringUTFChars(model_directory_path, nullptr);
    const char *output_file = env->GetStringUTFChars(output_file_path, nullptr);

    g_last_error = "Native GGUF engine is linked for arm64-v8a, but the safetensors-to-GGUF converter is not bundled yet. Model directory: ";
    g_last_error += model_directory == nullptr ? "<unavailable>" : model_directory;
    g_last_error += " Output: ";
    g_last_error += output_file == nullptr ? "<unavailable>" : output_file;

    if (model_directory != nullptr) {
        env->ReleaseStringUTFChars(model_directory_path, model_directory);
    }
    if (output_file != nullptr) {
        env->ReleaseStringUTFChars(output_file_path, output_file);
    }

    return 1;
}

extern "C"
JNIEXPORT jstring JNICALL
Java_com_example_horizonassimilator_conversion_NativeGgufEngine_nativeLastError(
        JNIEnv *env,
        jobject) {
    return env->NewStringUTF(g_last_error.c_str());
}
