package com.example.horizonassimilator.conversion

import android.os.Build
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.runBlocking
import kotlinx.coroutines.withContext
import java.io.File

data class NativeGgufRequest(
    val modelDirectory: File,
    val outputFile: File,
    val quantization: GgufQuantization,
    val modelFamily: ModelFamily
)

interface GgufEngine {
    suspend fun convert(
        request: NativeGgufRequest,
        onProgress: suspend (ConversionProgress) -> Unit
    ): Result<Unit>
}

class NativeGgufEngine : GgufEngine {
    override suspend fun convert(
        request: NativeGgufRequest,
        onProgress: suspend (ConversionProgress) -> Unit
    ): Result<Unit> {
        return runCatching {
            require(Build.SUPPORTED_64_BIT_ABIS.contains("arm64-v8a")) {
                "This conversion engine requires an arm64-v8a Android device."
            }
            require(request.modelDirectory.isDirectory) {
                "Model workspace does not exist."
            }

            onProgress(ConversionProgress(0.25f, "Starting native GGUF engine"))

            val resultCode = withContext(Dispatchers.Default) {
                val nativeProgress = NativeProgressCallback { progress, message ->
                    runBlocking {
                        onProgress(ConversionProgress(progress, message))
                    }
                }
                nativeConvertToGguf(
                    modelDirectoryPath = request.modelDirectory.absolutePath,
                    outputFilePath = request.outputFile.absolutePath,
                    quantization = request.quantization.label,
                    modelFamily = request.modelFamily.name,
                    progressCallback = nativeProgress
                )
            }

            check(resultCode == 0) {
                nativeLastError().ifBlank { "Native GGUF conversion failed with code $resultCode." }
            }

            onProgress(ConversionProgress(1f, "Native GGUF conversion complete"))
        }
    }

    private external fun nativeConvertToGguf(
        modelDirectoryPath: String,
        outputFilePath: String,
        quantization: String,
        modelFamily: String,
        progressCallback: NativeProgressCallback
    ): Int

    private external fun nativeLastError(): String

    private fun interface NativeProgressCallback {
        fun onProgress(progress: Float, message: String)
    }

    companion object {
        init {
            System.loadLibrary("horizon_gguf_engine")
        }
    }
}
