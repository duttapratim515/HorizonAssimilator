package com.example.horizonassimilator.conversion

import android.os.Build
import java.io.File

data class NativeGgufRequest(
    val modelDirectory: File,
    val outputFile: File
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

            val resultCode = nativeConvertToGguf(
                modelDirectoryPath = request.modelDirectory.absolutePath,
                outputFilePath = request.outputFile.absolutePath
            )

            check(resultCode == 0) {
                nativeLastError().ifBlank { "Native GGUF conversion failed with code $resultCode." }
            }

            onProgress(ConversionProgress(1f, "Native GGUF conversion complete"))
        }
    }

    private external fun nativeConvertToGguf(
        modelDirectoryPath: String,
        outputFilePath: String
    ): Int

    private external fun nativeLastError(): String

    companion object {
        init {
            System.loadLibrary("horizon_gguf_engine")
        }
    }
}
