package com.example.horizonassimilator.conversion

import android.content.Context
import kotlinx.coroutines.delay

class AndroidSafetensorsToGgufConverter(
    private val context: Context
) : SafetensorsToGgufConverter {

    override suspend fun convert(
        request: ConversionRequest,
        onProgress: suspend (ConversionProgress) -> Unit
    ): Result<Unit> {
        return runCatching {
            onProgress(ConversionProgress(0.05f, "Validating selected safetensors file"))
            require(request.input.displayName.endsWith(".safetensors", ignoreCase = true)) {
                "Select a .safetensors file before converting."
            }

            context.contentResolver.openInputStream(request.input.uri)?.use { input ->
                check(input.read() != -1) {
                    "The selected safetensors file is empty or cannot be read."
                }
            } ?: error("Unable to open the selected safetensors file.")

            delay(250)
            onProgress(ConversionProgress(0.20f, "Preparing GGUF output target"))

            context.contentResolver.openOutputStream(request.output, "wt")?.use { output ->
                output.flush()
            } ?: error("Unable to open the selected GGUF output file.")

            delay(250)
            onProgress(ConversionProgress(null, "Waiting for native GGUF converter engine"))

            throw ConverterEngineUnavailableException(
                "The Android conversion flow is wired, but the native safetensors-to-GGUF engine is not bundled yet."
            )
        }
    }
}
