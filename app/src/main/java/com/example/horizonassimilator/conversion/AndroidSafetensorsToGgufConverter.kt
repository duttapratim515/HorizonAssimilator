package com.example.horizonassimilator.conversion

import android.content.Context
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.delay
import kotlinx.coroutines.withContext
import java.io.File

class AndroidSafetensorsToGgufConverter(
    private val context: Context
) : SafetensorsToGgufConverter {
    private val workspacePreparer = ModelWorkspacePreparer(context)
    private val ggufEngine: GgufEngine = NativeGgufEngine()

    override suspend fun convert(
        request: ConversionRequest,
        onProgress: suspend (ConversionProgress) -> Unit
    ): Result<Unit> {
        return runCatching {
            onProgress(ConversionProgress(0.02f, "Creating local model workspace"))
            val workspace = workspacePreparer.prepare(
                model = request.input,
                onProgress = onProgress
            )

            onProgress(ConversionProgress(0.18f, "Validating staged model files"))
            val safetensorsFiles = workspace.safetensorsFiles
            require(safetensorsFiles.isNotEmpty()) {
                "Select at least one .safetensors file before converting."
            }

            val fileNames = workspace.files.map { it.file.name }
            require(fileNames.any { it.equals("config.json", ignoreCase = true) }) {
                "Add config.json from the model repository."
            }
            require(fileNames.any { it.equals("tokenizer.json", ignoreCase = true) || it.equals("tokenizer.model", ignoreCase = true) }) {
                "Add tokenizer.json or tokenizer.model from the model repository."
            }
            if (safetensorsFiles.size > 1) {
                require(fileNames.any { it.endsWith(".safetensors.index.json", ignoreCase = true) }) {
                    "Add the safetensors index JSON for this sharded model."
                }
            }

            safetensorsFiles.forEach { modelFile ->
                check(modelFile.file.length() > 0L) {
                    "${modelFile.file.name} is empty or cannot be read."
                }
            }

            workspace.files
                .filter { it.file.name.endsWith(".json", ignoreCase = true) || it.file.name.endsWith(".model", ignoreCase = true) }
                .forEach { modelFile ->
                    check(modelFile.file.length() > 0L) {
                        "${modelFile.file.name} is empty or cannot be read."
                    }
                }

            delay(250)
            onProgress(ConversionProgress(0.22f, "Preparing GGUF output target"))

            val engineOutput = File(workspace.directory, "converted.gguf")

            val engineResult = ggufEngine.convert(
                request = NativeGgufRequest(
                    modelDirectory = workspace.directory,
                    outputFile = engineOutput,
                    quantization = request.quantization
                ),
                onProgress = onProgress
            )

            engineResult.getOrThrow()

            require(engineOutput.exists() && engineOutput.length() > 0L) {
                "Native engine did not produce a GGUF file."
            }

            onProgress(ConversionProgress(0.95f, "Writing GGUF output"))
            withContext(Dispatchers.IO) {
                context.contentResolver.openOutputStream(request.output, "wt")?.use { output ->
                    engineOutput.inputStream().use { input ->
                        input.copyTo(output)
                    }
                } ?: error("Unable to open the selected GGUF output file.")
            }

            onProgress(ConversionProgress(1f, "GGUF output written"))
        }.recoverCatching { error ->
            if (error is UnsatisfiedLinkError) {
                throw ConverterEngineUnavailableException(
                    "The arm64 native GGUF engine could not be loaded. Check that the app was built with the NDK for arm64-v8a."
                )
            }
            throw error
        }
    }
}
