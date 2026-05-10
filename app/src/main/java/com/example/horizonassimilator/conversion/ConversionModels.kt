package com.example.horizonassimilator.conversion

import android.net.Uri

data class SelectedModel(
    val displayName: String,
    val files: List<ModelFile>
) {
    constructor(uri: Uri, displayName: String) : this(
        displayName = displayName,
        files = listOf(ModelFile(uri = uri, displayName = displayName))
    )

    val primaryUri: Uri
        get() = files.first().uri

    val safetensorsFiles: List<ModelFile>
        get() = files.filter { it.displayName.endsWith(".safetensors", ignoreCase = true) }
}

data class ModelFile(
    val uri: Uri,
    val displayName: String,
    val sizeBytes: Long? = null
)

fun String.isConfigMetadataFileName(): Boolean {
    return endsWith(".json", ignoreCase = true) &&
        !endsWith(".safetensors.index.json", ignoreCase = true) &&
        !contains("tokenizer", ignoreCase = true) &&
        (equals("config.json", ignoreCase = true) ||
            contains("config", ignoreCase = true) ||
            contains("model", ignoreCase = true))
}

fun String.isTokenizerMetadataFileName(): Boolean {
    return (endsWith(".json", ignoreCase = true) || endsWith(".model", ignoreCase = true)) &&
        (equals("tokenizer.json", ignoreCase = true) ||
            equals("tokenizer.model", ignoreCase = true) ||
            contains("tokenizer", ignoreCase = true))
}

sealed interface ConversionState {
    data object Idle : ConversionState
    data class Ready(
        val input: SelectedModel,
        val output: Uri? = null
    ) : ConversionState

    data class Running(
        val input: SelectedModel,
        val output: Uri,
        val progress: Float?,
        val message: String,
        val logs: List<String>
    ) : ConversionState

    data class Complete(
        val input: SelectedModel,
        val output: Uri,
        val logs: List<String>
    ) : ConversionState

    data class Failed(
        val input: SelectedModel?,
        val output: Uri?,
        val message: String,
        val logs: List<String>
    ) : ConversionState
}

data class ConversionRequest(
    val input: SelectedModel,
    val output: Uri,
    val quantization: GgufQuantization = GgufQuantization.F16,
    val modelFamily: ModelFamily = ModelFamily.LLAMA
)

data class ConversionProgress(
    val progress: Float?,
    val message: String
)

enum class GgufQuantization(
    val label: String,
    val description: String,
    val estimatedOutputRatio: Double,
    val nativeWriterEnabled: Boolean = false,
    val supportedFamilies: Set<ModelFamily> = setOf(ModelFamily.LLAMA, ModelFamily.QWEN)
) {
    F16(
        label = "F16",
        description = "Current native test mode. Highest fidelity, largest GGUF output.",
        estimatedOutputRatio = 1.0,
        nativeWriterEnabled = true,
        supportedFamilies = setOf(ModelFamily.LLAMA, ModelFamily.QWEN, ModelFamily.GEMMA)
    ),
    Q8_0(
        label = "Q8_0",
        description = "Validation mode. Smaller than F16, with broad tensor support for this native writer.",
        estimatedOutputRatio = 0.55,
        nativeWriterEnabled = true,
        supportedFamilies = setOf(ModelFamily.LLAMA, ModelFamily.QWEN, ModelFamily.GEMMA)
    ),
    Q6_K(
        label = "Q6_K",
        description = "Validation mode. Smaller than Q8_0, with K-quant packing enabled.",
        estimatedOutputRatio = 0.42,
        nativeWriterEnabled = true
    ),
    Q5_K_M(
        label = "Q5_K_M",
        description = "Validation mode. Medium 5-bit K-quant output for smaller mobile GGUF files.",
        estimatedOutputRatio = 0.35,
        nativeWriterEnabled = true
    ),
    Q5_0(
        label = "Q5_0",
        description = "Compatibility 5-bit quantization with 32-wide blocks for Qwen-sized rows.",
        estimatedOutputRatio = 0.39,
        nativeWriterEnabled = true,
        supportedFamilies = setOf(ModelFamily.QWEN)
    ),
    Q4_0(
        label = "Q4_0",
        description = "Compatibility 4-bit quantization with 32-wide blocks for Qwen-sized rows.",
        estimatedOutputRatio = 0.32,
        nativeWriterEnabled = true,
        supportedFamilies = setOf(ModelFamily.QWEN)
    ),
    Q4_K_M(
        label = "Q4_K_M",
        description = "Validation mode. Compact 4-bit K-quant output for mobile GGUF files.",
        estimatedOutputRatio = 0.29,
        nativeWriterEnabled = true
    ),
    Q4_K_S(
        label = "Q4_K_S",
        description = "Validation mode. Compact 4-bit K-quant output with the smallest enabled profile.",
        estimatedOutputRatio = 0.27,
        nativeWriterEnabled = true
    ),
    Q3_K_M(
        label = "Q3_K_M",
        description = "Validation mode. Smallest enabled K-quant output for aggressive mobile compression.",
        estimatedOutputRatio = 0.22,
        nativeWriterEnabled = true
    )
}

fun GgufQuantization.isSupportedFor(family: ModelFamily): Boolean {
    return nativeWriterEnabled && family in supportedFamilies
}

enum class ModelFamily(
    val label: String,
    val description: String,
    val nativeWriterEnabled: Boolean
) {
    LLAMA(
        label = "LLaMA / Mistral",
        description = "Current supported family for LLaMA-style safetensors.",
        nativeWriterEnabled = true
    ),
    QWEN(
        label = "Qwen",
        description = "Qwen2/Qwen2.5-style safetensors with BPE tokenizer metadata.",
        nativeWriterEnabled = true
    ),
    GEMMA(
        label = "Gemma",
        description = "Gemma-family safetensors. Tensor mapping is experimental.",
        nativeWriterEnabled = true
    ),
    PHI(
        label = "Phi",
        description = "Planned family. Tensor mapping is not enabled yet.",
        nativeWriterEnabled = false
    )
}
