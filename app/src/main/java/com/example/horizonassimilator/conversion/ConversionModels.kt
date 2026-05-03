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
    val output: Uri
)

data class ConversionProgress(
    val progress: Float?,
    val message: String
)
