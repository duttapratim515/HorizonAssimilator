package com.example.horizonassimilator.conversion

import android.net.Uri

data class SelectedModel(
    val uri: Uri,
    val displayName: String
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
