package com.example.horizonassimilator.ui.conversion

import android.content.Context
import android.net.Uri
import android.provider.OpenableColumns
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.dp
import com.example.horizonassimilator.conversion.AndroidSafetensorsToGgufConverter
import com.example.horizonassimilator.conversion.DownloadProgress
import com.example.horizonassimilator.conversion.ConversionProgress
import com.example.horizonassimilator.conversion.ConversionRequest
import com.example.horizonassimilator.conversion.ConversionState
import com.example.horizonassimilator.conversion.ModelDownloader
import com.example.horizonassimilator.conversion.SafetensorsToGgufConverter
import com.example.horizonassimilator.conversion.SelectedModel
import kotlinx.coroutines.launch

@Composable
fun ConversionScreen(
    onBackClick: () -> Unit = {}
) {
    val context = LocalContext.current
    val converter = remember(context) {
        AndroidSafetensorsToGgufConverter(context.applicationContext)
    }
    val downloader = remember(context) {
        ModelDownloader(context.applicationContext)
    }
    val scope = rememberCoroutineScope()

    var state by remember { mutableStateOf<ConversionState>(ConversionState.Idle) }
    var downloadUrl by remember { mutableStateOf("") }
    var downloadState by remember { mutableStateOf<ModelDownloadState>(ModelDownloadState.Idle) }

    val modelPicker = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.OpenDocument()
    ) { uri ->
        if (uri != null) {
            val model = SelectedModel(
                uri = uri,
                displayName = context.displayNameFor(uri)
            )
            state = ConversionState.Ready(input = model)
        }
    }

    val outputPicker = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.CreateDocument("application/octet-stream")
    ) { uri ->
        val ready = state as? ConversionState.Ready
        if (uri != null && ready != null) {
            scope.launch {
                runConversion(
                    converter = converter,
                    request = ConversionRequest(
                        input = ready.input,
                        output = uri
                    ),
                    onStateChange = { state = it }
                )
            }
        }
    }

    Column(
        modifier = Modifier
            .fillMaxSize()
            .verticalScroll(rememberScrollState())
            .padding(12.dp),
        verticalArrangement = Arrangement.spacedBy(12.dp)
    ) {
        OutlinedButton(onClick = onBackClick) {
            Text("Back")
        }

        Text(
            text = "Convert Model",
            style = MaterialTheme.typography.headlineSmall
        )

        Card(
            modifier = Modifier.fillMaxWidth()
        ) {
            Column(
                modifier = Modifier.padding(12.dp),
                verticalArrangement = Arrangement.spacedBy(12.dp)
            ) {
                Text(
                    text = selectedModelLabel(state),
                    style = MaterialTheme.typography.bodyLarge
                )

                Button(
                    onClick = { modelPicker.launch(arrayOf("*/*")) },
                    modifier = Modifier.fillMaxWidth()
                ) {
                    Text("Select safetensors")
                }

                UrlDownloadPanel(
                    url = downloadUrl,
                    state = downloadState,
                    onUrlChange = { downloadUrl = it },
                    onDownloadClick = {
                        scope.launch {
                            runModelDownload(
                                downloader = downloader,
                                sourceUrl = downloadUrl,
                                onDownloadStateChange = { downloadState = it },
                                onModelSelected = { model ->
                                    state = ConversionState.Ready(input = model)
                                }
                            )
                        }
                    }
                )

                Button(
                    onClick = {
                        val ready = state as? ConversionState.Ready
                        val fileName = ready?.input?.displayName
                            ?.replace(".safetensors", ".gguf", ignoreCase = true)
                            ?: "model.gguf"
                        outputPicker.launch(fileName)
                    },
                    enabled = state is ConversionState.Ready,
                    modifier = Modifier.fillMaxWidth()
                ) {
                    Text("Create GGUF")
                }
            }
        }

        ConversionStatusCard(state = state)
    }
}

@Composable
private fun UrlDownloadPanel(
    url: String,
    state: ModelDownloadState,
    onUrlChange: (String) -> Unit,
    onDownloadClick: () -> Unit
) {
    Column(
        verticalArrangement = Arrangement.spacedBy(8.dp)
    ) {
        OutlinedTextField(
            value = url,
            onValueChange = onUrlChange,
            label = { Text("Safetensors blob URL") },
            singleLine = true,
            modifier = Modifier.fillMaxWidth()
        )

        Button(
            onClick = onDownloadClick,
            enabled = url.isNotBlank() && state !is ModelDownloadState.Running,
            modifier = Modifier.fillMaxWidth()
        ) {
            Text("Download safetensors")
        }

        when (state) {
            ModelDownloadState.Idle -> Unit
            is ModelDownloadState.Running -> {
                if (state.progress != null) {
                    LinearProgressIndicator(
                        progress = { state.progress.coerceIn(0f, 1f) },
                        modifier = Modifier.fillMaxWidth()
                    )
                } else {
                    LinearProgressIndicator(
                        modifier = Modifier.fillMaxWidth()
                    )
                }
                Text(state.message)
            }
            is ModelDownloadState.Complete -> {
                Text("Downloaded ${state.model.displayName}")
            }
            is ModelDownloadState.Failed -> {
                Text(state.message, color = MaterialTheme.colorScheme.error)
            }
        }
    }
}

@Composable
private fun ConversionStatusCard(
    state: ConversionState
) {
    Card(
        modifier = Modifier.fillMaxWidth()
    ) {
        Column(
            modifier = Modifier.padding(12.dp),
            verticalArrangement = Arrangement.spacedBy(8.dp)
        ) {
            Text(
                text = "Status",
                style = MaterialTheme.typography.titleMedium
            )

            when (state) {
                ConversionState.Idle -> Text("No model selected.")
                is ConversionState.Ready -> Text("Ready to create a GGUF output file.")
                is ConversionState.Running -> {
                    if (state.progress != null) {
                        LinearProgressIndicator(
                            progress = { state.progress.coerceIn(0f, 1f) },
                            modifier = Modifier.fillMaxWidth()
                        )
                    } else {
                        LinearProgressIndicator(
                            modifier = Modifier.fillMaxWidth()
                        )
                    }
                    Text(state.message)
                    LogList(state.logs)
                }
                is ConversionState.Complete -> {
                    Text("Conversion complete.")
                    LogList(state.logs)
                }
                is ConversionState.Failed -> {
                    Text(state.message, color = MaterialTheme.colorScheme.error)
                    LogList(state.logs)
                }
            }
        }
    }
}

@Composable
private fun LogList(logs: List<String>) {
    if (logs.isEmpty()) {
        return
    }

    Spacer(modifier = Modifier.height(4.dp))
    logs.takeLast(8).forEach { log ->
        Text(
            text = log,
            style = MaterialTheme.typography.bodySmall
        )
    }
}

private sealed interface ModelDownloadState {
    data object Idle : ModelDownloadState
    data class Running(
        val progress: Float?,
        val message: String
    ) : ModelDownloadState

    data class Complete(
        val model: SelectedModel
    ) : ModelDownloadState

    data class Failed(
        val message: String
    ) : ModelDownloadState
}

private suspend fun runModelDownload(
    downloader: ModelDownloader,
    sourceUrl: String,
    onDownloadStateChange: (ModelDownloadState) -> Unit,
    onModelSelected: (SelectedModel) -> Unit
) {
    fun push(progress: DownloadProgress) {
        onDownloadStateChange(
            ModelDownloadState.Running(
                progress = progress.progress,
                message = progress.message
            )
        )
    }

    push(DownloadProgress(0f, "Starting download"))

    val result = downloader.downloadSafetensors(sourceUrl) { progress ->
        push(progress)
    }

    result.fold(
        onSuccess = { model ->
            onDownloadStateChange(ModelDownloadState.Complete(model))
            onModelSelected(model)
        },
        onFailure = { error ->
            onDownloadStateChange(
                ModelDownloadState.Failed(error.message ?: "Download failed.")
            )
        }
    )
}

private suspend fun runConversion(
    converter: SafetensorsToGgufConverter,
    request: ConversionRequest,
    onStateChange: (ConversionState) -> Unit
) {
    val logs = mutableListOf<String>()

    fun push(progress: ConversionProgress) {
        logs += progress.message
        onStateChange(
            ConversionState.Running(
                input = request.input,
                output = request.output,
                progress = progress.progress,
                message = progress.message,
                logs = logs.toList()
            )
        )
    }

    push(ConversionProgress(0f, "Starting conversion"))

    val result = converter.convert(request) { progress ->
        push(progress)
    }

    onStateChange(
        result.fold(
            onSuccess = {
                ConversionState.Complete(
                    input = request.input,
                    output = request.output,
                    logs = logs + "GGUF file created"
                )
            },
            onFailure = { error ->
                ConversionState.Failed(
                    input = request.input,
                    output = request.output,
                    message = error.message ?: "Conversion failed.",
                    logs = logs.toList()
                )
            }
        )
    )
}

private fun selectedModelLabel(state: ConversionState): String {
    return when (state) {
        ConversionState.Idle -> "Choose a safetensors file to begin."
        is ConversionState.Ready -> state.input.displayName
        is ConversionState.Running -> state.input.displayName
        is ConversionState.Complete -> state.input.displayName
        is ConversionState.Failed -> state.input?.displayName ?: "Choose a safetensors file to begin."
    }
}

private fun Context.displayNameFor(uri: Uri): String {
    return contentResolver.query(uri, null, null, null, null)?.use { cursor ->
        val nameIndex = cursor.getColumnIndex(OpenableColumns.DISPLAY_NAME)
        if (cursor.moveToFirst() && nameIndex >= 0) {
            cursor.getString(nameIndex)
        } else {
            null
        }
    } ?: uri.lastPathSegment ?: "model.safetensors"
}
