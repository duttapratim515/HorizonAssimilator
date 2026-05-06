package com.example.horizonassimilator.ui.conversion

import android.content.Context
import android.net.Uri
import android.provider.OpenableColumns
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
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
import com.example.horizonassimilator.conversion.ConversionLimits
import com.example.horizonassimilator.conversion.DeviceCapabilityStatus
import com.example.horizonassimilator.conversion.GgufQuantization
import com.example.horizonassimilator.conversion.ModelDownloader
import com.example.horizonassimilator.conversion.ModelFile
import com.example.horizonassimilator.conversion.SafetensorsToGgufConverter
import com.example.horizonassimilator.conversion.SelectedModel
import com.example.horizonassimilator.conversion.UrlValidation
import com.example.horizonassimilator.conversion.buildDeviceCapabilityPreflight
import com.example.horizonassimilator.conversion.toSizeLabel
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
    var selectedQuantization by remember { mutableStateOf(GgufQuantization.F16) }
    val urlValidation = remember(downloadUrl, downloader) {
        downloader.validateSafetensorsUrl(downloadUrl)
    }
    val readyModel = (state as? ConversionState.Ready)?.input
    val readyPreflight = remember(context, readyModel, selectedQuantization) {
        readyModel?.let { context.buildDeviceCapabilityPreflight(it, selectedQuantization) }
    }
    val canCreateGguf = readyPreflight?.let { preflight ->
        preflight.status != DeviceCapabilityStatus.Unsupported &&
            preflight.status != DeviceCapabilityStatus.TooLarge &&
            preflight.checks.all { it.passed }
    } == true

    val modelPicker = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.OpenMultipleDocuments()
    ) { uris ->
        if (uris.isNotEmpty()) {
            val files = uris.map { uri ->
                ModelFile(
                    uri = uri,
                    displayName = context.displayNameFor(uri),
                    sizeBytes = context.sizeBytesFor(uri)
                )
            }
            val model = SelectedModel(
                displayName = files.bundleDisplayName(),
                files = files
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
                        output = uri,
                        quantization = selectedQuantization
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
                    Text("Select model files")
                }

                ModelBundleSummary(state = state)

                QuantizationSelector(
                    selected = selectedQuantization,
                    onSelected = { selectedQuantization = it }
                )

                PreflightPanel(
                    state = state,
                    quantization = selectedQuantization
                )

                UrlDownloadPanel(
                    url = downloadUrl,
                    validation = urlValidation,
                    state = downloadState,
                    onUrlChange = {
                        downloadUrl = it
                        downloadState = ModelDownloadState.Idle
                    },
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
                            ?.replace(".json", ".gguf", ignoreCase = true)
                            ?.replace(" files", ".gguf", ignoreCase = true)
                            ?: "model.gguf"
                        outputPicker.launch(fileName)
                    },
                    enabled = canCreateGguf,
                    modifier = Modifier.fillMaxWidth()
                ) {
                    Text("Create GGUF")
                }
                if (state is ConversionState.Ready && !canCreateGguf) {
                    Text(
                        text = "Resolve the preflight checks before creating the GGUF.",
                        color = MaterialTheme.colorScheme.error
                    )
                }
            }
        }

        ConversionStatusCard(state = state)
    }
}

@Composable
private fun QuantizationSelector(
    selected: GgufQuantization,
    onSelected: (GgufQuantization) -> Unit
) {
    Column(
        verticalArrangement = Arrangement.spacedBy(8.dp)
    ) {
        Text(
            text = "Quantization",
            style = MaterialTheme.typography.titleMedium
        )
        Text(
            text = selected.description,
            style = MaterialTheme.typography.bodySmall
        )
        Text(
            text = "F16, Q8_0, Q6_K, and Q5_K_M are enabled for validation in this build.",
            style = MaterialTheme.typography.bodySmall,
            color = MaterialTheme.colorScheme.primary
        )
        GgufQuantization.entries.chunked(2).forEach { rowItems ->
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(8.dp)
            ) {
                rowItems.forEach { quantization ->
                    val enabled = quantization.nativeWriterEnabled && quantization != selected
                    OutlinedButton(
                        onClick = { onSelected(quantization) },
                        modifier = Modifier.weight(1f),
                        enabled = enabled
                    ) {
                        Text(quantization.label)
                    }
                }
                if (rowItems.size == 1) {
                    Spacer(modifier = Modifier.weight(1f))
                }
            }
        }
    }
}

@Composable
private fun PreflightPanel(
    state: ConversionState,
    quantization: GgufQuantization
) {
    val context = LocalContext.current
    val model = when (state) {
        is ConversionState.Ready -> state.input
        is ConversionState.Running -> state.input
        is ConversionState.Complete -> state.input
        is ConversionState.Failed -> state.input
        ConversionState.Idle -> null
    }

    Card(
        modifier = Modifier.fillMaxWidth()
    ) {
        Column(
            modifier = Modifier.padding(12.dp),
            verticalArrangement = Arrangement.spacedBy(8.dp)
        ) {
            Text(
                text = "Preflight Check",
                style = MaterialTheme.typography.titleMedium
            )

            if (model == null) {
                Text("Select model files to estimate device fit.")
                return@Column
            }

            val preflight = remember(context, model, quantization) {
                context.buildDeviceCapabilityPreflight(model, quantization)
            }
            val statusColor = when (preflight.status) {
                DeviceCapabilityStatus.Good -> MaterialTheme.colorScheme.primary
                DeviceCapabilityStatus.Heavy -> MaterialTheme.colorScheme.tertiary
                DeviceCapabilityStatus.TooLarge,
                DeviceCapabilityStatus.Unsupported -> MaterialTheme.colorScheme.error
            }

            Text(
                text = "${preflight.status.label}: ${preflight.status.description}",
                color = statusColor
            )
            Text("Quantization: ${quantization.label}")
            Text("Estimated GGUF: ${preflight.estimatedOutputBytes?.toSizeLabel() ?: "Unknown"}")
            Text("Estimated workspace: ${preflight.estimatedWorkspaceBytes?.toSizeLabel() ?: "Unknown"}")

            preflight.checks.forEach { check ->
                val prefix = if (check.passed) "OK" else "Check"
                Text("$prefix ${check.label}: ${check.value}. ${check.detail}")
            }
        }
    }
}

@Composable
private fun ModelBundleSummary(
    state: ConversionState
) {
    val model = when (state) {
        is ConversionState.Ready -> state.input
        is ConversionState.Running -> state.input
        is ConversionState.Complete -> state.input
        is ConversionState.Failed -> state.input
        ConversionState.Idle -> null
    } ?: return

    val fileNames = model.files.map { it.displayName }
    val safetensorsCount = model.safetensorsFiles.size
    val safetensorsBytes = model.safetensorsFiles.mapNotNull { it.sizeBytes }.sum()
    val hasUnknownSafetensorsSize = model.safetensorsFiles.any { it.sizeBytes == null }
    val hasConfig = fileNames.any { it.equals("config.json", ignoreCase = true) }
    val hasTokenizer = fileNames.any { it.equals("tokenizer.json", ignoreCase = true) || it.equals("tokenizer.model", ignoreCase = true) }
    val hasIndex = fileNames.any { it.endsWith(".safetensors.index.json", ignoreCase = true) }

    Column(
        verticalArrangement = Arrangement.spacedBy(4.dp)
    ) {
        Text("${model.files.size} files selected")
        Text("$safetensorsCount safetensors file${if (safetensorsCount == 1) "" else "s"}")
        if (hasUnknownSafetensorsSize) {
            Text("safetensors size: unknown")
        } else {
            Text("safetensors size: ${safetensorsBytes.toSizeLabel()}")
        }
        SafetensorsSizeWarning(totalBytes = safetensorsBytes, hasUnknownSize = hasUnknownSafetensorsSize)
        Text("config.json: ${if (hasConfig) "present" else "missing"}")
        Text("tokenizer: ${if (hasTokenizer) "present" else "missing"}")
        if (safetensorsCount > 1) {
            Text("safetensors index: ${if (hasIndex) "present" else "missing"}")
        }
    }
}

@Composable
private fun SafetensorsSizeWarning(
    totalBytes: Long,
    hasUnknownSize: Boolean
) {
    val message = when {
        hasUnknownSize -> "Size could not be read. The converter will verify it before running."
        totalBytes > ConversionLimits.MAX_SOURCE_BYTES -> "Over 20 GB. This app will block conversion by default."
        totalBytes >= ConversionLimits.LARGE_SOURCE_WARNING_BYTES -> "Very large for phone conversion. Expect long runtime, heat, and high storage use."
        totalBytes >= ConversionLimits.RECOMMENDED_SOURCE_BYTES -> "Large for mobile. A smaller GGUF will be more practical for HorizonText."
        else -> null
    } ?: return

    val color = if (totalBytes > ConversionLimits.MAX_SOURCE_BYTES) {
        MaterialTheme.colorScheme.error
    } else {
        MaterialTheme.colorScheme.onSurfaceVariant
    }

    Text(message, color = color)
}

@Composable
private fun UrlDownloadPanel(
    url: String,
    validation: UrlValidation,
    state: ModelDownloadState,
    onUrlChange: (String) -> Unit,
    onDownloadClick: () -> Unit
) {
    val showValidation = url.isNotBlank() || validation is UrlValidation.Invalid

    Column(
        verticalArrangement = Arrangement.spacedBy(8.dp)
    ) {
        OutlinedTextField(
            value = url,
            onValueChange = onUrlChange,
            label = { Text("Safetensors blob URL") },
            supportingText = {
                if (showValidation) {
                    Text(validation.message)
                }
            },
            isError = url.isNotBlank() && validation is UrlValidation.Invalid,
            singleLine = true,
            modifier = Modifier.fillMaxWidth()
        )

        Button(
            onClick = onDownloadClick,
            enabled = validation is UrlValidation.Valid && state !is ModelDownloadState.Running,
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
        ConversionState.Idle -> "Choose model files to begin."
        is ConversionState.Ready -> state.input.displayName
        is ConversionState.Running -> state.input.displayName
        is ConversionState.Complete -> state.input.displayName
        is ConversionState.Failed -> state.input?.displayName ?: "Choose model files to begin."
    }
}

private fun List<ModelFile>.bundleDisplayName(): String {
    val safetensors = filter { it.displayName.endsWith(".safetensors", ignoreCase = true) }
    return when {
        safetensors.size == 1 -> safetensors.first().displayName
        safetensors.size > 1 -> "${safetensors.size} safetensors files"
        size == 1 -> first().displayName
        else -> "$size model files"
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

private fun Context.sizeBytesFor(uri: Uri): Long? {
    return contentResolver.query(uri, null, null, null, null)?.use { cursor ->
        val sizeIndex = cursor.getColumnIndex(OpenableColumns.SIZE)
        if (cursor.moveToFirst() && sizeIndex >= 0 && !cursor.isNull(sizeIndex)) {
            cursor.getLong(sizeIndex).takeIf { it >= 0L }
        } else {
            null
        }
    }
}
