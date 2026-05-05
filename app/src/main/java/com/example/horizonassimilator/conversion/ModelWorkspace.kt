package com.example.horizonassimilator.conversion

import android.content.Context
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.io.File

data class ModelWorkspace(
    val directory: File,
    val files: List<ModelWorkspaceFile>
) {
    val safetensorsFiles: List<ModelWorkspaceFile>
        get() = files.filter { it.file.name.endsWith(".safetensors", ignoreCase = true) }
}

data class ModelWorkspaceFile(
    val source: ModelFile,
    val file: File
)

class ModelWorkspacePreparer(
    private val context: Context
) {
    suspend fun prepare(
        model: SelectedModel,
        onProgress: suspend (ConversionProgress) -> Unit
    ): ModelWorkspace = withContext(Dispatchers.IO) {
        validateSourceSize(model)

        val workspace = File(
            context.cacheDir,
            "model-workspaces/${System.currentTimeMillis()}-${model.displayName.toSafeDirectoryName()}"
        )
        val workspaceRoot = workspace.parentFile ?: context.cacheDir
        check(workspaceRoot.exists() || workspaceRoot.mkdirs()) {
            "Unable to create model workspace root."
        }
        validateFreeStorage(model, workspace)
        check(workspace.mkdirs()) {
            "Unable to create model workspace."
        }

        val copiedFiles = mutableListOf<ModelWorkspaceFile>()
        val usedNames = mutableSetOf<String>()

        model.files.forEachIndexed { index, modelFile ->
            val targetName = modelFile.displayName.uniqueFileName(usedNames)
            val targetFile = File(workspace, targetName)
            val progress = index.toFloat() / model.files.size.toFloat()

            onProgress(
                ConversionProgress(
                    progress = progress,
                    message = "Staging ${modelFile.displayName}"
                )
            )

            context.contentResolver.openInputStream(modelFile.uri)?.use { input ->
                targetFile.outputStream().use { output ->
                    input.copyTo(output)
                }
            } ?: error("Unable to open ${modelFile.displayName}.")

            check(targetFile.length() > 0L) {
                "${modelFile.displayName} is empty or cannot be copied."
            }

            copiedFiles += ModelWorkspaceFile(
                source = modelFile,
                file = targetFile
            )
        }

        onProgress(
            ConversionProgress(
                progress = 0.15f,
                message = "Model files staged"
            )
        )

        ModelWorkspace(
            directory = workspace,
            files = copiedFiles
        )
    }

    private fun validateSourceSize(model: SelectedModel) {
        val knownSafetensorsBytes = model.safetensorsFiles.mapNotNull { it.sizeBytes }.sum()
        val hasUnknownSize = model.safetensorsFiles.any { it.sizeBytes == null }

        if (!hasUnknownSize && knownSafetensorsBytes > ConversionLimits.MAX_SOURCE_BYTES) {
            error(
                "Selected safetensors total is ${knownSafetensorsBytes.toSizeLabel()}, above the 20 GB mobile conversion limit."
            )
        }
    }

    private fun validateFreeStorage(model: SelectedModel, workspace: File) {
        val knownSelectedBytes = model.files.mapNotNull { it.sizeBytes }.sum()
        if (knownSelectedBytes <= 0L) {
            return
        }

        val requiredBytes = knownSelectedBytes * ConversionLimits.WORKSPACE_STORAGE_MULTIPLIER
        val storageRoot = workspace.parentFile?.takeIf { it.exists() } ?: context.cacheDir
        val availableBytes = storageRoot.usableSpace

        if (availableBytes < requiredBytes) {
            error(
                "Not enough free storage. Need about ${requiredBytes.toSizeLabel()} available, but only ${availableBytes.toSizeLabel()} is free."
            )
        }
    }

    private fun String.toSafeDirectoryName(): String {
        return replace(Regex("[^A-Za-z0-9._-]"), "_").take(48).ifBlank { "model" }
    }

    private fun String.uniqueFileName(usedNames: MutableSet<String>): String {
        val safeName = replace(Regex("[\\\\/:*?\"<>|]"), "_").ifBlank { "model-file" }
        if (usedNames.add(safeName)) {
            return safeName
        }

        val baseName = safeName.substringBeforeLast('.', safeName)
        val extension = safeName.substringAfterLast('.', "")
        var suffix = 2

        while (true) {
            val candidate = if (extension.isBlank()) {
                "$baseName-$suffix"
            } else {
                "$baseName-$suffix.$extension"
            }

            if (usedNames.add(candidate)) {
                return candidate
            }

            suffix += 1
        }
    }
}
