package com.example.horizonassimilator.conversion

import android.app.ActivityManager
import android.content.Context
import android.os.Build
import android.os.StatFs
import kotlin.math.ceil

data class DeviceCapabilityPreflight(
    val status: DeviceCapabilityStatus,
    val arm64Supported: Boolean,
    val totalRamBytes: Long?,
    val freeStorageBytes: Long,
    val modelSizeBytes: Long?,
    val estimatedOutputBytes: Long?,
    val estimatedWorkspaceBytes: Long?,
    val checks: List<PreflightCheck>
)

data class PreflightCheck(
    val label: String,
    val value: String,
    val passed: Boolean,
    val detail: String
)

enum class DeviceCapabilityStatus(
    val label: String,
    val description: String
) {
    Good(
        label = "Good",
        description = "This model looks reasonable for on-device conversion."
    ),
    Heavy(
        label = "Heavy",
        description = "Conversion may be slow, hot, or memory hungry."
    ),
    TooLarge(
        label = "Too large",
        description = "The selected bundle is beyond the current mobile safety limits."
    ),
    Unsupported(
        label = "Unsupported",
        description = "This device cannot run the current native converter."
    )
}

fun Context.buildDeviceCapabilityPreflight(
    model: SelectedModel,
    quantization: GgufQuantization
): DeviceCapabilityPreflight {
    val arm64Supported = Build.SUPPORTED_64_BIT_ABIS.contains("arm64-v8a")
    val totalRamBytes = totalRamBytes()
    val freeStorageBytes = freeWorkspaceBytes()
    val hasUnknownModelSize = model.safetensorsFiles.any { it.sizeBytes == null }
    val modelSizeBytes = if (hasUnknownModelSize) {
        null
    } else {
        model.safetensorsFiles.sumOf { it.sizeBytes ?: 0L }
    }
    val estimatedOutputBytes = modelSizeBytes?.estimateOutputBytes(quantization)
    val estimatedWorkspaceBytes = modelSizeBytes?.let { sourceBytes ->
        sourceBytes * ConversionLimits.WORKSPACE_STORAGE_MULTIPLIER + (estimatedOutputBytes ?: 0L)
    }

    val fileNames = model.files.map { it.displayName }
    val safetensorsCount = model.safetensorsFiles.size
    val hasConfig = fileNames.any { it.equals("config.json", ignoreCase = true) }
    val hasTokenizer = fileNames.any { it.equals("tokenizer.json", ignoreCase = true) || it.equals("tokenizer.model", ignoreCase = true) }
    val hasIndex = fileNames.any { it.endsWith(".safetensors.index.json", ignoreCase = true) }

    val checks = listOf(
        PreflightCheck(
            label = "CPU",
            value = if (arm64Supported) "arm64-v8a" else "Not arm64",
            passed = arm64Supported,
            detail = if (arm64Supported) "Native GGUF engine can load." else "The converter is built for arm64-v8a only."
        ),
        PreflightCheck(
            label = "RAM",
            value = totalRamBytes?.toSizeLabel() ?: "Unknown",
            passed = totalRamBytes == null || totalRamBytes >= MIN_RAM_BYTES,
            detail = if (totalRamBytes != null && totalRamBytes < MIN_RAM_BYTES) {
                "Less than 4 GB RAM is likely too constrained."
            } else {
                "More RAM improves conversion stability."
            }
        ),
        PreflightCheck(
            label = "Free storage",
            value = freeStorageBytes.toSizeLabel(),
            passed = estimatedWorkspaceBytes == null || freeStorageBytes >= estimatedWorkspaceBytes,
            detail = estimatedWorkspaceBytes?.let { "Estimated need: ${it.toSizeLabel()}." } ?: "Need is estimated after file sizes are known."
        ),
        PreflightCheck(
            label = "Model size",
            value = modelSizeBytes?.toSizeLabel() ?: "Unknown",
            passed = modelSizeBytes == null || modelSizeBytes <= ConversionLimits.MAX_SOURCE_BYTES,
            detail = "Current safety limit: ${ConversionLimits.MAX_SOURCE_BYTES.toSizeLabel()}."
        ),
        PreflightCheck(
            label = "Bundle files",
            value = "$safetensorsCount safetensors",
            passed = safetensorsCount > 0 && hasConfig && hasTokenizer && (safetensorsCount <= 1 || hasIndex),
            detail = bundleDetail(safetensorsCount, hasConfig, hasTokenizer, hasIndex)
        )
    )

    val status = when {
        !arm64Supported -> DeviceCapabilityStatus.Unsupported
        modelSizeBytes != null && modelSizeBytes > ConversionLimits.MAX_SOURCE_BYTES -> DeviceCapabilityStatus.TooLarge
        estimatedWorkspaceBytes != null && freeStorageBytes < estimatedWorkspaceBytes -> DeviceCapabilityStatus.TooLarge
        checks.any { !it.passed } -> DeviceCapabilityStatus.Heavy
        modelSizeBytes == null -> DeviceCapabilityStatus.Heavy
        modelSizeBytes >= ConversionLimits.LARGE_SOURCE_WARNING_BYTES -> DeviceCapabilityStatus.Heavy
        totalRamBytes != null && totalRamBytes < STRONG_RAM_BYTES -> DeviceCapabilityStatus.Heavy
        estimatedWorkspaceBytes != null && estimatedWorkspaceBytes > freeStorageBytes * 7 / 10 -> DeviceCapabilityStatus.Heavy
        else -> DeviceCapabilityStatus.Good
    }

    return DeviceCapabilityPreflight(
        status = status,
        arm64Supported = arm64Supported,
        totalRamBytes = totalRamBytes,
        freeStorageBytes = freeStorageBytes,
        modelSizeBytes = modelSizeBytes,
        estimatedOutputBytes = estimatedOutputBytes,
        estimatedWorkspaceBytes = estimatedWorkspaceBytes,
        checks = checks
    )
}

private fun Context.totalRamBytes(): Long? {
    val activityManager = getSystemService(ActivityManager::class.java) ?: return null
    val memoryInfo = ActivityManager.MemoryInfo()
    activityManager.getMemoryInfo(memoryInfo)
    return memoryInfo.totalMem
}

private fun Context.freeWorkspaceBytes(): Long {
    val statFs = StatFs(cacheDir.absolutePath)
    return statFs.availableBytes
}

private fun Long.estimateOutputBytes(quantization: GgufQuantization): Long {
    return ceil(this * quantization.estimatedOutputRatio).toLong()
}

private fun bundleDetail(
    safetensorsCount: Int,
    hasConfig: Boolean,
    hasTokenizer: Boolean,
    hasIndex: Boolean
): String {
    val missing = buildList {
        if (safetensorsCount == 0) add(".safetensors")
        if (!hasConfig) add("config.json")
        if (!hasTokenizer) add("tokenizer")
        if (safetensorsCount > 1 && !hasIndex) add("safetensors index")
    }
    return if (missing.isEmpty()) {
        "Required model metadata is present."
    } else {
        "Missing ${missing.joinToString()}."
    }
}

private const val MIN_RAM_BYTES = 4L * 1024L * 1024L * 1024L
private const val STRONG_RAM_BYTES = 8L * 1024L * 1024L * 1024L
