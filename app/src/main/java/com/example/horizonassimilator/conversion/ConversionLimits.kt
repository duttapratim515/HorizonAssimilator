package com.example.horizonassimilator.conversion

object ConversionLimits {
    const val RECOMMENDED_SOURCE_BYTES: Long = 4L * 1024L * 1024L * 1024L
    const val LARGE_SOURCE_WARNING_BYTES: Long = 10L * 1024L * 1024L * 1024L
    const val MAX_SOURCE_BYTES: Long = 20L * 1024L * 1024L * 1024L
    const val WORKSPACE_STORAGE_MULTIPLIER: Long = 2L
}

fun Long.toSizeLabel(): String {
    val gib = this / 1_073_741_824.0
    val mib = this / 1_048_576.0
    return if (this >= 1_073_741_824L) {
        String.format("%.2f GB", gib)
    } else {
        String.format("%.1f MB", mib)
    }
}
