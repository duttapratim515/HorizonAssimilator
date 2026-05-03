package com.example.horizonassimilator.conversion

import android.content.Context
import android.net.Uri
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.io.File
import java.net.HttpURLConnection
import java.net.URI
import java.net.URL

data class DownloadProgress(
    val progress: Float?,
    val message: String
)

class ModelDownloader(
    private val context: Context
) {
    suspend fun downloadSafetensors(
        sourceUrl: String,
        onProgress: suspend (DownloadProgress) -> Unit
    ): Result<SelectedModel> = withContext(Dispatchers.IO) {
        runCatching {
            val validation = validateSafetensorsUrl(sourceUrl)
            require(validation is UrlValidation.Valid) { validation.message }

            val downloadUrl = validation.directDownloadUrl
            val fileName = downloadUrl.fileNameFromUrl()

            onProgress(DownloadProgress(0f, "Connecting to model host"))

            val connection = (URL(downloadUrl).openConnection() as HttpURLConnection).apply {
                connectTimeout = 15_000
                readTimeout = 30_000
                instanceFollowRedirects = true
            }

            try {
                connection.connect()
                val responseCode = connection.responseCode
                require(responseCode in 200..299) {
                    "Download failed with HTTP $responseCode."
                }

                val targetFile = File(context.cacheDir, fileName)
                val totalBytes = connection.contentLengthLong.takeIf { it > 0 }
                var copiedBytes = 0L
                val buffer = ByteArray(DEFAULT_BUFFER_SIZE)

                connection.inputStream.use { input ->
                    targetFile.outputStream().use { output ->
                        while (true) {
                            val read = input.read(buffer)
                            if (read == -1) {
                                break
                            }

                            output.write(buffer, 0, read)
                            copiedBytes += read

                            val progress = totalBytes?.let { copiedBytes.toFloat() / it.toFloat() }
                            val message = if (progress != null) {
                                "Downloading ${(progress * 100).toInt()}%"
                            } else {
                                "Downloading ${copiedBytes.toMegabytesLabel()}"
                            }
                            onProgress(DownloadProgress(progress, message))
                        }
                    }
                }

                require(targetFile.length() > 0L) {
                    "The downloaded safetensors file is empty."
                }

                onProgress(DownloadProgress(1f, "Download complete"))

                SelectedModel(
                    displayName = targetFile.name,
                    files = listOf(
                        ModelFile(
                            uri = Uri.fromFile(targetFile),
                            displayName = targetFile.name,
                            sizeBytes = targetFile.length()
                        )
                    )
                )
            } finally {
                connection.disconnect()
            }
        }
    }

    fun validateSafetensorsUrl(sourceUrl: String): UrlValidation {
        val trimmed = sourceUrl.trim()
        if (trimmed.isBlank()) {
            return UrlValidation.Invalid("Enter a Hugging Face safetensors URL.")
        }

        val uri = runCatching { URI(trimmed) }.getOrNull()
            ?: return UrlValidation.Invalid("Enter a valid URL.")

        if (!uri.scheme.equals("https", ignoreCase = true)) {
            return UrlValidation.Invalid("Use an HTTPS Hugging Face URL.")
        }

        if (!uri.host.equals("huggingface.co", ignoreCase = true)) {
            return UrlValidation.Invalid("Use a huggingface.co model file URL.")
        }

        val path = uri.path.orEmpty()
        if (!path.contains("/blob/") && !path.contains("/resolve/")) {
            return UrlValidation.Invalid("Use a Hugging Face file URL that contains /blob/ or /resolve/.")
        }

        if (!path.endsWith(".safetensors", ignoreCase = true)) {
            return UrlValidation.Invalid("The URL must point to a .safetensors file.")
        }

        return UrlValidation.Valid(trimmed.toDirectDownloadUrl())
    }

    private fun String.toDirectDownloadUrl(): String {
        return replace("/blob/", "/resolve/")
    }

    private fun String.fileNameFromUrl(): String {
        val path = substringBefore("?").trimEnd('/')
        return path.substringAfterLast('/').ifBlank { "model.safetensors" }
    }

    private fun Long.toMegabytesLabel(): String {
        return String.format("%.1f MB", this / 1_048_576f)
    }
}

sealed interface UrlValidation {
    val message: String

    data class Valid(
        val directDownloadUrl: String
    ) : UrlValidation {
        override val message: String = "URL looks good."
    }

    data class Invalid(
        override val message: String
    ) : UrlValidation
}
