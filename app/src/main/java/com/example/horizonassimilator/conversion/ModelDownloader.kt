package com.example.horizonassimilator.conversion

import android.content.Context
import android.net.Uri
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.io.File
import java.net.HttpURLConnection
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
            val downloadUrl = sourceUrl.trim().toDirectDownloadUrl()
            require(downloadUrl.startsWith("https://", ignoreCase = true)) {
                "Enter a valid HTTPS model URL."
            }

            val fileName = downloadUrl.fileNameFromUrl()
            require(fileName.endsWith(".safetensors", ignoreCase = true)) {
                "The URL must point to a .safetensors file."
            }

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
                    uri = Uri.fromFile(targetFile),
                    displayName = targetFile.name
                )
            } finally {
                connection.disconnect()
            }
        }
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
