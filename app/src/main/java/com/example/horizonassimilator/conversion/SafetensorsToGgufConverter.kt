package com.example.horizonassimilator.conversion

interface SafetensorsToGgufConverter {
    suspend fun convert(
        request: ConversionRequest,
        onProgress: suspend (ConversionProgress) -> Unit
    ): Result<Unit>
}

class ConverterEngineUnavailableException(
    message: String
) : IllegalStateException(message)
