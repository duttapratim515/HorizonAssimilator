package com.example.horizonassimilator.ui.home

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Card
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp

@Composable
fun QuantizationEducationScreen(
    onBackClick: () -> Unit = {}
) {
    Column(
        modifier = Modifier
            .fillMaxSize()
            .verticalScroll(rememberScrollState())
            .padding(16.dp),
        verticalArrangement = Arrangement.spacedBy(12.dp)
    ) {
        OutlinedButton(onClick = onBackClick) {
            Text("Back")
        }

        Text(
            text = "GGUF and Quantization",
            style = MaterialTheme.typography.headlineSmall
        )

        EducationCard(
            title = "safetensors",
            body = "safetensors files usually hold the original model weights from a Hugging Face repository. A complete local model also needs metadata such as config.json and tokenizer files."
        )
        EducationCard(
            title = "GGUF",
            body = "GGUF is a compact model format used by llama.cpp-style runtimes. HorizonAssimilator stages the selected model files locally, then asks the native GGUF engine to write a .gguf output."
        )
        EducationCard(
            title = "Conversion",
            body = "Conversion reads the source weights and model metadata, maps them into GGUF tensors, and writes a new local file. It needs temporary workspace storage because both the staged source and output can exist at the same time."
        )
        EducationCard(
            title = "Quantization",
            body = "Quantization stores weights with fewer bits. F16 keeps the most detail and uses the most storage. Q8_0, Q6_K, Q5_K_M, and Q4_K_M progressively reduce size. Q4_K_M is the recommended mobile default because it usually keeps useful quality while making the model much easier to store and run."
        )
    }
}

@Composable
private fun EducationCard(
    title: String,
    body: String
) {
    Card(
        modifier = Modifier.fillMaxWidth()
    ) {
        Column(
            modifier = Modifier.padding(12.dp),
            verticalArrangement = Arrangement.spacedBy(6.dp)
        ) {
            Text(
                text = title,
                style = MaterialTheme.typography.titleMedium
            )
            Text(body)
        }
    }
}
