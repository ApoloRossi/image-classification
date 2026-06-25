package com.apolodevsystem.imageclassification

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateListOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.draw.blur
import com.apolodevsystem.imageclassification.ui.theme.ImageClassificationTheme
import com.apolodevsystem.imageclassification.ml.ClassificationResult
import com.apolodevsystem.imageclassification.ml.NsfwTfliteClassifier
import com.google.android.gms.tflite.java.TfLite
import com.google.android.gms.tasks.Tasks
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import android.graphics.Bitmap
import android.graphics.BitmapFactory

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        setContent {
            ImageClassificationTheme {
                Scaffold(modifier = Modifier.fillMaxSize()) { innerPadding ->
                    AssetsNsfwScreen(modifier = Modifier.padding(innerPadding))
                }
            }
        }
    }
}

@Composable
fun AssetsNsfwScreen(modifier: Modifier = Modifier) {
    val context = androidx.compose.ui.platform.LocalContext.current
    val rows = remember { mutableStateListOf<ImageRow>() }
    var loading by remember { mutableStateOf(true) }
    var error by remember { mutableStateOf<String?>(null) }

    LaunchedEffect(Unit) {
        loading = true
        error = null
        rows.clear()

        try {
            withContext(Dispatchers.IO) {
                // Inicializa o runtime do TFLite via Google Play Services antes de qualquer inferência
                Tasks.await(TfLite.initialize(context))
            }

            withContext(Dispatchers.Default) {
                NsfwTfliteClassifier(context).use { classifier ->
                    val images = context.assets
                        .list("") // app/src/main/assets
                        .orEmpty()
                        .filter { it.isImageAsset() }
                        .sorted()

                    for (name in images) {
                        val bmp = loadBitmapFromAssets(context, name) ?: continue
                        val result = classifier.classify(bmp)
                        withContext(Dispatchers.Main) {
                            rows.add(ImageRow(name, bmp, result))
                        }
                    }
                }
            }
        } catch (t: Throwable) {
            error = t.message ?: t.toString()
        } finally {
            loading = false
        }
    }

    Column(modifier = modifier.fillMaxSize().padding(16.dp)) {
        Text(
            text = "Assets NSFW classifier",
            fontWeight = FontWeight.SemiBold,
        )
        Spacer(Modifier.height(6.dp))
        Text(
            text = if (loading) "Processando imagens do assets…" else "Imagens processadas: ${rows.size}",
            color = Color(0xFF666666),
        )
        error?.let {
            Spacer(Modifier.height(8.dp))
            Text(text = it, color = Color(0xFFB00020))
        }
        Spacer(Modifier.height(12.dp))

        LazyColumn(verticalArrangement = Arrangement.spacedBy(12.dp)) {
            items(rows, key = { it.assetName }) { row ->
                ImageResultRow(row = row)
            }
        }
    }
}

@Composable
private fun ImageResultRow(row: ImageRow) {
    val threshold = 0.70f
    val isNsfw = row.result.nsfwProbability >= threshold

    Row(
        modifier = Modifier
            .fillMaxWidth()
            .background(Color(0xFFF5F5F5))
            .padding(12.dp),
        verticalAlignment = Alignment.CenterVertically,
        horizontalArrangement = Arrangement.spacedBy(12.dp),
    ) {
        Box(modifier = Modifier.size(96.dp)) {
            Image(
                bitmap = row.bitmap.asImageBitmap(),
                contentDescription = row.assetName,
                modifier = Modifier
                    .fillMaxSize()
                    .then(if (isNsfw) Modifier.blur(14.dp) else Modifier),
                contentScale = ContentScale.Crop,
            )
            if (isNsfw) {
                Box(
                    modifier = Modifier
                        .fillMaxSize()
                        .background(Color.Black.copy(alpha = 0.18f))
                )
            }
        }

        Column(modifier = Modifier.weight(1f)) {
            Text(text = row.assetName, fontWeight = FontWeight.Medium)
            Spacer(Modifier.height(4.dp))
            Text(text = "Top: ${row.result.topLabel}")
            Text(text = "NSFW (porn+sexy): ${(row.result.nsfwProbability * 100f).format1()}%")
            Spacer(Modifier.height(4.dp))
            Text(
                text = row.result.probabilities.entries
                    .sortedByDescending { it.value }
                    .joinToString { (k, v) -> "$k ${(v * 100f).format1()}%" },
                color = Color(0xFF555555),
            )
        }
    }
}

private data class ImageRow(
    val assetName: String,
    val bitmap: Bitmap,
    val result: ClassificationResult,
)

private fun String.isImageAsset(): Boolean {
    val n = lowercase()
    return n.endsWith(".png") || n.endsWith(".jpg") || n.endsWith(".jpeg") || n.endsWith(".webp")
}

private fun loadBitmapFromAssets(context: android.content.Context, assetName: String): Bitmap? {
    return try {
        context.assets.open(assetName).use { input ->
            BitmapFactory.decodeStream(input)
        }
    } catch (_: Throwable) {
        null
    }
}

private fun Float.format1(): String = "%,.1f".format(this)