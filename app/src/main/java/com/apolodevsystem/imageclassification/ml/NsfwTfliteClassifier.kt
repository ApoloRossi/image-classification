package com.apolodevsystem.imageclassification.ml

import android.content.Context
import android.graphics.Bitmap
import org.tensorflow.lite.InterpreterApi
import org.tensorflow.lite.InterpreterApi.Options.TfLiteRuntime
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.image.ops.ResizeOp
import org.tensorflow.lite.support.common.ops.NormalizeOp
import org.tensorflow.lite.DataType
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer
import java.io.BufferedReader
import java.io.FileInputStream
import java.io.InputStreamReader
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel

class NsfwTfliteClassifier(
    private val context: Context,
    modelAssetPath: String = "model.tflite",
    labelsAssetPath: String = "labels.txt",
) : AutoCloseable {

    private val interpreter: InterpreterApi
    private val labels: List<String>

    init {
        labels = loadLabels(labelsAssetPath)
        interpreter = InterpreterApi.create(
            loadModelFile(modelAssetPath),
            InterpreterApi.Options().setRuntime(TfLiteRuntime.FROM_SYSTEM_ONLY),
        )
    }

    fun classify(bitmap: Bitmap): ClassificationResult {
        val inputTensor = interpreter.getInputTensor(0)
        val shape = inputTensor.shape() // [1, height, width, 3]
        val height = shape.getOrNull(1) ?: 224
        val width = shape.getOrNull(2) ?: 224

        val imageProcessor = ImageProcessor.Builder()
            .add(ResizeOp(height, width, ResizeOp.ResizeMethod.BILINEAR))
            .add(NormalizeOp(0f, 255f))
            .build()

        var tImage = TensorImage(DataType.FLOAT32)
        tImage.load(bitmap)
        tImage = imageProcessor.process(tImage)

        val outputTensor = interpreter.getOutputTensor(0)
        val outShape = outputTensor.shape() // [1, numLabels]
        val numLabels = outShape.lastOrNull() ?: labels.size

        val outputBuffer = TensorBuffer.createFixedSize(intArrayOf(1, numLabels), outputTensor.dataType())
        interpreter.run(tImage.buffer, outputBuffer.buffer.rewind())

        val raw = outputBuffer.floatArray
        val probs = normalizeScores(raw)

        val byLabel = labels
            .take(probs.size)
            .mapIndexed { idx, label -> label to probs[idx] }
            .toMap()

        val nsfwProb = (byLabel["porn"] ?: 0f) + (byLabel["sexy"] ?: 0f)
        val top = byLabel.entries.maxByOrNull { it.value }?.key ?: "unknown"

        return ClassificationResult(
            topLabel = top,
            nsfwProbability = nsfwProb.coerceIn(0f, 1f),
            probabilities = byLabel,
        )
    }

    override fun close() {
        interpreter.close()
    }

    private fun loadLabels(assetPath: String): List<String> {
        context.assets.open(assetPath).use { input ->
            BufferedReader(InputStreamReader(input)).use { reader ->
                return reader.readLines()
                    .map { it.trim() }
                    .filter { it.isNotEmpty() }
            }
        }
    }

    private fun loadModelFile(assetPath: String): MappedByteBuffer {
        context.assets.openFd(assetPath).use { afd ->
            FileInputStream(afd.fileDescriptor).use { inputStream ->
                val fileChannel = inputStream.channel
                return fileChannel.map(FileChannel.MapMode.READ_ONLY, afd.startOffset, afd.declaredLength)
            }
        }
    }

    private fun normalizeScores(x: FloatArray): FloatArray {
        // Some exported models already return probabilities (0..1 summing ~1).
        // Applying softmax again distorts them, so we only softmax when needed.
        if (x.isEmpty()) return x
        val inRange = x.all { it in 0f..1f }
        val sum = x.sum()
        if (inRange && sum in 0.95f..1.05f) return x
        return softmax(x)
    }

    private fun softmax(x: FloatArray): FloatArray {
        if (x.isEmpty()) return x
        val max = x.maxOrNull() ?: 0f
        var sum = 0.0
        val exp = DoubleArray(x.size)
        for (i in x.indices) {
            val e = kotlin.math.exp((x[i] - max).toDouble())
            exp[i] = e
            sum += e
        }
        val out = FloatArray(x.size)
        if (sum == 0.0) return out
        for (i in x.indices) out[i] = (exp[i] / sum).toFloat()
        return out
    }
}

data class ClassificationResult(
    val topLabel: String,
    val nsfwProbability: Float,
    val probabilities: Map<String, Float>,
)

