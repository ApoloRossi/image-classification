# ImageClassification — NSFW Classifier for Android

Android app that classifies images from the `assets/` folder using a local TensorFlow Lite model, blurring any content detected as NSFW.

---

## How it works

1. On launch, the app lists all images in `assets/`
2. Each image is fed into `NsfwTfliteClassifier`, which runs inference on-device with the bundled `model.tflite`
3. Results are displayed in a scrollable list with the top label, per-class probabilities, and NSFW score
4. Images scoring ≥ 70% NSFW (`porn` + `sexy` combined) are blurred automatically

---

## TensorFlow Lite — local library setup

The project uses the **official TFLite runtime + Support Library** directly from Maven — no custom AAR or local `.jar` needed.

### `app/build.gradle.kts`

```kotlin
android {
    androidResources {
        noCompress += "tflite"   // prevents compression so the model can be memory-mapped
    }
}

dependencies {
    implementation("org.tensorflow:tensorflow-lite:2.16.1")
    implementation("org.tensorflow:tensorflow-lite-support:0.4.4")
}
```

> `noCompress += "tflite"` is required. Without it, Android compresses the asset and `FileChannel.map()` fails at runtime.

### Model & labels

Place files in `app/src/main/assets/`:

```
assets/
├── model.tflite   # exported TFLite model (float32, input [1, H, W, 3])
└── labels.txt     # one class name per line, e.g.: drawings, hentai, neutral, porn, sexy
```

---

## Classifier usage

```kotlin
// Instantiate once; reuse across multiple images; close when done
NsfwTfliteClassifier(context).use { classifier ->
    val bitmap: Bitmap = /* load your image */
    val result: ClassificationResult = classifier.classify(bitmap)

    println(result.topLabel)          // e.g. "neutral"
    println(result.nsfwProbability)   // combined porn + sexy score, 0..1
    println(result.probabilities)     // map of every label → score
}
```

`NsfwTfliteClassifier` implements `AutoCloseable`, so `use { }` automatically releases the interpreter.

### What the classifier does internally

```
Bitmap
  └─► TensorImage (FLOAT32)
        └─► ResizeOp  →  model's native H × W (read from tensor shape)
        └─► NormalizeOp(mean=0, std=255)  →  values in [0, 1]
              └─► Interpreter.run()
                    └─► softmax (only if model doesn't already output probabilities)
                          └─► ClassificationResult
```

---

## Project structure

```
app/src/main/
├── assets/
│   ├── model.tflite
│   └── labels.txt
└── java/com/apolodevsystem/imageclassification/
    ├── MainActivity.kt              # Compose UI — loads assets, shows results
    └── ml/
        └── NsfwTfliteClassifier.kt  # TFLite wrapper
```

---

## Requirements

| Item | Version |
|---|---|
| Min SDK | 24 (Android 7.0) |
| Target SDK | 36 |
| Kotlin | — |
| Compose BOM | via `libs.versions.toml` |
| TFLite runtime | 2.16.1 |
| TFLite Support | 0.4.4 |

---

## Adding your own images

Drop any `.png`, `.jpg`, `.jpeg`, or `.webp` file into `app/src/main/assets/` and rebuild. The app picks them up automatically at runtime — no code changes needed.
