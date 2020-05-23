<div align="center">
<p>
    <img width="300" src="https://user-images.githubusercontent.com/28896432/82730828-96652b80-9d3d-11ea-948c-9ebd44e48596.png">
</p>

<h1>Transformers Android Demo<br/ >(Tensorflow Lite &amp; Pytorch Mobile)</h1>

<!-- <div align="left">
Transformers for Android on-device infererence (Tensorflow Lite &amp; Pytorch Mobile)
</div> -->

</div>

## Sentiment Classification with ELECTRA-Small

**Sentiment classification** finetuned on Movie Review Dataset ([IMDB English Dataset](https://www.kaggle.com/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews), [NSMC Korean Dataset](https://github.com/e9t/nsmc)). **Both English and Korean are supported.**

> Available models:
>
> 1. "Original" TorchScript ELECTRA-Small (53MB)
> 2. "Original" TFLite ELECTRA-Small (53MB)
> 3. FP16 post-training-quantized TFLite ELECTRA-Small (26MB)
> 4. "hybrid" (8-bits precision weights) post-training-quantized TFLite ELECTRA-Small (13MB)

## Demo

Most of the assets are from [Official Pytorch Android Code](https://github.com/pytorch/android-demo-app). (Tested with Galaxy S10)

📱 [APK Download Link](https://drive.google.com/open?id=1DvVSl5gC5pk_VEgvlLEp6sqBwx3wqlFB) 📱

<div float="left">
    <img width="200" style="margin-right: 10px" src="https://user-images.githubusercontent.com/28896432/82734749-f5379e80-9d57-11ea-9a2e-5c3f1fe654c7.jpg">
    <img width="200" style="margin-right: 10px" src="https://user-images.githubusercontent.com/28896432/82734750-f668cb80-9d57-11ea-92b0-3cbef929570b.jpg">
    <img width="200" style="margin-right: 10px" src="https://user-images.githubusercontent.com/28896432/82734752-f668cb80-9d57-11ea-9ced-ebbec7053d2e.jpg">
    <img width="200" style="margin-right: 10px"  src="https://user-images.githubusercontent.com/28896432/82734753-f7016200-9d57-11ea-8dc9-13d01c9f0857.jpg">
</div>

### Build the demo app

<details><summary>Android Studio</summary>

#### Prerequisites

- If you don't have already, install [Android Studio](https://developer.android.com/studio/index.html), following the instructions on the website.
- Android Studio 3.2 or later.
- Install Android SDK and Android NDK using Android Studio UI.
- You need an Android device and Android development environment with minimum API 26.
- The `libs` directory contains a custom build of [TensorFlow Lite with TensorFlow ops built-in](https://www.tensorflow.org/lite/guide/ops_select), which is used by the app. It results in a bigger binary than the "normal" build but allows compatibility with ELECTRA-Small.

#### Building

- Open Android Studio, and from the Welcome screen, select `Open an existing Android Studio project`.
- From the Open File or Project window that appears, select the directory where you cloned this repo.
- You may also need to install various platforms and tools according to error messages.
- If it asks you to use Instant Run, click Proceed Without Instant Run.

#### Running

- You need to have an Android device plugged in with developer options enabled at this point. See [here](https://developer.android.com/studio/run/device) for more details on setting up developer devices.
- Click `Run` to run the demo app on your Android device.

</details>

<details><summary>Gradle (Command Line)</summary>

If [Android SDK](https://developer.android.com/studio/index.html#command-tools) and [Android NDK](https://developer.android.com/ndk/downloads) are already installed you can install this application to the connected android device with:

```
./gradlew installDebug
```

</details>

## Dependencies

### 1. Android

To convert the `original model` to `tflite` format, it has to use `select TensorFlow Ops`.

```python
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.target_spec.supported_ops = [tf.lite.OpsSet.SELECT_TF_OPS]
```

🚨 Using the transformers tflite model, **you should build aar file by yourself.** (Please check this [documentation for tflite ops](https://www.tensorflow.org/lite/guide/ops_select)) 🚨

In this app, I used the same aar file provided from [huggingface demo app](https://github.com/huggingface/tflite-android-transformers/tree/master/bert/libs). (The `libs` directory contains a custom build of aar.)

```gradle
dependencies {
    implementation 'org.pytorch:pytorch_android:1.5.0'
    implementation 'org.pytorch:pytorch_android_torchvision:1.5.0'
    // implementation 'org.tensorflow:tensorflow-lite:2.1.0'
    // implementation 'org.tensorflow:tensorflow-lite-select-tf-ops:0.0.0-nightly'
    implementation(name: 'tensorflow-lite-with-select-tf-ops-0.0.0-nightly', ext: 'aar')
}
```

### 2. Python

- torch==1.5.0
- transformers==2.9.1
- tensorflow==2.1.0

🚨 Highly recommend to use `tensorflow v2.1.0` instead of `tensorflow v2.2.0`. TF-Lite conversion is not working in `tensorflow v2.2.0`. ([Related Issue](https://github.com/huggingface/transformers/issues/3905)) 🚨

## Convert the model to TFLite or TorchScript

※ The models are already uploaded on huggingface s3. They will be automatically downloaded during build. If you want to download `fp16` or `8bits` model, uncomment the line in [download.gradle](./app/download.gradle).

🚨 **TFLite conversion isn't working with CPU environment, working well with GPU.** 🚨

You should specify **the input shape(=max_seq_len)** for model conversion.

```bash
# torchscript
$ python3 model_converter/{$TASK_NAME}/jit_compile.py --max_seq_len 40
# tflite (default)
$ python3 model_converter/{$TASK_NAME}/tflite_converter.py --max_seq_len 40
# tflite (fp16)
$ python3 model_converter/{$TASK_NAME}/tflite_converter.py --max_seq_len 40 --model fp16
# tflite (8bits)
$ python3 model_converter/{$TASK_NAME}/tflite_converter.py --max_seq_len 40 --model 8bits
```

## More Details

### 1. Length &amp; Padding

`MAX_SEQ_LEN` is set as 40 in this app. You may change this one by yourself.

- You should be cautious about the input shape when you converting the model(`--max_seq_len` option in python script)
- Also you need to change the `MAX_SEQ_LEN` in android source code.

```java
private static final int MAX_SEQ_LEN = 40;
```

- **In TFLite, dynamic input size is not supported!** ([Related Issue](https://github.com/tensorflow/tensorflow/issues/24607)) So if the input shape doesn't match with `max_seq_len`, it crashes:( You should pad the input sequence for tflite model.
- But in torchscipt, even though we specified the input shape when converting the model, variable lengths are also possible. So I didn't padded the sequence for pytorch demo. If you want to pad the sequence for pytorch demo, please change the variable as below.

```java
private static final boolean PAD_TO_MAX_LENGTH = true;
```

### 2. FP16 & 8Bits on TFLite

I've already uploaded `fp16` and `8bits` tflite model on `huggingface s3`. (English & Korean both)

If you want to use those models, uncomment the line in `download.gradle` as below. They will be **automatically downloaded during gradle build**.

```gradle
task downloadLiteModel {
    def downloadFiles = [
        // "https://s3.amazonaws.com/models.huggingface.co/bert/monologg/koelectra-small-finetuned-sentiment/nsmc_small_fp16.tflite" : "nsmc_small_fp16.tflite",
        //  "https://s3.amazonaws.com/models.huggingface.co/bert/monologg/koelectra-small-finetuned-sentiment/nsmc_small_8bits.tflite": "nsmc_small_8bits.tflite",
    ]
}
```

Also you need to change the `MODEL_PATH` on Activity.

```java
// 1. fp16
private static final String MODEL_PATH = "imdb_small_fp16.tflite";
// 2. 8bits hybrid
private static final String MODEL_PATH = "imdb_small_8bits.tflite";
```

### 3. Slow inference when using TorchScript

At the first time running the inference using torchscript, the inference is quite slow. After the first pass, inference time comes back as normal.

It seems that the first time running the `forward` might do some _preheating work_. (Not sure about it...)

- [[Related Issue 1]](https://github.com/huggingface/transformers/issues/1477), [[Related Issue 2]](https://github.com/huggingface/transformers/issues/902), [[Related Issue 3]](https://discuss.pytorch.org/t/torchscript-model-inference-slow-in-python/68951), [[Related Issue 4]](https://quabr.com/60232846/too-slow-first-run-torchscript-model-and-its-implementation-in-flask)

## Reference

- [ELECTRA](https://github.com/google-research/electra)
- [KoELECTRA](https://github.com/monologg/KoELECTRA)
- [Huggingface Transformers](https://github.com/huggingface/transformers)
- [Pytorch Android Demo](https://github.com/pytorch/android-demo-app)
- [Huggingface TFLite Android Demo](https://github.com/huggingface/tflite-android-transformers)
- [TFLite Documentation](https://github.com/tensorflow/examples/tree/master/lite)
- [TFLite Java API](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/experimental/support/java/README.md)
- [IMDB Dataset](https://www.kaggle.com/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)
- [NSMC Dataset](https://github.com/e9t/nsmc)
