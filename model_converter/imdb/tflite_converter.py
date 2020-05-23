import os
import sys
import argparse
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

import tensorflow as tf
from transformers import ElectraTokenizer

from model import TFElectraForSequenceClassification

parser = argparse.ArgumentParser()
# NOTE This should be same as the setting of the android!!!
parser.add_argument("--max_seq_len", default=40, type=int, help="Maximum sequence length")
parser.add_argument("--model", default="default", help="Conversion (Default, fp16, 8bits)")
args = parser.parse_args()

tokenizer = ElectraTokenizer.from_pretrained("monologg/electra-small-finetuned-imdb")
model = TFElectraForSequenceClassification.from_pretrained("monologg/electra-small-finetuned-imdb",
                                                           from_pt=True)

input_spec = tf.TensorSpec([1, args.max_seq_len], tf.int32)
model._set_inputs(input_spec, training=False)

print(model.inputs)
print(model.outputs)

# Tokenize input text
text = "This movie is awesome lol!"
encode_inputs = tokenizer.encode_plus(
    text,
    return_tensors="tf",
    max_length=args.max_seq_len,
    pad_to_max_length=True
)

outputs = model(encode_inputs["input_ids"])
print(outputs[0])

converter = tf.lite.TFLiteConverter.from_keras_model(model)

if args.model == "default":
    # 1. For normal conversion:
    converter.target_spec.supported_ops = [tf.lite.OpsSet.SELECT_TF_OPS]
    tflite_model = converter.convert()
    open("app/src/main/assets/nsmc_small.tflite", "wb").write(tflite_model)
elif args.model == "fp16":
    # 2. For conversion with FP16 quantization:
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
    converter.target_spec.supported_types = [tf.float16]
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.experimental_new_converter = True
    tflite_model = converter.convert()
    open("app/src/main/assets/nsmc_small_fp16.tflite", "wb").write(tflite_model)
elif args.model == "8bits":
    # 3. For conversion with hybrid quantization:
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
    converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
    converter.experimental_new_converter = True
    tflite_model = converter.convert()
    open("app/src/main/assets/nsmc_small_8bits.tflite", "wb").write(tflite_model)
else:
    raise ValueError("Only default, fp16, 8bits available!")
