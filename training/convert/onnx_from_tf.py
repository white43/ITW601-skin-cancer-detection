import os
import argparse

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import tf2onnx
import keras

cli_opts = argparse.ArgumentParser()
cli_opts.add_argument("--model", required=True)
options = cli_opts.parse_args()

model = keras.models.load_model(options.model)

tf2onnx.convert.from_keras(
    model,
    input_signature=(tf.TensorSpec(model.input_shape, tf.float32, name="input_layer"),),
    output_path=".".join(options.model.split(".")[0:-1]) + ".onnx",
)

