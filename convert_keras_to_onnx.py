import os
import sys
import argparse

from tensorflow.python.keras import backend as K
from tensorflow.python.keras.models import load_model

import onnx
import keras2onnx

keras_model_path = sys.argv[1]

model = load_model(keras_model_path)
onnx_model = keras2onnx.convert_keras(model, model.name)
onnx.save_model(onnx_model, keras_model_path + ".onnx")
