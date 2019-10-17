import tensorflow as tf
import argparse
import os
from trainers import img_pipelines
from config import config
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", type=str, required=True,
	help="path to h5 file")
ap.add_argument("-d", "--dataset", type=str, required=True,
	help="path to representative dataset")
args = vars(ap.parse_args())

save_pb_dir = './models'
model_fname = args["model"]
data_dir = args["dataset"]
tflite_model_file = os.path.splitext(os.path.basename(model_fname))[0]+'.tflite'

num_inputs = 100
val = img_pipelines.categorical_pipeline(data_dir, mode='accept_nth',
    batch_size=1, offset=1,raw=True)

def representative_dataset_gen():
  for inp,output in val:
    # Get sample input data as a numpy array in a method of your choosing.
    yield [inp[0].astype(np.float32),inp[1].astype(np.float32)]

converter = tf.lite.TFLiteConverter.from_keras_model_file(model_fname)
# converter.target_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset_gen
# converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
tflite_quant_model = converter.convert()

# converter = tf.lite.TFLiteConverter.from_keras_model_file(model_fname)
# tflite_model = converter.convert()
# converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
with open(tflite_model_file, "wb") as f:
    f.write(tflite_quant_model)
