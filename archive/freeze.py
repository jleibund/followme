import tensorflow as tf
from tensorflow.python.framework import graph_io
from tensorflow.keras.models import load_model
import argparse
import os
from config import config
import subprocess

# Clear any previous session.
tf.keras.backend.clear_session()

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", type=str, required=True,
	help="path to h5 file")
args = vars(ap.parse_args())


save_pb_dir = './models'
model_fname = args["dataset"]
save_pb_name = os.path.splitext(os.path.basename(model_fname))[0]+'.pb'
save_tflite_name = os.path.splitext(os.path.basename(model_fname))[0]+'.tflite'
def freeze_graph(graph, session, output, save_pb_dir='.', save_pb_name='frozen_model.pb', save_pb_as_text=False):
    with graph.as_default():
        graphdef_inf = tf.graph_util.remove_training_nodes(graph.as_graph_def())
        graphdef_frozen = tf.graph_util.convert_variables_to_constants(session, graphdef_inf, output)
        graph_io.write_graph(graphdef_frozen, save_pb_dir, save_pb_name, as_text=save_pb_as_text)
        return graphdef_frozen

# This line must be executed before loading Keras model.
tf.keras.backend.set_learning_phase(0)

model = load_model(model_fname)

session = tf.keras.backend.get_session()

INPUT_NODE = [t.op.name for t in model.inputs]
OUTPUT_NODE = [t.op.name for t in model.outputs]
print(INPUT_NODE, OUTPUT_NODE)
frozen_graph = freeze_graph(session.graph, session, [out.op.name for out in model.outputs], save_pb_dir=save_pb_dir,save_pb_name=save_pb_name)

# tflite_convert --output_file=cnn2.tflite --graph_def_file=./models/cnn2-categorical-raw-1565559150.pb --inference_type=QUANTIZED_UINT8 --input_arrays=input_1,input_2 --output_arrays=angle_out/Softmax,throttle_out/Softmax --mean_values=128,128 --std_dev_values=127,127 --default_ranges_min=0 --default_ranges_max=6

subprocess.call(['tflite_convert','--output_file=./models/%s'%save_tflite_name,'--graph_def_file=./models/%s'%save_pb_name,'--inference_type=QUANTIZED_UINT8',
				'--input_arrays=input_1,input_2','--output_arrays=angle_out/Softmax,throttle_out/Softmax','--mean_values=128,128',
				'--std_dev_values=127,127','--default_ranges_min=0','--default_ranges_max=6'])

subprocess.call(['edgetpu_compiler','-s','./models/%s'%save_tflite_name,'-o',save_pb_dir])

# pb_file = os.path.join(save_pb_dir,save_pb_name)
# output_dir = './model'
# mo_tf_path = './mo_tf.py'
# (resolution_width,resolution_height) = config.recording.resolution
# input_width = resolution_width
# input_height = resolution_height-config.camera.crop_top - config.camera.crop_bottom
# input_shape=(input_height, input_width, 3)
# input_shape_str = str(input_shape).replace(' ','')+',(2)'
# input_layers = 'dense_3_input,conv2d_1_input'
# print('input shape: %s'%input_shape_str)
# subprocess.call(['python',mo_tf_path,'--input_model',pb_file,'--output_dir',output_dir,'--input',
#                  input_layers,'--input_shape',input_shape_str,'--data_type','FP16'])
