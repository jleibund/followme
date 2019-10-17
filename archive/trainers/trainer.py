import os
import time
import matplotlib
import matplotlib.pyplot as plt
import cv2
import sys
matplotlib.use("Agg")

import numpy as np

from keras.models import Model, Sequential
from keras.layers import Input, Activation, Dense, Convolution2D, Dropout, Flatten
from keras.optimizers import SGD
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import TensorBoard, ReduceLROnPlateau, ModelCheckpoint, EarlyStopping
from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Dropout
from keras.layers.core import Dense
from keras.layers import Flatten
from keras.layers import Input
from keras.models import Model
from keras.layers import concatenate

from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report

# import the necessary packages
from .learningratefinder import LearningRateFinder
from .clr_callback import CyclicLR

from config import config
import methods

from trainers.generators import file_generators
from trainers import histograms
from trainers import img_pipelines
from sklearn.model_selection import train_test_split

LRFIND_PLOT_PATH = os.path.sep.join(["output", "lrfind_plot.png"])
TRAINING_PLOT_PATH = os.path.sep.join(["output", "training_plot.png"])
CLR_PLOT_PATH = os.path.sep.join(["output", "clr_plot.png"])

offset = 4

lr = config.training.LR
epochs = lr = config.training.NUM_EPOCHS
gen_batch = config.training.BATCH_SIZE
min_lr = config.training.MIN_LR
max_lr = config.training.MAX_LR
clr_method = config.training.CLR_METHOD
step_size = config.training.STEP_SIZE
val_batch = 32
val_stride = 20

min_delta=.0001

now = str(int(time.time()))

def train_categorical(data_dir, track, lr_find=False):

    hist = histograms.angles_histogram(data_dir)
    print(hist[0])
    print(hist[1])

    # input_shape=(99-config.camera.crop_top - config.camera.crop_bottom, 132, 3)
    (resolution_width,resolution_height) = config.recording.resolution
    input_width = resolution_width
    input_height = resolution_height-config.camera.crop_top - config.camera.crop_bottom
    input_shape=(input_height, input_width, 3)
    print("Image Input shape: " + str(input_shape))

    im_count = file_generators.file_count(data_dir)
    steps_per_epoch = im_count / gen_batch
    gen = img_pipelines.categorical_pipeline(data_dir, mode='reject_nth',
        batch_size=gen_batch, offset=offset)
    val = img_pipelines.categorical_pipeline(data_dir, mode='accept_nth',
        batch_size=val_batch, offset=offset)

    # val_list = list(val)
    # sub_list = []
    # res_list = []
    # for i in range(val_batch):
    #     sub_list.append(val_list[i][0])
    #     res_list.append(val_list[i][1])

    cnnInputShape = (input_height,input_width, 3)
    mlpInputShape = (len(config.training.columns),)
    inputs = [Input(shape=mlpInputShape),Input(shape=cnnInputShape)]

    model = create_steering_throttle(inputs)

    losses = {
        "angle_out": "categorical_crossentropy",
        "throttle_out": "categorical_crossentropy"
    }
    loss_weights = {'angle_out': 0.9, 'throttle_out': 0.9}
    # loss_weights = {"angle_out": 1.0}

    # compile the model using mean absolute percentage error as our loss,
    # implying that we seek to minimize the absolute percentage difference
    # between our price *predictions* and the *actual prices*
    # opt = Adam(lr=lr, decay=lr / epochs)
    # opt = "rmsprop"

    # opt = SGD(lr=config.training.MIN_LR, decay=config.training.MIN_LR / epochs, momentum=0.9, nesterov=True)
    opt = SGD(lr=min_lr, momentum=0.9)
    model.compile(loss=losses, loss_weights=loss_weights, optimizer=opt,metrics=["accuracy"])

    print(model.summary())

    # check to see if we are attempting to find an optimal learning rate
    # before training for the full number of epochs
    if lr_find:
    	# initialize the learning rate finder and then train with learning
    	# rates ranging from 1e-10 to 1e+1
    	print("[INFO] finding learning rate...")
    	lrf = LearningRateFinder(model)
    	lrf.find(
    		gen,
    		1e-10, 1e+1,
    		stepsPerEpoch=np.ceil((im_count / float(gen_batch))),
    		batchSize=gen_batch)

    	# plot the loss for the various learning rates and save the
    	# resulting plot to disk
    	lrf.plot_loss()
    	plt.savefig(LRFIND_PLOT_PATH)

    	# gracefully exit the script so we can adjust our learning rates
    	# in the config and then train the network for our full set of
    	# epochs
    	print("[INFO] learning rate finder complete")
    	print("[INFO] examine plot and adjust learning rates before training")
    	sys.exit(0)


    models_dir = os.path.abspath(os.path.expanduser(config.training.models_dir))
    model_path = os.path.join(models_dir,
        track + '-' + now + '.h5')
    logs_dir = os.path.abspath(os.path.expanduser(config.training.logs_dir))
    log_path = os.path.join(logs_dir,
    track + '-' + now)

    methods.create_file(model_path)

    tb = TensorBoard(log_path)

    # otherwise, we have already defined a learning rate space to train
    # over, so compute the step size and initialize the cyclic learning
    # rate method
    stepSize = step_size * (im_count // gen_batch)
    clr = CyclicLR(
    	mode=clr_method,
    	base_lr=min_lr,
    	max_lr=max_lr,
    	step_size=stepSize)


    model_cp = ModelCheckpoint(model_path, monitor='val_loss',
                               save_best_only=True, mode='min', period=1)

    e_stop = EarlyStopping(monitor='val_loss', mode='auto', min_delta=min_delta, patience=config.training.PATIENCE)

    print("Best model saved in " + model_path)

    # train the network
    print("[INFO] training network...")
    H = model.fit_generator(gen,
                               epochs=epochs,
                               verbose=1,
                               steps_per_epoch=steps_per_epoch,
                               validation_data=val,
                               validation_steps=im_count / (val_batch * val_stride),
                               callbacks=[clr, tb, model_cp])
                               # callbacks=[clr, tb, model_cp, e_stop])

    min_loss = np.min(H.history['val_loss'])

    print("[INFO] evaluating network...")
    # val_list = list(val)
    # sub_list = []
    # for i in range(gen_batch):
    #     sub_list.append(val_list[i])

    # predictions = model.predict(sub_list, batch_size=val_batch)
    # yaw_classes = [str(n) for n in range(0,config.model.yaw_bins)]
    # print(classification_report(res_list.argmax(axis=1),
    #  	predictions.argmax(axis=1), target_names=yaw_classes))
    print(H.history.keys())
    # construct a plot that plots and saves the training history
    N = np.arange(0, epochs)
    plt.style.use("ggplot")
    plt.figure()
    for key in H.history.keys():
        plt.plot(N, H.history[key], label=key)
    plt.title("Training Loss and Accuracy")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend(loc="lower left")
    plt.savefig(TRAINING_PLOT_PATH)

    # plot the learning rate history
    N = np.arange(0, len(clr.history["lr"]))
    plt.figure()
    plt.plot(N, clr.history["lr"])
    plt.title("Cyclical Learning Rate (CLR)")
    plt.xlabel("Training Iterations")
    plt.ylabel("Learning Rate")
    plt.savefig(CLR_PLOT_PATH)

    return min_loss

def create_steering_throttle(inputs,regress=False):
    drop = 0.25
    mlp = create_mlp(inputs[0],regress=False)
    cnn = create_cnn(inputs[1])
    # cnn = create_nvidia2(inputs[1])
    combinedInput = concatenate([mlp, cnn])
    x = Dense(256)(combinedInput)
    x = Activation("relu")(x)
    x = Dropout(drop)(x)
    x = Dense(128)(combinedInput)
    x = Activation("relu")(x)
    x = Dropout(drop)(x)
    # x = Dense(64)(combinedInput)
    # x = Activation("relu")(x)

    if regress:
        steering_branch = Dense(1, activation="linear",name='angle_out')(x)
        throttle_branch = Dense(1, activation='linear', name='throttle_out')(x)
    else:
        steering_branch = Dense(config.model.yaw_bins, activation="softmax",name='angle_out')(x)
        throttle_branch = Dense(config.model.throttle_bins, activation='softmax', name='throttle_out')(x)
    model = Model(inputs=inputs, outputs=[steering_branch,throttle_branch])
    return model

def create_mlp(inputs, regress=False):
    # define our MLP network
    # chanDim = -1
    drop = 0.25
    x = inputs
    x = Dense(4)(x)
    x = Activation("relu")(x)
    x = Dropout(drop)(x)
    x = Dense(4)(x)
    x = Activation("relu")(x)
    x = Dropout(drop)(x)
    x = Dense(4)(x)
    x = Activation("relu")(x)

    # check to see if the regression node should be added
    if regress:
        x = Dense(2)(x)
        x = Activation("linear")(x)

    # return Model(inputs, x)
    return x

def create_nvidia1(inputs):

    x = inputs
    x = Convolution2D(3, (5,5), strides=(2,2),padding="same")(x)
    x = Activation("relu")(x)
    x = Convolution2D(24, (5,5), strides=(2,2),padding="same")(x)
    x = Activation("relu")(x)
    x = Convolution2D(36, (5,5), strides=(2,2),padding="same")(x)
    x = Activation("relu")(x)
    x = Convolution2D(48, (3,3), strides=(1,1),padding="same")(x)
    x = Activation("relu")(x)
    x = Convolution2D(64, (3,3), strides=(1,1),padding="same")(x)
    x = Dropout(0.25)(x)
    x = Flatten(name='flattened')(x)
    x = Dense(50)(x)
    x = Activation("relu")(x)
    x = Dense(10)(x)

    return x


def create_cnn(inputs, filters=(24, 32, 64), regress=False):
    # initialize the input shape and channel dimension, assuming
    # TensorFlow/channels-last ordering
    chanDim = -1
    drop = 0.25
    # loop over the number of filters
    for (i, f) in enumerate(filters):
        # if this is the first CONV layer then set the input
        # appropriately
        if i == 0:
            x = inputs

        # CONV => RELU => BN => POOL
        x = Conv2D(f, (1, 1), strides=(1,1), padding="same")(x)
        x = BatchNormalization(axis=chanDim)(x)
        x = Activation("relu")(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        # x = Dropout(drop)(x)

    x = Flatten(name='flattened')(x)
    x = Dropout(drop)(x)
    x = Dense(256)(x)
    x = Activation("relu")(x)
    x = Dense(128)(x)
    x = Activation("relu")(x)

    # check to see if the regression node should be added
    if regress:
        x = Dense(2, activation="linear")(x)


    # return the CNN
    # return Model(inputs, x)
    return x

def create_cnn2(inputs, filters=(24, 32, 64), regress=False):
    # initialize the input shape and channel dimension, assuming
    # TensorFlow/channels-last ordering
    chanDim = -1
    drop = 0.25
    # loop over the number of filters
    for (i, f) in enumerate(filters):
        # if this is the first CONV layer then set the input
        # appropriately
        if i == 0:
            x = inputs

        # CONV => RELU => BN => POOL
        x = Conv2D(f, (3, 3), strides=(1,1), padding="same")(x)
        x = BatchNormalization(axis=chanDim)(x)
        x = Activation("relu")(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        # x = Dropout(drop)(x)

    x = Flatten(name='flattened')(x)
    x = Dropout(drop)(x)
    x = Dense(256)(x)
    x = Activation("relu")(x)
    x = Dense(128)(x)
    x = Activation("relu")(x)
    x = Dense(64)(x)
    x = Activation("relu")(x)

    # check to see if the regression node should be added
    if regress:
        x = Dense(2, activation="linear")(x)


    # return the CNN
    # return Model(inputs, x)
    return x
