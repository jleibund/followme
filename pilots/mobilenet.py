'''

mobilenet.py

Mobilenet + SORT-driven basic steering and throttle.

'''
from __future__ import division
import asyncio
from sklearn.preprocessing import MinMaxScaler
from PIL import Image
import math
from pilots.sort import Sort #create instance of the SORT tracker
from pilots.pid import PID
import numpy as np
import methods
from config import config
from .pilot import BasePilot
from sklearn.preprocessing import MinMaxScaler
import cv2
import time
import logging
from methods import min_abs
import base64

import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)
px = 300

class MobileNet(BasePilot):
    '''
    A pilot based on a CNN with categorical output
    '''

    def __init__(self, rover, model_path=config.mobilenet.model_path, **kwargs):
        self.fps = 0
        self.yaw = 0
        self.throttle = 0
        self.rover = rover
        self.f_time = 0
        # path to the edgetpu model for mobilenet SSD
        self.model_path = model_path
        self.t = None
        self.stopped = False
        self.selected = False
        self.frame_buffer = None
        self.target = None
        # using a PID to control throttle
        self.pid = PID(float(config.mobilenet.P),float(config.mobilenet.I),float(config.mobilenet.D))
        super(MobileNet, self).__init__(**kwargs)

    async def decide(self):
        return methods.yaw_to_angle(self.yaw), self.throttle, self.frame_time

    async def start(self):
        model_path = self.model_path
        from tflite_runtime.interpreter import Interpreter
        from tflite_runtime.interpreter import load_delegate
        # attempt to load the edgetpu model, preferring the coral compiled version
        if (model_path.endswith('edgetpu.tflite')):
            interpreter = Interpreter(model_path=model_path,experimental_delegates=[load_delegate('libedgetpu.so.1.0')])
        else:
            interpreter = Interpreter(model_path=model_path)
        await asyncio.sleep(1)

        # allocate model tensors and get input/output details
        interpreter.allocate_tensors()
        self.model = interpreter
        self.input_details = self.model.get_input_details()
        self.output_details = self.model.get_output_details()
        self.frame_time = 0

        # create instance of the SORT tracker
        mot_tracker = Sort()
        target = None
        avf_t = config.model.throttle_average_factor
        target_area = float(config.mobilenet.target_area)
        min_area = float(config.mobilenet.min_area)
        min_height = float(config.mobilenet.min_height)
        max_height = float(config.mobilenet.max_height)
        target_height = float(config.mobilenet.target_height)
        total_area = None
        w = None
        h = None
        crop_top = int(config.camera.crop_top)
        crop_bottom = int(config.camera.crop_bottom)
        width = None

        while not self.stopped:
            # do not run when not "on"
            if not self.selected:
                await asyncio.sleep(0.1)
                continue

            start_time = time.time()

            # get the sensor readings and the image from the camera
            sensors = self.rover.sensor_reading
            image = None
            frame_time = None
            try:
                image = self.rover.vision_sensor.read()
                frame_time = self.rover.vision_sensor.frame_time
            except Exception as e:
                pass

            # only continue if current cam frame time is greater than mobilnet last frame time
            if image is not None and frame_time > self.frame_time:

                # get width/height if first time around loop
                if w is None:
                    h,w,_ = image.shape
                    # substitute width (square image)
                    width = int((w-h)/2)
                    total_area = w*h

                # crop and resize the image and prepare array for tensors
                cropped_img = image[0:h,width:(w-width)]
                resized_img = cv2.resize(cropped_img, (px,px), interpolation = cv2.INTER_AREA)
                img_arr = np.expand_dims(resized_img, axis=0)

                # set the tensor using uint8 and invoke
                self.model.set_tensor(self.input_details[0]['index'], img_arr.astype(np.uint8))
                s1 = time.time()
                self.model.invoke()
                t1 = int((time.time()-s1)*1000)

                # get the response arrays
                locs = self.model.get_tensor(self.output_details[0]['index'])[0]
                classes = self.model.get_tensor(self.output_details[1]['index'])[0]
                scores = self.model.get_tensor(self.output_details[2]['index'])[0]
                detections = self.model.get_tensor(self.output_details[3]['index'])[0]
                dets = []

                # produces 10 results
                for i in range(10):
                    score = scores[i]
                    c = int(classes[i])
                    # if the class is a person (class 0) and its high enough confidence
                    if c == 0 and score > config.mobilenet.object_confidence:
                        # adjust dims back to full image sizes and append, retain the score for the SORT kalman tracker
                        y1 = int(locs[i][0]*h)
                        x1 = int(locs[i][1]*h)+width
                        y2 = int(locs[i][2]*h)
                        x2 = int(locs[i][3]*h)+width
                        dets.append([x1,y1,x2,y2,score])

                # run candidates through the tracker
                s2 = time.time()
                trackers = mot_tracker.update(np.array(dets))
                t2 = int((time.time()-s2)*1000)
                
                # we will try to find our existing target or if not there, reset and pick the largest/closest person in frame
                largest = None
                found = None
                yaw = 0
                throttle = 0
                for d in trackers:
                    # for given candidate, if target is set see if there is a match on this iteration
                    if target is not None and target[4]==d[4]:
                        found = d
                        break
                    # otherwise keep track of the largest box
                    if largest is None:
                        largest = d
                    elif ((largest[2]-largest[0])<(d[2]-d[0])):
                        largest = d

                # if we didn't find the target but we have a largest box
                if found is None and largest is not None:
                    current_height = (largest[2]-largest[0])
                    current_width = (largest[3]-largest[1])
                    current_area = current_height * current_width

                    # as long as the box height is larger than the min height (rover shouldn't take off after tiny boxes!)
                    if current_height > min_height:
                        # set the target and the found box
                        target = largest
                        found = target

                        # reset the PID with set point as the target height, which is fixed (config file)
                        self.pid.clear()
                        self.pid.SetPoint = target_height
                        logging.info("Reset target")
                    else:
                        # otherwise reset target
                        target = None
                elif found is None:
                    target = None

                # if both target and found are set
                if found is not None and target is not None:
                    # compute the yaw and found height
                    yaw = (((found[0]+((found[2]-found[0])/2))/w)*2)-1
                    found_height = (found[3]-found[1])

                    # if larger than max height, apply the brake
                    if found_height > max_height:
                        throttle = -1.0
                        logging.info('Mobilenet: cutoff found_height %.1f %sms'%(found_height,int((time.time()-start_time)*1000)))
                    else:
                        # otherwise update PID and convert PID output to throttle proportion
                        s3 = time.time()
                        self.pid.update(found_height)
                        t3 = int((time.time()-s3)*1000)
                        throttle = self.pid.output/target_height

                        # average throttle using factor for smooth acceleration /  decelleration
                        throttle = avf_t * self.throttle + (1.0 - avf_t) * throttle
                        logging.info('Mobilenet: yaw %.3f throttle %.3f height %.1f %sms [t1: %sms, t2: %sms, t3: %sms]'%(yaw,throttle,found_height,int((time.time()-start_time)*1000),t1,t2,t3))

                    # save detection on the rover
                    self.target = found
                else:
                    # remove detection from the rover
                    self.target = None

                # adjust yaw in incremental steps (smoothing per step)
                yaw_step = config.model.yaw_step
                if abs(yaw - self.yaw) < yaw_step:
                    self.yaw = yaw
                elif yaw < self.yaw:
                    yaw = self.yaw - yaw_step
                elif yaw > self.yaw:
                    yaw = self.yaw + yaw_step

                self.yaw = yaw
                self.frame_buffer = image
                self.throttle = throttle
                stop_time = time.time()
                self.frame_time = stop_time
                self.f_time = stop_time - start_time
                self.fps = 1/self.f_time
            await asyncio.sleep(0.001)

    def stop(self):
        self.stopped = True

    def pname(self):
        return "MobileNet"
