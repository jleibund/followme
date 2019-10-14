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
        self.model_path = model_path
        self.t = None
        self.stopped = False
        self.selected = False
        self.frame = None
        self.frame_buffer = None
        self.copied_time = None
        self.base64_time = None
        self.base64_buffer = None
        self.pid = PID(float(config.mobilenet.P),float(config.mobilenet.I),float(config.mobilenet.D))
        super(MobileNet, self).__init__(**kwargs)

    async def decide(self):
        return methods.yaw_to_angle(self.yaw), self.throttle, self.frame_time
        #return methods.yaw_to_angle(self.yaw), 0, self.frame_time

    async def start(self):
        model_path = self.model_path
        if (model_path.endswith('edgetpu.tflite')):
            from tflite_runtime.interpreter import load_delegate
            load_delegate('libedgetpu.so.1.0')
        await asyncio.sleep(1)
        from tflite_runtime.interpreter import Interpreter
        from tflite_runtime.interpreter import load_delegate
        if (model_path.endswith('edgetpu.tflite')):
            interpreter = Interpreter(model_path=model_path,experimental_delegates=[load_delegate('libedgetpu.so.1.0')])
        else:
            interpreter = Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        self.model = interpreter
        self.input_details = self.model.get_input_details()
        self.output_details = self.model.get_output_details()
        self.frame_time = 0
        mot_tracker = Sort() #create instance of the SORT tracker
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
            if not self.selected:
                await asyncio.sleep(0.5)
                continue

            start_time = time.time()
            sensors = self.rover.sensor_reading
            image = None
            frame_time = None
            try:
                image = self.rover.vision_sensor.read()
                frame_time = self.rover.vision_sensor.frame_time
            except Exception as e:
                pass

            if image is not None and frame_time > self.frame_time:
                if w is None:
                    h,w,_ = image.shape
                    width = int((w-h)/2)
                    total_area = w*h

                cropped_img = image[0:h,width:(w-width)]
                resized_img = cv2.resize(cropped_img, (px,px), interpolation = cv2.INTER_AREA)
                img_arr = np.expand_dims(resized_img, axis=0)

                lowy=-1
                if config.training.brake:
                    lowy=0

                self.model.set_tensor(self.input_details[0]['index'], img_arr.astype(np.uint8))
                self.model.invoke()
                locs = self.model.get_tensor(self.output_details[0]['index'])[0]
                classes = self.model.get_tensor(self.output_details[1]['index'])[0]
                scores = self.model.get_tensor(self.output_details[2]['index'])[0]
                detections = self.model.get_tensor(self.output_details[3]['index'])[0]
                dets = []
                for i in range(10):
                    score = scores[i]
                    c = int(classes[i])
                    if c == 0 and score > config.mobilenet.object_confidence:
                        y1 = int(locs[i][0]*h)
                        x1 = int(locs[i][1]*h)+width
                        y2 = int(locs[i][2]*h)
                        x2 = int(locs[i][3]*h)+width
                        dets.append([x1,y1,x2,y2,score])

                trackers = mot_tracker.update(np.array(dets))

                largest = None
                found = None
                yaw = 0
                throttle = 0
                for d in trackers:
                    if target is not None and target[4]==d[4]:
                        found = d
                        break
                    if largest is None:
                        largest = d
                    elif ((largest[2]-largest[0])<(d[2]-d[0])):
                        largest = d

                if found is None and largest is not None:
                    current_height = (largest[2]-largest[0])
                    current_width = (largest[3]-largest[1])
                    current_area = current_height * current_width
                    if current_height > min_height:
                        target = largest
                        found = target
                        self.pid.clear()
                        self.pid.SetPoint = target_height
                        logging.info("Reset target")
                    else:
                        target = None
                elif found is None:
                    target = None

                if found is not None and target is not None:
                    yaw = (((found[0]+((found[2]-found[0])/2))/w)*2)-1
                    found_width = (found[2]-found[0])
                    found_height = (found[3]-found[1])
                    found_area = found_width * found_height
                    percent_height = (found_height/h)
                    if found_height > max_height:
                        throttle = -1.0
                        logging.info('Mobilenet: cutoff found_height %.1f %sms'%(found_height,int(self.f_time*1000)))
                    else:
                        self.pid.update(found_height)
                        throttle = self.pid.output/target_height
                        throttle = avf_t * self.throttle + (1.0 - avf_t) * throttle
                        logging.info('Mobilenet: yaw %.3f throttle %.3f height %.1f %sms'%(yaw,throttle,found_height,int(self.f_time*1000)))
                    self.rover.detection = found
                else:
                    self.rover.detection = None

                yaw_step = config.model.yaw_step
                if abs(yaw - self.yaw) < yaw_step:
                    self.yaw = yaw
                elif yaw < self.yaw:
                    yaw = self.yaw - yaw_step
                elif yaw > self.yaw:
                    yaw = self.yaw + yaw_step

                self.yaw = yaw
                self.frame = image
                self.throttle = throttle
                stop_time = time.time()
                self.frame_time = stop_time
                self.f_time = stop_time - start_time
                self.fps = 1/self.f_time
            else:
                await asyncio.sleep(0.001)

    def image(self):
        '''
        Returns the JPEG image buffer corresponding to
        the current frame. Caches result for
        efficiency.
        '''

        if self.frame_time > self.copied_time and self.frame is not None:
            # self.frame_buffer = self.frame.copy()
            self.frame_buffer = self.frame
            self.copied_time = time.time()
        return self.frame_buffer

    def base64(self):
        '''
        Returns a base-64 encoded string corresponding
        to the current frame. Caches result for
        efficiency.
        '''
        if self.frame_time > self.base64_time:
            image_buffer = self.image()
            if image_buffer is not None:
                base64_buffer = base64.b64encode(image_buffer).decode('ascii')
                self.base64_buffer = base64_buffer
                self.base64_time = time.time()

        return self.base64_buffer

    def stop(self):
        self.stopped = True

    def pname(self):
        return "MobileNet"
