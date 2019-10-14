import time
import os
import shutil

from PIL import Image

import methods
from config import config
import cv2
import pandas as pd

class FileRecorder(object):
    '''
    Represents a recorder that writes to image files
    '''

    def __init__(self, rover):
        self.rover = rover
        self.instance_path = self.make_instance_dir(
            config.recording.session_dir)
        super(FileRecorder, self).__init__()

    def make_instance_dir(self, sessions_path):
        '''
        Create a directory for the current session based on time,
        and a global sessions directory if it does not exist.
        '''
        real_path = os.path.join("/", os.path.expanduser(sessions_path))
        if not os.path.isdir(real_path):
            os.makedirs(real_path)
        instance_name = time.strftime('%Y_%m_%d__%I_%M_%S_%p')
        instance_path = os.path.join(real_path, instance_name)
        if not os.path.isdir(instance_path):
            os.makedirs(instance_path)
        return instance_path

    def record_frame(self):
        '''
        Record a single image buffer, with frame index, angle and throttle values
        as its filename
        '''
        image_buffer = self.rover.vision_sensor.read()
        sensors = self.rover.sensor_reading
        angle = self.rover.pilot_angle
        throttle = self.rover.pilot_throttle

        if (config.recording.motion_only and throttle * -1.0 < config.recording.throttle_threshold or
               abs(angle) < config.recording.steering_threshold):
           self.is_recording = False
           return
        if image_buffer is None or sensors is None:
            return

        sensors['time'] = [methods.current_milis()]
        sensors['frame_count'] = [self.frame_count]

        self.is_recording = True
        file_angle = int(angle * 10)
        file_throttle = int(throttle * 1000)
        filepath = self.create_img_filepath(
            self.instance_path,
            self.frame_count,
            file_angle,
            file_throttle,
            sensors)
        scaled_image = cv2.resize(image_buffer,config.recording.resolution)
        cv2.imwrite(filepath,scaled_image)
        self.frame_count += 1

    def create_img_filepath(self, directory, frame_count,
                            angle, throttle, sensors, file_type='jpg', prefix='frame'):
        '''
        Generate the complete filepath for saving an image
        '''
        filepath = str("%s/" %
                       directory +
                       prefix+"_" +
                       str(frame_count).zfill(5) +
                       "_ttl_" +
                       str(throttle) +
                       "_agl_" +
                       str(angle) +
                       "_sl_" +
                       str(sensors['sonar_left']) +
                       "_sr_" +
                       str(sensors['sonar_right']) +
                       "_mil_" +
                       str(methods.current_milis()) +
                       '.' + file_type)
        return filepath
