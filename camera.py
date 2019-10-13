import io
import os
import time
import base64
import io
import logging
import numpy as np
import imutils
import cv2
import asyncio
from PIL import Image
from config import config

class PiVideoStream(object):

    def __init__(self, resolution=config.camera.resolution,
            framerate=config.camera.framerate,
            rotation=config.camera.rotation,
            vflip=config.camera.vflip,
            fisheye=config.camera.fisheye,
            **kwargs):

        super(PiVideoStream, self).__init__(resolution, **kwargs)
        self.resolution = resolution
        self.frame = np.zeros(
                shape=(
                    self.resolution[1],
                    self.resolution[0],
                    3))
        self.frame_time = 0
        self.copied_time = 0
        self.base64_time = 0
        self.image_time = 0
        self.image_buffer = None
        self.frame_buffer = None
        self.base64_buffer = None

        self.framerate = framerate
        self.rotation = rotation
        self.vflip = vflip
        self.exposure_mode = "sports"

        # self.fisheye = fisheye
        # self.dim1=None
        # self.dim2=None
        # self.dim3=None
        # self.map1=None
        # self.map2=None
        self.new_K=None
        self.balance=0.0
        # (self.map1, self.map2) = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), K, DIM, cv2.CV_16SC2)
        self.frame = None
        self.stopped = False

    async def start(self):
        import picamera
        from picamera.array import PiRGBArray

        with picamera.PiCamera() as camera:
            camera.resolution = self.resolution
            camera.framerate = self.framerate
            camera.rotation = self.rotation
            camera.vflip = self.vflip
            camera.hflip = self.vflip
            camera.exposure_mode = "sports"
            rawCapture = PiRGBArray(camera, size=self.resolution)

            logging.info("PiVideoStream loaded.. .warming camera")
            camera.start_preview()
            await asyncio.sleep(2)
            logging.info("PiVideoStream starting continuous stream")
            for f in camera.capture_continuous(
                    rawCapture, format="bgr", use_video_port=True):
                frame = f.array
                if frame is None:
                    await asyncio.sleep(0.1)
                    continue

                # if self.fisheye:
                #     frame = cv2.remap(frame, self.map1, self.map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
                    # if not self.dim1:
                    #     self.dim1 = frame.shape[:2][::-1]
                    #     assert self.dim1[0]/self.dim1[1] == DIM[0]/DIM[1], "Image to undistort needs to have same aspect ratio as the ones used in calibration"
                    # if not self.dim2:
                    #     self.dim2 = self.dim1
                    # if not self.dim3:
                    #     self.dim3 = self.dim2
                    # if self.new_K is None:
                    #     scaled_K = K * self.dim1[0] / DIM[0]  # The values of K is to scale with image dimension.
                    #     scaled_K[2][2] = 1.0  # Except that K[2][2] is always 1.0
                    #     self.new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(scaled_K, D, self.dim2, np.eye(3), balance=self.balance)
                    #     self.map1, self.map2 = cv2.fisheye.initUndistortRectifyMap(scaled_K, D, np.eye(3), self.new_K, self.dim3, cv2.CV_16SC2)
                    #     print("here!")

                (h,w) = frame.shape[:2]
                self.frame = frame
                rawCapture.truncate(0)
                self.frame_time = time.time()
                await asyncio.sleep(0.001)
                print("camera frame")
                if self.stopped:
                    return

    def read(self):
        '''
        Returns the JPEG image buffer corresponding to
        the current frame. Caches result for
        efficiency.
        '''

        if self.frame_time > self.copied_time and self.frame is not None:
            # self.frame_buffer = self.frame.copy()
            image_buffer = self.frame
            if image_buffer is not None:
                # image_buffer = cv2.resize(image_buffer,config.recording.resolution)
                if config.camera.crop_top or config.camera.crop_bottom:
                    h, w, _ = image_buffer.shape
                    t = config.camera.crop_top
                    l = h - config.camera.crop_bottom
                    image_buffer = image_buffer[t:l, :]

            self.frame_buffer = image_buffer
            self.copied_time = time.time()
        return self.frame_buffer

    def image(self):
        if self.frame_time > self.image_time and self.frame is not None:
            image_buffer = self.read()
            if image_buffer is not None:
                retval, encoded = cv2.imencode('.jpg', image_buffer)
                self.image_buffer = encoded
                self.image_time = time.time()
        return self.image_buffer

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
