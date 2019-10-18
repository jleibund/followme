'''

camera.py

PiCamera sensor implementation.

'''

import io
import os
import time
import io
import logging
import numpy as np
import imutils
import cv2
import asyncio
from PIL import Image
from config import config
from .sensor import BaseSensor

class PiVideoStream(BaseSensor):

    def __init__(self, resolution=config.camera.resolution,
            framerate=config.camera.framerate,
            rotation=config.camera.rotation,
            vflip=config.camera.vflip,
            **kwargs):
        super(PiVideoStream, self).__init__(**kwargs)

        self.resolution = resolution
        self.frame = np.zeros(
                shape=(
                    self.resolution[1],
                    self.resolution[0],
                    3))
        self.frame_time = 0
        self.copied_time = 0
        self.frame_buffer = None
        self.framerate = framerate
        self.rotation = rotation
        self.vflip = vflip
        self.exposure_mode = "sports"
        self.balance=0.0
        self.frame = None
        self.stopped = False

    async def start(self):
        try:
            import picamera
            from picamera.array import PiRGBArray
            with picamera.PiCamera() as camera:
                camera.resolution = self.resolution
                camera.framerate = self.framerate
                camera.rotation = self.rotation
                camera.vflip = self.vflip
                camera.hflip = self.vflip
                camera.exposure_mode = self.exposure_mode
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

                    self.frame = frame
                    rawCapture.truncate(0)
                    self.frame_time = time.time()
                    await asyncio.sleep(0.001)
                    if self.stopped:
                        return
        except:
            logging.error("Could not load picamera, running in test mode")
            from imutils.video.webcamvideostream import WebcamVideoStream
            stream = WebcamVideoStream(src=0).start()
            logging.info("Starting WebcamVideoStream stream")
            while not self.stopped:
                frame = stream.read()
                if frame is None:
                    await asyncio.sleep(0.1)
                    continue
                self.frame = frame
                self.frame_time = time.time()
                await asyncio.sleep(0.001)


    def read(self):
        '''
        Returns the JPEG image buffer corresponding to
        the current frame. Caches result for
        efficiency.
        '''

        if self.frame_time > self.copied_time and self.frame is not None:
            image_buffer = self.frame
            if image_buffer is not None:
                if config.camera.resize:
                    (w1, h1) = config.camera.resize
                    (w2, h2) = config.camera.resolution
                    h,w,_ = image.shape
                    width = int((w-h)/2)
                    cropped_img = image[0:h,width:(w-width)]
                    resized_img = cv2.resize(cropped_img, (w1,h1), interpolation = cv2.INTER_AREA)
                    image_buffer = resized_img

            self.frame_buffer = image_buffer
            self.copied_time = time.time()
        return self.frame_buffer

    def stop(self):
        self.stopped = True
