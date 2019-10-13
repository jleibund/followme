import sys
import asyncio
import logging
import os
from navio2 import mpu9250
from navio2 import util
from config import config

class IMU(object):
    def __init__(self, **kwargs):
        super(IMU, self).__init__(**kwargs)
        self.cached = ([0,0,0],[0,0,0],[0,0,0])
        self.stopped = False

    async def start(self):
        util.check_apm()
        self.imu = mpu9250.MPU9250()
        self.imu.initialize()
        await asyncio.sleep(2)
        while not self.stopped:
            m9a, m9g, m9m = self.imu.getMotion9()
            await asyncio.sleep(0.1)
            self.cached = (m9a,m9g,m9m)
            print("reading")

    def read(self):
        '''
        Returns the JPEG image buffer corresponding to
        the current frame. Caches result for
        efficiency.
        '''
        return self.cached

    def stop(self):
        self.stopped = True
