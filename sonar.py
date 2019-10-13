import asyncio
import sys
import logging
import os
import aiofiles
from config import config
from navio2 import util
from navio2 import adc

TRIGGER_LEFT=504
TRIGGER_RIGHT=503
CHANNEL_LEFT=5
CHANNEL_RIGHT=4
READINGS=3
mVcm=4.883
cmToFt=0.0328084

class StereoSonar(object):
    def __init__(self, readings=config.sonar.readings, **kwargs):
        super(StereoSonar, self).__init__(**kwargs)
        self.readings = readings
        self.cached = (0,0)
        self.stopped = False

    async def start(self):
        util.check_apm()
        self.adc = adc.ADC()
        await asyncio.gather(__setup_pin(TRIGGER_LEFT),__setup_pin(TRIGGER_RIGHT))
        await asyncio.sleep(2)
        while not self.stopped:
            left_volts = await __read_pin(self.adc,TRIGGER_LEFT,CHANNEL_LEFT)
            right_volts = await __read_pin(self.adc,TRIGGER_RIGHT,CHANNEL_RIGHT)
            self.cached = (left_volts,right_volts)

    def read(self):
        '''
        Returns the JPEG image buffer corresponding to
        the current frame. Caches result for
        efficiency.
        '''
        return self.cached

    def stop(self):
        self.stopped = True

    async def __setup_pin(self,pin):
        spin = '%s'%pin
        try:
            async with aiofiles.open('/sys/class/gpio/unexport','w') as f:
                await f.write(spin)
        except:
            pass

        try:
            async with aiofiles.open('/sys/class/gpio/export','w') as f:
                await f.write(spin)
        except:
            pass
        await asyncio.sleep(0.1)

        async with aiofiles.open('/sys/class/gpio/gpio%s/direction'%pin,'w') as f:
            await f.write('out')
        __set_pin(pin,False)

    async def __set_pin(self,pin,on):
        value = '1' if on else '0'
        async with aiofiles.open('/sys/class/gpio/gpio%s/value'%pin,'w') as f:
            await f.write(value)

    async def __read_pin(self,pin,channel):
        await asyncio.sleep(0.05)
        await __set_pin(pin,False)
        await asyncio.sleep(0.01)
        await __set_pin(pin,True)
        await asyncio.sleep(0.01)
        value=0
        for i in range (0,READINGS):
            volts = self.adc.read(channel)
            value += volts
            await asyncio.sleep(0.001)
        reading = value / READINGS
        await asyncio.sleep(0.01)
        await __set_pin(pin,False)
        await asyncio.sleep(0.05)
        return int(reading)

    @staticmethod
    def volts_to_distance(mV):
        return (mV/mVcm)*cmToFt
