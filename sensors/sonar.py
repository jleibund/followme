import asyncio
import sys
import logging
import os
import aiofiles
from config import config
from navio2 import util
from navio2 import adc

class StereoSonar(object):
    mVcm=4.883
    cmToFt=0.0328084

    def __init__(self, readings=config.sonar.readings,
                        trigger_left=config.sonar.trigger_left,
                        trigger_right=config.sonar.trigger_right,
                        channel_left=config.sonar.channel_left,
                        channel_right=config.sonar.channel_right,
                        **kwargs):
        super(StereoSonar, self).__init__(**kwargs)
        self.readings = readings
        self.cached = (0,0)
        self.stopped = False
        self.trigger_left=trigger_left
        self.trigger_right=trigger_right
        self.channel_left=channel_left
        self.channel_right=channel_right

    async def start(self):
        logging.info("Start StereoSonar")
        util.check_apm()
        logging.info("APM checked")
        try:
            self.adc = adc.ADC()
        except:
            logging.error("Could not load ADC, no sonar available")
            return
        logging.info("Created ADC")
        await asyncio.gather(self.__setup_pin(self.trigger_left),self.__setup_pin(self.trigger_right))
        logging.info("Setup Sonar Pins")
        await asyncio.sleep(2)
        while not self.stopped:
            left_volts = await self.__read_pin(self.trigger_left,self.channel_left)
            self.cached = (left_volts,self.cached[1])
            right_volts = await self.__read_pin(self.trigger_right,self.channel_right)
            self.cached = (self.cached[0],right_volts)

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
        await self.__set_pin(pin,False)

    async def __set_pin(self,pin,on):
        value = '1' if on else '0'
        async with aiofiles.open('/sys/class/gpio/gpio%s/value'%pin,'w') as f:
            await f.write(value)

    async def __read_pin(self,pin,channel):
        await asyncio.sleep(0.04)
        await self.__set_pin(pin,False)
        await asyncio.sleep(0.01)
        await self.__set_pin(pin,True)
        await asyncio.sleep(0.01)
        value=0
        for i in range (0,self.readings):
            volts = self.adc.read(channel)
            value += volts
            await asyncio.sleep(0.001)
        reading = value / self.readings
        await asyncio.sleep(0.01)
        await self.__set_pin(pin,False)
        await asyncio.sleep(0.04)
        return int(reading)

    @staticmethod
    def volts_to_distance(mV):
        return (mV/mVcm)*cmToFt
