'''

sensor.py

Sensor base class.

'''

class BaseSensor(object):

    async def start(self):
        '''
        Start receiving values from the sensor
        '''
        pass

    def read(self):
        '''
        Read sensor value
        '''
        pass
