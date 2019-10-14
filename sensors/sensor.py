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

    def update(self):
        '''
        Performs sensor update steps. This is called
        in a separate thread
        '''
        pass
