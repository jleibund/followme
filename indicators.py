import methods
from config import config
from navio2 import leds

class NAVIO2LED(object):
    '''
    Abstraction of the NAVIO2 LED indicator
    '''

    def __init__(self, **kwargs):

        self.led = leds.Led()
        super(NAVIO2LED, self).__init__(**kwargs)

    def set_state(self, state):
        if state == "warmup":
            self.led.setColor('Yellow')
        elif state == "ready":
            self.led.setColor('Blue')
        elif state == "standby":
            self.led.setColor('Green')
        elif state == "recording":
            self.led.setColor('Red')
        elif state == "error":
            self.led.setColor('Magenta')
