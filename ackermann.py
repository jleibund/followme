'''
mixers.py
Classes to wrap motor controllers into a functional drive unit.
'''

from __future__ import division
import math
import methods
from config import config
import atexit
import math
import logging

class NAVIO2PWM(object):
    '''
    NAVIO2 PWM driver
    '''
    def __init__(self, channel, frequency=50):
        from navio2 import pwm as navio_pwm
        from navio2 import util
        util.check_apm()
        self.pwm = navio_pwm.PWM(channel)
        self.pwm.initialize()
        self.channel = channel
        self.pwm.set_period(frequency)

    def update(self, value):
        '''
        Accepts an input [-1, 1] and applies it as
        a PWM with RC-style duty cycle [1, 2].
        '''
        assert(value <= 1 and -1 <= value)
        pwm_val = 1.5 + value * 0.5
        self.pwm.set_duty_cycle(pwm_val)

class AckermannSteeringMixer(object):
    '''
    Mixer for vehicles steered by changing the
    angle of the front wheels.
    This is used for RC car-type vehicles.
    '''

    def __init__(self, steering_driver, throttle_driver):
        self.steering_driver = steering_driver
        self.throttle_driver = throttle_driver

    @staticmethod
    def from_channels(throttle_channel=config.ackermann_car.throttle_channel,
                        steering_channel=config.ackermann_car.steering_channel):
        logging.info("Setting up Ackermann car")
        try:
            throttle_driver = NAVIO2PWM(throttle_channel)
            steering_driver = NAVIO2PWM(steering_channel)
            return AckermannSteeringMixer(
                steering_driver=steering_driver,
                throttle_driver=throttle_driver)
        except Exception as e:
            logging.error("Could not load ackermann: %s"%e)

    def update(self, throttle, angle):
        throttle = min(1, max(-1, -throttle))
        yaw = min(1, max(-1, methods.angle_to_yaw(angle)))
        if not config.car.reverse_steering:
            yaw = -yaw
        if config.car.reverse_throttle:
            throttle = -throttle

        # scaling throttle
        if throttle > 0:
            throttle = throttle / config.car.throttle_scaling
        self.throttle_driver.update(throttle)
        self.steering_driver.update(yaw)
