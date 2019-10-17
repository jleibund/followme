import math

from PIL import Image, ImageOps

import methods
from config import config


def angle_to_sin(generator):
    '''
    Generator that converts a steering angle to sinus
    '''
    for inp, out in generator:
        yield inp, [math.sin(math.radians(out[0])),out[1]]

def angle_to_yaw(generator):
    '''
    Generator that converts angle values to yaw [-1, 1]
    '''
    for inp, out in generator:
        yield inp, [float(out[0])/config.car.max_steering_angle,out[1]]

def yaw_to_log(generator):
    '''
    Generator that log scales yaw values in the same range
    '''
    for inp, out in generator:
        yield inp, [math.copysign(math.log((abs(out[0])+1)*9, 10), yaw),out[1]]
