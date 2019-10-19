from __future__ import division

import os
import subprocess
import re
import time
import asyncio
from threading import Thread
from multiprocessing import Process, Queue
import numpy as np

from config import config


'''
IOLOOP and Threads
'''
def start_loop(tasks):
    if tasks is None:
        return
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    for task in tasks:
        loop.run_until_complete(task.start())

def start_thread(tasks):
    if tasks is None:
        return
    thread = Thread(target=start_loop,args=(tasks))
    thread.daemon = True
    thread.start()
    return thread

def start_process(tasks):
    if tasks is None:
        return
    process = Process(target=start_loop,args=(tasks))
    process.start()
    return process


'''
BINNING
functions to help convert between floating point numbers and categories.
'''


def to_index(a, low=-1.0, high=1.0, bins=config.model.output_size):
    step = (high - low) / bins
    b = min(int((a - low) / step), bins - 1)
    return b


def from_index(b, low=-1.0, high=1.0, bins=config.model.output_size):
    step = (high - low) / bins
    a = (b + 0.5) * step + low
    return a


def to_one_hot(y, low=-1.0, high=1.0, bins=config.model.output_size):
    arr = np.zeros(bins)
    arr[to_index(y, low=low, high=high, bins=bins)] = 1
    return arr

def to_one_hot_2d(x, y, lowx=-1.0, highx=1.0, lowy=-1.0, highy=1.0, binsx=config.model.yaw_bins, binsy=config.model.throttle_bins):
    arr = np.zeros(binsx*binsy)
    yidx = to_index(y, low=lowy, high=highy, bins=binsy)
    xidx = to_index(x, low=lowx, high=highx, bins=binsx)
    idx = (yidx*binsx)+xidx
    arr[idx] = 1
    return arr

def from_one_hot(y,low=-1.0, high=1.0, bins=config.model.output_size):
    v = np.argmax(y)
    v = from_index(v,low=low,high=high,bins=bins)
    return v

def from_one_hot_2d(h, lowx=-1, highx=1.0, lowy=-1.0, highy=1.0, binsx=config.model.yaw_bins, binsy=config.model.throttle_bins):
    v = np.argmax(h)
    yidx = int(v/binsx)
    xidx = v%binsx
    x = from_index(xidx,low=lowx,high=highx,bins=binsx)
    y = from_index(yidx,low=lowy,high=highy,bins=binsy)
    return (x,y)


'''
ANGLE CONVERSIONS
functions to help converting between angles and yaw input values.
'''


def angle_to_yaw(angle, limit=config.car.max_steering_angle):
    '''
    Convert from angle to yaw
    '''
    return angle / float(limit)


def yaw_to_angle(yaw, limit=config.car.max_steering_angle):
    '''
    Convert from yaw to angle
    '''
    return yaw * float(limit)


'''
I2C TOOLS
functions to help with discovering i2c devices
'''


def i2c_addresses(bus_index):
    '''
    Get I2C Addresses using i2cdetect.
    Unfortunately the alternative, simpler implementation
    using smbus does not detect NAVIO2 properly, so it's
    needed that i2cdetect is called.
    '''
    addresses = []

    p = subprocess.Popen(['i2cdetect', '-y', '1'], stdout=subprocess.PIPE,)
    for i in range(0, 9):
        line = str(p.stdout.readline())
        for match in re.finditer("[0-9][0-9]:.*[0-9][0-9]", line):
            for number in re.finditer("[0-9][0-9](?!:)", match.group()):
                addresses.append('0x' + number.group())
    return addresses


def board_type():
    '''
    Guess the available board type based on the
    I2C addresses found.
    '''
    addresses = i2c_addresses(1)
    if not addresses:
        return None
    if '0x40' in addresses:
        return 'navio'
    if '0x77' in addresses:
        return 'navio2'
    elif '0x60' in addresses:
        return 'adafruit'


'''
TIME TOOLS
'''


def current_milis():
    '''
    Return the current time in miliseconds
    '''
    return int(round(time.time() * 1000))

'''
MISC
'''

def min_abs(vm, v):
    if vm is None:
        return v
    if abs(vm) <= abs(v):
        return vm
    sign = -1 if vm < 0 else 1
    return abs(v) * sign


def parse_img_filepath(filepath):
    '''
    Parse an image filename and derive angle
    and throttle values
    '''
    f = filepath.split('/')[-1]
    f = f[:-6] #remove ".jpg"
    f = f.split('_')

    throttle = float(f[3]) * 0.001
    angle = float(f[5]) * 0.1
    sonar_left = round(float(f[7]))
    sonar_right = round(float(f[9]))
    milliseconds = round(float(f[11]))

    return angle, throttle, milliseconds, sonar_left, sonar_right

def create_file(path):
    '''
    Create a file at path if not exist
    '''
    def mkdir_p(path):
        try:
            os.makedirs(path)
        except OSError as exc: # Python >2.5
            # if exc.errno == errno.EEXIST and os.path.isdir(path):
            #     pass
            # else: raise
            pass

    def touch(fname):
        try:
            os.utime(fname, None)
        except OSError:
            open(fname, 'a').close()

    mkdir_p(os.path.dirname(path))
    touch(path)
