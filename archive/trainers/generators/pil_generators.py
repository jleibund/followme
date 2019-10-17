import os
import time
import math
import random
import psutil
from random import randint as ri

import numpy as np

from PIL import Image, ImageOps, ImageChops, ImageDraw, ImageFont

#import pyblur

import methods
from config import config

import logging


def image_generator(generator):
    '''
    Generator that yields images from files.
    '''
    for inp, out in generator:
        file_path = inp[1]
        try:
            with Image.open(file_path) as img:
                angle, throttle, ms, sensor_left, sensor_right = methods.parse_img_filepath(file_path)
                yield [[sensor_left,sensor_right],img], [angle,throttle]
        except IOError as err:
            logging.warning(err)

def image_crop(generator, top=config.camera.crop_top,
        bottom=config.camera.crop_bottom):
    '''
    Generator that crops the top pixels of an image
    '''
    for inp, out in generator:
        img = inp[1]
        w, h = img.size
        yield [inp[0],img.crop((0, top, w, h-bottom))], out

def image_resize(generator, size=config.recording.resolution):
    '''
    Generator that resizes images
    '''
    for inp, out in generator:
        img = inp[1]
        img = img.resize(size, Image.ANTIALIAS)
        yield [inp[0],img], out

def image_mirror(generator):
    '''
    Generator that augments batches of images and telemetry
    through mirroring
    '''
    for inp, out in generator:
        sonar = inp[0]
        img = inp[1]
        if random.uniform(0.0, 1.0) > 0.5:
            img = ImageOps.mirror(img)
            yield [[sonar[1],sonar[0]],img], [-out[0],out[1]]
        else:
            yield [inp[0],img], [out[0],out[1]]

def image_rotate(generator, prob=0.4, max_angle=9):
    '''
    Generator that augments batches of images and telemetry
    through random rotation
    '''
    for inp, out in generator:
        img = inp[1]
        if prob < random.uniform(0, 1):
            image_angle = random.uniform(-max_angle, max_angle)
            dst_im = Image.new(
                "RGBA", img.size, (ri(
                    0, 255), ri(
                    0, 255), ri(
                    0, 255)))
            rot = img.rotate(image_angle, resample=Image.BICUBIC, expand=False)
            dst_im.paste(rot)
            img = dst_im.convert('RGB')
        yield [inp[0],img], out

def image_voffset(generator, prob=0.4, max_pixels=5):
    '''
    Generator that vertically offsets an image
    '''
    for inp, out in generator:
        img = inp[1]
        if prob < random.uniform(0, 1):
            image_offset = random.randrange(-max_pixels, max_pixels)
            dst_im = Image.new(
                "RGBA", img.size, (ri(
                    0, 255), ri(
                    0, 255), ri(
                    0, 255)))
            rot = ImageChops.offset(img, 0, image_offset)
            dst_im.paste(rot)
            img = dst_im.convert('RGB')
        yield [inp[0],img], out

def array_generator(generator):
    '''
    Generator that yields a numpy array from a PIL image
    '''
    for inp, outp in generator:
        img = inp[1]
        yield [inp[0],np.array(img)], outp

def show_image(generator, k=280):
    '''
    Accepts a generator and shows images on screen with a prompt to continue
    '''
    for inp, out in generator:
        img = inp[1]
        if type(img) is np.ndarray:
            img = (img + 1.) * 128.
            img = img.astype(np.uint8)
            img = Image.fromarray(img, 'RGB')
        if img.height < img.width:
            factor = float(k) / img.height
        else:
            factor = float(k) / img.width
        img = img.resize((int(img.width* factor), int(img.height * factor)))
        draw = ImageDraw.Draw(img)
        font = ImageFont.load_default()
        draw.text((9, 9),str(out[0]),(255,0,0),font=font)
        img.show()
        raw_input("Press Enter to continue...")
        # hide image
        for proc in psutil.process_iter():
            if proc.name() == "display":
                proc.kill()
