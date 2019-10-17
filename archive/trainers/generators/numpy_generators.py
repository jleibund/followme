import random

import numpy as np

import methods
from config import config


def category_generator(generator):
    '''
    Generator that yields categorical angle from scalar
    '''
    max_steering_angle = config.car.max_steering_angle
    max_sonar = config.sonar.max_mV
    for inp, out in generator:
        sonar_left = inp[0][0]/max_sonar
        sonar_right = inp[0][1]/max_sonar
        yaw = out[0]
        throttle = out[1]
        lowy=-1
        if config.training.brake and throttle < 0:
            throttle = 0
            lowy=0
        # val = methods.to_one_hot_2d(yaw,throttle,lowx=-max_steering_angle,highx=max_steering_angle,lowy=lowy,highy=1,
        #                          binsx=config.model.yaw_bins, binsy=config.model.throttle_bins)
        yaw_binned = methods.to_one_hot(yaw,low=-max_steering_angle,high=max_steering_angle,bins=config.model.yaw_bins)
        throttle_binned = methods.to_one_hot(throttle,low=lowy,high=1,bins=config.model.throttle_bins)
        sonars = np.array([sonar_left,sonar_right])
        yield [sonars,inp[1]], [yaw_binned,throttle_binned]
        # yield ([sonars,inp[1]], {'angle_out': yaw_binned, 'throttle_out': throttle_binned})


def brightness_shifter(generator, min_shift=-0.1, max_shift=0.1):
    '''
    Generator that shifts brightness of an np array
    '''
    for inp, out in generator:
        img_array = inp[1]
        shift_value = random.uniform(min_shift, max_shift)
        shift = np.array([shift_value,
                          shift_value,
                          shift_value])
        img_out = np.clip(img_array + shift, -1.0, 0.9999)
        yield [inp[0],img_out], out


def batch_image_generator(generator, batch_size=32):
    '''
    Generator that bundles images and telemetry into batches.
    '''
    X1_b = []
    X2_b = []
    Y1_b = []
    Y2_b = []
    for X, Y in generator:
        X1_b.append(X[0])
        X2_b.append(X[1])
        Y1_b.append(Y[0])
        Y2_b.append(Y[1])
        if len(X1_b) == batch_size:
            yield [np.array(X1_b),np.array(X2_b)], {'angle_out':np.array(Y1_b),'throttle_out':np.array(Y2_b)}
            X1_b = []
            X2_b = []
            Y1_b = []
            Y2_b = []
    yield [np.array(X1_b),np.array(X2_b)], {'angle_out':np.array(Y1_b),'throttle_out':np.array(Y2_b)}


def center_normalize(generator):
    '''
    Generators that zero-centers and normalizes image data
    '''
    for inp, Y in generator:
        X = inp[1]
        X = X / 128.
        X = X - 1.  # zero-center
        yield [inp[0],X], Y


def equalize_probs(generator,
                   prob=config.training.equalize_prob_strength,
                   symmetric=config.training.symmetric):
    '''
    Generators that attempts to equalize the number of times
    each bin has appeared in the stream
    This could be done to accept bins instead of angles
    however it is more practical to place it early in the
    pipeline before the angle is converted to category
    '''
    size = config.model.yaw_bins
    max_steering_angle = config.car.max_steering_angle
    picks = np.ones(size)
    for inp, out in generator:
        inp_idx = methods.to_index(out[0],
                                   low=-max_steering_angle,
                                   high=max_steering_angle,
                                   bins=config.model.yaw_bins)
        pick_mean = np.mean(picks)
        if picks[inp_idx] > pick_mean and random.uniform(0, 1) < prob:
            continue
        picks[inp_idx] += 1
        if symmetric and inp_idx != size - inp_idx - 1:
            picks[size - inp_idx - 1] += 1
        yield inp, out


def nth_select(generator, nth, mode='reject_nth', offset=0):
    '''
    Generator that either selects or discards nth input
    including offset
    '''
    assert(mode in ['accept_nth', 'reject_nth'])
    counter = 0
    for inp, out in generator:
        is_nth = counter % nth == offset
        if (is_nth and mode == 'accept_nth' or
            not is_nth and mode == 'reject_nth'):
            yield inp, out
        counter += 1


def gaussian_noise(generator, scale=config.training.noise_scale):
    '''
    Generator that adds gaussian noise to an 2d numpy
    array (an image)
    '''
    for inp in generator:
        img = inp[1]
        scale_instance = np.random.randn() * scale
        noisy = np.random.randn(*inp.shape) * scale_instance
        img = np.clip(img + noisy, -1.0, 0.9999)
        yield [inp[0],img], out
