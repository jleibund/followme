import numpy as np

from trainers.generators.file_generators import filename_generator

from config import config

def angles_histogram(image_dir):
    '''
    Accepts a path and generates a histogram based on all angles
    of training file in path.
    '''
    angles = []
    for _, out in filename_generator(image_dir):
        angles.append(out[0])
    return np.histogram(angles, config.model.yaw_bins)
