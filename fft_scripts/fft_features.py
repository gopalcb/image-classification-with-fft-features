import os
import numpy as np
from PIL import Image, ImageOps
from matplotlib import pyplot as plt
from skimage import io
import math
import cmath
import random


def compute_fourier_space_features(coeff_data, feature_name):
    # compute spectral rotation angle
    def compute_phase_angle(z):
        phase = cmath.phase(z)
        return phase

    # Compute sector index
    def compute_sector(z):
        phase_d = cmath.phase(z)
        v = (phase_d/22.5)//1 #get only integer
        if v == 16:
            return 0

        return v

    # Compute distance given r,c
    def compute_distance(z):
        x, y = z.real, z.imag
        d = math.sqrt(x*x+y*y)
        return d

    computed_custom_features = []
    for z in coeff_data:
        r, c = z.real, z.imag
        if feature_name == 'angle':
            computed_custom_features.append(compute_phase_angle(z))

        if feature_name == 'sector':
            computed_custom_features.append(compute_sector(z))

        if feature_name == 'distance':
            computed_custom_features.append(compute_distance(z))

    return computed_custom_features