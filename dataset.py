import os
import numpy as np
from PIL import Image, ImageOps
from matplotlib import pyplot as plt
from skimage import io
import math
import cmath
import random


def load_images(data_root='data'):
    """
    load image pixle values into numpy array
    the class labels are extracted from image name
    """
    image_files = os.listdir(data_root)
    image_np_dict = {
        'deer': [],
        'horse': [],
        'squirrel': [],
        'tiger': []
    }

    print('INFO: reading images..')
    for file in image_files:
        # print(f'INFO: reading image file: {file}')
        label = file.split('-')[0]
        img = io.imread(f'{data_root}/{file}', as_gray=True)
        # img = img / np.max(img)
        image_np_dict[label].append(img)

    # image_np_dict['deer'] = np.array(image_np_dict['deer'])
    # image_np_dict['horse'] = np.array(image_np_dict['horse'])
    # image_np_dict['squirrel'] = np.array(image_np_dict['squirrel'])
    # image_np_dict['tiger'] = np.array(image_np_dict['tiger'])
    class_labels = list(image_np_dict.keys())

    print('INFO: reading images complete')
    return image_np_dict, class_labels


