import os
import numpy as np
from PIL import Image, ImageOps
from matplotlib import pyplot as plt
from skimage import io
import math
import cmath
import random
from dataset import *


def compute_fourier_coefficients_and_magnitude(image_np_dict):
    """
    1. compute fourier transformation for each image
    2. compute the magnitude of fourier coefficients

    params:
      image_np_dict: {'label': [d, x, y]}

    return:
      coefficients_np_dict: {'label': [d, x, y]}
      mag_np_dict: {'label': [d, x, y]}
    """
    mag_np_dict = {
        'deer': [],
        'horse': [],
        'squirrel': [],
        'tiger': []
    }
    coefficients_np_dict = {
        'deer': [],
        'horse': [],
        'squirrel': [],
        'tiger': []
    }
    for label, data_list in image_np_dict.items():
        print(f'INFO: computing fourier coefficients of {label} data')
        for i, data in enumerate(data_list):
            # print(f'INFO: computing index: {i}')
            fourier_coefficients = np.fft.fftshift(np.fft.fft2(data))
            coefficients_np_dict[label].append(fourier_coefficients)
            magnitudes_np = abs(fourier_coefficients)
            # print(magnitudes_np)
            mag_np_dict[label].append(magnitudes_np)

    print('INFO: computing fourier coefficients complete')
    return coefficients_np_dict, mag_np_dict


def flatten_and_sort_coefficients_and_magnitudes(coefficients_np_dict, mag_np_dict):
    """
    flatten and sort fourier coefficients and magnitudes
    """
    flatten_mag_dict = {
        'deer': [],
        'horse': [],
        'squirrel': [],
        'tiger': []
    }
    flatten_coeff_dict = {
        'deer': [],
        'horse': [],
        'squirrel': [],
        'tiger': []
    }

    for label, data_list in mag_np_dict.items():
        print(f'INFO: flattening and sorting {label} data')
        for i, data in enumerate(data_list):
            flatten_mag_data = data.flatten()
            flatten_coeff_data = coefficients_np_dict[label][i].flatten()
            sorted_indices = np.argsort(flatten_mag_data)[::-1]

            flatten_mag_data = flatten_mag_data[sorted_indices]
            flatten_mag_dict[label].append(flatten_mag_data)

            flatten_coeff_data = flatten_coeff_data[sorted_indices]
            flatten_coeff_dict[label].append(flatten_coeff_data)

    print('INFO: processing complete')
    return flatten_coeff_dict, flatten_mag_dict

# flatten_dict = flatten_and_sort_magnitudes(mag_np_dict)


def filter_flatten_data_by_max_value(flatten_dict, cutoff):
    """
    captured the fourier magnitude where magnitude >= cutoff
    """
    filtered_data_dict = {
        'deer': [],
        'horse': [],
        'squirrel': [],
        'tiger': []
    }

    for label, data_list in flatten_dict.items():
        print(f'INFO: filtering magnitude of {label} - cutoff: {cutoff}')
        for i, data in enumerate(data_list):
            items = []
            for n in data:
                if n >= cutoff:
                    items.append(n)

            items = np.array(items)
            # print(f'INFO: total value count: {len(items)}')
            filtered_data_dict[label].append(items)

    print('INFO: processing complete')
    return filtered_data_dict


# filtered_data_dict = filter_flatten_data_by_max_value(flatten_dict, 5000)


def grip_first_n_items(mag_np_dict, limit):
    """
    capture first n magnitude values
    """
    gripped_data_dict = {
        'deer': [],
        'horse': [],
        'squirrel': [],
        'tiger': []
    }

    for label, data_list in mag_np_dict.items():
        print(f'INFO: gripping date of {label} - limit: {limit}')
        for i, data in enumerate(data_list):
            items = data[0:limit]
            print(f'INFO: total value count: {len(items)}')
            gripped_data_dict[label].append(items)

    print('INFO: processing complete')
    return gripped_data_dict


# gripped_data_dict = grip_first_n_items(flatten_dict, 100)



def get_normalized_n_data_samples(flatten_coeff_dict, flatten_mag_dict, sample_sizes, fs_features):
    """
    get multiple samples and apply normalization
    """
    n_samples = {}
    for size, dim in sample_sizes.items():
        sample_data_dict = {
            'deer': [],
            'horse': [],
            'squirrel': [],
            'tiger': []
        }
        print(f'INFO: sampling and normalizing - dim: {dim}')
        # sample = grip_first_n_items(gripped_data_dict, size)

        for label, data_list in flatten_mag_dict.items():
            for i, data in enumerate(data_list):
                mag_norm = data[0:size]
                mag_norm = mag_norm / np.max(mag_norm)
                mag_norm = mag_norm.reshape(dim)

                fft_features = [mag_norm]
                coeff_data = flatten_coeff_dict[label][i][0:size]

                # compute fourier space features
                for fs_feature in fs_features:
                    fft_features.append(compute_fourier_space_features(coeff_data, fs_feature))

                # print(mag_norm)
                # sample_data_dict[label].append(fft_features)
                sample_data_dict[label].append(mag_norm)

        n_samples[dim] = sample_data_dict

    print('INFO: sampling and normalization complete')
    return n_samples


# n_samples = get_normalized_n_data_samples(gripped_data_dict, [500, 700, 1000, 1200, 1500, 2000, 2500, 3000])


def train_test_split(n_samples):
  """
  split dataset into train and test set
  number of trainset = 28 out of 40
  """
  trainset_size_wise, train_labels_size_wise = {}, {}
  testset_size_wise, test_labels_size_wise = {}, {}

  for size, data_dict in n_samples.items():
    print(f'INFO: spliting data into train and testset of size: {size}')
    trainset, train_labels = [], []
    testset, test_labels = [], []

    for label, data_list in data_dict.items():
      trainset.append(data_list[0: 7])
      train_labels.append(label)

      testset.append(data_list[7: 10])
      test_labels.append(label)

    # trainset = np.array(trainset)
    print('INFO: train and testset shape')
    # print(trainset.shape)
    # train_labels = np.array(train_labels)
    trainset_size_wise[size] = trainset
    train_labels_size_wise[size] = train_labels

    # testset = np.array(testset)
    # print(testset.shape)
    # test_labels = np.array(test_labels)
    testset_size_wise[size] = testset
    test_labels_size_wise[size] = test_labels

  return trainset_size_wise, train_labels_size_wise, testset_size_wise, test_labels_size_wise


def compute_fft_custom_features(coefficients_np_dict):
  """
  compute fourier space custom features: phase angle, sector index and distance
  along with magnitude

  params:
    coefficients_np_dict: {'label': [d, x, y]}

  return:
    train/test custom_features_dict: {'label': [d, x, y]}
  """
  print('INFO: computing fourier space custom features including magnitude')
  train_custom_features_dict = {
      'deer': [],
      'horse': [],
      'squirrel': [],
      'tiger': []
  }
  test_custom_features_dict = {
      'deer': [],
      'horse': [],
      'squirrel': [],
      'tiger': []
  }
  grip_len = 64*64

  # traverse coefficients_np_dict and compute custom features
  for label, data_list in coefficients_np_dict.items():

    custom_features_train, custom_features_test = [], []
    for i, data in enumerate(data_list):
      magnitudes = abs(data)

      mag_flatten = magnitudes.flatten()
      coeff_flatten = data.flatten()

      sorted_indices = np.argsort(mag_flatten)
      mag_flatten = mag_flatten[sorted_indices][0:grip_len]
      coeff_flatten = coeff_flatten[sorted_indices][0:grip_len]

      phase_angles, sectors, distances = [], [], []
      for j, mag in enumerate(mag_flatten):
        # compute phase angle feature
        z = coeff_flatten[j]
        phase = cmath.phase(z)
        phase_angles.append(phase)

        # compute sector feature
        phase_d = cmath.phase(z)
        v = (phase_d/22.5)//1 # get only integer
        if v == 16:
            sector = 0
        else:
          sector = v
        sectors.append(sector)

        # compute distance
        x, y = z.real, z.imag
        d = math.sqrt(x*x+y*y)
        distances.append(d)

      custom_feature = np.array([mag_flatten, phase_angles, sectors, distances])

      if i <= 6:
        custom_features_train.append(custom_feature)
      else:
        custom_features_test.append(custom_feature)

    train_custom_features_dict[label].append(custom_features_train)
    test_custom_features_dict[label].append(custom_features_test)

  print('INFO: processing complete')
  return train_custom_features_dict, test_custom_features_dict



def get_dataset_features_and_labels():
    """
    make function calls and compute needed features and class labels
    """
    print(f"{'-'*15} LOADING IMAGES {'-'*15}")
    image_np_dict, class_labels = load_images()

    print(f"{'-'*15} COMPUTE FOURIER COEFFICIENTS {'-'*15}")
    coefficients_np_dict, mag_np_dict = compute_fourier_coefficients_and_magnitude(image_np_dict)

    print(f"{'-'*15} FLATTEN AND SORT MAGNITUDE {'-'*15}")
    flatten_coeff_dict, flatten_mag_dict = flatten_and_sort_coefficients_and_magnitudes(coefficients_np_dict, mag_np_dict)

    print(f"{'-'*15} DATA SAMPLING {'-'*15}")
    sample_sizes = {
        # 64: (8, 8),
        # 256: (16, 16),
        # 1024: (32, 32),
        # 4096: (64, 64),
        # 16384: (128, 128),
        # 65536: (256, 256),
        # 262144: (512, 512)
    }
    n_samples = get_normalized_n_data_samples(flatten_coeff_dict, flatten_mag_dict, sample_sizes, [])

    print(f"{'-'*15} COMPUTING CUSTOM FFT FEATURES {'-'*15}")
    # get mag with custom features
    # total features: [mag, angle, sector, distance]
    # for 256,256 shape: [65536, 65536, 65536, 65536]
    train_custom_features_dict, test_custom_features_dict = compute_fft_custom_features(coefficients_np_dict)

    print(f"{'-'*15} TRAIN TEST SPLIT {'-'*15}")
    trainset_size_wise, train_labels_size_wise, testset_size_wise, test_labels_size_wise = train_test_split(
        n_samples
    )
    print('INFO: data processing complete!')

    return {
        'n_samples': n_samples, 'image_np_dict': image_np_dict, 'class_labels': class_labels,
        'coefficients_np_dict': coefficients_np_dict, 'mag_np_dict': mag_np_dict,
        'flatten_coeff_dict': flatten_coeff_dict, 'flatten_mag_dict': flatten_mag_dict,
        'trainset_size_wise': trainset_size_wise, 'train_labels_size_wise': train_labels_size_wise,
        'testset_size_wise': testset_size_wise, 'test_labels_size_wise': test_labels_size_wise,
        'train_custom_features_dict': train_custom_features_dict, 'test_custom_features_dict': test_custom_features_dict
    }