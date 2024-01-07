"""
project entry point
"""

from train import *
from test import *
from fft import *


def TRAIN():
    """
    train model
    """

    """
    get processed features
    """
    features_dict = get_dataset_features_and_labels()

    # train with fft magnitude only
    train(features_dict)

    # train with fft magnitude and following custom fft features:
    # phase angle, sector index and distance
    train(features_dict)


def TEST():
    """
    test model
    """
    """
    get processed features
    """
    features_dict = get_dataset_features_and_labels()

    # test with fft magnitude only
    evaluate_accuracy(features_dict)

    # test with fft magnitude and following custom fft features:
    # phase angle, sector index and distance
    evaluate_custom_fft_features_accuracy(features_dict)


def TEST_INFERENCE():
    """
    test mode by providing a new image
    """
    label = extract_results_from_prediction('/data/deer-1')
    
    return label == 'deer'
