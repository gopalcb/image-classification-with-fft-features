from model import *
import tensorflow as tf
import keras
import numpy as np


classes_map = {
    'tiger': 0,
    'deer': 1,
    'horse': 2,
    'squirrel': 3
}

def evaluate_accuracy(features_dict):
  """
  evaluate model accuracy of models trained with magnitude data only
  """
  testset_size_wise = features_dict['testset_size_wise']
  test_labels_size_wise = features_dict['test_labels_size_wise']

  for size, test_data in testset_size_wise.items():
    print(f'INFO: validating for DIM: {size}')
    testset, test_lbls = [], []

    for i, data_3d in enumerate(test_data):
      # print(data_3d)
      lbl = test_labels_size_wise[size][i]
      # print(lbl)
      for j, data in enumerate(data_3d):
        # print(data.shape)
        testset.append(data)
        test_lbls.append(classes_map[lbl])

    testset = np.array(testset)
    test_lbls = np.array(test_lbls)
    print(f'trainset shape: {testset.shape}')
    print(f'train label shape: {test_lbls.shape}')

    model = keras.models.load_model(f'model-{str(size)}.keras')
    test_loss, test_acc = model.evaluate(testset,  test_lbls, verbose=2)
    print(f'INFO: test loss: {test_loss}, test accuracy: {test_acc}')


def evaluate_custom_fft_features_accuracy(features_dict):
  """
  test model for custom fft features
  """

  test_custom_features_dict = features_dict['test_custom_features_dict']

  for label, test_data in test_custom_features_dict.items():
    print(f'INFO: testing for custom fft features')
    test_data_ = test_data[0]
    # print(train_data_.shape)
    testset, test_lbls = [], []

    for i, data in enumerate(test_data_):
      testset.append(data)
      test_lbls.append(classes_map[label])

    testset = np.array(testset)
    test_lbls = np.array(test_lbls)
    print(f'trainset shape: {testset.shape}')
    print(f'train label shape: {test_lbls.shape}')

    model = keras.models.load_model(f'model-custom-fft.keras')
    test_loss, test_acc = model.evaluate(testset,  test_lbls, verbose=2)
    print(f'INFO: test loss: {test_loss}, test accuracy: {test_acc}')


# evaluate_custom_fft_features_accuracy(features_dict)