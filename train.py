from model import *
import tensorflow as tf
import keras
import numpy as np


# HYPERPARAMETERS
EPOCHS = 10
LR = 0.0001
classes_map = {
    'tiger': 0,
    'deer': 1,
    'horse': 2,
    'squirrel': 3
}

def train(features_dict):
  """
  train model with different input shape
  """
  trainset_size_wise = features_dict['trainset_size_wise']
  train_labels_size_wise = features_dict['train_labels_size_wise']

  for size, train_data in trainset_size_wise.items():
    print(f'INFO: training for DIM: {size}')
    trainset, train_lbls = [], []

    for i, data_3d in enumerate(train_data):
      # print(data_3d)
      lbl = train_labels_size_wise[size][i]
      # print(lbl)
      for j, data in enumerate(data_3d):
        # print(data.shape)
        trainset.append(data)
        train_lbls.append(classes_map[lbl])

    trainset = np.array(trainset)
    train_lbls = np.array(train_lbls)
    print(f'trainset shape: {trainset.shape}')
    print(f'train label shape: {train_lbls.shape}')

    model = build_model(size)
    model.fit(
        trainset,
        train_lbls,
        batch_size=5,
        epochs=EPOCHS
    )
    model.save(f'model-{str(size)}.keras')


def train_custom_fft_features(features_dict):
  """
  train model using custom fft features
  """
  # 'train_custom_features_dict': train_custom_features_dict, 'test_custom_features_dict': test_custom_features_dict
  train_custom_features_dict = features_dict['train_custom_features_dict']
  # print(train_custom_features_dict)

  trainset, train_lbls = [], []
  for label, train_data in train_custom_features_dict.items():
    print(f'INFO: training for custom fft features')
    train_data_ = train_data[0]

    for i, data in enumerate(train_data_):
      trainset.append(data)
      train_lbls.append(classes_map[label])

  trainset = np.array(trainset)
  print(label)
  train_lbls = np.array(train_lbls)
  print(f'trainset shape: {trainset.shape}')
  print(f'train label shape: {train_lbls.shape}')

  model = build_model_for_custom_fft_features()
  model.fit(
      trainset,
      train_lbls,
      batch_size=5,
      epochs=EPOCHS
  )
  model.save(f'model-custom-fft.keras')