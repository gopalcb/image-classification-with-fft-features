import tensorflow as tf
import keras
import numpy as np
from skimage import io


def inference_with_fft_magnitude(input_path):
  """
  make inference using the models that are trained with fft magnitude of different shapes
  """

  # initialize the model for making inference
  print(f'INFO: making inference - input path: {input_path}')
  size = (256, 256)
  grip_len = 256*256

  print(f'INFO: loading model for data shape: {size}')
  model = keras.models.load_model(f'model-{str(size)}.keras')
  probability_model = tf.keras.Sequential(
      [model, tf.keras.layers.Softmax()]
  )
  print('INFO: model loaded!')

  # read image data
  print(f'INFO: reading input image and computing fourier coefficients')
  img = io.imread(input_path, as_gray=True)
  # compute fourier coefficients
  fourier_coefficients = np.fft.fftshift(np.fft.fft2(img))
  magnitudes_np = abs(fourier_coefficients)

  # flatten coeffients and magnitude 2D array
  magnitudes_np_flatten = magnitudes_np.flatten()
  fourier_coefficients_flatten = fourier_coefficients.flatten()

  # apply arg sort
  sorted_indices = np.argsort(magnitudes_np_flatten)[::-1]
  magnitudes_np_flatten = magnitudes_np_flatten[sorted_indices]
  fourier_coefficients_flatten = fourier_coefficients_flatten[sorted_indices]

  # grip first grip_len elements
  print(f'INFO: grip first {grip_len} elements')
  magnitudes_np_grip = magnitudes_np_flatten[0:grip_len]
  fourier_coefficients_grip = fourier_coefficients_flatten[0:grip_len]

  # reshape
  magnitude_np_reshape = magnitudes_np_grip.reshape(size)
  magnitude_np_reshape = np.array([magnitude_np_reshape])
  fourier_coefficients_reshape = fourier_coefficients_grip.reshape(size)
  fourier_coefficients_reshape = np.array([fourier_coefficients_reshape])
  print(magnitudes_np_grip.shape)

  # inference
  print(f'INFO: generating prediction matrix')
  predictions = probability_model.predict(magnitude_np_reshape)
  return predictions



def inference_with_custom_fft_features(input_path):
  """
  inference data using the model that is trained with custom fft features
  """

  # initialize the model for making inference
  print(f'INFO: making inference - input path: {input_path}')
  size = (64, 64)
  grip_len = 64*64

  print(f'INFO: loading model for data shape: {size}')
  model = keras.models.load_model(f'model-custom-fft.keras')
    probability_model = tf.keras.Sequential(
        [model, tf.keras.layers.Softmax()]
    )
  print('INFO: model loaded!')

  # read image data
  print(f'INFO: reading input image and computing fourier coefficients')
  img = io.imread(input_path, as_gray=True)
  # compute fourier coefficients
  fourier_coefficients = np.fft.fftshift(np.fft.fft2(img))
  magnitudes_np = abs(fourier_coefficients)

  # flatten coeffients and magnitude 2D array
  magnitudes_np_flatten = magnitudes_np.flatten()
  fourier_coefficients_flatten = fourier_coefficients.flatten()

  # apply arg sort
  sorted_indices = np.argsort(magnitudes_np_flatten)[::-1]
  magnitudes_np_flatten = magnitudes_np_flatten[sorted_indices]
  fourier_coefficients_flatten = fourier_coefficients_flatten[sorted_indices]

  # grip first grip_len elements
  print(f'INFO: grip first {grip_len} elements')
  magnitudes_np_grip = magnitudes_np_flatten[0:grip_len]
  fourier_coefficients_grip = fourier_coefficients_flatten[0:grip_len]

  # compute custom fft features
  phase_angles, sectors, distances = [], [], []
  for j, mag in enumerate(magnitudes_np_grip):
    # compute phase angle feature
    z = fourier_coefficients_grip[j]
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

  custom_feature = np.array([magnitudes_np_grip, phase_angles, sectors, distances])

  # reshape
  custom_feature = np.array([custom_feature])
  print(custom_feature.shape)

  # inference
  print(f'INFO: generating prediction matrix')
  predictions = probability_model.predict(custom_feature)
  return predictions


def extract_results_from_prediction(input_path):
  """
  get predicted class label
  """
  classes_map = {
      'tiger': 0,
      'deer': 1,
      'horse': 2,
      'squirrel': 3
  }

  # predictions = inference_with_custom_fft_features('data/deer-1.jpg')
  predictions = inference_with_custom_fft_features(input_path)
  # print(predictions)
  label = ''
  for i, pred in enumerate(predictions[0]):
    if pred == 1:
      label = list(classes_map.keys()) [list(classes_map.values()).index(i)]

  return label


# extract_results_from_prediction()
