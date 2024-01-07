import tensorflow as tf
import keras


def build_model(input_shape):
  """
  build neural network classifier for different input shape
  """
  # print(f'nn in shape: {input_shape}')
  if input_shape == (8, 8):
    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(8, 8)),

        tf.keras.layers.Dense(64, activation='relu'),
        # tf.keras.layers.BatchNormalization(),

        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        # tf.keras.layers.BatchNormalization(),

        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        # tf.keras.layers.BatchNormalization(),

        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dropout(0.1),
        # tf.keras.layers.BatchNormalization(),

        tf.keras.layers.Dense(16, activation='relu'),
        # tf.keras.layers.BatchNormalization(),

        tf.keras.layers.Dense(4)
    ])

  if input_shape == (16, 16):
    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(16, 16)),

        tf.keras.layers.Dense(256, activation='relu'),
        # tf.keras.layers.BatchNormalization(),

        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        # tf.keras.layers.BatchNormalization(),

        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        # tf.keras.layers.BatchNormalization(),

        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        # tf.keras.layers.BatchNormalization(),

        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        # tf.keras.layers.BatchNormalization(),

        tf.keras.layers.Dense(16, activation='relu'),
        # tf.keras.layers.BatchNormalization(),

        tf.keras.layers.Dense(4)
    ])

  if input_shape == (32, 32):
    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(32, 32)),

        tf.keras.layers.Dense(1024, activation='relu'),
        # tf.keras.layers.BatchNormalization(),

        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        # tf.keras.layers.BatchNormalization(),

        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        # tf.keras.layers.BatchNormalization(),

        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        # tf.keras.layers.BatchNormalization(),

        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        # tf.keras.layers.BatchNormalization(),

        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        # tf.keras.layers.BatchNormalization(),

        tf.keras.layers.Dense(16, activation='relu'),
        # tf.keras.layers.BatchNormalization(),

        tf.keras.layers.Dense(4)
    ])

  else:
    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(input_shape)),

        tf.keras.layers.Dense(1024, activation='relu'),
        # tf.keras.layers.BatchNormalization(),

        tf.keras.layers.Dense(2048, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        # tf.keras.layers.BatchNormalization(),

        tf.keras.layers.Dense(2048, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        # tf.keras.layers.BatchNormalization(),

        tf.keras.layers.Dense(1048, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        # tf.keras.layers.BatchNormalization(),

        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        # tf.keras.layers.BatchNormalization(),

        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        # tf.keras.layers.BatchNormalization(),

        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        # tf.keras.layers.BatchNormalization(),

        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        # tf.keras.layers.BatchNormalization(),

        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        # tf.keras.layers.BatchNormalization(),

        tf.keras.layers.Dense(16, activation='relu'),
        # tf.keras.layers.BatchNormalization(),

        tf.keras.layers.Dense(4)
    ])

  model.compile(
      optimizer='adam',
      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
      metrics=['accuracy']
  )

  return model


def build_model_for_custom_fft_features():
  """
  build neural network classifier for custom fft features
  """
  model = tf.keras.Sequential([
      tf.keras.layers.Flatten(input_shape=(4, 4096)),

      tf.keras.layers.Dense(1024, activation='relu'),
      # tf.keras.layers.BatchNormalization(),

      tf.keras.layers.Dense(1024, activation='relu'),
      tf.keras.layers.Dropout(0.3),
      # tf.keras.layers.BatchNormalization(),

      tf.keras.layers.Dense(512, activation='relu'),
      tf.keras.layers.Dropout(0.2),
      # tf.keras.layers.BatchNormalization(),

      tf.keras.layers.Dense(512, activation='relu'),
      tf.keras.layers.Dropout(0.2),
      # tf.keras.layers.BatchNormalization(),

      tf.keras.layers.Dense(256, activation='relu'),
      tf.keras.layers.Dropout(0.2),
      # tf.keras.layers.BatchNormalization(),

      tf.keras.layers.Dense(128, activation='relu'),
      tf.keras.layers.Dropout(0.2),
      # tf.keras.layers.BatchNormalization(),

      tf.keras.layers.Dense(64, activation='relu'),
      tf.keras.layers.Dropout(0.2),
      # tf.keras.layers.BatchNormalization(),

      tf.keras.layers.Dense(16, activation='relu'),
      # tf.keras.layers.BatchNormalization(),

      tf.keras.layers.Dense(4)
  ])
  
  model.compile(
      optimizer='adam',
      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
      metrics=['accuracy']
  )

  return model

# build_model((8, 8)).summary()
