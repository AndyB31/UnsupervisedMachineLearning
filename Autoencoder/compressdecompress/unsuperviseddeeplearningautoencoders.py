# -*- coding: utf-8 -*-
"""UnsupervisedDeepLearningAutoEncoders.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1EyRJVLQFMlOikThE_sy2ve-aChTfss5j
"""

import tensorflow as tf  
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

(x_train, y_train),(x_test, y_test) = tf.keras.datasets.mnist.load_data()  # loads the popular "mnist" training dataset
x_train = x_train/255.0  # scales the data. pixel values range from 0 to 255, so this makes it range 0 to 1
x_test = x_test/255.0  # scales the data. pixel values range from 0 to 255, so this makes it range 0 to 1

plt.imshow(x_train[0], cmap="gray")

def autoencoder_create():
  #encoderpart
  encoder_input = keras.Input(shape=(28, 28, 1), name='img')
  x = keras.layers.Flatten()(encoder_input)
  encoder_output = keras.layers.Dense(64, activation="relu")(x)
  encoder = keras.Model(encoder_input, encoder_output, name='encoder')
  #decoderpart
  decoder_input = keras.layers.Dense(64, activation="relu")(encoder_output)
  x = keras.layers.Dense(784, activation="sigmoid")(decoder_input)
  decoder_output = keras.layers.Reshape((28, 28, 1))(x)
  lr_schedule = keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=1e-2,
    decay_steps=1000000000,
    decay_rate=1e-6)
  opt = tf.keras.optimizers.Adam(lr_schedule)
  return keras.Model(encoder_input, decoder_output, name='autoencoder'), opt, encoder

autoencoder,opt, encoder = autoencoder_create()

autoencoder.summary()

epochs = 5

autoencoder.compile(opt, loss='mse')
for epoch in range(epochs):

  history = autoencoder.fit(
    x_train,
    x_train,
    epochs=1, 
    batch_size=32, validation_split=0.10
      )   
  autoencoder.save(f"models/AE-{epoch+1}.model")

example = encoder.predict([ x_test[0].reshape(-1, 28, 28, 1) ])

print(example[0].shape)
print(example[0])

plt.imshow(example[0].reshape((8,8)), cmap="gray")

plt.imshow(x_test[0], cmap="gray")

ae_out = autoencoder.predict([ x_test[0].reshape(-1, 28, 28, 1) ])
img = ae_out[0]  # predict is done on a vector, and returns a vector, even if its just 1 element, so we still need to grab the 0th
plt.imshow(ae_out[0], cmap="gray")