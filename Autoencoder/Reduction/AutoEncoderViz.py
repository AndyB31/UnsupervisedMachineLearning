#Autoencoder

import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

(x_train, y_train),(x_test, y_test) = tf.keras.datasets.mnist.load_data()  # loads the popular "mnist" training dataset
x_train = x_train/255.0  # scales the data. pixel values range from 0 to 255, so this makes it range 0 to 1
x_test = x_test/255.0  # scales the data. pixel values range from 0 to 255, so this makes it range 0 to 1

plt.imshow(x_train[0], cmap="gray")

def autoencoder(input_shape):
  input = keras.Input(shape=input_shape)
  f = keras.layers.Flatten()(input)
  e = keras.layers.Dense(256,activation="relu")(f)
  e = keras.layers.Dense(128,activation="relu")(e)
  lat_space = keras.layers.Dense(64, activation="relu")(e)
  d = keras.layers.Dense(128, activation="relu")(lat_space)
  d = keras.layers.Dense(256, activation="relu")(d)
  regen = keras.layers.Dense(input_shape[0] * input_shape[1], activation="sigmoid")(d)
  reshaped = keras.layers.Reshape((28,28))(regen)

  autoencoder = keras.Model(input, reshaped)
  encoder = keras.Model(input, lat_space)
  decoder = keras.Model(lat_space, regen)
  opt = keras.optimizers.Adam()
  return autoencoder, encoder, decoder, opt

input_shape = (28,28,1)
autoencoder,encoder, decoder,opt = autoencoder(input_shape)

autoencoder.compile(opt, loss='mse')
autoencoder.fit(x_train,  x_train, epochs=20,  validation_data=[x_test, x_test])