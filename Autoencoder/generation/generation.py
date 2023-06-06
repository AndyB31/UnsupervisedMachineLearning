#Autoencoder

import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


(x_train, y_train),(x_test, y_test) = tf.keras.datasets.mnist.load_data()  # loads the popular "mnist" training dataset
x_train = x_train/255.0  # scales the data. pixel values range from 0 to 255, so this makes it range 0 to 1
x_test = x_test/255.0  # scales the data. pixel values range from 0 to 255, so this makes it range 0 to 1

plt.imshow(x_train[0], cmap="gray")

def autoencoder(input_shape):
  input = keras.Input(shape=input_shape)
  f = keras.layers.Flatten()(input)
  e = keras.layers.Dense(256,activation="sigmoid")(f)
  e = keras.layers.Dense(128,activation="sigmoid")(e)
  lat_space = keras.layers.Dense(2, activation="sigmoid")(e)
  d = keras.layers.Dense(128, activation="sigmoid")(lat_space)
  d = keras.layers.Dense(256, activation="sigmoid")(d)
  regen = keras.layers.Dense(input_shape[0] * input_shape[1], activation="sigmoid")(d)
  reshaped = keras.layers.Reshape((28,28))(regen)

  autoencoder = keras.Model(input, reshaped)
  encoder = keras.Model(input, lat_space)
  decoder = keras.Model(lat_space, reshaped)
  opt = keras.optimizers.Adam()
  return autoencoder, encoder, decoder, opt

input_shape = (28,28)
autoencoder,encoder, decoder,opt = autoencoder(input_shape)

autoencoder.compile(opt, loss='mse')
autoencoder.fit(x_train,  x_train, epochs=10,  validation_data=[x_test, x_test])

encoded = np.array(encoder(x_test))
#3D plot
x = encoded[:,0]
y = encoded[:,1]
#z = encoded[:,2]



# Couleurs correspondant à chaque catégorie
unique_values = np.unique(y)
count = len(unique_values)
colors = list(mcolors.TABLEAU_COLORS.values())[:count]

# Création du scatter plot en 3D avec des couleurs différentes pour chaque catégorie
fig = plt.figure()
ax = fig.add_subplot(111)#, projection='3d')
ax.scatter(x, y, c=y_test, cmap=plt.cm.get_cmap('viridis', len(colors)), s=1)

# Affichage du plot
plt.show(interactive=True)



def generate_images(decoder):
  images = []
  for i in np.arange(0, 1, 0.1):
    for j in np.arange(0, 1, 0.1):
      latent_vector = np.array([[i, j]])
      generated_image = decoder.predict(latent_vector)
      images.append(generated_image.reshape(28, 28))

  fig, axes = plt.subplots(10, 10, figsize=(10, 10))
  fig.subplots_adjust(hspace=0.5, wspace=0.5)

  for i, ax in enumerate(axes.flat):
    ax.imshow(images[i], cmap='gray')
    ax.axis('off')
    ax.set_title(f"i={i / 10:.1f}, j={i % 10 / 10:.1f}")

  plt.show()

generate_images(decoder)
