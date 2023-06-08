import tensorflow as tf
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

def build_generator(latent_dim):
    inp = tf.keras.layers.Input(latent_dim)
    layer = tf.keras.layers.Dense(256, activation='leaky_relu')(inp)
    layer = tf.keras.layers.Dense(512, activation='leaky_relu')(layer)
    layer = tf.keras.layers.Dense(1024, activation='leaky_relu')(layer)
    layer = tf.keras.layers.Dense(784, activation='tanh')(layer)
    out = tf.keras.layers.Reshape((28, 28, 1))(layer)

    model = tf.keras.Model(inp, out)
    return model

# Définition du discriminateur
def build_discriminator():
    inp = tf.keras.layers.Input((28,28,1))
    layer = tf.keras.layers.Flatten()(inp)
    layer = tf.keras.layers.Dropout(0.4)(layer)
    layer = tf.keras.layers.Dense(1024, activation='leaky_relu')(layer)
    layer = tf.keras.layers.Dropout(0.4)(layer)
    layer = tf.keras.layers.Dense(512, activation='tanh')(layer)
    layer = tf.keras.layers.Dropout(0.4)(layer)
    layer = tf.keras.layers.Dense(512, activation='tanh')(layer)
    out = tf.keras.layers.Dense(1, activation='sigmoid')(layer)

    model = tf.keras.Model(inp,out)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001, beta_1=0.5), loss="binary_crossentropy")#, metrcis=["accuracy"])

    return model

def GAN(gen, disc):
    disc.trainable = False

    gan = tf.keras.Sequential()
    gan.add(gen)
    gan.add(disc)

    gan.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001, beta_1=0.5), loss="binary_crossentropy")

    return gan

latent_dim = 100

discriminator = build_discriminator()
generator = build_generator(latent_dim)
gan = GAN(generator, discriminator)


(train_images, train_labels), (_, _) = tf.keras.datasets.mnist.load_data()
train_images = train_images.astype('float32')
# Calcul de la moyenne et de l'écart type
mean = np.mean(train_images)
std = np.std(train_images)

# Standardisation du jeu de données
x_train = (train_images - mean) / std



batch_size = 128
half_batch = 64
epochs = 100
loss = []

for epoch in range(epochs):
    print("EPOCH ", epoch)
    for j in tqdm(range(len(x_train) // batch_size)):
        noise = np.random.randn(half_batch, latent_dim)

        real_images, y_real = x_train[np.random.randint(0, len(x_train), half_batch)].reshape(half_batch, 28, 28, 1), np.ones(half_batch).reshape(half_batch,1)
        fake_images, y_fake = generator.predict(noise), np.zeros(half_batch).reshape(half_batch,1)

        x, y = np.vstack((real_images, fake_images)), np.vstack((y_real, y_fake))

        disc_loss = discriminator.train_on_batch(x, y)

        batch_noise = np.random.randn(batch_size, latent_dim)
        gan_loss = gan.train_on_batch(batch_noise, np.zeros(batch_size).reshape(batch_size,1))

        loss.append([disc_loss, gan_loss])

    fig, axes = plt.subplots(5,5, figsize=(12,12))
    print("losses --> ",disc_loss, "  ", gan_loss)
    for ii in range(5):
        for jj in range(5):
            axes[ii, jj].imshow(generator.predict(np.random.randn(latent_dim).reshape(1, latent_dim)).reshape(28,28), cmap="gray")

    plt.show()
    plt.close()


