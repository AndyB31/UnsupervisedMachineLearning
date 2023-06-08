import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Définition du générateur
def build_generator(latent_dim):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(256, input_dim=latent_dim, activation='leaky_relu'))
    model.add(tf.keras.layers.Dense(512, activation='leaky_relu'))
    model.add(tf.keras.layers.Dense(784, activation='linear'))
    model.add(tf.keras.layers.Reshape((28, 28, 1)))
    return model

# Définition du discriminateur
def build_discriminator():
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Flatten(input_shape=(28, 28, 1)))
    model.add(tf.keras.layers.Dense(256, activation='leaky_relu'))
    model.add(tf.keras.layers.Dense(64, activation='leaky_relu'))
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
    return model


discriminator_accuracy = tf.keras.metrics.BinaryAccuracy()
@tf.function
def train_discriminator(images, loss):
    noise = tf.random.normal([images.shape[0], latent_dim])

    with tf.GradientTape() as disc_tape:
        generated_images = generator(noise)

        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)

        disc_loss_real = loss(tf.ones_like(real_output), real_output)
        disc_loss_fake = loss(tf.zeros_like(fake_output), fake_output)
        disc_loss = disc_loss_real + disc_loss_fake
    # Calcul de l'accuracy du discriminateur
    real_accuracy = discriminator_accuracy(tf.ones_like(real_output), real_output)
    fake_accuracy = discriminator_accuracy(tf.zeros_like(fake_output), fake_output)

    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

    return disc_loss, real_accuracy, fake_accuracy

@tf.function
def train_generator(loss):
    noise = tf.random.normal([batch_size, latent_dim])

    with tf.GradientTape() as gen_tape:
        generated_images = generator(noise, training=True)
        fake_output = discriminator(generated_images, training=False)

        gen_loss = loss(tf.ones_like(fake_output), fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))

    return gen_loss

# Construction du générateur et du discriminateur
loss = tf.keras.losses.BinaryCrossentropy()
latent_dim = 100
generator = build_generator(latent_dim)
discriminator = build_discriminator()

# Initialisation des poids
generator.build(input_shape=(None, latent_dim))
discriminator.build(input_shape=(None, 28, 28, 1))
generator.summary()
discriminator.summary()
generator_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)

# Chargement et préparation du jeu de données MNIST
(train_images, train_labels), (_, _) = tf.keras.datasets.mnist.load_data()
train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')
# Calcul de la moyenne et de l'écart type
mean = np.mean(train_images)
std = np.std(train_images)

# Standardisation du jeu de données
x_train = (train_images - mean) / std
#train_images = (train_images - 127.5) / 127.5

# Entraînement du GAN
batch_size = 128
epochs = 10000



for epoch in range(epochs):
    # Création du jeu de données
    train_dataset = tf.data.Dataset.from_tensor_slices(x_train).shuffle(60000).batch(batch_size)
    for real_images in train_dataset:
        # Entraînement du discriminateur
        disc_loss, real_accuracy, fake_accuracy = train_discriminator(real_images,loss)

        # Entraînement du générateur
        for i in range(2):
            gen_loss = train_generator(loss)

    if epoch % 5 == 0:
        print(f"Epoch {epoch}/{epochs}")
        print("Losses: Generator {:.4f}, Discriminator {:.4f}".format(gen_loss, disc_loss))
        print("Accuracies: Real {:.2f}%, Fake {:.2f}%".format(real_accuracy * 100, fake_accuracy * 100))
        print()

    if epoch % 50 == 0:
        random_latent_vectors = tf.random.normal(shape=(10, latent_dim))
        generated_images = generator(random_latent_vectors, training=False)

        for i in range(generated_images.shape[0]):
            img = tf.clip_by_value(generated_images[i, :, :, 0] * std + mean, 0, 255)
            img = img.numpy().astype(np.uint8)
            plt.imshow(img, cmap='gray')
            plt.axis('off')
            plt.show()

    discriminator_accuracy.reset_states()

