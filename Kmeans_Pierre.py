import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd

def kmeans(X, k, num_iterations):
    num_samples, num_features = X.shape

    initial_centers = X[np.random.choice(num_samples, k, replace=False)]

    for _ in tqdm(range(num_iterations)):
        distances = np.linalg.norm(X[:, np.newaxis] - initial_centers, axis=-1)

        classes = np.argmin(distances, axis=-1)

        # Mise à jour des centres de cluster
        new_centers = np.array([np.mean(X[classes == i], axis=0) for i in range(k)])

        # Vérification de la convergence
        if np.allclose(initial_centers, new_centers):
            break

        initial_centers = new_centers

    return initial_centers, classes

def kmeans2(X, k, num_iterations):
    initial_centers = tf.random.shuffle(X)[:k]

    # Création des variables pour les centres de cluster
    centers = tf.Variable(initial_centers)

    # Création du graphe TensorFlow
    for _ in tqdm(range(num_iterations)):
        expanded_centers = tf.expand_dims(centers, axis=0)
        expanded_points = tf.expand_dims(X, axis=1)

        distances = tf.reduce_sum(tf.square(expanded_points - expanded_centers), axis=-1)
        assignments = tf.argmin(distances, axis=-1)

        mask = tf.one_hot(assignments, depth=k)
        mask_expanded = tf.expand_dims(mask, axis=-2)

        centers = tf.reduce_sum(tf.expand_dims(X, axis=-1) * mask_expanded, axis=1) / tf.reduce_sum(mask_expanded, axis=1)

    return centers, assignments



(x_train,y_train),(x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train2 = np.array([x.flatten() for x in x_train])
X = tf.constant(x_train2)


k = 10

# Exécution de l'algorithme des k-means
centers, assignments = kmeans(x_train2, k, num_iterations=100)

# Récupération des résultats
print("Centers:")
print(centers)

print("Assignments:")
print(assignments)


df = pd.DataFrame({"class":assignments, "label":y_train})
df = df.groupby("class")["label"].apply(list)
for i in range(len(df)):
    vecteur = df[i]  # Obtenir le vecteur de la colonne "label"
    positions = range(len(vecteur))  # Positions des barres
    plt.bar(vecteur, positions)  # Tracer le diagramme en barres

    # Étiquettes des axes et titre
    plt.xlabel('Index')
    plt.ylabel('Valeurs')
    plt.title(f'Diagramme en barres - Classe {i}')

    # Afficher le diagramme en barres
    plt.show()


# Tracer le diagramme en barres
plt.bar(df.index, df.values)

# Étiquettes des axes et titre
plt.xlabel('Éléments')
plt.ylabel('Occurrences')
plt.title('Diagramme en barres des occurrences')

# Afficher le diagramme en barres
plt.show()





