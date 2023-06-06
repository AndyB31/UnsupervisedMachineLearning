import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import matplotlib.colors as mcolors
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
#X = tf.constant(x_train2)


k = 30

# Exécution de l'algorithme des k-means
centers, assignments = kmeans(x_train2, k, num_iterations=100)

# Récupération des résultats
print("Centers:")
print(centers)

print("Assignments:")
print(assignments)

#=============================

# Exemple de données de clusters prédits et de classes réelles
clusters_predits = assignments
classes_reelles = y_train

# Compter les occurrences de chaque paire (cluster, classe)
occurrences = {}
for cluster, classe in zip(clusters_predits, classes_reelles):
    key = (cluster, classe)
    occurrences[key] = occurrences.get(key, 0) + 1

# Préparer les données pour le bar plot
categories = list(set(clusters_predits))
valeurs = [occurrences.get((cluster, classe), 0) for cluster in categories for classe in np.arange(k)]

# Générer une liste de couleurs pour chaque classe
couleurs_classes = list(mcolors.TABLEAU_COLORS.values())[:k]  # Génère k couleurs

# Tracer le bar plot avec les couleurs des classes
plt.bar(range(len(valeurs)), valeurs, color=couleurs_classes)

# Étiquettes des axes et titre
plt.xlabel('Cluster')
plt.ylabel('Occurrences')
plt.title('Correspondance entre clusters prédits et classes réelles')


# Ajouter des barres verticales toutes les 10 itérations
for i in range(0, len(valeurs), 9):
    plt.axvline(i, color='red', linestyle='--')

# Afficher le bar plot
plt.show()


