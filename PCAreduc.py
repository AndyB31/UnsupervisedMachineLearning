#PCA reduction
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from mpl_toolkits.mplot3d import Axes3D

n_components = 3
# Load the MNIST dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Combine the training and testing data
X = np.concatenate((x_train, x_test), axis=0)
Y = np.concatenate((y_train, y_test), axis=0)

# Flatten the images into vectors
X = X.reshape(X.shape[0], -1)

# Convert data to float and scale it between 0 and 1
X = X.astype(float) / 255.0

# Calculate the mean vector
mean_vector = np.mean(X, axis=0)

# Subtract the mean vector from the data
X_centered = X - mean_vector

# calculating the covariance matrix of the mean-centered data.
cov_mat = np.cov(X_centered, rowvar=False)

# Calculating Eigenvalues and Eigenvectors of the covariance matrix
eigen_values, eigen_vectors = np.linalg.eigh(cov_mat)

# sort the eigenvalues in descending order
sorted_index = np.argsort(eigen_values)[::-1]

sorted_eigenvalue = eigen_values[sorted_index]
# similarly sort the eigenvectors
sorted_eigenvectors = eigen_vectors[:, sorted_index]

# select the first n eigenvectors, n is desired dimension
# of our final reduced data.
eigenvector_subset = sorted_eigenvectors[:, 0:n_components]

X_reduced = np.dot(X_centered, eigenvector_subset)

x = X_reduced[:,0]
y = X_reduced[:,1]
z = X_reduced[:,2]



# Couleurs correspondant à chaque catégorie
unique_values = np.unique(y)
count = len(unique_values)
colors = list(mcolors.TABLEAU_COLORS.values())[:count]

# Création du scatter plot en 3D avec des couleurs différentes pour chaque catégorie
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x, y, z, c=Y, cmap=plt.cm.get_cmap('viridis', len(colors)), s=1)

# Affichage du plot
plt.show(interactive=True)
