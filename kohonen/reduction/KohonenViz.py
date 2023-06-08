import numpy as np
from numpy.ma.core import ceil
from scipy.spatial import distance #distance calculation
from sklearn.preprocessing import MinMaxScaler #normalisation
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score #scoring
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from matplotlib import animation, colors
from tqdm import tqdm
import tensorflow as tf
tf.keras.datasets.mnist.load_data(path="mnist.npz")

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Data Normalisation
def minmax_scaler(data):
  scaler = MinMaxScaler()
  scaled = scaler.fit_transform(data)
  return scaled

# Euclidean distance
def e_distance(x,y):
  return distance.euclidean(x,y)

# Manhattan distance
def m_distance(x,y):
  return distance.cityblock(x,y)

# Best Matching Unit search
def winning_neuron(data, t, som, num_rows, num_cols):
  winner = [0,0]
  shortest_distance = np.sqrt(data.shape[1]) # initialise with max distance
  input_data = data[t]
  for row in range(num_rows):
    for col in range(num_cols):
      distance = e_distance(som[row][col], data[t])
      if distance < shortest_distance:
        shortest_distance = distance
        winner = [row,col]
  return winner

# Learning rate and neighbourhood range calculation
def decay(step, max_steps,max_learning_rate,max_m_dsitance):
  coefficient = 1.0 - (np.float64(step)/max_steps)
  learning_rate = coefficient*max_learning_rate
  neighbourhood_range = ceil(coefficient * max_m_dsitance)
  return learning_rate, neighbourhood_range

def display_mnist(xs, row: int = None, col: int = None, width: int = None):
    if row:
        if not width:
            width = 3 * (row + 1) if row else 12
        fig, axes = plt.subplots(row + 1, col, figsize=(12, width))
    else:
        fig, axes = plt.subplots((xs.shape[0] // 4) + 1, 5, figsize=(12, 3*((xs.shape[0] // 4) + 1)))
    axes = axes.flatten()

    for i in range(len(xs)):
        axes[i].imshow(xs[i], cmap='gray')
        axes[i].axis('off')

    plt.tight_layout()
    plt.show()
class KohonenMaps:
  def __init__(self, rows: int = 10, cols: int = 10, max_m_distance: int = 4, max_learning_rate: int = 0.5, steps: int = int(1000)):
    self.rows = rows
    self.cols = cols
    self.mx_m_dist = max_m_distance
    self.mx_lr = max_learning_rate
    self.steps = steps

  def fit(self, X, random_state: int = 42, Y = None, show_step:bool = False):
    self.x_norm = minmax_scaler(X)
    np.random.seed(random_state)
    self.kmap = np.random.random_sample(size=(self.rows, self.cols, self.x_norm.shape[1]))

    # start training iterations
    for step in tqdm(range(self.steps)):
      if (step + 1) % 200 == 0:
        if show_step:
          self.show(step=step, img = True)
        else:
          print("Iteration: ", step + 1) # print out the current iteration for every 1k
      learning_rate, neighbourhood_range = decay(step, self.steps, self.mx_lr, self.mx_m_dist)

      t = np.random.randint(0, high = self.x_norm.shape[0]) # random index of traing data
      winner = winning_neuron(self.x_norm, t, self.kmap, self.rows, self.cols)
      for row in range(self.rows):
        for col in range(self.cols):
          if m_distance([row, col], winner) <= neighbourhood_range:
            self.kmap[row][col] += learning_rate * (self.x_norm[t] - self.kmap[row][col]) #update neighbour's weight

    print("Kmap training completed")

  def show(self, y = None, step:int = None, img: bool = False):
    if img:
      fig, axes = plt.subplots(self.rows, self.cols, figsize=(15, 15))
      for r in range(self.rows):
        for c in range(self.cols):
          axes[r, c].imshow(self.kmap[r, c].reshape(28, 28))
          axes[r, c].axis('off')
      plt.tight_layout()
      plt.show()
      return
x_rs = x_train.reshape(x_train.shape[0], x_train.shape[1] * x_train.shape[2])

som = KohonenMaps(25, 25, 6)

som.fit(x_rs, show_step=True)