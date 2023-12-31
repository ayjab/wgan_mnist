import tensorflow as tf
from keras import backend
from keras.datasets.mnist import load_data
from numpy.random import randn
from numpy.random import randint
from numpy import expand_dims, ones
import matplotlib.pyplot as plt
import numpy as np

# Define the Wasserstein loss function
def wasserstein_loss(y_true, y_pred):
    return -tf.keras.backend.mean(y_true * y_pred)

# Define the CenterAround class that centers the weights around a reference value
class CenterAround(tf.keras.constraints.Constraint):
  def __init__(self, ref_value):
    self.ref_value = ref_value

  def __call__(self, w):
    return backend.clip(w, -self.ref_value, self.ref_value)

  def get_config(self):
    return {'ref_value': self.ref_value}

# Function to load real samples from the MNIST dataset
def load_real_samples():
    (trainX, trainy), (_, _) = load_data()
    # Chose which number to use for training
    selected_ix = trainy == 3
    X = trainX[selected_ix]
    X = expand_dims(X, axis=-1)
    X = X.astype('float32')
    X = (X - 127.5) / 127.5
    return X

# Function to generate real samples from the dataset
def generate_real_samples(dataset, n_samples):
    ix = randint(0, dataset.shape[0], n_samples)
    X = dataset[ix]
    y = -ones((n_samples, 1))
    return X, y

# Function to generate latent points as input for the generator
def generate_latent_points(latent_dim, n_samples):
    x_input = randn(latent_dim * n_samples)
    x_input = x_input.reshape(n_samples, latent_dim)
    return x_input

# Function to plot generated images using the generator
def generate_images(generator, latent_dim, n_samples):
    x_input = generate_latent_points(latent_dim, n_samples)
    X = generator.predict(x_input)
    for i in range(n_samples):
        plt.subplot(int(np.sqrt(n_samples)), int(np.sqrt(n_samples)), 1 + i)
        plt.axis('off')
        plt.imshow(X[i, :, :, 0], cmap='gray_r')
    plt.show()

# use the generator to generate n fake examples, with class labels
def generate_fake_samples(generator, latent_dim, n_samples):
    x_input = generate_latent_points(latent_dim, n_samples)
    X = generator.predict(x_input)
    y = ones((n_samples, 1))
    return X, y