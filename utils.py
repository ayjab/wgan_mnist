import tensorflow as tf
from keras import backend
from keras.datasets.mnist import load_data
from numpy.random import randn
from numpy.random import randint
from numpy import expand_dims, ones

def wasserstein_loss(y_true, y_pred):
    return -tf.keras.backend.mean(y_true * y_pred)

class CenterAround(tf.keras.constraints.Constraint):
  def __init__(self, ref_value):
    self.ref_value = ref_value

  def __call__(self, w):
    return backend.clip(w, -self.ref_value, self.ref_value)

  def get_config(self):
    return {'ref_value': self.ref_value}

def load_real_samples():
	(trainX, trainy), (_, _) = load_data()
	selected_ix = trainy == 7
	X = trainX[selected_ix]
	X = expand_dims(X, axis=-1)
	X = X.astype('float32')
	X = (X - 127.5) / 127.5
	return X

def generate_real_samples(dataset, n_samples):
	ix = randint(0, dataset.shape[0], n_samples)
	X = dataset[ix]
	y = -ones((n_samples, 1))
	return X, y

def generate_latent_points(latent_dim, n_samples):
	x_input = randn(latent_dim * n_samples)
	x_input = x_input.reshape(n_samples, latent_dim)
	return x_input

def generate_fake_samples(generator, latent_dim, n_samples):
	x_input = generate_latent_points(latent_dim, n_samples)
	X = generator.predict(x_input)
	y = ones((n_samples, 1))
	return X, y