import tensorflow as tf
from keras import backend
from keras.datasets.mnist import load_data
from numpy.random import randn
from numpy.random import randint
from numpy import expand_dims, ones

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
    # Load the MNIST dataset
    (trainX, trainy), (_, _) = load_data()
    # Select only the images of the digit 3
    selected_ix = trainy == 3
    X = trainX[selected_ix]
    # Expand the dimensions of the images
    X = expand_dims(X, axis=-1)
    # Convert the images to float32 type
    X = X.astype('float32')
    # Normalize the images to be between -1 and 1
    X = (X - 127.5) / 127.5
    return X

# Function to generate real samples from the dataset
def generate_real_samples(dataset, n_samples):
    # Randomly select a number of images
    ix = randint(0, dataset.shape[0], n_samples)
    X = dataset[ix]
    # Assign a label of -1 to the images
    y = -ones((n_samples, 1))
    return X, y

# Function to generate latent points as input for the generator
def generate_latent_points(latent_dim, n_samples):
    # Generate random points in the latent space
    x_input = randn(latent_dim * n_samples)
    # Reshape the points to be in the correct format for the generator
    x_input = x_input.reshape(n_samples, latent_dim)
    return x_input

# Function to generate fake samples with the generator
def generate_images(generator, latent_dim, n_samples):
    # Generate points in the latent space
    x_input = generate_latent_points(latent_dim, n_samples)
    # Predict outputs
    X = generator.predict(x_input)
    # Plot the result
    for i in range(n_samples):
        pyplot.subplot(int(np.sqrt(n_samples)), int(np.sqrt(n_samples)), 1 + i)
        pyplot.axis('off')
        pyplot.imshow(X[i, :, :, 0], cmap='gray_r')
    pyplot.show()
