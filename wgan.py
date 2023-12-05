# Import necessary libraries and modules
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, Dense, Flatten, Reshape, BatchNormalization, LeakyReLU
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.optimizers import RMSprop
from utils import CenterAround, wasserstein_loss

# Define the WGAN class
class WGAN:
    # Initialize the class with image shape and latent dimension
    def __init__(self, img_shape, latent_dim):
        self.img_shape = img_shape
        self.latent_dim = latent_dim

        # Build and compile the discriminator and generator models
        self.discriminator = self.build_discriminator()
        self.generator = self.build_generator()

        # Build and compile the WGAN model
        self.wgan = self.build_wgan()

    # Method to build the discriminator model
    def build_discriminator(self):
        # Initialize the weights and constraints
        init = RandomNormal(stddev=0.02)
        const = CenterAround(0.01)
        # Create a sequential model
        model = Sequential()
        # Add layers to the model
        model.add(Conv2D(64, (4,4), strides=(2,2), padding='same', kernel_initializer=init, kernel_constraint=const, input_shape=in_shape))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.2))
        model.add(Conv2D(64, (4,4), strides=(2,2), padding='same', kernel_initializer=init, kernel_constraint=const))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.2))
        model.add(Flatten())
        model.add(Dense(1))
        # Define the optimizer
        opt = RMSprop(lr=0.00005)
        # Compile the model
        model.compile(loss=wasserstein_loss, optimizer=opt)
        return model

    # Method to build the generator model
    def build_generator(self):
        # Initialize the weights
        init = RandomNormal(stddev=0.02)
        # Create a sequential model
        model = Sequential()
        # Define the number of nodes
        n_nodes = 128 * 7 * 7
        # Add layers to the model
        model.add(Dense(n_nodes, kernel_initializer=init, input_dim=latent_dim))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Reshape((7, 7, 128)))
        model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding='same', kernel_initializer=init))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.2))
        model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding='same', kernel_initializer=init))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.2))
        model.add(Conv2D(1, (7,7), activation='tanh', padding='same', kernel_initializer=init))
        return model
    
    # Method to build the WGAN model
    def build_wgan(self):
        # Set the discriminator to non-trainable
        self.discriminator.trainable = False
        # Define the input layer
        z = layers.Input(shape=(self.latent_dim,))
        # Generate an image from the generator
        img = self.generator(z)
        # Determine the validity of the image by the discriminator
        validity = self.discriminator(img)
        # Define the optimizer
        opt = RMSprop(lr=0.00005)
        # Compile the model
        validity.compile(loss=wasserstein_loss, optimizer=opt)
        # Return the model
        return tf.keras.Model(z, validity)