import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, Dense, Flatten, Reshape, BatchNormalization, LeakyReLU
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.optimizers import RMSprop
from utils import CenterAround, wasserstein_loss

class WGAN:
    def __init__(self, img_shape, latent_dim):
        self.img_shape = img_shape
        self.latent_dim = latent_dim

        # Build and compile the discriminator and generator models
        self.discriminator = self.build_discriminator()
        self.generator = self.build_generator()

        # Build and compile the WGAN model
        self.wgan = self.build_wgan()

    def build_discriminator(self):
    	init = RandomNormal(stddev=0.02)
    	const = CenterAround(0.01)
    	model = Sequential()
    	model.add(Conv2D(64, (4,4), strides=(2,2), padding='same', kernel_initializer=init, kernel_constraint=const, input_shape=in_shape))
    	model.add(BatchNormalization())
    	model.add(LeakyReLU(alpha=0.2))
    	model.add(Conv2D(64, (4,4), strides=(2,2), padding='same', kernel_initializer=init, kernel_constraint=const))
    	model.add(BatchNormalization())
    	model.add(LeakyReLU(alpha=0.2))
    	model.add(Flatten())
    	model.add(Dense(1))
    	opt = RMSprop(lr=0.00005)
    	model.compile(loss=wasserstein_loss, optimizer=opt)
    	return model

    def build_generator(self):
    	init = RandomNormal(stddev=0.02)
    	model = Sequential()
    	n_nodes = 128 * 7 * 7
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
    
    def build_wgan(self):
        self.discriminator.trainable = False
        z = layers.Input(shape=(self.latent_dim,))
        img = self.generator(z)
        validity = self.discriminator(img)
    	opt = RMSprop(lr=0.00005)
        validity.compile(loss=wasserstein_loss, optimizer=opt)
        return tf.keras.Model(z, validity)