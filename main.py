# Import necessary functions and classes from different modules
from utils import generate_real_samples, generate_fake_samples, generate_latent_points, load_real_samples, generate_images
from numpy import ones
from matplotlib import pyplot
from statistics import mean
from wgan import WGAN
from tensorflow.keras.models import load_model

# Define the training function for the WGAN
def train(g_model, c_model, gan_model, dataset, latent_dim, n_epochs=10, n_batch=64, n_critic=5):
    # Calculate the number of batches per epoch
    bat_per_epo = int(dataset.shape[0] / n_batch)
    # Calculate the total number of training steps
    n_steps = bat_per_epo * n_epochs
    # Calculate the size of half a batch
    half_batch = int(n_batch / 2)
    # Initialize lists to store the history of losses
    c1_hist, c2_hist, g_hist = list(), list(), list()
    # Loop over the total number of training steps
    for i in range(n_steps):
        # Initialize lists to store the losses for this step
        c1_tmp, c2_tmp = list(), list()
        # Train the critic for n_critic times
        for _ in range(n_critic):
            # Generate real samples and train the critic on them
            X_real, y_real = generate_real_samples(dataset, half_batch)
            c_loss1 = c_model.train_on_batch(X_real, y_real)
            c1_tmp.append(c_loss1)
            # Generate fake samples and train the critic on them
            X_fake, y_fake = generate_fake_samples(g_model, latent_dim, half_batch)
            c_loss2 = c_model.train_on_batch(X_fake, y_fake)
            c2_tmp.append(c_loss2)
        # Store the average losses for this step
        c1_hist.append(mean(c1_tmp))
        c2_hist.append(mean(c2_tmp))
        # Generate points in the latent space as input for the generator
        X_gan = generate_latent_points(latent_dim, n_batch)
        # Create inverted labels for the fake samples
        y_gan = -ones((n_batch, 1))
        # Train the generator (through the gan model, where the critic is frozen)
        g_loss = gan_model.train_on_batch(X_gan, y_gan)
        g_hist.append(g_loss)
        # Print the progress
        print('> %d, c1=%.3f, c2=%.3f g=%.3f' % (i+1, c1_hist[-1], c2_hist[-1], g_loss))

	pyplot.plot(c1_hist, label='crit_real')
	pyplot.plot(c2_hist, label='crit_fake')
	pyplot.plot(g_hist, label='gen')
	pyplot.legend()
	pyplot.savefig('plot_line_plot_loss.png')
	pyplot.close()

# Load the real samples from the MNIST dataset
dataset = load_real_samples()
# Define the dimensionality of the latent space
latent_dim = 50
# Create a WGAN
wgan = WGAN(dataset.shape, latent_dim)
# Build the critic and the generator
discriminator = wgan.build_discriminator()
generator = wgan.build_discriminator()
# Build the combined model
wgan_model = wgan.build_wgan()

# Train the WGAN
train(generator, discriminator, wgan_model, dataset, latent_dim)

# Save the generator model for future use
generator.save('generator_model.h5')
# Save the discriminator model for future use
discriminator.save('discriminator_model.h5')

'''
# Load the generator model
generator = load_model('generator_model.h5', custom_objects={'CenterAround': CenterAround, 'wasserstein_loss': wasserstein_loss})

# Load the discriminator model
discriminator = load_model('discriminator_model.h5', custom_objects={'CenterAround': CenterAround, 'wasserstein_loss': wasserstein_loss})
'''

# Generate 25 images
generate_images(generator, latent_dim, 25)