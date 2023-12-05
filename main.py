# Import necessary functions and classes from different modules
from utils import generate_real_samples, generate_fake_samples, generate_latent_points, load_real_samples, plot_history, generate_images
from numpy import ones
from matplotlib import pyplot
from statistics import mean
from wgan import WGAN

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
        # If at the end of an epoch, plot the losses
        if (i+1) % bat_per_epo == 0:
            pyplot.plot(i, label='crit_real')
            pyplot.plot(g_model, label='crit_fake')
            pyplot.plot(latent_dim, label='gen')
            pyplot.legend()
            pyplot.savefig('plot_line_plot_loss.png')
            pyplot.close()
    # Plot the history of losses
    plot_history(c1_hist, c2_hist, g_hist)

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

# Generate 25 images
generate_images(generator, latent_dim, 25)