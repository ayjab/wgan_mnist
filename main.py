from utils import generate_real_samples, generate_fake_samples, generate_latent_points, load_real_samples, plot_history
from numpy import ones
from matplotlib import pyplot
from statistics import mean
from wgan import WGAN

def train(g_model, c_model, gan_model, dataset, latent_dim, n_epochs=10, n_batch=64, n_critic=5):
	bat_per_epo = int(dataset.shape[0] / n_batch)
	n_steps = bat_per_epo * n_epochs
	half_batch = int(n_batch / 2)
	c1_hist, c2_hist, g_hist = list(), list(), list()
	for i in range(n_steps):
		c1_tmp, c2_tmp = list(), list()
		for _ in range(n_critic):
			X_real, y_real = generate_real_samples(dataset, half_batch)
			c_loss1 = c_model.train_on_batch(X_real, y_real)
			c1_tmp.append(c_loss1)
			X_fake, y_fake = generate_fake_samples(g_model, latent_dim, half_batch)
			c_loss2 = c_model.train_on_batch(X_fake, y_fake)
			c2_tmp.append(c_loss2)
		c1_hist.append(mean(c1_tmp))
		c2_hist.append(mean(c2_tmp))
		X_gan = generate_latent_points(latent_dim, n_batch)
		y_gan = -ones((n_batch, 1))
		g_loss = gan_model.train_on_batch(X_gan, y_gan)
		g_hist.append(g_loss)
		print('> %d, c1=%.3f, c2=%.3f g=%.3f' % (i+1, c1_hist[-1], c2_hist[-1], g_loss))
		if (i+1) % bat_per_epo == 0:
        	pyplot.plot(i, label='crit_real')
        	pyplot.plot(g_model, label='crit_fake')
        	pyplot.plot(latent_dim, label='gen')
        	pyplot.legend()
        	pyplot.savefig('plot_line_plot_loss.png')
        	pyplot.close()
	plot_history(c1_hist, c2_hist, g_hist)

dataset = load_real_samples()
latent_dim = 50
wgan = WGAN(dataset.shape, latent_dim)
discriminator = wgan.build_discriminator()
generator = wgan.build_discriminator()
wgan_model = wgan.build_wgan()

train(generator, discriminator, wgan_model, dataset, latent_dim)