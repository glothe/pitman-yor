import numpy as np
from jax import random
import jax.numpy as jnp

import numpyro
from numpyro.distributions import *

import matplotlib.pyplot as plt

from utils import mix_weights
from sampler import sample_posterior

def richardson_component_prior(data):
	# Component prior parameters, as described by Richardson et al
	# https://people.maths.bris.ac.uk/~mapjg/papers/RichardsonGreenRSSB.pdf
	# p735

	mu_bar = data.mean(axis=0)
	R = np.abs(data - mu_bar).max()
	sigma2_mu = .5 * R * R

	return mu_bar, sigma2_mu

def sample_data_miller(rng_key, N: int):
	data = np.hstack((
		Normal(4, 1).sample(rng_key, (int(.45 * N),)),
		Normal(6, .2).sample(rng_key, (int(.3 * N),)),
		Normal(8, .6).sample(rng_key, (N - int(.3 * N) - int(.45 * N),))))
	return data

def gaussian_DPMM(data, alpha: float = 1, T: int = 10):
	mu_bar, sigma2_mu = richardson_component_prior(data)

	with numpyro.plate("beta_plate", T-1):
		beta = numpyro.sample("beta", Beta(1, alpha))

	with numpyro.plate("component_plate", T):
		mu = numpyro.sample("mu", Normal(mu_bar, jnp.sqrt(sigma2_mu)))

		kappa = numpyro.sample("kappa", Gamma(2, sigma2_mu))
		sigma2_inv = numpyro.sample("sigma2_inv", Gamma(.5, kappa))

	with numpyro.plate("data", data.shape[0]):
		z = numpyro.sample("z", Categorical(mix_weights(beta)))
		numpyro.sample("obs", Normal(mu[z], jnp.sqrt(1/sigma2_inv[z])), obs=data)

def make_gaussian_dpmm_gibbs_fn(data):
	def gibbs_fn(rng_key, gibbs_sites, hmc_sites):
		beta = hmc_sites['beta']
		mu = hmc_sites['mu']
		sigma2_inv = hmc_sites['sigma2_inv']

		T, = mu.shape
		assert beta.shape == (T-1,)
		assert sigma2_inv.shape == (T,)

		N, = data.shape

		log_probs = Normal(loc=mu, scale=jnp.sqrt(1/sigma2_inv)).log_prob(data[:, None])
		assert log_probs.shape == (N, T)

		log_weights = jnp.log(mix_weights(beta))
		assert log_weights.shape == (T,)

		logits = log_probs + log_weights[None,:]
		assert logits.shape == (N, T)

		with numpyro.plate("z", N):
			z = CategoricalLogits(logits).sample(rng_key)
			assert z.shape == (N,)
	
		return {'z':z}
	return gibbs_fn

def main_gaussian_dpmm():
	Nsamples = 1000
	T = 10
	t = np.arange(T+1)
	y = np.zeros(T+1)
	rng_key = random.PRNGKey(0)
	repeat = 5

	for Npoints in (200, 400, 500):
		for _ in range(repeat):
			data = sample_data_miller(rng_key, Npoints)
			y += sample_posterior(rng_key, gaussian_DPMM, data, Nsamples, T=T, alpha=1,
				# Uncomment the line below to use HMCGibbs
				#	gibbs_fn=make_gaussian_dpmm_gibbs_fn(data), gibbs_sites=['z'],
				)
		y /= repeat
		plt.plot(t, y, label=f"N={Npoints}")
		plt.scatter(t, y)

	plt.legend()
	plt.show()

if __name__ == '__main__':
	main_gaussian_dpmm()
