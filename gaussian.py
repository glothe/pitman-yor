import numpy as np
from jax import random
import jax.numpy as jnp

import numpyro
from numpyro.distributions import *

import matplotlib.pyplot as plt

from utils import mix_weights
from sampler import sample_posterior


def richardson_component_prior(data: jnp.ndarray):
	""" 
	Compute the parameters of the parameters of the Gaussian prior distrbution on Gaussian components,
	as described by Richardson et al. in
	https://people.maths.bris.ac.uk/~mapjg/papers/RichardsonGreenRSSB.pdf, p735

	Args:
		data (jnp.ndarray(shape=(Nsamples, dim))): input data

	Returns:
		mu_bar (float): prior component mean
		sigma2_mu (float): prior component variance 
	""" 

	mu_bar = data.mean(axis=0)
	R = np.abs(data - mu_bar).max()
	sigma2_mu = .5 * R * R

	return mu_bar, sigma2_mu

def gaussian_DPMM(data: jnp.ndarray, alpha: float = 1, T: int = 10):
	Nsamples = data.shape
	mu_bar, sigma2_mu = richardson_component_prior(data)

	with numpyro.plate("beta_plate", T-1):
		beta = numpyro.sample("beta", Beta(1, alpha))

	with numpyro.plate("component_plate", T):
		mu = numpyro.sample("mu", Normal(mu_bar, jnp.sqrt(sigma2_mu)))
		kappa = numpyro.sample("kappa", Gamma(2, sigma2_mu))
		sigma2 = numpyro.sample("sigma2", InverseGamma(.5, kappa))

	with numpyro.plate("data", Nsamples):
		z = numpyro.sample("z", Categorical(mix_weights(beta)))
		numpyro.sample("obs", Normal(mu[z], jnp.sqrt(sigma2[z])), obs=data)
		
def multivariate_gaussian_DPMM(data: jnp.ndarray, alpha: float = 1, T: int = 10):
	Nsamples, Ndim = data.shape
	mu_bar, sigma2_mu = richardson_component_prior(data)

	with numpyro.plate("beta_plate", T-1):
		beta = numpyro.sample("beta", Beta(1, alpha))

	with numpyro.plate("component_plate", T):
		mu = numpyro.sample("mu", MultivariateNormal(mu_bar, sigma2_mu * jnp.eye(Ndim)))
		kappa = numpyro.sample("kappa", Gamma(2, sigma2_mu))

		# TODO prior on component variance
		sigma2 = numpyro.sample("sigma2", InverseGamma(.5, kappa))

	with numpyro.plate("data", Nsamples):
		z = numpyro.sample("z", Categorical(mix_weights(beta)))
		
		loc = mu[z]
		cov = sigma2[z,None,None] * jnp.eye(Ndim)
		numpyro.sample("obs", MultivariateNormal(loc, cov), obs=data)

def make_gaussian_DPMM_gibbs_fn(data: jnp.ndarray):
	Nsamples, Ndim = data.shape

	def gibbs_fn(rng_key: random.PRNGKey, gibbs_sites, hmc_sites):
		beta = hmc_sites['beta']
		mu = hmc_sites['mu']
		sigma2 = hmc_sites['sigma2']

		T, = mu.shape
		assert beta.shape == (T-1,)
		assert sigma2_inv.shape == (T,)

		N, = data.shape

		log_probs = Normal(loc=mu, scale=jnp.sqrt(sigma2_inv) * jnp.eye(Ndim)).log_prob(data[:, None])
		assert log_probs.shape == (N, T)

		log_weights = jnp.log(mix_weights(beta))
		assert log_weights.shape == (T,)

		logits = log_probs + log_weights[None,:]
		assert logits.shape == (Nsamples, T)

		with numpyro.plate("z", Nsamples):
			z = CategoricalLogits(logits).sample(rng_key)
			assert z.shape == (Nsamples,)
	
		return {'z':z}
	return gibbs_fn

def make_multivariate_gaussian_DPMM_gibbs_fn(data: jnp.ndarray):
	Nsamples, Ndim = data.shape

	def gibbs_fn(rng_key: random.PRNGKey, gibbs_sites, hmc_sites):
		beta = hmc_sites['beta']
		mu = hmc_sites['mu']
		sigma2 = hmc_sites['sigma2']
		cov = sigma2[:, None, None] * jnp.eye(Ndim)

		print(cov.shape)

		# TODO not working yet
		log_probs = MultivariateNormal(mu, cov).log_prob(data[...,None])
		
		print(log_probs.shape)

		log_weights = jnp.log(mix_weights(beta))
		logits = log_probs + log_weights[None,:]

		with numpyro.plate("z", Nsamples):
			z = CategoricalLogits(logits).sample(rng_key)
	
		return {'z':z}
	return gibbs_fn