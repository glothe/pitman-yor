import numpy as np
from jax import random

from numpyro.distributions import Normal, Poisson

import matplotlib.pyplot as plt

from sampler import sample_posterior
from gaussian import gaussian_DPMM, make_gaussian_DPMM_gibbs_fn
from poisson import poisson_DPMM, make_poisson_DPMM_gibbs_fn


# Gaussian DPMM experiment
def sample_data_miller(rng_key: random.PRNGKey, N: int) -> np.ndarray:
	"""
	Sample N points from a 1D 3-component gaussian mixture with parameters from Miller et al:
	"Inconsistency of Pitman–Yor Process Mixtures for the Number of Components"

	Returns:
		data (np.ndarray(shape=(N), dtype=float))

	"""
	data = np.hstack((
		Normal(4, 1).sample(rng_key, (int(.45 * N),)),
		Normal(6, .2).sample(rng_key, (int(.3 * N),)),
		Normal(8, .6).sample(rng_key, (N - int(.3*N) - int(.45*N),))
	))
	return data[:, None]

def main_gaussian_DPMM():
	Nsamples = 1000
	T = 10
	t = np.arange(T+1)
	rng_key = random.PRNGKey(0)
	repeat = 1

	for Npoints in (200, 400, 500):
		y = np.zeros(T+1)
		for _ in range(repeat):
			data = sample_data_miller(rng_key, Npoints)
			y += sample_posterior(rng_key, gaussian_DPMM, data, Nsamples, T=T, alpha=1,
				# Uncomment the line below to use HMCGibbs
				#	gibbs_fn=make_gaussian_DPMM_gibbs_fn(data), gibbs_sites=['z'],
				)
		y /= repeat
		plt.plot(t, y, label=f"N={Npoints}")
		plt.scatter(t, y)

	plt.legend()
	plt.show()


# Poisson DPMM experiment
def sample_data_poisson(rng_key: random.PRNGKey, N: int):
	data = np.hstack((
		Poisson(4).sample(rng_key, (int(.45 * N),)),
		Poisson(6).sample(rng_key, (int(.3 * N),)),
		Poisson(8).sample(rng_key, (N - int(.3 * N) - int(.45 * N),))))
	return data

def main_poisson_DPMM():
	Nsamples = 1000
	T = 10
	t = np.arange(T+1)
	rng_key = random.PRNGKey(0)
	repeat = 1

	for Npoints in (200, 400, 500):
		y = np.zeros(T+1)
		for _ in range(repeat):
			data = sample_data_poisson(rng_key, Npoints)
			y += sample_posterior(rng_key, poisson_DPMM, data, Nsamples, T=T, alpha=1,
				# Uncomment the line below to use HMCGibbs
					gibbs_fn=make_poisson_DPMM_gibbs_fn(data), gibbs_sites=['z'],
				)
		y /= repeat
		plt.plot(t, y, label=f"N={Npoints}")
		plt.scatter(t, y)

	plt.legend()
	plt.show()

if __name__ == '__main__':
	main_gaussian_DPMM()
	# main_poisson_DPMM()
