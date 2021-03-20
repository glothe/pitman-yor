import numpy as np
from jax import random

from numpyro.distributions import Normal, Poisson

import matplotlib.pyplot as plt

from sampler import sample_posterior
from gaussian import gaussian_DPMM, make_gaussian_DPMM_gibbs_fn
from poisson import poisson_DPMM, make_poisson_DPMM_gibbs_fn
from utils import compute_PY_prior


def make_synthetic_experiment(sample_data , model, make_gibbs_fn):
	"""
	Template for small synthetic experiments
	"""
	Nsamples = 1000
	T = 10
	t = np.arange(T+1)
	rng_key = random.PRNGKey(0)
	repeat = 1
	n_values = [200, 500, 1000, 2000]
	alpha = 1
	sigma = 0

	priors = compute_PY_prior(alpha, sigma, n_values)
	for Npoints, prior in zip(n_values, priors):
		y = np.zeros(T+1)
		for _ in range(repeat):
			data = sample_data(rng_key, Npoints)
			y += sample_posterior(rng_key, model, data, Nsamples, T=T, alpha=1,
				# Uncomment the line below to use HMCGibbs
				#	gibbs_fn=make_gibbs_fn(data), gibbs_sites=['z'],
				)
		y /= repeat
		plt.plot(t, y, label=f"N={Npoints}", marker='o')
		plt.plot(t, prior[:T+1], label=f"Prior with N={Npoints}",
				color=plt.gca().lines[-1].get_color(), linestyle='dashed')
	plt.legend()
	plt.show()


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
	return data

def main_gaussian_DPMM():
	make_synthetic_experiment(sample_data_miller, gaussian_DPMM, make_gaussian_DPMM_gibbs_fn)

# Poisson DPMM experiment
def sample_data_poisson(rng_key: random.PRNGKey, N: int):
	data = np.hstack((
		Poisson(4).sample(rng_key, (int(.45 * N),)),
		Poisson(6).sample(rng_key, (int(.3 * N),)),
		Poisson(8).sample(rng_key, (N - int(.3 * N) - int(.45 * N),))))
	return data

def main_poisson_DPMM():
	make_synthetic_experiment(sample_data_poisson, poisson_DPMM, make_poisson_DPMM_gibbs_fn)

if __name__ == '__main__':
	#main_gaussian_DPMM()
	main_poisson_DPMM()
