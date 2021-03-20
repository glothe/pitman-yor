import numpy as np
import jax.numpy as jnp
import jax.random as random

import numpyro
from numpyro.infer import Predictive, MCMC, BarkerMH, NUTS, Predictive, HMCGibbs, DiscreteHMCGibbs


def compute_n_clusters_distribution(z: np.ndarray, T: int) -> np.ndarray:
	""" 
	Compute the empirical distribution of the number of clusters from sampled data.

	Args:
		z (np.ndarray(shape=(Nsamples, Npoints), dtype=int)): cluster assignments
		T (int) : maximum number of cluster

	Returns:
		Normalized histogram of cluster size (np.ndarray(shape=(T+1), dtype=float)) 
	"""

	Nsamples, _ = z.shape
	counts = np.zeros(shape=Nsamples, dtype=int)
	for i, A in enumerate(z):
		counts[i] = len(np.unique(A))

	nclusters, cluster_counts = np.unique(counts, return_counts=True)

	y = np.zeros(T+1)
	y[nclusters] = cluster_counts / np.sum(cluster_counts)
	return y

def sample_posterior_with_predictive(
		rng_key: random.PRNGKey,
		model,
		data: np.ndarray,
		Nsamples: int = 1000,
		alpha: float = 1,
		T: int = 10):

	kernel = NUTS(model)
	mcmc = MCMC(kernel, num_samples=Nsamples, num_warmup=500)

	mcmc.run(rng_key, data=data, alpha=alpha, T=T)
	samples = mcmc.get_samples()

	predictive = Predictive(model, posterior_samples=samples, return_sites=["z"])
	z = predictive(rng_key, data)["z"]
	return compute_n_clusters_distribution(z, T)

def sample_posterior_gibbs(
		rng_key: random.PRNGKey,
		model,
		data: np.ndarray,
		Nsamples: int = 1000,
		alpha: float = 1,
		T: int = 10,
		gibbs_fn=None,
		gibbs_sites=None):
	assert gibbs_fn is not None
	assert gibbs_sites is not None

	Npoints = len(data)

	inner_kernel = NUTS(model)
	kernel = HMCGibbs(inner_kernel, gibbs_fn=gibbs_fn, gibbs_sites=gibbs_sites)
	mcmc = MCMC(kernel, num_samples=Nsamples, num_warmup=Nsamples)
	mcmc.run(rng_key, data=data, alpha=alpha, T=T)
	samples = mcmc.get_samples()

	z = samples['z']
	assert z.shape == (Nsamples, Npoints)

	return compute_n_clusters_distribution(z, T)

def sample_posterior(
		rng_key: random.PRNGKey,
		model,
		data: np.ndarray,
		Nsamples: int = 1000,
		alpha: float = 1,
		T: int = 10,
		gibbs_fn=None,
		gibbs_sites=None):

	if gibbs_fn is None or gibbs_sites is None:
		return sample_posterior_with_predictive(rng_key, model, data, Nsamples, alpha, T)
	else:
		return sample_posterior_gibbs(rng_key, model, data, Nsamples, alpha, T, gibbs_fn, gibbs_sites)


