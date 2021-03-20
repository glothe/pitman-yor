
import jax.numpy as jnp
import jax.random as random

import numpyro
from numpyro.distributions import Beta, Gamma, Categorical, Poisson, CategoricalLogits

from utils import mix_weights


def poisson_DPMM(data: jnp.ndarray, alpha: float = 1, T: int = 10):
	with numpyro.plate("beta_plate", T-1):
		beta = numpyro.sample("beta", Beta(1, alpha))

	with numpyro.plate("component_plate", T):
		rate = numpyro.sample("rate", Gamma(1, 1))

	with numpyro.plate("data", data.shape[0]):
		z = numpyro.sample("z", Categorical(mix_weights(beta)))
		numpyro.sample("obs", Poisson(rate[z]), obs=data)

def make_poisson_DPMM_gibbs_fn(data: jnp.ndarray):
	def gibbs_fn(rng_key: random.PRNGKey, gibbs_sites, hmc_sites):
		rate = hmc_sites['rate']
		beta = hmc_sites['beta']

		T, = rate.shape
		assert beta.shape == (T-1,)

		N, = data.shape

		log_probs = Poisson(rate).log_prob(data[:, None])
		assert log_probs.shape == (N, T)

		log_weights = jnp.log(mix_weights(beta))
		assert log_weights.shape == (T,)

		logits = log_probs + log_weights[None,:]
		assert logits.shape == (N, T)

		z = CategoricalLogits(logits).sample(rng_key)
		assert z.shape == (N,)
	
		return {'z':z}
	return gibbs_fn