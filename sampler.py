import numpy as np
import jax.numpy as jnp
import jax.random as random

import numpyro
from numpyro.infer import Predictive, MCMC, BarkerMH, NUTS, Predictive, HMCGibbs, DiscreteHMCGibbs


NUM_WARMUP = 1000


def sample_posterior_with_predictive(
        rng_key: random.PRNGKey,
        model,
        data: np.ndarray,
        Nsamples: int = 1000,
        alpha: float = 1,
        sigma: float = 0,
        T: int = 10):

    kernel = NUTS(model)
    mcmc = MCMC(kernel, num_samples=Nsamples, num_warmup=NUM_WARMUP)

    mcmc.run(rng_key, data=data, alpha=alpha, sigma=sigma, T=T)
    samples = mcmc.get_samples()

    predictive = Predictive(model, posterior_samples=samples, return_sites=["z"])
    return predictive(rng_key, data)["z"]

def sample_posterior_gibbs(
        rng_key: random.PRNGKey,
        model,
        data: np.ndarray,
        Nsamples: int = 1000,
        alpha: float = 1,
        sigma: float = 0,
        T: int = 10,
        gibbs_fn=None,
        gibbs_sites=None):
    assert gibbs_fn is not None
    assert gibbs_sites is not None

    Npoints = len(data)

    inner_kernel = NUTS(model)
    kernel = HMCGibbs(inner_kernel, gibbs_fn=gibbs_fn, gibbs_sites=gibbs_sites)
    mcmc = MCMC(kernel, num_samples=Nsamples, num_warmup=NUM_WARMUP)
    mcmc.run(rng_key, data=data, alpha=alpha, sigma=sigma, T=T)
    samples = mcmc.get_samples()

    z = samples['z']
    assert z.shape == (Nsamples, Npoints)

    return z

def sample_posterior(
        rng_key: random.PRNGKey,
        model,
        data: np.ndarray,
        Nsamples: int = 1000,
        alpha: float = 1,
        sigma: float = 0,
        T: int = 10,
        gibbs_fn=None,
        gibbs_sites=None):
    """ 
        Sample 'Nsamples' points from the posterior distribution.

        Returns:
            z (np.ndarray(shape=(Nsamples, Npoints), dtype=int))
    """

    if gibbs_fn is None or gibbs_sites is None:
        return sample_posterior_with_predictive(rng_key, model, data, Nsamples,
                                                alpha, sigma, T)
    else:
        return sample_posterior_gibbs(rng_key, model, data, Nsamples,
                                      alpha, sigma, T,
                                      gibbs_fn, gibbs_sites)


