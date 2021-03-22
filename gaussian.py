import numpy as np
from jax import random
import jax.numpy as jnp

import numpyro
from numpyro.distributions import *

import matplotlib.pyplot as plt

from typing import Callable, Dict

from utils import mix_weights, sample_beta_DP, sample_beta_PY
from sampler import sample_posterior


def richardson_component_prior(data: jnp.ndarray):
    """ 
    Compute the parameters of the parameters of the Gaussian prior distrbution on Gaussian components,
    as described by Richardson et al. in
    https://people.maths.bris.ac.uk/~mapjg/papers/RichardsonGreenRSSB.pdf, p735

    Args:
        data (jnp.ndarray(shape=(Npoints, dim))): input data

    Returns:
        mu_bar (float): prior component mean
        sigma2_mu (float): prior component variance 
    """ 

    mu_bar = data.mean(axis=0)
    R = np.abs(data - mu_bar).max()
    sigma2_mu = .5 * R * R

    return mu_bar, sigma2_mu

def gaussian_DPMM(data: jnp.ndarray, alpha: float = 1, T: int = 10):
    Npoints, = data.shape
    mu_bar, sigma2_mu = richardson_component_prior(data)

    beta = sample_beta_PY(alpha=alpha, T=T)

    with numpyro.plate("component_plate", T):
        mu = numpyro.sample("mu", Normal(mu_bar, jnp.sqrt(sigma2_mu)))
        kappa = numpyro.sample("kappa", Gamma(2, sigma2_mu))
        sigma2 = numpyro.sample("sigma2", InverseGamma(.5, kappa))

    with numpyro.plate("data", Npoints):
        z = numpyro.sample("z", Categorical(mix_weights(beta)))
        numpyro.sample("obs", Normal(mu[z], jnp.sqrt(sigma2[z])), obs=data)
        
def multivariate_gaussian_DPMM(data: jnp.ndarray, alpha: float = 1, sigma: float = 0, T: int = 10):
    Npoints, Ndim = data.shape
    mu_bar, sigma2_mu = richardson_component_prior(data)

    beta = sample_beta_PY(alpha=alpha, sigma=sigma, T=T)

    with numpyro.plate("component_plate", T):
        mu = numpyro.sample("mu", MultivariateNormal(mu_bar, sigma2_mu * jnp.eye(Ndim)))

        # http://pyro.ai/examples/lkj.html
        with numpyro.plate("dim", Ndim):
            theta = numpyro.sample("theta", HalfCauchy(1))
        L_omega = numpyro.sample("L_omega", LKJCholesky(Ndim, 1))
        L_Omega = jnp.sqrt(theta.T[:,:, None]) * L_omega

    with numpyro.plate("data", Npoints):
        z = numpyro.sample("z", Categorical(mix_weights(beta)))

        assert mu.shape == (T, Ndim)
        assert L_Omega.shape == (T, Ndim, Ndim)

        numpyro.sample("obs", MultivariateNormal(mu[z], scale_tril=L_Omega[z]), obs=data)

def multivariate_gaussian_DPMM_isotropic(data: jnp.ndarray, alpha: float = 1, sigma: float = 0, T: int = 10):
    Npoints, Ndim = data.shape

    mu_bar, sigma2_mu = richardson_component_prior(data)
    assert mu_bar.shape == (Ndim,)
    assert isinstance(sigma2_mu, float)

    beta = sample_beta_PY(alpha=alpha, sigma=sigma, T=T)
    assert beta.shape == (T-1,)

    with numpyro.plate("component_plate", T):
        mu = numpyro.sample("mu", MultivariateNormal(mu_bar, sigma2_mu*np.eye(Ndim)))
        assert mu.shape == (T, Ndim), (mu.shape, T, Ndim)

        kappa = numpyro.sample("kappa", Gamma(2, sigma2_mu))
        assert kappa.shape == (T,), (kappa.shape, T)

        # This line seems to make everything fail
        sigma2 = numpyro.sample("sigma2_inv", InverseGamma(.5, kappa))

        # variances = sigma2[:, None, None] * jnp.eye(Ndim)

    with numpyro.plate("data", Npoints):
        z = numpyro.sample("z", Categorical(mix_weights(beta)))

        # TODO use the actual variance here
        numpyro.sample("obs", MultivariateNormal(mu[z], jnp.eye(Ndim)), obs=data)

def make_gaussian_DPMM_gibbs_fn(data: jnp.ndarray) -> \
    Callable[[random.PRNGKey, Dict[str, jnp.ndarray], Dict[str, jnp.ndarray]], Dict[str, jnp.ndarray]]:
    Npoints, = data.shape
    def gibbs_fn(rng_key: random.PRNGKey,
                 gibbs_sites: Dict[str, jnp.ndarray],
                 hmc_sites: Dict[str, jnp.ndarray]
                 ) -> Dict[str, jnp.ndarray]:
        beta = hmc_sites['beta']
        mu = hmc_sites['mu']
        sigma2 = hmc_sites['sigma2']

        T, = mu.shape
        assert beta.shape == (T-1,)
        assert sigma2.shape == (T,)

        log_probs = Normal(loc=mu, scale=jnp.sqrt(sigma2)).log_prob(data[:, None])
        assert log_probs.shape == (Npoints, T)

        log_weights = jnp.log(mix_weights(beta))
        assert log_weights.shape == (T,)

        logits = log_probs + log_weights[None,:]
        assert logits.shape == (Npoints, T)

        with numpyro.plate("z", Npoints):
            z = CategoricalLogits(logits).sample(rng_key)
            assert z.shape == (Npoints,)
    
        return {'z':z}
    return gibbs_fn

def make_multivariate_gaussian_DPMM_gibbs_fn(data: jnp.ndarray) -> \
    Callable[[random.PRNGKey, Dict[str, jnp.ndarray], Dict[str, jnp.ndarray]], Dict[str, jnp.ndarray]]:
    Npoints, Ndim = data.shape
    def gibbs_fn(rng_key: random.PRNGKey,
                 gibbs_sites: Dict[str, jnp.ndarray],
                 hmc_sites: Dict[str, jnp.ndarray]
                 ) -> Dict[str, jnp.ndarray]:
        beta = hmc_sites['beta']
        mu = hmc_sites['mu']
        theta = hmc_sites['theta']
        L_omega = hmc_sites['L_omega']
        L_Omega = jnp.sqrt(theta.T[:,:, None]) * L_omega

        T, _ = mu.shape

        assert beta.shape == (T-1,)
        assert mu.shape == (T, Ndim)
        assert theta.shape == (Ndim, T)
        assert L_omega.shape == (T, Ndim, Ndim)
        assert L_Omega.shape == (T, Ndim, Ndim)

        log_probs = MultivariateNormal(loc=mu, scale_tril=L_Omega).log_prob(data[:, None])
        assert log_probs.shape == (Npoints, T)

        log_weights = jnp.log(mix_weights(beta))
        assert log_weights.shape == (T,)

        logits = log_probs + log_weights[None,:]
        assert logits.shape == (Npoints, T)

        with numpyro.plate("z", Npoints):
            z = CategoricalLogits(logits).sample(rng_key)
        assert z.shape == (Npoints,)
        return {'z':z}
    return gibbs_fn
