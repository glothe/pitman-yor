
import jax.numpy as jnp
import jax.random as random
import numpy as np

import numpyro
from numpyro.distributions import Beta, Gamma, Categorical, Poisson, CategoricalLogits

from typing import Tuple

from utils import mix_weights, sample_beta_DP, sample_beta_PY


def poisson_DPMM(data: jnp.ndarray, alpha: float = 1, sigma: float = 0, T: int = 10):
    beta = sample_beta_PY(alpha, sigma, T)

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

def constant_condition_3_PY(alpha: float, sigma: float, n: int, t: np.ndarray) -> np.ndarray:
    c = (t + 1)/(alpha + sigma * t) * (1 - (1+sigma)/n)
    return c

def explicit_upper_bound(data: np.ndarray,
                         t: np.ndarray,
                         params_PY: Tuple[float, float] = (1, 0),
                         params_prior: Tuple[float, float] = (1, 1),
                         ) -> np.ndarray:
    n, = data.shape

    bounds_R, counts = np.unique(data, return_counts=True) 
    max_R = bounds_R[-1]
    cardinals_R = np.cumsum(counts)
    assert cardinals_R.shape == bounds_R.shape
    # for R = bounds_R[i], |{x in data, x <= R}| = cardinals_R[i]

    ratios_cardinals_R = (cardinals_R[None,:] - t[:,None])/n
    valid_cardinals_R = (ratios_cardinals_R > 0)
    ratios_cardinals_R[~valid_cardinals_R] = 1
    if not valid_cardinals_R.any():
        print(f"Warning explicit_upper_bound for n = {n}: no valid R found, not enough points")
    
    # constant from condition 3
    c_condition_3 = constant_condition_3_PY(*params_PY, n, t)

    # compute the marginal m
    alpha, beta = params_prior
    ratios = (1 + (alpha - 1)/(np.arange(0, max_R) + 1)) / (1 + beta)
    assert ratios.shape == (max_R,)

    ratios_with_init = np.pad(ratios, (1, 0), mode='constant',
                              constant_values=beta**alpha/(1+beta)**alpha)
    assert ratios_with_init.shape == (max_R+1,)

    full_marginals = np.cumprod(ratios_with_init)
    assert full_marginals.shape == (max_R+1,)

    selected_marginals = full_marginals[bounds_R]
    assert selected_marginals.shape == bounds_R.shape
    
    # compute the constant c from condition 4
    c_condition_4 = 1/selected_marginals

    admissible_ub = t[:,None] * c_condition_3[:,None] * c_condition_4[None,:] \
                    / (t[:,None] * c_condition_3[:,None] * c_condition_4[None:] \
                    + ratios_cardinals_R)
    admissible_ub[~valid_cardinals_R] = 1

    ub = np.amin(admissible_ub, axis=-1)
    assert ub.shape == t.shape

    ub[0] = 1.
    return ub
    



    

