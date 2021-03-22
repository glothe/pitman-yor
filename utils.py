import numpy as np
import jax.numpy as jnp

import numpyro
from numpyro.distributions import *

from typing import Iterable, List


def mix_weights(beta: jnp.ndarray) -> jnp.ndarray:
    T = beta.shape[-1]
    batched = beta.shape[:-1]

    beta1m_cumprod = (1 - beta).cumprod(-1)
    assert beta1m_cumprod.shape == (*batched, T)

    res = jnp.pad(beta, (0, 1), mode='constant', constant_values=1) \
            * jnp.pad(beta1m_cumprod, (1, 0), mode='constant', constant_values=1)
    assert res.shape == (*batched, T+1)

    return res

def sample_beta_DP(alpha: float, T:int = 10) -> jnp.ndarray:
    with numpyro.plate("beta_plate", T-1):
        beta = numpyro.sample("beta", Beta(1, alpha))
    return beta

def sample_beta_PY(alpha: float, sigma: float = 0, T: int = 10) -> jnp.ndarray:
    with numpyro.plate("beta_plate", T-1):
        beta = numpyro.sample("beta", Beta(1 - sigma, alpha + sigma * np.arange(1, T)))
    assert beta.shape == (T-1,), (beta.shape, T)

    return beta


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
    counts = np.zeros(shape=Nsamples, dtype=int) # number of clusters of each sample
    for i, A in enumerate(z):
        counts[i] = len(np.unique(A))

    nclusters, cluster_counts = np.unique(counts, return_counts=True)

    y = np.zeros(T+1)
    y[nclusters] = cluster_counts / np.sum(cluster_counts)
    return y

def compute_cluster_size_distribution(z: np.ndarray) -> np.ndarray:
    """ 
        Compute emipirical cluster sizes.
    """
    Nsamples, Npoints = z.shape
    counts = np.zeros(Npoints + 1)  # number of clusters of size m
    
    for i, sample in enumerate(z):
        clusters, size = np.unique(sample, return_counts=True)
        np.add.at(counts, size, 1)

    counts /= Nsamples
    return counts

def compute_PY_prior(alpha: float, sigma: float, n_values: Iterable[int]) -> List[np.ndarray]:
    """
    Compute recursively the distrbution of the number of clusters
    in a PY(alpha, sigma) model, for n in n_values
    """
    n_max = max(n_values)
    results = [None] * len(n_values)
    next_n_value = 0

    # For n = 1
    P = np.zeros(n_max)
    P[0] = 1.
    next_P = np.zeros_like(P)
    indices = np.arange(n_max)

    # induction
    for n in range(1, n_max):
        if n_values[next_n_value] == n:
            # Include the value zero with prob 0
            results[next_n_value] = np.pad(P[:n], (1, 0), mode='constant', constant_values=0)
            assert results[next_n_value].shape == (n_values[next_n_value]+1,)
            next_n_value += 1
        next_P[:] = 0
        next_P[:n]    += P[:n] *     (n - sigma * indices[:n])/(alpha + n)
        next_P[1:n+1] += P[:n] * (alpha + sigma * indices[:n])/(alpha + n)
        P[:n+1] = next_P[:n+1]

    assert next_n_value == len(n_values) - 1
    assert n_values[next_n_value] == n_max

    results[-1] = np.pad(P[:n_max], (1, 0), mode='constant', constant_values=0)
    assert results[-1].shape == (n_max+1,)

    return results
