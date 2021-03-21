import numpy as np
from jax import random

from numpyro.distributions import Normal, Poisson

import matplotlib.pyplot as plt

from sampler import sample_posterior
from gaussian import gaussian_DPMM, make_gaussian_DPMM_gibbs_fn
from poisson import poisson_DPMM, make_poisson_DPMM_gibbs_fn, explicit_upper_bound
from utils import compute_PY_prior, compute_n_clusters_distribution, compute_cluster_size_distribution


# Use HMCGibbs
USE_GIBBS = False


def make_synthetic_experiment(sample_data: np.ndarray, model, make_gibbs_fn, explict_ub=None):
    """
    Template for small synthetic experiments
    """
    rng_key = random.PRNGKey(0)

    # Sampling parameters
    Nsamples = 1000
    repeat = 1
    n_values = [100, 200, 500, 1000]  #, [200, 500, 1000, 2000]

    # DPMM/PYMM parameters
    T = 10                # max number of component in the truncated stick breaking representation
    t = np.arange(T + 1)
    alpha = 1
    sigma = 0

    # Plotting parameters
    fig, (ax0, ax1) = plt.subplots(1, 2)

    priors = compute_PY_prior(alpha, sigma, n_values)
    for Npoints, prior in zip(n_values, priors):
        cluster_count = np.zeros(T + 1)  # cluster count histogram
        upper_bound = np.zeros(T + 1)
        
        cluster_size = np.zeros(Npoints+1)

        # Repeat the experiment 
        for _ in range(repeat):
            data = sample_data(rng_key, Npoints)
            z = sample_posterior(rng_key, model, data, Nsamples,
                    T=T, alpha=1,
                    gibbs_fn=make_gibbs_fn(data) if USE_GIBBS else None,
                    gibbs_sites=['z'] if USE_GIBBS else None,
                )

            cluster_count += compute_n_clusters_distribution(z, T)
            if explict_ub is not None:
                upper_bound += explict_ub(data, t, params_PY=(alpha, sigma))

            cluster_size += compute_cluster_size_distribution(z)

        cluster_count /= repeat
        cluster_size /= repeat

        # Plot cluster count histograms (ax0)
        ax0.plot(t, cluster_count, label=f"N={Npoints}", marker='o')

        color = ax0.lines[-1].get_color()
        ax0.plot(t, prior[:T+1], label=f"Prior N={Npoints}", color=color, linestyle='dashed')

        if explict_ub is not None:
            upper_bound /= repeat
            ax0.plot(t, upper_bound, label=f"Upper bound N={Npoints}", color=color, linestyle='dotted')

        # Plot cluster size histograms (ax1)
        frac = np.arange(Npoints + 1) / Npoints
        ax1.plot(frac, cluster_size, color=color, label=f"N={Npoints}")

    ax0.axhline(y=1, color='black', linewidth=0.3, linestyle='dotted')
    ax0.set(title="Number of cluster")
    ax0.legend()

    ax1.axhline(y=1, color='black', linewidth=0.3, linestyle='dotted')
    ax1.set(xlabel="Fraction of total size", title="Size of clusters")
    ax1.legend()

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
    make_synthetic_experiment(sample_data_poisson, poisson_DPMM, make_poisson_DPMM_gibbs_fn, explict_ub=explicit_upper_bound)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--distribution", type=str, help="Component distribution", default="poisson",
        choices=["poisson", "gaussian"])

    args = parser.parse_args()

    if args.distribution == "poisson":
        main_poisson_DPMM()
    elif args.distribution == "gaussian":
        main_gaussian_DPMM()
    else:
        raise ValueError()
