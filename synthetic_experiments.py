"""
Investigate data size dependency on synthetic datasets. 
"""

import numpy as np
from jax import random

from numpyro.distributions import Normal, Poisson

import matplotlib.pyplot as plt

from sampler import sample_posterior
from gaussian import gaussian_DPMM, make_gaussian_DPMM_gibbs_fn
from poisson import poisson_DPMM, make_poisson_DPMM_gibbs_fn, explicit_upper_bound
from utils import compute_PY_prior, compute_n_clusters_distribution, compute_cluster_size_distribution


USE_GIBBS = True  # Use HMCGibbs
N_SAMPLES = 1000
REPEATS = 1


def make_synthetic_experiment(sample_data, model, make_gibbs_fn, explicit_ub=None):
    """
    Template for small synthetic experiments with varying size of observed data. 

    Args:
        sample_data: sampling function
        model: model function

    """
    rng_key = random.PRNGKey(0)

    # Sampling parameters
    n_values = [50, 200] # [200 #, 500, 1000, 2000]

    # DPMM/PYMM parameters
    T = 20  # max number of component in the truncated stick breaking representation
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
        for _ in range(REPEATS):
            data = sample_data(rng_key, Npoints)
            z = sample_posterior(rng_key, model, data, N_SAMPLES,
                    T=T, alpha=1,
                    gibbs_fn=make_gibbs_fn(data) if USE_GIBBS else None,
                    gibbs_sites=['z'] if USE_GIBBS else None,
                )

            cluster_count += compute_n_clusters_distribution(z, T)
            if explicit_ub is not None:
                upper_bound += explicit_ub(data, t, params_PY=(alpha, sigma))

            cluster_size += compute_cluster_size_distribution(z)

        cluster_count /= REPEATS
        cluster_size /= REPEATS

        # Plot cluster count histograms (ax0)
        ax0.plot(t, cluster_count, label=f"N={Npoints}")

        color = ax0.lines[-1].get_color()
        ax0.plot(t, prior[:T+1], label=f"Prior N={Npoints}", color=color, linestyle='dashed', lw=1)

        if explicit_ub is not None:
            upper_bound /= REPEATS
            ax0.plot(t, upper_bound, label=f"Upper bound N={Npoints}", color=color, linestyle='dotted', lw=1)

        # Plot cluster size histograms (ax1)
        bins = np.linspace(0, 1, 10, endpoint=True)
        frac = np.arange(0, Npoints + 1) / Npoints

        # TODO : use an actual histogram ?
        # Overlaying histograms doesn't really look good.
        hist, edges = np.histogram(frac, bins, density=True, weights=cluster_size)
        ax1.plot(0.5 * (edges[1:] + edges[:-1]), hist, color=color, label=f"N={Npoints}")

    ax0.axhline(y=1, color='black', linewidth=0.3, linestyle='dotted')
    ax0.set(title=r"Number of clusters", xlabel="$t$", ylabel=r"$P(T_n=t|X_{1:N})$")
    ax0.legend()

    ax1.set(xlabel="Fraction of total size", title="Size of clusters")
    ax1.legend()

    plt.show()


def make_synthetic_experiment_alpha(sample_data, model, make_gibbs_fn, explicit_ub=None):
    rng_key = random.PRNGKey(0)

    # Sampling parameters
    Npoints = 1000
    data = sample_data(rng_key, Npoints)

    # DPMM/PYMM parameters
    T = 20
    t = np.arange(T + 1)

    for alpha in [0.5, 1, 2]:
        z = sample_posterior(rng_key, model, data, N_SAMPLES,
                alpha=alpha, sigma=0, T=T, 
                gibbs_fn=make_gibbs_fn(data) if USE_GIBBS else None,
                gibbs_sites=['z'] if USE_GIBBS else None,
            )
        cluster_count = compute_n_clusters_distribution(z, T)
        plt.plot(t, cluster_count, label=r"$\alpha={}$".format(alpha))
        color = plt.gca().lines[-1].get_color()

        # Upper bound
        if explicit_ub is not None:
            upper_bound = explicit_ub(data, t, params_PY=(alpha, 0))
            plt.plot(t, upper_bound, label=r"Upper bound $\alpha={}$".format(alpha), color=color,
                     linestyle="dotted", lw=1)

        # Prior
        prior = compute_PY_prior(alpha, 0, [Npoints])[0]
        plt.plot(t, prior[:T+1], label=r"Prior $\alpha={}$".format(alpha), color=color,
                 linestyle="dashed", lw=1)
        # plt.axvline(alpha*np.log(Npoints), color=color, lw=1, linestyle="dashed")

    plt.legend()
    plt.title(r"Impact of $\alpha$")
    plt.xlabel("Number of clusters")
    plt.show()

def make_synthetic_experiment_sigma(sample_data, model, make_gibbs_fn, explicit_ub=None):
    rng_key = random.PRNGKey(0)

    # Sampling parameters
    Npoints = 1000
    data = sample_data(rng_key, Npoints)

    # DPMM/PYMM parameters
    T = 20
    t = np.arange(T + 1)

    for sigma in [0, 0.1, .5]:
        z = sample_posterior(rng_key, model, data, N_SAMPLES,
                alpha=1, sigma=sigma, T=T, 
                gibbs_fn=make_gibbs_fn(data) if USE_GIBBS else None,
                gibbs_sites=['z'] if USE_GIBBS else None,
            )
        cluster_count = compute_n_clusters_distribution(z, T)
        plt.plot(t, cluster_count, label=r"$\sigma={}$".format(sigma))
        color = plt.gca().lines[-1].get_color()

        # Upper bound
        if explicit_ub is not None:
            upper_bound = explicit_ub(data, t, params_PY=(1, sigma))
            plt.plot(t, upper_bound, label=r"Upper bound $\sigma={}$".format(sigma), color=color,
                     linestyle="dotted", lw=1)

        # Prior
        prior = compute_PY_prior(1, sigma, [Npoints])[0]
        plt.plot(t, prior[:T+1], label=r"Prior $\sigma={}$".format(sigma), color=color,
                 linestyle="dashed", lw=1)
        # plt.axvline(alpha*np.log(Npoints), color=color, lw=1, linestyle="dashed")

    plt.legend()
    plt.title(r"Impact of $\sigma$")
    plt.xlabel("Number of clusters")
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
    make_synthetic_experiment(
        sample_data=sample_data_miller,
        model=gaussian_DPMM,
        make_gibbs_fn=make_gaussian_DPMM_gibbs_fn
    )

# Poisson DPMM experiment
def sample_data_poisson(rng_key: random.PRNGKey, N: int):
    data = np.hstack((
        Poisson(4).sample(rng_key, (int(.45 * N),)),
        Poisson(6).sample(rng_key, (int(.3 * N),)),
        Poisson(8).sample(rng_key, (N - int(.3 * N) - int(.45 * N),))))
    return data

def main_poisson_DPMM():
    make_synthetic_experiment_alpha(
        sample_data=sample_data_poisson,
        model=poisson_DPMM,
        make_gibbs_fn=make_poisson_DPMM_gibbs_fn,
        explicit_ub=explicit_upper_bound
    )

    make_synthetic_experiment_sigma(
        sample_data=sample_data_poisson,
        model=poisson_DPMM,
        make_gibbs_fn=make_poisson_DPMM_gibbs_fn,
        explicit_ub=explicit_upper_bound
    )

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
