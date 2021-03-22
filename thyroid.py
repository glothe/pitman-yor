import numpy as np
from jax import random

from sklearn.decomposition import PCA

import matplotlib.pyplot as plt

from gaussian import multivariate_gaussian_DPMM, make_multivariate_gaussian_DPMM_gibbs_fn
from sampler import sample_posterior


N_SAMPLES = 1000
REPEATS = 1


def load_data(file_name: str = "../data/thyroid_train.dat") -> (np.ndarray, np.ndarray):
    x = np.genfromtxt(file_name, usecols=[0, 1, 2, 3, 4, 5], dtype=np.float)
    clusters = np.genfromtxt(file_name, usecols=[6], dtype=np.float).astype(np.int)
    return x, clusters

def plot_data(x: np.ndarray, clusters: np.ndarray):
    classes = np.unique(clusters)
    limit = 500

    pca = PCA(n_components=2)
    x_pca = np.ascontiguousarray(pca.fit_transform(x))

    for c in classes:
        xc = x_pca[clusters == c]
        plt.scatter(xc[:limit, 0], xc[:limit, 1], alpha=.2)
    
    plt.show()

def plot_posterior(data: np.ndarray):
    T = 10
    t = np.arange(T+1)
    rng_key = random.PRNGKey(0)
    repeat = 1

    for Npoints in (50,):
        y = np.zeros(T+1)
        for _ in range(repeat):
            # TODO: take random subsample for 'repeat' > 1
            data_sub = np.random.choice(data, size=Npoints, replace=False)

            y += sample_posterior(rng_key, multivariate_gaussian_DPMM, data_sub, Nsamples, T=T, alpha=1,
                # Uncomment the line below to use HMCGibbs - not working yet
                #	gibbs_fn=make_multivariate_gaussian_DPMM_gibbs_fn(data), gibbs_sites=['z'],
                )

        y /= repeat
        plt.plot(t, y, label=f"N={Npoints}")
        plt.scatter(t, y)

    plt.legend()
    plt.show()

def plot_clusters(data: np.ndarray):
    T = 10
    t = np.arange(T + 1)
    Npoints = 100

    rng_key = random.PRNGKey(0)

    x = data[:Npoints]
    z = sample_posterior(rng_key, multivariate_gaussian_DPMM, x, Nsamples=1, T=T, alpha=1)

    pca = PCA(n_components=2)
    x_pca = np.ascontiguousarray(pca.fit_transform(x))

    for c in np.unique(z):
        xc = x_pca[z == c]
        plt.scatter(xc[:, 0], xc[:, 1], alpha=.5)

    plt.show()


if __name__ == "__main__":
    x, clusters = load_data()

    # plot_data(x, clusters)
    # plot_posterior(x)
    plot_clusters(x)