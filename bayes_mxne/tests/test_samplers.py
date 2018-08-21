# Author : Alex Gramfort, <alexandre.gramfort@inria.fr>

import numpy as np
from numpy.testing import assert_array_equal, assert_array_almost_equal

from scipy.integrate import quad

from bayes_mxne.samplers import _cond_gamma_hyperprior_sampler
from bayes_mxne.samplers import _sc_slice_sampler


def test_gamma_hyperprior_sampler_random_state():
    """Test that hyperprior sampler handles the numpy random state properly"""
    size = 10
    beta = 1.
    coupling = 1.
    couplings = coupling * np.ones(size)

    rng = np.random.RandomState(42)
    gammas1 = _cond_gamma_hyperprior_sampler(couplings, beta, rng)
    gammas2 = _cond_gamma_hyperprior_sampler(couplings, beta, rng)
    rng = 42
    gammas3 = _cond_gamma_hyperprior_sampler(couplings, beta, rng)

    assert_array_equal(gammas1, gammas3)
    assert np.all(gammas1 != gammas2)


def test_gamma_sampler():
    """Test gamma sampler"""

    size = 100000
    beta = 1.
    coupling = 1.
    couplings = coupling * np.ones(size)
    bins = 50

    # Run gamma hyperprior sampler
    random_state = 42
    gammas = _cond_gamma_hyperprior_sampler(couplings, beta, random_state)

    xmin, xmax = np.min(gammas), np.max(gammas)
    xx = np.linspace(xmin, xmax, 1000)

    def dist(xx):
        """Gamma distribution one wants to sample from"""
        return np.exp(- coupling / xx) * np.exp(- xx / beta)

    # Compute the normalizing constant
    Z, _ = quad(dist, 1e-5, bins)

    hist, bin_edges = np.histogram(gammas, normed=True, bins=bins)
    bin_midpoints = (bin_edges[:-1] + bin_edges[1:]) / 2.

    assert_array_almost_equal(dist(bin_midpoints) / Z, hist, decimal=2)


def test_sc_slice_sampler():
    """Test Slice Sampler"""
    (a, b, c, d), x0, ss_n_samples = (1,) * 4, 0., 20
    random_state = np.random.RandomState(42)
    size = 500
    samples = [_sc_slice_sampler(a, b, c, d, x0, ss_n_samples, random_state)
               for k in range(size)]
    samples = np.array(samples)

    def dist(xx):
        return np.exp(-a * xx ** 2 + b * xx - c * np.sqrt(xx ** 2 + d))

    xx = np.linspace(-2, 3, 300)
    Z, _ = quad(dist, -5, 5)

    hist, bin_edges = np.histogram(samples, normed=True, bins=20)
    bin_midpoints = (bin_edges[:-1] + bin_edges[1:]) / 2.

    assert_array_almost_equal(dist(bin_midpoints) / Z, hist, decimal=1)
