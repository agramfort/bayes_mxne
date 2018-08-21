# Authors: Yousra Bekhti <yousra.bekhti@gmail.com>
#          Alexandre Gramfort <alexandre.gramfort@inria.fr>


from __future__ import print_function

from numpy.testing import assert_equal
import numpy as np

from scipy.linalg.special_matrices import toeplitz
from scipy import linalg

from mne.inverse_sparse.mxne_optim import norm_l2inf

from bayes_mxne import mm_mixed_norm_bayes


def test_mm_mixed_norm_bayes():
    """Basic test of the mm_mixed_norm_bayes function"""
    # First we define the problem size and the location of the active sources.
    n_features = 16
    n_samples = 24
    n_times = 5

    X_true = np.zeros((n_features, n_times))
    # Active sources at indices 10 and 30
    X_true[5, :] = 2.
    X_true[10, :] = 2.

    # Construction of a covariance matrix
    rng = np.random.RandomState(0)
    # Set the correlation of each simulated source
    corr = [0.6, 0.95]
    cov = []
    for c in corr:
        this_cov = toeplitz(c ** np.arange(0, n_features // len(corr)))
        cov.append(this_cov)

    cov = np.array(linalg.block_diag(*cov))

    # Simulation of the design matrix / forward operator
    G = rng.multivariate_normal(np.zeros(n_features), cov, size=n_samples)

    # Simulation of the data with some noise
    M = G.dot(X_true)
    M += 0.1 * np.std(M) * rng.randn(n_samples, n_times)
    n_orient = 1

    # Define the regularization parameter and run the solver
    lambda_max = norm_l2inf(np.dot(G.T, M), n_orient)
    lambda_ref = 0.3 * lambda_max
    K = 10
    random_state = 0  # set random seed to make results replicable
    out = mm_mixed_norm_bayes(
        M, G, lambda_ref, n_orient=n_orient, K=K, return_lpp=True,
        random_state=random_state)

    Xs, active_sets = out[0]
    lpp_samples, rel_res_samples, block_norm_samples, lppMAP = out[1]

    freq_occ = np.mean(active_sets, axis=0)
    assert_equal(np.argsort(freq_occ)[-2:], [10, 5])
    assert lpp_samples.shape == (K,)
    assert rel_res_samples.shape == (K,)
    assert block_norm_samples.shape == (K,)
    assert lppMAP.shape == (K,)

    out = mm_mixed_norm_bayes(
        M, G, lambda_ref, n_orient=n_orient, K=K, return_samples=True,
        random_state=random_state)

    (Xs, active_sets), (X_samples, gamma_samples) = out

    freq_occ = np.mean(active_sets, axis=0)
    assert_equal(np.argsort(freq_occ)[-2:], [10, 5])
    assert lpp_samples.shape == (K,)
    assert rel_res_samples.shape == (K,)
    assert block_norm_samples.shape == (K,)
    assert lppMAP.shape == (K,)
    assert X_samples.shape == (K, n_features, n_times, 2)
    assert gamma_samples.shape == (K, n_features, 2)
