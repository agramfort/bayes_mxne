"""
=====================================
Plot a demonstration on a toy problem
=====================================

This example demonstrates the difficulty to recover sources
which have stronger correlation between the columns of the
gain matrix G.
"""
# Authors: Yousra Bekhti <yousra.bekhti@gmail.com>
#          Alexandre Gramfort <alexandre.gramfort@inria.fr>

# License: BSD (3-clause)

import numpy as np
from scipy import linalg
from scipy.linalg.special_matrices import toeplitz
import matplotlib.pyplot as plt

from mne.inverse_sparse.mxne_optim import norm_l2inf
from bayes_mxne import mm_mixed_norm_bayes

print(__doc__)

###############################################################################
# Construction of simulated data
# ------------------------------
#
# First we define the problem size and the location of the active sources.
n_features = 40
n_samples = 15
n_times = 10

X_true = np.zeros((n_features, n_times))
# Active sources at indices 10 and 30
X_true[10, :] = 2.
X_true[30, :] = 2.

###############################################################################
# Construction of a covariance matrix
rng = np.random.RandomState(0)
# Set the correlation of each simulated source
corr = [0.6, 0.95]
cov = []
for c in corr:
    this_cov = toeplitz(c ** np.arange(0, n_features // len(corr)))
    cov.append(this_cov)

cov = np.array(linalg.block_diag(*cov))

###############################################################################
# Simulation of the design matrix / forward operator
G = rng.multivariate_normal(np.zeros(n_features), cov, size=n_samples)

###############################################################################
# Simulation of the data with some noise
M = G.dot(X_true)
M += 0.3 * np.std(M) * rng.randn(n_samples, n_times)
n_orient = 1

###############################################################################
# Define the regularization parameter and run the solver
# ------------------------------------------------------
lambda_max = norm_l2inf(np.dot(G.T, M), n_orient)

lambda_ref = 0.1 * lambda_max
K = 2000
Xs, active_sets, lpp_samples, _, _, lpp_Xs = \
    mm_mixed_norm_bayes(M, G, lambda_ref, n_orient=n_orient, K=K, verbose=True)

freq_occ = np.mean(active_sets, axis=0)

###############################################################################
# Plot the covariance to see the correlation of the neighboring
# sources around each simulated one (10 and 30).

plt.matshow(cov)
plt.title('Covariance')

# Plot the active support of the solution
plt.figure(figsize=(6.4, 3.3))
plt.stem(np.mean(active_sets, axis=0))
plt.xlabel('Features')
plt.ylabel('Support %')
plt.tight_layout()
plt.show()

tmp = np.array(active_sets).astype(float)
as_cov = np.dot(tmp.T, tmp) / K

# Active set covariance
plt.matshow(as_cov, origin='lower left')
plt.clim([0, 0.05])
plt.gca().xaxis.set_ticks_position('bottom')
plt.title('Active set covariance')
plt.colorbar()
plt.show()
