"""
============================================================
Experiment 1: Variable feature correlation with a 1d problem
============================================================

This example aims to illustrate the difficulty to recover sources
in the presence of stronger correlation between the columns of the
gain matrix G.

First, we construct G as a random matrix in the following way: its rows are
drawn from a Gaussian distribution with zero mean and a block-diagonal
covariance matrix C = block_diag(C1, C2), where (C1)i,j = 0.5|i−j|,
(C2)i,j = 0.95|i−j|, i, j = 1,...,10. Then, each column is normalized to have
unit l2 norm. First figure illustrates the set-up (See Figure 1 in paper [1]).

X is a sparse vector with a value of 1 at index 4 and 14.
We want to illustrate that due to the asymmetry in the design, the correct
recovery of the source at index 14 is more difcult due to the stronger
correlation in the second block of columns.

We generate M by adding Gaussian white noise with standard deviation
equal to 0.2 max(GX). We first run the MM algorithm 1 using a uniform
initializzation, i.e. w = ones(n_features), with lambda
set to 0.2 lambda_max. lambda_max is the smallest regularization value for
which no source is found as active using an l2,1 regularization (Ndiaye et al
2015, Strohmeier et al 2016). With the MNE implementation of the MM
solver.

It does not recover an X supported at locations 4 and 14, i.e. it is not able
to locate the sources correctly. Then, we run algorithm 3 from paper
the same settings for the majorization-minimization (MM) algorithm as
before to obtain chains of posterior samples, and the corresponding posterior
modes. We also show that one can find a better local minimum with algorithm 3.

We finally cluster the modes based on their spatial support. This reveals
multiple modes in the posterior. Figure 2 depicts the spatial support of the
modes listed based on the relative frequency with which they were found.
It reveals that, indeed, there is a larger uncertainty in the location of the
second source and that in this scenario, the support
of the mode which is found most often coincides with that of the true solution.

The example aims to replicate experiment 1 in the paper and the figure 1 and 2.

Reference:

[1] Bekhti, Y., Lucka, F., Salmon, J., & Gramfort, A. (2018). A hierarchical
Bayesian perspective on majorization-minimization for non-convex sparse
regression: application to M/EEG source imaging. Inverse Problems, Volume 34,
Number 8.

"""
# Authors: Yousra Bekhti <yousra.bekhti@gmail.com>
#          Alexandre Gramfort <alexandre.gramfort@inria.fr>
#          Felix Lucka <f.lucka@ucl.ac.uk>

# License: BSD (3-clause)

import numpy as np
from scipy import linalg
from scipy.linalg.special_matrices import toeplitz
import matplotlib.pyplot as plt

from mne.inverse_sparse.mxne_optim import iterative_mixed_norm_solver
from bayes_mxne import mm_mixed_norm_bayes
from bayes_mxne.utils import unique_rows

print(__doc__)

###############################################################################
# Construction of simulated data
# ------------------------------
#
# First we define the problem size and the location of the active sources.
n_features = 20
n_samples = 10
n_times = 1
lambda_percent = 20.
K = 1000

X_true = np.zeros((n_features, n_times))
# Active sources at indices 10 and 30
X_true[4, :] = 1.
X_true[14, :] = 1.

###############################################################################
# Construction of a covariance matrix
rng = np.random.RandomState(0)
# Set the correlation of each simulated source
corr = [0.5, 0.95]
cov = []
for c in corr:
    this_cov = toeplitz(c ** np.arange(0, n_features // len(corr)))
    cov.append(this_cov)

cov = np.array(linalg.block_diag(*cov))

plt.matshow(cov)
plt.title('True Covariance')

###############################################################################
# Simulation of the design matrix / forward operator
G = rng.multivariate_normal(np.zeros(n_features), cov, size=n_samples)

plt.matshow(G.T.dot(G))
plt.title("Feature covariance")

###############################################################################
# Simulation of the data with some noise
M = G.dot(X_true)
M += 0.2 * np.max(np.abs(M)) * rng.randn(n_samples, n_times)

###############################################################################
# Define the regularization parameter and run the MM solver
# ---------------------------------------------------------

lambda_max = np.max(np.linalg.norm(np.dot(G.T, M), axis=1))
lambda_ref = lambda_percent / 100. * lambda_max

X_mm, active_set_mm, E = \
    iterative_mixed_norm_solver(M, G, lambda_ref, n_mxne_iter=10)

pobj_l2half_X_mm = E[-1]

print("Found support: %s" % np.where(active_set_mm)[0])

###############################################################################
# Run the solver
# --------------

Xs, active_sets, lpp_samples, lpp_Xs, pobj_l2half_Xs = \
    mm_mixed_norm_bayes(M, G, lambda_ref, K=K)

# Plot if we found better local minima then the first result found be the
plt.figure()
plt.hist(pobj_l2half_Xs, bins=20, label="Modes obj.")
plt.axvline(pobj_l2half_X_mm, label="MM obj.")
plt.legend()

###############################################################################
# Plot the frequency of the supports
# ----------------------------------

unique_supports = unique_rows(active_sets)
n_modes = len(unique_supports)

print('Number of modes identified: %d' % n_modes)

# Now get frequency of each support
frequency = np.empty(len(unique_supports))
for k, support in enumerate(unique_supports):
    frequency[k] = np.mean(np.sum(active_sets !=
                                  support[np.newaxis, :], axis=1) == 0)

# Sort supports by frequency
order = np.argsort(frequency)[::-1]
unique_supports = unique_supports[order]
frequency = frequency[order]

# Plot support frequencies in a colorful way
C = unique_supports * np.arange(n_features, dtype=float)[np.newaxis, :]
C[C == 0] = np.nan
plt.matshow(C, cmap=plt.cm.spectral)
plt.xticks(range(20))
plt.yticks(range(n_modes), ["%2.1f%%" % (100 * f,) for f in frequency])
plt.ylabel("Support Freqency")
plt.xlabel('Features')
plt.grid('on', alpha=0.5)
plt.gca().xaxis.set_ticks_position('bottom')

# Plot a matrix which shows in its (i, j)th entry the frequency with which
# locations i and j are simultaneously found active in a mode estimate.
as_cov = np.dot(active_sets.T, active_sets.astype(float)) / K

# Active set covariance
plt.matshow(as_cov)
plt.clim([0, 1])
plt.gca().xaxis.set_ticks_position('bottom')
plt.title('Active set covariance')
plt.colorbar()
plt.show()
