"""
=========================================================
Experiment 2: Show relative frequencies with a 1d problem
=========================================================

While the posterior mode whose support coincided with that of the true
solution was also found with the highest relative frequency, it is not clear
whether this frequency is a reliable indication of the mode’s true relative
posterior mass. In general, this question is difficult to examine for high
dimensional problems. Nonetheless, here we constructed an example to at least
show that the frequencies are consistent: we now draw the rows of a 10 × 10
matrix G from a Gaussian distribution with zero mean and the covariance matrix
(Toeplitz with 0.95 correlation).

Then, we set G = [G, G], i.e. the first and last 10 columns of G are exactly
the same. This means that the regression problem (1) and the posterior
distribution are invariant with respect to switching the first and last 10
entries. Every mode has a corresponding copy ‘on the other side’, which should
be found with the same relative frequency.

The example aims to replicate experiment 2 in the paper and the figure 4.

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
lambda_percent = 50.
K = 5000

X_true = np.zeros((n_features, n_times))
# Active sources at indices 4 and 14
X_true[4, :] = 1.
X_true[14, :] = 1.

###############################################################################
# Construction of a covariance matrix
rng = np.random.RandomState(0)
# Set the correlation of each simulated source
corr = 0.95
cov = toeplitz(corr ** np.arange(0, n_features // 2))

###############################################################################
# Simulation of the design matrix / forward operator
G = rng.multivariate_normal(np.zeros(len(cov)), cov, size=n_samples)
G = np.concatenate((G, G), axis=1)
G /= np.linalg.norm(G, axis=0)[np.newaxis, :]  # normalize columns

plt.matshow(G.T.dot(G))
plt.title("Feature covariance")

###############################################################################
# Simulation of the data with some noise
M = G.dot(X_true)
M += 0.2 * np.max(np.abs(M)) * rng.randn(n_samples, n_times)

###############################################################################
# Define the regularization parameter and run the solver
# ------------------------------------------------------

lambda_max = np.max(np.linalg.norm(np.dot(G.T, M), axis=1))
lambda_ref = lambda_percent / 100. * lambda_max

X_mm, active_set_mm, _ = \
    iterative_mixed_norm_solver(M, G, lambda_ref, n_mxne_iter=10)

print("Found support: %s" % np.where(active_set_mm)[0])

###############################################################################
# Run the solver
# --------------

Xs, active_sets, lpp_samples, lpp_Xs, pobj_l2half_Xs = \
    mm_mixed_norm_bayes(M, G, lambda_ref, K=K)

# we plot the log posterior probability to check when the sampler reached
# its statinary phase
plt.figure()
plt.plot(lpp_samples, label='log post. prob. chain samples')
plt.plot(lpp_Xs, label='log post. prob. Xs (full MAP)')
plt.xlabel('Samples')
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
plt.matshow(C, cmap=plt.cm.Set1)
plt.xticks(range(20))
plt.yticks(range(n_modes), ["%2.1f%%" % (100 * f,) for f in frequency])
plt.ylabel("Support Freqency")
plt.xlabel('Features')
plt.grid('on', alpha=0.5)
plt.gca().xaxis.set_ticks_position('bottom')
plt.tight_layout()
plt.show()
