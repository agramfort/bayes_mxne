# Authors: Yousra Bekhti <yousra.bekhti@gmail.com>
#          Alexandre Gramfort <alexandre.gramfort@inria.fr>
#          Felix Lucka <f.lucka@ucl.ac.uk>
#          Joseph Salmon <joseph.salmon@telecom-paristech.fr>


import numpy as np
from numpy.linalg import norm

from mne.inverse_sparse.mxne_optim import (mixed_norm_solver,
                                           groups_norm2)

from .samplers import _L21_gamma_hypermodel_sampler
from .pyrtnorm import check_random_state


# The L21GammaHyperModelOptimizer
def compute_block_norms(w, n_orient):
    return np.sqrt(groups_norm2(w.copy(), n_orient))


def _neg_log_post_prob(G, X, M, n_orient, gamma, alpha, beta, n_times):
    R = norm(np.dot(G, X) - M, 'fro')
    relRes = R / norm(M, 'fro')
    nlpp = 1 / 2 * R ** 2
    XBlocNorms_ = compute_block_norms(X, n_orient)
    gammaNonZero = np.where(gamma)[0]
    nlpp = nlpp + np.sum(XBlocNorms_[gammaNonZero] / gamma[gammaNonZero])
    nlpp = nlpp + np.sum(gamma / beta) - (alpha - 1 - n_times * n_orient) \
        * np.sum(np.log(gamma[gammaNonZero]))

    return nlpp, relRes


def _log_posterior_prob(Xsamples, gammaChain, G, M, n_orient, beta):
    """Compute the log posterior probability.

    Parameters
    ----------
    Xsamples : array, shape (n_dipoles, n_times, K)
        The K samples on the chain.
    gammaChain : list
        list containing the chain of K gamma_samples.
    G : array, shape (n_sensors, n_dipoles)
        The forward operator.
    M : array, shape (n_sensors, n_times)
        The data.
    n_orient : int
        The number of orientation (1 : fixed or 3 : free or loose).
    beta : float
        The beta parameter of the prior distribution.

    Returns
    -------
    lppSamples : array, shape (K,)
        The log posterior probability of the samples. See eq (12)
        in paper.
    relResSamples : array, shape (K,)
        The Frobenius norms of the residuals for the K samples of the chain,
        normalized by the Frobenius norm of the data M.
    blockNormSamples : array, shape (K,)
        The sum of the block norms for the K samples on the chain.
    """
    K = Xsamples.shape[2]
    lppSamples = np.zeros((K,))
    relResSamples = np.zeros((K,))
    blockNormSamples = np.zeros((K,))

    normM = norm(M, 'fro')

    for i in range(K):
        relResSamples[i] = norm(np.dot(G, Xsamples[:, :, i]) - M, 'fro')
        lppSamples[i] = - 1. / 2 * relResSamples[i] ** 2
        relResSamples[i] = relResSamples[i] / normM
        blockNormSamples[i] = \
            np.sum(compute_block_norms(Xsamples[:, :, i], n_orient))
        XGammaRatio = compute_block_norms(Xsamples[:, :, i], n_orient) \
            / gammaChain[:, i]
        lppSamples[i] -= sum(XGammaRatio[~np.isnan(XGammaRatio)])
        lppSamples[i] -= sum(gammaChain[:, i] / beta)

    return lppSamples, relResSamples, blockNormSamples


def _energy_l2half_reg(M, G, X, active_set, lambda_l2half, n_orient):
    reg = lambda_l2half * np.sqrt(compute_block_norms(X, n_orient)).sum()
    return norm(np.dot(G[:, active_set], X) - M, 'fro') + reg


def _L21_gamma_hypermodel_optimizer(G, M, gamma, alpha, beta, n_orient,
                                    maxIter, lambdaRef, epsilon=1.e-6,
                                    verbose=False):
    n_locations = G.shape[1] // n_orient
    n_times = M.shape[1]

    nu = alpha - 1 - n_times * n_orient
    X = np.zeros((n_locations * n_orient, n_times))

    energy = np.zeros((maxIter,))
    relResiduum = np.zeros((maxIter,))
    # print('starting posterior optimization')

    active_set = np.ones((G.shape[1],), dtype='bool')
    Gbar = np.zeros(G.shape)
    for i_iter in range(maxIter):
        if verbose:
            print("iter %s - active set %s" % (i_iter, active_set.sum()))
        Xold = X.copy()
        X = np.zeros((n_locations * n_orient, n_times))

        # update X
        gamma_bar = np.tile(gamma, [n_orient, 1]).ravel(order='F')
        Gbar = G * (lambdaRef * gamma_bar)

        Xbis, active_set, _ = mixed_norm_solver(
            M, Gbar, lambdaRef, maxit=3000, tol=1e-4, n_orient=n_orient,
            debias=False, verbose=False)

        X[active_set, :] = Xbis

        X = (X.T * (lambdaRef * gamma_bar)).T

        # update gamma
        XBlocNorms = compute_block_norms(X, n_orient)
        gamma = beta * (nu + np.sqrt(nu ** 2 + XBlocNorms / beta)) + epsilon

        # compute relative change in X and energy
        XChange = 1.
        if X.shape[0] == Xold.shape[0]:
            XChange = norm(X - Xold, 'fro')
        if XChange > 0:
            XChange = 0.5 * XChange / (norm(X, 'fro') + norm(Xold, 'fro'))
        # else:
        #     break

        energy[i_iter], relResiduum[i_iter] = _neg_log_post_prob(
            G, X, M, n_orient, gamma[active_set[::n_orient]],
            alpha, beta, n_times)

    # output
    return X, active_set, gamma, energy, relResiduum


def mm_mixed_norm_bayes(M, G, lambda_ref, n_orient=1, K=900, scK=1, ssK=1,
                        n_burnin=0, maxiter=10, return_samples=False,
                        random_state=42, verbose=False):
    """Run MM solver with K MCMC samples as initilization.

    Parameters
    ----------
    M : array, shape (n_samples, n_times)
        The data.
    G : array, shape (n_samples, n_features)
        The forward operator.
    lambda_ref : float
        Regularization parameter tau * lambda_max
    n_orient : int
        The number of orientation (1 : fixed or 3 : free or loose).
        Used for M/EEG application as there is 3 features per
        physical locations. We have n_locations = n_features // n_orient.
    K : int
        length of the MCMC chain
    scK : int
        number of Slice-Within-Gibbs steps, we use 1 for now (seems fine)
    ssK : int
        number of slice sampling steps, we use 1 for now (seems fine)
    n_burnin : int
        The number of samples in the burnin phase
    maxiter : int
        The number of interations in the L21 solver.
    return_samples : bool
        If True, the samples on returned. XXX which samples?
    random_state : int | None
        An integer to fix the seed of the numpy random number
        generator. Necessary to have replicable results.
    verbose : bool
        If True print info on optimization.

    Returns
    -------
    Xs : list of length K
        Æll modes found with the K MCMC initilization.
    As : ndarray, shape (K, n_features)
        Active sets of modes found with the K MCMC initilization.
    lppSamples : array, shape (K,)
        The log posterior probability of the samples. See eq (12)
        in paper.
    lppMAP : array, shape (K,)
        The log posterior probability of the Xs obtained
        after full-MAP estimation.
    pobj : array, shape (K,)
        The primal objective solved by MM solver.
    X_samples : array, shape (K, n_features, n_times, 2)
        The X samples along the chain.
        Warning this can be big.
        Only returned if return_samples is True.
    gamma_samples : array, shape (K, n_features, 2)
        The gamma samples along the chain.
        Warning this can be big.
        Only returned if return_samples is True.
    """
    rng = check_random_state(random_state)

    n_features = G.shape[1]
    n_locations = n_features // n_orient
    n_times = M.shape[1]

    a = n_orient * n_times + 1
    b = 4 / lambda_ref ** 2

    gamma_init = np.ones((n_locations,)) / lambda_ref
    lambda_map = 1. / np.mean(gamma_init)

    # we start from the a uniform initilization and
    X_sample = np.zeros((G.shape[1], M.shape[1], 1))
    gamma_sample = np.ones((n_locations, 1)) / lambda_ref

    lppSamples = np.zeros((K,))
    relResSamples = np.zeros((K,))
    blockNormSamples = np.zeros((K,))
    lppMAP = np.zeros((K,))
    relResMAP = np.zeros((K,))
    blockNormMAP = np.zeros((K,))
    solution_support = np.zeros((K, n_locations))
    pobj_l2half = np.zeros((K,))

    X_new_MAP = {}
    as_new_MAP = {}
    energy = {}

    Xs = list()
    As = np.empty((K, n_features), dtype=bool)
    X_samples = []
    gamma_samples = []

    for k in range(K):
        # compute new sample by continuing the chain at the last sample
        # and doing 1 step
        X_sample, gamma_sample = _L21_gamma_hypermodel_sampler(
            M, G, X0=X_sample[:, :, -1], gammas=gamma_sample[:, -1],
            n_orient=n_orient, beta=b, n_burnin=n_burnin,
            n_samples=2, sc_n_samples=scK, ss_n_samples=ssK,
            random_state=rng, verbose=verbose)

        # compute new full-MAP estimate
        X_new_MAP[k], as_new_MAP[k], gamma_new_MAP, energy[k], _ = \
            _L21_gamma_hypermodel_optimizer(G, M, gamma_sample[:, -1], a, b,
                                            n_orient, maxiter, lambda_map,
                                            verbose=verbose)

        # we compute and plot the log posterior probability, the relative
        # residual and the block norm to monitor the sampler
        lppSamples[k], relResSamples[k], blockNormSamples[k] = \
            _log_posterior_prob(X_sample[:, :, -1:], gamma_sample[:, -1:], G,
                                M, n_orient, b)
        lppMAP[k], relResMAP[k], blockNormMAP[k] = _log_posterior_prob(
            X_new_MAP[k][:, :, np.newaxis], gamma_new_MAP[:, np.newaxis], G, M,
            n_orient, b)

        block_norms_new = compute_block_norms(X_new_MAP[k], n_orient)
        block_norms_new = (block_norms_new > 0.05 * block_norms_new.max())
        solution_support[k, :] = block_norms_new

        pobj_l2half[k] = _energy_l2half_reg(
            M, G, X_new_MAP[k][as_new_MAP[k]], as_new_MAP[k], lambda_ref,
            n_orient)
        if verbose:
            print("sample %s - lppSamp %s - relResSamp %s - lppMAP %s - "
                  "relResSamp %s" % (k, lppSamples[k], relResSamples[k],
                                     lppMAP[k], relResMAP[k]))
        Xs.append(X_new_MAP[k][as_new_MAP[k]])
        As[k] = as_new_MAP[k]
        if return_samples:
            X_samples.append(X_sample)
            gamma_samples.append(gamma_sample)

    # out = Xs, As, lppSamples, relResSamples, blockNormSamples, lppMAP
    out = Xs, As, lppSamples, lppMAP, pobj_l2half
    if return_samples:
        out += (np.array(X_samples), np.array(gamma_samples))
    return out
