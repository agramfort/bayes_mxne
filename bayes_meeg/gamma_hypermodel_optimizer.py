
import numpy as np
from numpy.linalg import norm

from mne.inverse_sparse.mxne_optim import (mixed_norm_solver,
                                           groups_norm2)
from bayes_meeg.samplers import L21_gamma_hypermodel_sampler


# The L21GammaHyperModelOptimizer
def compute_block_norms(w, n_orient):
    return np.sqrt(groups_norm2(w.copy(), n_orient))


def neg_log_post_prob(G, X, M, n_orient, gamma, alpha, beta, n_times):
    R = norm(np.dot(G, X) - M, 'fro')
    relRes = R / norm(M, 'fro')
    nlpp = 1 / 2 * R ** 2
    XBlocNorms_ = compute_block_norms(X, n_orient)
    gammaNonZero = np.where(gamma)[0]
    nlpp = nlpp + np.sum(XBlocNorms_[gammaNonZero] / gamma[gammaNonZero])
    nlpp = nlpp + np.sum(gamma / beta) - (alpha - 1 - n_times * n_orient) \
        * np.sum(np.log(gamma[gammaNonZero]))

    return nlpp, relRes


def log_posterior_prob(XChain, gammaChain, G, M, n_orient, beta):
    """Compute the log posterior probability.

    XChain : list
        list containing the chain of K X_samples.
    gammaChain : list
        list containing the chain of K gamma_samples.
    G : array, shape (n_sensors, n_dipoles)
        The forward operator.
    M : array, shape (n_sensors, n_times)
        The data.
    n_orient : int
        The number of orientation (1 : fixed or 3 : free or loose).
    beta : float
    """
    K = XChain.shape[2]
    lpChain = np.zeros((K,))
    relResChain = np.zeros((K,))
    blockNormChain = np.zeros((K,))

    normM = norm(M, 'fro')

    for i in range(K):
        relResChain[i] = norm(np.dot(G, XChain[:, :, i]) - M, 'fro')
        lpChain[i] = - 1 / 2 * relResChain[i] ** 2
        relResChain[i] = relResChain[i] / normM
        blockNormChain[i] = sum(compute_block_norms(XChain[:, :, i], n_orient))
        XGammaRatio = compute_block_norms(XChain[:, :, i], n_orient) \
            / gammaChain[:, i]
        lpChain[i] = lpChain[i] - sum(XGammaRatio[~np.isnan(XGammaRatio)])
        lpChain[i] = lpChain[i] - sum(gammaChain[:, i] / beta)

    return lpChain, relResChain, blockNormChain


def energy_l2half_reg(M, G, X, active_set, lambda_l2half, n_orient):
    reg = lambda_l2half * np.sqrt(compute_block_norms(X, n_orient)).sum()
    return norm(np.dot(G[:, active_set], X) - M, 'fro') + reg


def L21_gamma_hypermodel_optimizer(G, M, gamma, alpha, beta, n_orient, maxIter,
                                   lambdaRef, epsilon=1.e-6):
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

        energy[i_iter], relResiduum[i_iter] = neg_log_post_prob(
            G, X, M, n_orient, gamma[active_set[::n_orient]],
            alpha, beta, n_times)

    # output
    return X, active_set, gamma, energy, relResiduum


def mm_mixed_norm_bayes(M, G, lambda_ref, n_orient=1, K=900, scK=1, ssK=1,
                        n_burnin=0, maxiter=10, return_lpp=False,
                        return_samples=False):
    """Run MM solver with K MCMC samples as initilization.

    Parameters
    ----------
    M : array, shape (n_sensors, n_times)
        The data.
    G : array, shape (n_sensors, n_dipoles)
        The forward operator.
    lambda_ref : float
        Regularization parameter tau * lambda_max
    n_orient : int
        The number of orientation (1 : fixed or 3 : free or loose).
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
    return_lpp : bool
        It True the LPP is returned. XXX LPP?
    return_samples : bool
        If True, the samples on returned. XXX which samples?

    Returns
    -------
    Xs : list
        list of all solutions using the K MCMC initilization
    As : list
        list of all active sets of each solution
    XXX : fix given the return_samples and return_lpp
    """
    n_locations = G.shape[1] // n_orient
    n_times = M.shape[1]

    a = n_orient * n_times + 1
    b = 4 / lambda_ref ** 2

    gamma_init = np.ones((n_locations,)) / lambda_ref
    lambda_map = 1 / np.mean(gamma_init)

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
    energy_l2half_reg_vec = np.zeros((K,))

    X_new_MAP = {}
    as_new_MAP = {}
    energy = {}

    Xs = []
    As = []
    X_samples = []
    gamma_samples = []

    for k in range(K):
        # compute new sample by continuing the chain at the last sample
        # and doing 1 step
        X_sample, gamma_sample = L21_gamma_hypermodel_sampler(
            M, G, X_sample[:, :, -1], gamma_sample[:, -1], n_orient,
            b, n_burnin, 2, scK, ssK)

        # compute new full-MAP estimate
        X_new_MAP[k], as_new_MAP[k], gamma_new_MAP, energy[k], _ = \
            L21_gamma_hypermodel_optimizer(G, M, gamma_sample[:, -1], a, b,
                                           n_orient, maxiter, lambda_map)

        # we compute and plot the log posterior probability, the relative
        # residual and the block norm to monitor the sampler
        lppSamples[k], relResSamples[k], blockNormSamples[k] = \
            log_posterior_prob(X_sample[:, :, -1:], gamma_sample[:, -1:], G, M,
                               n_orient, b)
        lppMAP[k], relResMAP[k], blockNormMAP[k] = log_posterior_prob(
            X_new_MAP[k][:, :, np.newaxis], gamma_new_MAP[:, np.newaxis], G, M,
            n_orient, b)

        block_norms_new = compute_block_norms(X_new_MAP[k], n_orient)
        block_norms_new = (block_norms_new > 0.05 * block_norms_new.max())
        solution_support[k, :] = block_norms_new

        energy_l2half_reg_vec[k] = energy_l2half_reg(
            M, G, X_new_MAP[k][as_new_MAP[k]], as_new_MAP[k], lambda_ref,
            n_orient)
        print("sample %s - lppSamp %s - relResSamp %s - lppMAP %s - relResSamp"
              " %s"
              % (k, lppSamples[k], relResSamples[k], lppMAP[k], relResMAP[k]))
        Xs.append(X_new_MAP[k][as_new_MAP[k]])
        As.append(as_new_MAP[k])
        if return_samples:
            X_samples.append(X_sample)
            gamma_samples.append(gamma_samples)

    out = Xs, np.array(As)
    if return_lpp:
        out = out, lppSamples, relResSamples, blockNormSamples, lppMAP
    if return_samples:
        out = out, X_samples, gamma_samples
    return out
