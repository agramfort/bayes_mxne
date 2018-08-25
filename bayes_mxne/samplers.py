# Authors: Yousra Bekhti <yousra.bekhti@gmail.com>
#          Alexandre Gramfort <alexandre.gramfort@inria.fr>
#          Felix Lucka <f.lucka@ucl.ac.uk>
#          Joseph Salmon <joseph.salmon@telecom-paristech.fr>


from math import log, sqrt, log1p, exp
import numpy as np
from scipy import linalg
from numba import njit, float64

from mne.inverse_sparse.mxne_optim import groups_norm2

from .pyrtnorm import rtnorm
from .utils_random import check_random_state
from .utils_random import _copy_np_state, _copyback_np_state, \
    get_np_state_ptr


@njit(float64(float64, float64), nogil=True)
def _cond_gamma_hyperprior_sampler_aux(coupling, beta):
    # compute maximal value
    if coupling == 0:
        gamma = - beta * log(np.random.rand())
        return gamma

    gammaMax = sqrt(beta * coupling)
    # corresponding probability
    logPmax = - coupling / gammaMax - gammaMax / beta
    # compute point in which exponetial tail umbrella is used
    xCeil = (beta * coupling) / gammaMax + gammaMax
    # compute the probability mass of the umbrella over and below xCeil and
    # total mass
    logPMassOverCeil = log(beta) - xCeil / beta
    logPMassBelowXCeil = logPmax + log(xCeil)
    logPMassTotal = logPMassOverCeil + \
        log1p(exp(logPMassBelowXCeil - logPMassOverCeil))

    while True:
        w, u, v = np.random.rand(), np.random.rand(), np.random.rand()
        # flip a coin in which area of the umbrella we are, use logarithms
        if (log(w) + logPMassTotal) < logPMassOverCeil:
            # we are in the tail, generate an exponentially distributed
            # random variable truncated to [xCeil,infty]
            logV = log(v) - xCeil / beta  # rescale probability v to [0,pCeil]
            gamma = - beta * logV
            # acceptance step
            if (log(u) * beta) < (coupling / logV):
                break
        else:
            # draw from rectangle
            gamma = v * xCeil
            # acceptance step
            if (log(u) + logPmax) < (-coupling / gamma - gamma / beta):
                break

    return gamma


def _cond_gamma_hyperprior_sampler(coupling, beta, random_state=None):
    r"""Sample from distribution of the form

    p(gamma) \prop exp(- coupling / gamma) exp(- gamma / beta)
    """
    rng = check_random_state(random_state)
    hyperprior_sample = _cond_gamma_hyperprior_sampler_aux

    # Put the state of the numpy rng to Numba
    ptr = get_np_state_ptr()
    _copy_np_state(rng, ptr)

    # hyperprior_sample = \
    #     use_numba_random(random_state)(_cond_gamma_hyperprior_sampler_aux)
    if isinstance(coupling, float):
        gamma = hyperprior_sample(coupling, beta)
    else:
        gamma = np.empty(len(coupling))
        for i in range(len(coupling)):
            gamma[i] = hyperprior_sample(coupling[i], beta)

    # Update the state of the numpy rng from Numba rng stae
    _copyback_np_state(rng, ptr)

    return gamma


def _sc_slice_sampler(a, b, c, d, x0, n_samples, random_state):
    r"""Sample from

    p(x) \prop exp(-a x^2 + b x - c \sqrt{x^2 + d})
    """
    rng = check_random_state(random_state)
    if not(a == 0 and b == 0):
        sigma = 1. / sqrt(2. * a)
        mu = b / (2. * a)
    else:
        raise ValueError('this should not happen')

    x = x0
    for k in range(n_samples):
        # sample aux variable y
        log_gy = -c * (sqrt(x**2 + d))

        t = rng.rand()
        log_y = log_gy + log(t)

        # solve for xi
        xi = sqrt((-log_y / c)**2 - d)

        if xi > 0:  # otherwise, there is no interval to sample from
            x = rtnorm(a=-xi, b=xi, mu=mu, sigma=sigma, size=1,
                       random_state=random_state)
        else:
            x = 0

    return x


def _L21_gamma_hypermodel_sampler(M, G, X0, gammas, n_orient, beta, n_burnin,
                                  n_samples, sc_n_samples=10,
                                  ss_n_samples=200,
                                  random_state=None,
                                  verbose=False):
    """Run Gamma sampler

    Parameters
    ----------
    M : array, shape (n_samples, n_times)
        The data.
    G : array, shape (n_samples, n_features)
        The forward operator / design matrix.
    X0 : array, shape (n_features, n_times)
        The initial X.
    gammas : array, shape (n_locations, n_times)
        The initial gammas.
    n_orient : int
        The number of orientation (1 : fixed or 3 : free or loose).
        Used for M/EEG application as there is 3 features per
        physical locations. We have n_locations = n_features // n_orient.
    bela : float
        The beta scale paraemter of the gamma distribution
    n_burnin : int
        The number of iterations in the burnin phase.
    n_samples : int
        The number of sampels to draw.
    ss_n_samples : int
        The number of samples in the slice sampler.
    random_state : int | None
        An integer to fix the seed of the numpy random number
        generator. Necessary to have replicable results.
    verbose : bool
        If True print info on optimization.

    Returns
    -------
    XChain : array, shape (n_features, n_times, n_samples)
        The X samples along the chain.
    gammaChain : array, shape (n_locations, n_samples)
        The gamma samples along the chain.
    """
    rng = check_random_state(random_state)
    n_dipoles = G.shape[1]
    n_locations = n_dipoles // n_orient
    _, n_times = M.shape

    XChain = np.zeros((n_dipoles, n_times, n_samples))
    gammaChain = np.zeros((n_locations, n_samples))

    # precompute some terms
    GColSqNorm = np.sum(G ** 2, axis=0)
    GTM = np.dot(G.T, M)
    GTG = np.dot(G.T, G)

    if not X0.all():
        X = np.zeros((n_locations * n_orient, n_times))
    else:
        X = X0

    for k in range(-n_burnin, n_samples):
        if verbose:
            print("Running iter %d" % k)
        # update X by single component Gibbs sampler
        # initialize with 0 instead of current state (this had a proper reason,
        # but we should re-examine)
        # X = np.zeros((n_locations * n_orient, n_times))

        for kSCGibbs in range(sc_n_samples):
            # print(" -- Running SC iter %d" % kSCGibbs)
            randLocOrder = rng.permutation(n_locations)
            for jLoc in randLocOrder:
                # a only depends on the location
                a = GColSqNorm[jLoc] / 2.
                c = 1. / gammas[jLoc]

                # extract X for this location
                XLoc = X[jLoc * n_orient: (jLoc + 1) * n_orient, :]
                XLocSqNorm = linalg.norm(XLoc, 'fro') ** 2

                # update all time points and all dir without random shuffle
                for jTime in range(n_times):
                    for jDir in range(n_orient):
                        # get corresponding dipole, time and block index
                        jComp = jDir + jLoc * n_orient
                        XjComp = X[jComp, jTime]
                        # compute b and d
                        b = GTM[jComp, jTime] - np.dot(X[:, jTime].T,
                                                       GTG[:, jComp]) + \
                            2 * a * XjComp
                        d = XLocSqNorm - XjComp**2
                        # call slice sampler
                        XjComp = _sc_slice_sampler(
                            a, b, c, d, XjComp, ss_n_samples, rng)
                        # update auxillary variables
                        XLocSqNorm = d + XjComp**2
                        X[jComp, jTime] = XjComp

        # check for instabilities cause by insufficient sampling steps,
        # usually leading to an explosion of the residual
        # if (linalg.norm(G.dot(X) - M, 'fro') / linalg.norm(M, 'fro')) > 10:
        #     raise ValueError('relative residual exceeded threshold, '
        #                      'the sampler is likely to diverge due to '
        #                      'insufficient precision in the block-sampling')

        # update gamma by umbrella sampler
        # Compute the amplitudes of the sources for one hyperparameter
        XBlkNorm = np.sqrt(groups_norm2(X.copy(), n_orient))
        gammas = _cond_gamma_hyperprior_sampler(XBlkNorm, beta, rng)

        # store results
        if k >= 0:
            XChain[:, :, k] = X
            gammaChain[:, k] = gammas

    return XChain, gammaChain


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from scipy.integrate import quad

    size = 10000
    beta = 1.
    coupling = 1.
    couplings = coupling * np.ones(size)

    import time
    t0 = time.time()
    gammas = _cond_gamma_hyperprior_sampler(couplings, beta)
    print(time.time() - t0)

    plt.close('all')

    xmin, xmax = np.min(gammas), np.max(gammas)
    xx = np.linspace(xmin, xmax, 1000)

    def dist(xx):
        return np.exp(- coupling / xx) * np.exp(- xx / beta)

    Z, _ = quad(dist, 1e-5, 20)

    plt.figure()
    plt.hist(gammas, normed=True, bins=20)
    plt.plot(xx, dist(xx) / Z, 'r', linewidth=2)
    plt.show()

    (a, b, c, d), x0, n_samples = (1,) * 4, 0., 1000
    chain = _sc_slice_sampler(a, b, c, d, x0, n_samples)

    def dist(xx):
        return np.exp(-a * xx ** 2 + b * xx - c * np.sqrt(xx ** 2 + d))
    xx = np.linspace(-2, 3, 300)

    Z, _ = quad(dist, -5, 5)

    plt.figure()
    plt.hist(chain, normed=True, bins=20)
    plt.plot(xx, dist(xx) / Z, 'r', linewidth=2)
    plt.show()
