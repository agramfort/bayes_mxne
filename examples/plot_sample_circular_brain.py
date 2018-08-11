"""
===================================
Circular plot of the configurations
===================================

This example demonstrates how to run a MM solver using K different
MCMC initilization. Plot then on a circular plot all the configurations
of the source localization.
"""
# Authors: Yousra Bekhti <yousra.bekhti@gmail.com>
#          Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>

# License: BSD (3-clause)

from copy import deepcopy
import numpy as np

import mne
from mne.datasets import sample

from mne.inverse_sparse.mxne_inverse import \
    (_prepare_gain, is_fixed_orient, _make_sparse_stc)
from mne.inverse_sparse.mxne_optim import norm_l2inf

from bayes_meeg.gamma_hypermodel_optimizer import (mm_mixed_norm_bayes,
                                                   compute_block_norms)
from bayes_meeg.config_plots import energy_l2half_reg, circular_brain_plot

###############################################################################
# Let us read in the `fif` file for MNE sample dataset corresponding
# to the forward, the evoked and the covariance matrix.
data_path = sample.data_path()

fwd_fname = data_path + '/MEG/sample/sample_audvis-meg-eeg-oct-6-fwd.fif'
ave_fname = data_path + '/MEG/sample/sample_audvis-ave.fif'
cov_fname = data_path + '/MEG/sample/sample_audvis-shrunk-cov.fif'

subjects_dir = data_path + '/subjects'
condition = 'Left Auditory'

# Read noise covariance matrix
noise_cov = mne.read_cov(cov_fname)

# Handling average file
evoked = mne.read_evokeds(ave_fname, condition=condition, baseline=(None, 0))
evoked.crop(tmin=0.04, tmax=0.18)
evoked = evoked.pick_types(eeg=False, meg=True)

# Handling forward solution
forward = mne.read_forward_solution(fwd_fname)
forward = mne.convert_forward_solution(forward, surf_ori=True)


###############################################################################
# Run solver method

def apply_solver(evoked, forward, noise_cov, loose=0.2, depth=0.8, K=2000):
    all_ch_names = evoked.ch_names
    # put the forward solution in fixed orientation if it's not already
    if loose is None and not is_fixed_orient(forward):
        forward = deepcopy(forward)
        forward = mne.convert_forward_solution(forward, force_fixed=True)

    gain, gain_info, whitener, source_weighting, mask = _prepare_gain(
        forward, evoked.info, noise_cov, pca=False, depth=depth,
        loose=loose, weights=None, weights_min=None)

    n_locations = gain.shape[1]
    sel = [all_ch_names.index(name) for name in gain_info['ch_names']]
    M = evoked.data[sel]

    # Whiten data
    M = np.dot(whitener, M)

    n_orient = 1 if is_fixed_orient(forward) else 3

    # The value of lambda for which the solution will be all zero
    lambda_max = norm_l2inf(np.dot(gain.T, M), n_orient)

    lambda_ref = 0.1 * lambda_max

    out = mm_mixed_norm_bayes(
        M, gain, lambda_ref, n_orient=n_orient, K=K, return_lpp=True)

    (Xs, active_sets), _, _, _, _ = out

    solution_support = np.zeros((K, n_locations))
    stcs, obj_fun = [], []
    for k in range(K):
        X = np.zeros((n_locations, Xs[k].shape[1]))
        X[active_sets[k]] = Xs[k]
        block_norms_new = compute_block_norms(X, n_orient)
        block_norms_new = (block_norms_new > 0.05 * block_norms_new.max())
        solution_support[k, :] = block_norms_new

        stc = _make_sparse_stc(Xs[k], active_sets[k], forward, tmin=0.,
                               tstep=1. / evoked.info['sfreq'])
        stcs.append(stc)
        obj_fun.append(energy_l2half_reg(M, gain, stc.data, active_sets[k],
                       lambda_ref, n_orient))
    return solution_support, stcs, obj_fun


###############################################################################
# Apply your solver

loose, depth, K = None, 0.8, 5
out = apply_solver(evoked, forward, noise_cov, loose=loose,
                   depth=depth, K=K)
solution_support, stcs, obj_fun = out

###############################################################################
# Plotting of all configurations intro a circular plot
circular_brain_plot(forward, solution_support, stcs, obj_fun)
