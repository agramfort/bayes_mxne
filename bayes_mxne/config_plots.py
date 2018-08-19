"""Config file for plots."""

from os.path import expanduser
import numpy as np
from numpy.linalg import norm

import mne
from mne.datasets import sample
from mne.inverse_sparse.mxne_optim import groups_norm2
from mne.viz import circular_layout, plot_connectivity_circle

data_path = expanduser('~') + '/Dropbox/bayes_mxne_data/'


def circular_brain_plot(forward, solution_support, stcs, obj_fun,
                        label_name=False, plot_circular=True,
                        plot_labels=True, n_burnin=0, vmin=0., vmax=10.,
                        plot_hist=False, subplot=111, title='', fig=None,
                        colorbar=True):
    import matplotlib.pylab as plt

    indices = list()
    vertices_lh = list()
    vertices_rh = list()

    support = np.unique(np.where(solution_support)[1])
    indices.append(support)
    for stc in stcs:
        vertices_lh.append(stc.vertices[0])
        vertices_rh.append(stc.vertices[1])

    indices = np.unique(np.concatenate(np.array(indices)))
    vertices_lh = np.unique(np.concatenate(np.array(vertices_lh)))
    vertices_rh = np.unique(np.concatenate(np.array(vertices_rh)))

    n_sources = solution_support.shape[1]
    # Get n_used of the left hemi
    n_used = forward['src'][0]['nuse']
    n_rh = forward['src'][1]['nuse']
    #
    indices_lh = indices[np.where(indices < n_used)[0]]
    indices_rh = indices[np.where(indices >= n_used)[0]]

    indices_sym = np.sort(np.concatenate(np.array(
        [indices, indices_lh + n_used, indices_rh - n_rh])))

    #
    n_support = len(indices_sym)
    iindices = [list(indices_sym).index(ii) for ii in indices]

    # Remove burnin
    solution_support = solution_support[n_burnin:, :]
    stcs = stcs[n_burnin:]

    # Plot histogram of the different objective function values
    # obtained with the MCMC init after burnin.
    if plot_hist:
        out = plt.hist(obj_fun)
        ymin, ymax = 0, out[0].max()
        plt.vlines(obj_fun[0], ymin=ymin, ymax=ymax)
        plt.ylim([ymin, ymax])
        plt.yticks(fontsize=16)
        plt.xticks(fontsize=16)
        plt.xlabel('Objective function values', fontsize=16)
        plt.legend(['MM solution', 'MCMC initializations'], fontsize=16)
        plt.show()

    # ####
    ind = np.unique(np.where(solution_support)[1])
    # ####

    # Take the results for all indices
    data_support_res = solution_support[:, indices]

    # Construct cooeccurrence matrix
    cooecc = np.zeros((n_sources, n_sources))
    ixgrid = np.ix_(indices, indices)
    cooecc[ixgrid] = np.dot(data_support_res.T, data_support_res)

    ixgrid = np.ix_(indices_sym, indices_sym)
    cooecc = cooecc[ixgrid]

    # Read labels
    # data_path = sample.data_path()
    subjects_dir = sample.data_path() + '/subjects'
    labels = mne.read_labels_from_annot('sample', parc='aparc.a2009s',
                                        subjects_dir=subjects_dir)

    # First, we reorder the labels based on their location in the left hemi
    label_names = [label.name for label in labels]

    lh_labels = [name for name in label_names if name.endswith('lh')]

    # Get the y-location of the label
    label_ypos = list()
    for name in lh_labels:
        idx = label_names.index(name)
        ypos = np.mean(labels[idx].pos[:, 1])
        label_ypos.append(ypos)

    # Reorder the labels based on their location
    lh_labels = [label for (yp, label) in sorted(zip(label_ypos, lh_labels))]

    # Find the corresponding label name to the vertices
    lh_names = list()
    length = 0
    for vertex in vertices_lh:
        for label in labels[::2]:
            if vertex in label.vertices:
                lh_names.append(label.name)
        if len(lh_names) != length + 1:
            lh_names.append(labels[56].name)
        length += 1

    rh_names = list()
    length = 0
    for vertex in vertices_rh:
        for label in labels[1::2]:
            if vertex in label.vertices:
                rh_names.append(label.name)

        if len(rh_names) != length + 1:
            rh_names.append(labels[57].name)
        length += 1

    names = lh_names + rh_names
    names_sym = np.array([''] * n_support, dtype='U30')
    names_sym[iindices] = names
    for ii in np.where(names_sym == '')[0]:
        if indices_sym[ii] < n_used:
            ind = np.where(indices == indices_sym[ii] + n_rh)[0]
            if ind.shape[0] == 0:
                ind = np.where(indices == indices_sym[ii] - 2 + n_rh)[0]
            name = np.array(names)[ind][0][:-3] + '-lh'
        else:
            ind = np.where(indices == indices_sym[ii] - n_used)[0]
            if ind.shape[0] == 0:
                ind = np.where(indices == indices_sym[ii] - 2 - n_used)[0]
            name = np.array(names)[ind][0][:-3] + '-rh'
        names_sym[ii] = name

    names = names_sym
    dipole_colors = list()
    names_lh = list()
    # For each found label find its color
    for label in names:
        if label[:-3] != 'n':
            idx = label_names.index(label)
            if labels[idx].color == (0., 0., 0., 1.):
                labels[idx].color = (0.5, 0.5, 0.5, 1.)
            dipole_colors.append(labels[idx].color)
            names_lh.append(label[:-3])
        else:
            dipole_colors.append((0., 0., 0., 1.))
            names_lh.append('none')
    names_lh = names_lh[:n_support // 2]

    seen_labels = list()
    node_order = list()
    # Find the good order for names and cooecc
    for label in lh_labels:
        if label not in seen_labels and label[:-3] in names_lh:
            node_order.append(np.where(np.array(names_lh) == label[:-3])[0])
            seen_labels.append(label)

    lh_order = list(np.concatenate(np.array(node_order)))

    node_order = lh_order[::-1]

    # colors = list(np.array(dipole_colors)[node_order])
    node_order.extend(list(np.array(lh_order) + len(lh_order)))

    node_width = 2 * 180. / n_support

    label_indices = range(n_support)
    node_angles = circular_layout(label_indices, node_order, start_pos=90,
                                  group_boundaries=[0, len(label_indices) / 2])

    dipole_colors = dipole_colors[:n_support // 2]
    colors = np.concatenate(np.array([dipole_colors, dipole_colors]))

    if not label_name:
        names = np.array([''] * len(indices_sym), dtype=np.dtype((str, 35)))

    if plot_circular:
        fig, ax = plot_connectivity_circle(cooecc, names, n_lines=None,
                                           node_angles=node_angles,
                                           node_colors=colors,
                                           node_width=node_width,
                                           vmin=vmin, vmax=vmax, fig=fig,
                                           subplot=subplot, title=title,
                                           padding=4, textcolor='black',
                                           facecolor='white',
                                           colormap='viridis',
                                           node_edgecolor=None,
                                           colorbar=colorbar)

        plot_bars_circular(n_support, node_angles, cooecc.diagonal(), ax)
    if plot_labels:
        brain_labels(labels, names_sym, subjects_dir)


def plot_bars_circular(n_support, node_angles, freqs, ax):
    # Plot the interaction circle with bars
    width, bottom = 2 * np.pi / n_support, 10
    theta = node_angles * np.pi / 180.
    ax.bar(theta, freqs / 10., width=width, bottom=bottom,
           color='black', edgecolor='black', align='center')


def brain_labels(labels, names, subjects_dir, stc=None,
                 title=None, hemi='both', view=['med'], save=False,
                 fname='', dataset='sample_LAud'):
    from surfer import Brain
    n_support = names.shape[0]
    label_names = [label.name for label in labels]
    # Plot the selected labels in a Brain
    brain = Brain('sample', hemi=hemi, surf='inflated',
                  subjects_dir=subjects_dir, title=title,
                  views=view, background='white')
    for label in np.unique(names[:n_support // 2]):
        if hemi == 'both':
            # Left hemi
            idx = label_names.index(label)
            brain.add_label(labels[idx], color=labels[idx].color)
            # Right hemi
            idx = label_names.index(label[:-3] + '-rh')
            brain.add_label(labels[idx], color=labels[idx].color)
        elif hemi == 'lh':
            # Left hemi
            idx = label_names.index(label)
            brain.add_label(labels[idx], color=labels[idx].color)
        elif hemi == 'rh':
            # Right hemi
            idx = label_names.index(label[:-3] + '-rh')
            brain.add_label(labels[idx], color=labels[idx].color)

    if save:
        brain.save_montage('paper_figures/images/' + fname + '_' + view[0] +
                           '_' + hemi + '.png', order=view, border_size=1)


def plot_vertices(vertices_lh, vertices_rh, alpha=0.5, save=False,
                  fname=None, simulated=False, col='green'):
    from surfer import Brain
    # Read labels
    subjects_dir = sample.data_path() + '/subjects'
    lh_vertex = 114573  # Auditory label
    rh_vertex = 53641  # Medial occipital label

    brain = Brain('sample', hemi='lh', surf='inflated',
                  subjects_dir=subjects_dir, title='lh',
                  background='white')

    if simulated:
        brain.add_foci(lh_vertex, coords_as_verts=True, color='red',
                       hemi='lh')
    brain.add_foci(vertices_lh, coords_as_verts=True, color=col,
                   alpha=alpha, hemi='lh')

    if save:
        brain.save_montage('paper_figures/images/' + fname + '_lat_lh.png',
                           order=['lat'], border_size=1)

    brain = Brain('sample', hemi='rh', surf='inflated',
                  subjects_dir=subjects_dir, title='rh', views=['lat'],
                  background='white')

    if simulated:
        brain.add_foci(rh_vertex, coords_as_verts=True, color='red',
                       hemi='rh')
    brain.add_foci(vertices_rh, coords_as_verts=True, color=col,
                   alpha=alpha, hemi='rh')

    if save:
        brain.save_montage('paper_figures/images/' + fname + '_lat_rh.png',
                           order=['lat'], border_size=1)

    return brain


def compute_block_norms(w, nDir):
    return np.sqrt(groups_norm2(w.copy(), nDir))


def energy_l2half_reg(M, G, X, active_set, lambda_l2half, nDir):
    reg = lambda_l2half * np.sqrt(compute_block_norms(X, nDir)).sum()
    return norm(np.dot(G[:, active_set], X) - M, 'fro') + reg


def plot_heat_maps(dataset, meg_type, eeg_type, samp, ori, ico, smooth=7.,
                   n_burnin=0, n_best_init=-1, colorbar=False, save=False):
    lh_vertex = 114573  # Auditory label
    rh_vertex = 53641  # Medial occipital label

    # Get src
    fname = data_path + 'datasets/sample_audvis-%s-fwd.fif' % ico
    fwd = mne.read_forward_solution(fname, surf_ori=True, force_fixed=True)
    src = fwd['src']

    fname = 'results/%s_full_map_with_MCMC_init_eeg_%s_meg_%s%s%s.npy' \
        % (dataset, eeg_type, meg_type, samp, ori)

    data = np.load(data_path + fname, encoding='latin1')[0]
    solution_support = data['solution_support']

    # Remove burnin
    solution_support = solution_support[n_burnin:, :]

    vertices = [src[0]['vertno'], src[1]['vertno']]
    stc = mne.SourceEstimate(solution_support.T, vertices, tmin=0., tstep=1.,
                             subject='sample')

    norm = np.linalg.norm(stc.data, axis=0) ** 2
    stc._data /= norm
    stc_to = stc.morph('sample', smooth=smooth)

    brain = stc_to.copy().mean().plot('sample', hemi='lh', figure=1,
                                      clim={'lims': [1.e-4, 5.e-4, 1.e-1],
                                            'kind': 'value'}, time_label=None,
                                      background='white', colorbar=colorbar)
    if dataset == 'simulated':
        brain.add_foci(lh_vertex, coords_as_verts=True, color='green',
                       hemi='lh')

    if save:
        fname = '%s_heat_map_eeg_%s_meg_%s%s%s_lat_lh.png' \
            % (dataset, eeg_type, meg_type, ori, samp)

        brain.save_montage('paper_figures/images/' + fname,
                           order=['lat'], border_size=1)
        brain.close()

    brain = stc_to.copy().mean().plot('sample', hemi='rh', figure=2,
                                      views=['lat'],
                                      clim={'lims': [1.e-4, 5.e-4, 1.e-1],
                                            'kind': 'value'},
                                      time_label=None, background='white',
                                      colorbar=colorbar)
    if dataset == 'simulated':
        brain.add_foci(rh_vertex, coords_as_verts=True, color='green',
                       hemi='rh')

    if save:
        fname = '%s_heat_map_eeg_%s_meg_%s%s%s_lat_rh.png' \
            % (dataset, eeg_type, meg_type, ori, samp)

        brain.save_montage('paper_figures/images/' + fname,
                           order=['lat'], border_size=1)
        brain.close()

    return stc_to
