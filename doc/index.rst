.. bayes_mxne documentation master file, created by
   sphinx-quickstart on Mon May 23 16:22:52 2016.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Documentation of the bayes_mxne package
=======================================

Installation
============

We recommend the `Anaconda Python distribution <https://www.continuum.io/downloads>`_. To install ``bayes_mxne``, you first need to install its dependencies which are MNE and numba.

For instructions on how to install MNE see: http://martinos.org/mne/stable/install_mne_python.html

For numba::

	$ conda install numba

should do it.

If you want to install the latest version of the code (nightly) use::

	$ pip install https://api.github.com/repos/agramfort/bayes_mxne/zipball/master

If you do not have admin privileges on the computer, use the ``--user`` flag
with `pip`. To upgrade, use the ``--upgrade`` flag provided by `pip`.

To check if everything worked fine, you can do::

	$ python -c 'import bayes_mxne'

and it should not give any error messages.

Bug reports
===========

Use the `github issue tracker <https://github.com/agramfort/bayes_mxne/issues>`_ to report bugs.

Cite
====

[1] Bekhti, Y., Lucka, F., Salmon, J., & Gramfort, A. (2018). "`A hierarchical Bayesian perspective on majorization-minimization for non-convex sparse regression: application to M/EEG source imaging <http://iopscience.iop.org/article/10.1088/1361-6420/aac9b3/pdf>`_."
Inverse Problems, Volume 34, Number 8.
