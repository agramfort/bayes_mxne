M/EEG Source Localization
=========================

This repository hosts the code to solve the M/EEG inverse problem. It improves the majorization-minimization techniques by probing the multimodal posterior density using Markov Chain Monte-Carlo (MCMC) techniques.

More details in **[1]** to see how this method reveals the different modes of the posterior distribution in order to explore and quantify the inherent uncertainty and ambiguity of such ill-posed inference procedure. In the context of M/EEG, each mode corresponds to a plausible configuration of neural sources, which is crucial for data interpretation, especially in clinical contexts.

Cite
----
[1] Bekhti, Y., Lucka, F., Salmon, J., & Gramfort, A. (2018). A hierarchical Bayesian perspective on majorization-minimization for non-convex sparse regression: application to M/EEG source imaging. Inverse Problems. [paper](http://iopscience.iop.org/article/10.1088/1361-6420/aac9b3/pdf)
