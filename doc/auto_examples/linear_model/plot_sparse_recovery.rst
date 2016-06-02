

.. _example_linear_model_plot_sparse_recovery.py:


============================================================
Sparse recovery: feature selection for sparse linear models
============================================================

Given a small number of observations, we want to recover which features
of X are relevant to explain y. For this :ref:`sparse linear models
<l1_feature_selection>` can outperform standard statistical tests if the
true model is sparse, i.e. if a small fraction of the features are
relevant.

As detailed in :ref:`the compressive sensing notes
<compressive_sensing>`, the ability of L1-based approach to identify the
relevant variables depends on the sparsity of the ground truth, the
number of samples, the number of features, the conditioning of the
design matrix on the signal subspace, the amount of noise, and the
absolute value of the smallest non-zero coefficient [Wainwright2006]
(http://statistics.berkeley.edu/tech-reports/709.pdf).

Here we keep all parameters constant and vary the conditioning of the
design matrix. For a well-conditioned design matrix (small mutual
incoherence) we are exactly in compressive sensing conditions (i.i.d
Gaussian sensing matrix), and L1-recovery with the Lasso performs very
well. For an ill-conditioned matrix (high mutual incoherence),
regressors are very correlated, and the Lasso randomly selects one.
However, randomized-Lasso can recover the ground truth well.

In each situation, we first vary the alpha parameter setting the sparsity
of the estimated model and look at the stability scores of the randomized
Lasso. This analysis, knowing the ground truth, shows an optimal regime
in which relevant features stand out from the irrelevant ones. If alpha
is chosen too small, non-relevant variables enter the model. On the
opposite, if alpha is selected too large, the Lasso is equivalent to
stepwise regression, and thus brings no advantage over a univariate
F-test.

In a second time, we set alpha and compare the performance of different
feature selection methods, using the area under curve (AUC) of the
precision-recall.



.. rst-class:: horizontal


    *

      .. image:: images/plot_sparse_recovery_001.png
            :scale: 47

    *

      .. image:: images/plot_sparse_recovery_002.png
            :scale: 47

    *

      .. image:: images/plot_sparse_recovery_003.png
            :scale: 47

    *

      .. image:: images/plot_sparse_recovery_004.png
            :scale: 47




**Python source code:** :download:`plot_sparse_recovery.py <plot_sparse_recovery.py>`

.. literalinclude:: plot_sparse_recovery.py
    :lines: 41-

**Total running time of the example:**  6.10 seconds
( 0 minutes  6.10 seconds)
    