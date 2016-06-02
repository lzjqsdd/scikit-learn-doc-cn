

.. _example_linear_model_plot_bayesian_ridge.py:


=========================
Bayesian Ridge Regression
=========================

Computes a Bayesian Ridge Regression on a synthetic dataset.

See :ref:`bayesian_ridge_regression` for more information on the regressor.

Compared to the OLS (ordinary least squares) estimator, the coefficient
weights are slightly shifted toward zeros, which stabilises them.

As the prior on the weights is a Gaussian prior, the histogram of the
estimated weights is Gaussian.

The estimation of the model is done by iteratively maximizing the
marginal log-likelihood of the observations.



.. rst-class:: horizontal


    *

      .. image:: images/plot_bayesian_ridge_001.png
            :scale: 47

    *

      .. image:: images/plot_bayesian_ridge_002.png
            :scale: 47

    *

      .. image:: images/plot_bayesian_ridge_003.png
            :scale: 47




**Python source code:** :download:`plot_bayesian_ridge.py <plot_bayesian_ridge.py>`

.. literalinclude:: plot_bayesian_ridge.py
    :lines: 19-

**Total running time of the example:**  0.24 seconds
( 0 minutes  0.24 seconds)
    