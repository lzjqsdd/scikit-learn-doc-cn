

.. _example_linear_model_plot_theilsen.py:


====================
Theil-Sen Regression
====================

Computes a Theil-Sen Regression on a synthetic dataset.

See :ref:`theil_sen_regression` for more information on the regressor.

Compared to the OLS (ordinary least squares) estimator, the Theil-Sen
estimator is robust against outliers. It has a breakdown point of about 29.3%
in case of a simple linear regression which means that it can tolerate
arbitrary corrupted data (outliers) of up to 29.3% in the two-dimensional
case.

The estimation of the model is done by calculating the slopes and intercepts
of a subpopulation of all possible combinations of p subsample points. If an
intercept is fitted, p must be greater than or equal to n_features + 1. The
final slope and intercept is then defined as the spatial median of these
slopes and intercepts.

In certain cases Theil-Sen performs better than :ref:`RANSAC
<ransac_regression>` which is also a robust method. This is illustrated in the
second example below where outliers with respect to the x-axis perturb RANSAC.
Tuning the ``residual_threshold`` parameter of RANSAC remedies this but in
general a priori knowledge about the data and the nature of the outliers is
needed.
Due to the computational complexity of Theil-Sen it is recommended to use it
only for small problems in terms of number of samples and features. For larger
problems the ``max_subpopulation`` parameter restricts the magnitude of all
possible combinations of p subsample points to a randomly chosen subset and
therefore also limits the runtime. Therefore, Theil-Sen is applicable to larger
problems with the drawback of losing some of its mathematical properties since
it then works on a random subset.



.. rst-class:: horizontal


    *

      .. image:: images/plot_theilsen_001.png
            :scale: 47

    *

      .. image:: images/plot_theilsen_002.png
            :scale: 47




**Python source code:** :download:`plot_theilsen.py <plot_theilsen.py>`

.. literalinclude:: plot_theilsen.py
    :lines: 36-

**Total running time of the example:**  2.04 seconds
( 0 minutes  2.04 seconds)
    