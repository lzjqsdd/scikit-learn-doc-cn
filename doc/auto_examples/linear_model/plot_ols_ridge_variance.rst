

.. _example_linear_model_plot_ols_ridge_variance.py:


=========================================================
Ordinary Least Squares and Ridge Regression Variance
=========================================================
Due to the few points in each dimension and the straight
line that linear regression uses to follow these points
as well as it can, noise on the observations will cause
great variance as shown in the first plot. Every line's slope
can vary quite a bit for each prediction due to the noise
induced in the observations.

Ridge regression is basically minimizing a penalised version
of the least-squared function. The penalising `shrinks` the
value of the regression coefficients.
Despite the few data points in each dimension, the slope
of the prediction is much more stable and the variance
in the line itself is greatly reduced, in comparison to that
of the standard linear regression



.. rst-class:: horizontal


    *

      .. image:: images/plot_ols_ridge_variance_001.png
            :scale: 47

    *

      .. image:: images/plot_ols_ridge_variance_002.png
            :scale: 47




**Python source code:** :download:`plot_ols_ridge_variance.py <plot_ols_ridge_variance.py>`

.. literalinclude:: plot_ols_ridge_variance.py
    :lines: 23-

**Total running time of the example:**  0.24 seconds
( 0 minutes  0.24 seconds)
    