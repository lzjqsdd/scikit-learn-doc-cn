

.. _example_gaussian_process_gp_diabetes_dataset.py:


========================================================================
Gaussian Processes regression: goodness-of-fit on the 'diabetes' dataset
========================================================================

In this example, we fit a Gaussian Process model onto the diabetes
dataset.

We determine the correlation parameters with maximum likelihood
estimation (MLE). We use an anisotropic squared exponential
correlation model with a constant regression model. We also use a
nugget of 1e-2 to account for the (strong) noise in the targets.

We compute a cross-validation estimate of the coefficient of
determination (R2) without reperforming MLE, using the set of correlation
parameters found on the whole dataset.


**Python source code:** :download:`gp_diabetes_dataset.py <gp_diabetes_dataset.py>`

.. literalinclude:: gp_diabetes_dataset.py
    :lines: 21-
    