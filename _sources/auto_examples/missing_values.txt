

.. _example_missing_values.py:


======================================================
Imputing missing values before building an estimator
======================================================

This example shows that imputing the missing values can give better results
than discarding the samples containing any missing value.
Imputing does not always improve the predictions, so please check via cross-validation.
Sometimes dropping rows or using marker values is more effective.

Missing values can be replaced by the mean, the median or the most frequent
value using the ``strategy`` hyper-parameter.
The median is a more robust estimator for data with high magnitude variables
which could dominate results (otherwise known as a 'long tail').

Script output::

  Score with the entire dataset = 0.56
  Score without the samples containing missing values = 0.48
  Score after imputation of the missing values = 0.55

In this case, imputing helps the classifier get close to the original score.
  


**Python source code:** :download:`missing_values.py <missing_values.py>`

.. literalinclude:: missing_values.py
    :lines: 25-
    