

.. _example_ensemble_plot_adaboost_regression.py:


======================================
Decision Tree Regression with AdaBoost
======================================

A decision tree is boosted using the AdaBoost.R2 [1] algorithm on a 1D
sinusoidal dataset with a small amount of Gaussian noise.
299 boosts (300 decision trees) is compared with a single decision tree
regressor. As the number of boosts is increased the regressor can fit more
detail.

.. [1] H. Drucker, "Improving Regressors using Boosting Techniques", 1997.




.. image:: images/plot_adaboost_regression_001.png
    :align: center




**Python source code:** :download:`plot_adaboost_regression.py <plot_adaboost_regression.py>`

.. literalinclude:: plot_adaboost_regression.py
    :lines: 15-

**Total running time of the example:**  0.46 seconds
( 0 minutes  0.46 seconds)
    