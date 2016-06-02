

.. _example_ensemble_plot_adaboost_twoclass.py:


==================
Two-class AdaBoost
==================

This example fits an AdaBoosted decision stump on a non-linearly separable
classification dataset composed of two "Gaussian quantiles" clusters
(see :func:`sklearn.datasets.make_gaussian_quantiles`) and plots the decision
boundary and decision scores. The distributions of decision scores are shown
separately for samples of class A and B. The predicted class label for each
sample is determined by the sign of the decision score. Samples with decision
scores greater than zero are classified as B, and are otherwise classified
as A. The magnitude of a decision score determines the degree of likeness with
the predicted class label. Additionally, a new dataset could be constructed
containing a desired purity of class B, for example, by only selecting samples
with a decision score above some value.




.. image:: images/plot_adaboost_twoclass_001.png
    :align: center




**Python source code:** :download:`plot_adaboost_twoclass.py <plot_adaboost_twoclass.py>`

.. literalinclude:: plot_adaboost_twoclass.py
    :lines: 19-

**Total running time of the example:**  4.11 seconds
( 0 minutes  4.11 seconds)
    