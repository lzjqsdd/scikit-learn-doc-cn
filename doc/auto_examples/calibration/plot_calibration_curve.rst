

.. _example_calibration_plot_calibration_curve.py:


==============================
Probability Calibration curves
==============================

When performing classification one often wants to predict not only the class
label, but also the associated probability. This probability gives some
kind of confidence on the prediction. This example demonstrates how to display
how well calibrated the predicted probabilities are and how to calibrate an
uncalibrated classifier.

The experiment is performed on an artificial dataset for binary classification
with 100.000 samples (1.000 of them are used for model fitting) with 20
features. Of the 20 features, only 2 are informative and 10 are redundant. The
first figure shows the estimated probabilities obtained with logistic
regression, Gaussian naive Bayes, and Gaussian naive Bayes with both isotonic
calibration and sigmoid calibration. The calibration performance is evaluated
with Brier score, reported in the legend (the smaller the better). One can
observe here that logistic regression is well calibrated while raw Gaussian
naive Bayes performs very badly. This is because of the redundant features
which violate the assumption of feature-independence and result in an overly
confident classifier, which is indicated by the typical transposed-sigmoid
curve.

Calibration of the probabilities of Gaussian naive Bayes with isotonic
regression can fix this issue as can be seen from the nearly diagonal
calibration curve. Sigmoid calibration also improves the brier score slightly,
albeit not as strongly as the non-parametric isotonic regression. This can be
attributed to the fact that we have plenty of calibration data such that the
greater flexibility of the non-parametric model can be exploited.

The second figure shows the calibration curve of a linear support-vector
classifier (LinearSVC). LinearSVC shows the opposite behavior as Gaussian
naive Bayes: the calibration curve has a sigmoid curve, which is typical for
an under-confident classifier. In the case of LinearSVC, this is caused by the
margin property of the hinge loss, which lets the model focus on hard samples
that are close to the decision boundary (the support vectors).

Both kinds of calibration can fix this issue and yield nearly identical
results. This shows that sigmoid calibration can deal with situations where
the calibration curve of the base classifier is sigmoid (e.g., for LinearSVC)
but not where it is transposed-sigmoid (e.g., Gaussian naive Bayes).



.. rst-class:: horizontal


    *

      .. image:: images/plot_calibration_curve_001.png
            :scale: 47

    *

      .. image:: images/plot_calibration_curve_002.png
            :scale: 47


**Script output**::

  Logistic:
          Brier: 0.099
          Precision: 0.872
          Recall: 0.851
          F1: 0.862
  
  Naive Bayes:
          Brier: 0.118
          Precision: 0.857
          Recall: 0.876
          F1: 0.867
  
  Naive Bayes + Isotonic:
          Brier: 0.098
          Precision: 0.883
          Recall: 0.836
          F1: 0.859
  
  Naive Bayes + Sigmoid:
          Brier: 0.109
          Precision: 0.861
          Recall: 0.871
          F1: 0.866
  
  Logistic:
          Brier: 0.099
          Precision: 0.872
          Recall: 0.851
          F1: 0.862
  
  SVC:
          Brier: 0.163
          Precision: 0.872
          Recall: 0.852
          F1: 0.862
  
  SVC + Isotonic:
          Brier: 0.100
          Precision: 0.853
          Recall: 0.878
          F1: 0.865
  
  SVC + Sigmoid:
          Brier: 0.099
          Precision: 0.874
          Recall: 0.849
          F1: 0.861



**Python source code:** :download:`plot_calibration_curve.py <plot_calibration_curve.py>`

.. literalinclude:: plot_calibration_curve.py
    :lines: 44-

**Total running time of the example:**  2.94 seconds
( 0 minutes  2.94 seconds)
    