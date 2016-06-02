

.. _example_calibration_plot_calibration_multiclass.py:


==================================================
Probability Calibration for 3-class classification
==================================================

This example illustrates how sigmoid calibration changes predicted
probabilities for a 3-class classification problem. Illustrated is the
standard 2-simplex, where the three corners correspond to the three classes.
Arrows point from the probability vectors predicted by an uncalibrated
classifier to the probability vectors predicted by the same classifier after
sigmoid calibration on a hold-out validation set. Colors indicate the true
class of an instance (red: class 1, green: class 2, blue: class 3).

The base classifier is a random forest classifier with 25 base estimators
(trees). If this classifier is trained on all 800 training datapoints, it is
overly confident in its predictions and thus incurs a large log-loss.
Calibrating an identical classifier, which was trained on 600 datapoints, with
method='sigmoid' on the remaining 200 datapoints reduces the confidence of the
predictions, i.e., moves the probability vectors from the edges of the simplex
towards the center. This calibration results in a lower log-loss. Note that an
alternative would have been to increase the number of base estimators which
would have resulted in a similar decrease in log-loss.



.. rst-class:: horizontal


    *

      .. image:: images/plot_calibration_multiclass_000.png
            :scale: 47

    *

      .. image:: images/plot_calibration_multiclass_001.png
            :scale: 47


**Script output**::

  Log-loss of
   * uncalibrated classifier trained on 800 datapoints: 1.280 
   * classifier trained on 600 datapoints and calibrated on 200 datapoint: 0.534



**Python source code:** :download:`plot_calibration_multiclass.py <plot_calibration_multiclass.py>`

.. literalinclude:: plot_calibration_multiclass.py
    :lines: 24-

**Total running time of the example:**  0.64 seconds
( 0 minutes  0.64 seconds)
    