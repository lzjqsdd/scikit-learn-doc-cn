

.. _example_model_selection_plot_confusion_matrix.py:


================
Confusion matrix
================

Example of confusion matrix usage to evaluate the quality
of the output of a classifier on the iris data set. The
diagonal elements represent the number of points for which
the predicted label is equal to the true label, while
off-diagonal elements are those that are mislabeled by the
classifier. The higher the diagonal values of the confusion
matrix the better, indicating many correct predictions.

The figures show the confusion matrix with and without
normalization by class support size (number of elements
in each class). This kind of normalization can be
interesting in case of class imbalance to have a more
visual interpretation of which class is being misclassified.

Here the results are not as good as they could be as our
choice for the regularization parameter C was not the best.
In real life applications this parameter is usually chosen
using :ref:`grid_search`.




.. rst-class:: horizontal


    *

      .. image:: images/plot_confusion_matrix_001.png
            :scale: 47

    *

      .. image:: images/plot_confusion_matrix_002.png
            :scale: 47


**Script output**::

  Confusion matrix, without normalization
  [[13  0  0]
   [ 0 10  6]
   [ 0  0  9]]
  Normalized confusion matrix
  [[ 1.    0.    0.  ]
   [ 0.    0.62  0.38]
   [ 0.    0.    1.  ]]



**Python source code:** :download:`plot_confusion_matrix.py <plot_confusion_matrix.py>`

.. literalinclude:: plot_confusion_matrix.py
    :lines: 26-

**Total running time of the example:**  0.31 seconds
( 0 minutes  0.31 seconds)
    