

.. _example_model_selection_plot_learning_curve.py:


========================
Plotting Learning Curves
========================

On the left side the learning curve of a naive Bayes classifier is shown for
the digits dataset. Note that the training score and the cross-validation score
are both not very good at the end. However, the shape of the curve can be found
in more complex datasets very often: the training score is very high at the
beginning and decreases and the cross-validation score is very low at the
beginning and increases. On the right side we see the learning curve of an SVM
with RBF kernel. We can see clearly that the training score is still around
the maximum and the validation score could be increased with more training
samples.



.. rst-class:: horizontal


    *

      .. image:: images/plot_learning_curve_001.png
            :scale: 47

    *

      .. image:: images/plot_learning_curve_002.png
            :scale: 47




**Python source code:** :download:`plot_learning_curve.py <plot_learning_curve.py>`

.. literalinclude:: plot_learning_curve.py
    :lines: 16-

**Total running time of the example:**  9.19 seconds
( 0 minutes  9.19 seconds)
    