

.. _example_model_selection_plot_roc.py:


=======================================
Receiver Operating Characteristic (ROC)
=======================================

Example of Receiver Operating Characteristic (ROC) metric to evaluate
classifier output quality.

ROC curves typically feature true positive rate on the Y axis, and false
positive rate on the X axis. This means that the top left corner of the plot is
the "ideal" point - a false positive rate of zero, and a true positive rate of
one. This is not very realistic, but it does mean that a larger area under the
curve (AUC) is usually better.

The "steepness" of ROC curves is also important, since it is ideal to maximize
the true positive rate while minimizing the false positive rate.

Multiclass settings
-------------------

ROC curves are typically used in binary classification to study the output of
a classifier. In order to extend ROC curve and ROC area to multi-class
or multi-label classification, it is necessary to binarize the output. One ROC
curve can be drawn per label, but one can also draw a ROC curve by considering
each element of the label indicator matrix as a binary prediction
(micro-averaging).

Another evaluation measure for multi-class classification is
macro-averaging, which gives equal weight to the classification of each
label.

.. note::

    See also :func:`sklearn.metrics.roc_auc_score`,
             :ref:`example_model_selection_plot_roc_crossval.py`.




.. rst-class:: horizontal


    *

      .. image:: images/plot_roc_001.png
            :scale: 47

    *

      .. image:: images/plot_roc_002.png
            :scale: 47




**Python source code:** :download:`plot_roc.py <plot_roc.py>`

.. literalinclude:: plot_roc.py
    :lines: 38-

**Total running time of the example:**  0.21 seconds
( 0 minutes  0.21 seconds)
    