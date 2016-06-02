

.. _example_svm_plot_separating_hyperplane_unbalanced.py:


=================================================
SVM: Separating hyperplane for unbalanced classes
=================================================

Find the optimal separating hyperplane using an SVC for classes that
are unbalanced.

We first find the separating plane with a plain SVC and then plot
(dashed) the separating hyperplane with automatically correction for
unbalanced classes.

.. currentmodule:: sklearn.linear_model

.. note::

    This example will also work by replacing ``SVC(kernel="linear")``
    with ``SGDClassifier(loss="hinge")``. Setting the ``loss`` parameter
    of the :class:`SGDClassifier` equal to ``hinge`` will yield behaviour
    such as that of a SVC with a linear kernel.

    For example try instead of the ``SVC``::

        clf = SGDClassifier(n_iter=100, alpha=0.01)




.. image:: images/plot_separating_hyperplane_unbalanced_001.png
    :align: center




**Python source code:** :download:`plot_separating_hyperplane_unbalanced.py <plot_separating_hyperplane_unbalanced.py>`

.. literalinclude:: plot_separating_hyperplane_unbalanced.py
    :lines: 27-

**Total running time of the example:**  0.07 seconds
( 0 minutes  0.07 seconds)
    