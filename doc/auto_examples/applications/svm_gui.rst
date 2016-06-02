

.. _example_applications_svm_gui.py:


==========
Libsvm GUI
==========

A simple graphical frontend for Libsvm mainly intended for didactic
purposes. You can create data points by point and click and visualize
the decision region induced by different kernels and parameter settings.

To create positive examples click the left mouse button; to create
negative examples click the right button.

If all examples are from the same class, it uses a one-class SVM.



**Python source code:** :download:`svm_gui.py <svm_gui.py>`

.. literalinclude:: svm_gui.py
    :lines: 16-
    