

.. _example_plot_kernel_approximation.py:


==================================================
Explicit feature map approximation for RBF kernels
==================================================

An example illustrating the approximation of the feature map
of an RBF kernel.

.. currentmodule:: sklearn.kernel_approximation

It shows how to use :class:`RBFSampler` and :class:`Nystroem` to
approximate the feature map of an RBF kernel for classification with an SVM on
the digits dataset. Results using a linear SVM in the original space, a linear
SVM using the approximate mappings and using a kernelized SVM are compared.
Timings and accuracy for varying amounts of Monte Carlo samplings (in the case
of :class:`RBFSampler`, which uses random Fourier features) and different sized
subsets of the training set (for :class:`Nystroem`) for the approximate mapping
are shown.

Please note that the dataset here is not large enough to show the benefits
of kernel approximation, as the exact SVM is still reasonably fast.

Sampling more dimensions clearly leads to better classification results, but
comes at a greater cost. This means there is a tradeoff between runtime and
accuracy, given by the parameter n_components. Note that solving the Linear
SVM and also the approximate kernel SVM could be greatly accelerated by using
stochastic gradient descent via :class:`sklearn.linear_model.SGDClassifier`.
This is not easily possible for the case of the kernelized SVM.

The second plot visualized the decision surfaces of the RBF kernel SVM and
the linear SVM with approximate kernel maps.
The plot shows decision surfaces of the classifiers projected onto
the first two principal components of the data. This visualization should
be taken with a grain of salt since it is just an interesting slice through
the decision surface in 64 dimensions. In particular note that
a datapoint (represented as a dot) does not necessarily be classified
into the region it is lying in, since it will not lie on the plane
that the first two principal components span.

The usage of :class:`RBFSampler` and :class:`Nystroem` is described in detail
in :ref:`kernel_approximation`.




.. rst-class:: horizontal


    *

      .. image:: images/plot_kernel_approximation_001.png
            :scale: 47

    *

      .. image:: images/plot_kernel_approximation_002.png
            :scale: 47




**Python source code:** :download:`plot_kernel_approximation.py <plot_kernel_approximation.py>`

.. literalinclude:: plot_kernel_approximation.py
    :lines: 44-

**Total running time of the example:**  2.07 seconds
( 0 minutes  2.07 seconds)
    