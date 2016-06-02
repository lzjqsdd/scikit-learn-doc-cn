

.. _example_neighbors_plot_kde_1d.py:


===================================
Simple 1D Kernel Density Estimation
===================================
This example uses the :class:`sklearn.neighbors.KernelDensity` class to
demonstrate the principles of Kernel Density Estimation in one dimension.

The first plot shows one of the problems with using histograms to visualize
the density of points in 1D. Intuitively, a histogram can be thought of as a
scheme in which a unit "block" is stacked above each point on a regular grid.
As the top two panels show, however, the choice of gridding for these blocks
can lead to wildly divergent ideas about the underlying shape of the density
distribution.  If we instead center each block on the point it represents, we
get the estimate shown in the bottom left panel.  This is a kernel density
estimation with a "top hat" kernel.  This idea can be generalized to other
kernel shapes: the bottom-right panel of the first figure shows a Gaussian
kernel density estimate over the same distribution.

Scikit-learn implements efficient kernel density estimation using either
a Ball Tree or KD Tree structure, through the
:class:`sklearn.neighbors.KernelDensity` estimator.  The available kernels
are shown in the second figure of this example.

The third figure compares kernel density estimates for a distribution of 100
samples in 1 dimension.  Though this example uses 1D distributions, kernel
density estimation is easily and efficiently extensible to higher dimensions
as well.



.. rst-class:: horizontal


    *

      .. image:: images/plot_kde_1d_001.png
            :scale: 47

    *

      .. image:: images/plot_kde_1d_002.png
            :scale: 47

    *

      .. image:: images/plot_kde_1d_003.png
            :scale: 47




**Python source code:** :download:`plot_kde_1d.py <plot_kde_1d.py>`

.. literalinclude:: plot_kde_1d.py
    :lines: 29-

**Total running time of the example:**  0.59 seconds
( 0 minutes  0.59 seconds)
    