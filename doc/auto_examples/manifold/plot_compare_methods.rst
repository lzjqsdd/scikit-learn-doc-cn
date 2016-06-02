

.. _example_manifold_plot_compare_methods.py:


=========================================
 Comparison of Manifold Learning methods
=========================================

An illustration of dimensionality reduction on the S-curve dataset
with various manifold learning methods.

For a discussion and comparison of these algorithms, see the
:ref:`manifold module page <manifold>`

For a similar example, where the methods are applied to a
sphere dataset, see :ref:`example_manifold_plot_manifold_sphere.py`

Note that the purpose of the MDS is to find a low-dimensional
representation of the data (here 2D) in which the distances respect well
the distances in the original high-dimensional space, unlike other
manifold-learning algorithms, it does not seeks an isotropic
representation of the data in the low-dimensional space.



.. rst-class:: horizontal





**Python source code:** :download:`plot_compare_methods.py <plot_compare_methods.py>`

.. literalinclude:: plot_compare_methods.py
    :lines: 21-

**Total running time of the example:**  0.00 seconds
( 0 minutes  0.00 seconds)
    