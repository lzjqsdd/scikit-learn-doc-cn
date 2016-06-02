

.. _example_manifold_plot_manifold_sphere.py:


=============================================
Manifold Learning methods on a severed sphere
=============================================

An application of the different :ref:`manifold` techniques
on a spherical data-set. Here one can see the use of
dimensionality reduction in order to gain some intuition
regarding the manifold learning methods. Regarding the dataset,
the poles are cut from the sphere, as well as a thin slice down its
side. This enables the manifold learning techniques to
'spread it open' whilst projecting it onto two dimensions.

For a similar example, where the methods are applied to the
S-curve dataset, see :ref:`example_manifold_plot_compare_methods.py`

Note that the purpose of the :ref:`MDS <multidimensional_scaling>` is
to find a low-dimensional representation of the data (here 2D) in
which the distances respect well the distances in the original
high-dimensional space, unlike other manifold-learning algorithms,
it does not seeks an isotropic representation of the data in
the low-dimensional space. Here the manifold problem matches fairly
that of representing a flat map of the Earth, as with
`map projection <http://en.wikipedia.org/wiki/Map_projection>`_



.. rst-class:: horizontal





**Python source code:** :download:`plot_manifold_sphere.py <plot_manifold_sphere.py>`

.. literalinclude:: plot_manifold_sphere.py
    :lines: 29-

**Total running time of the example:**  0.00 seconds
( 0 minutes  0.00 seconds)
    