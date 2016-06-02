

.. _example_cluster_plot_digits_linkage.py:


=============================================================================
Various Agglomerative Clustering on a 2D embedding of digits
=============================================================================

An illustration of various linkage option for agglomerative clustering on
a 2D embedding of the digits dataset.

The goal of this example is to show intuitively how the metrics behave, and
not to find good clusters for the digits. This is why the example works on a
2D embedding.

What this example shows us is the behavior "rich getting richer" of
agglomerative clustering that tends to create uneven cluster sizes.
This behavior is especially pronounced for the average linkage strategy,
that ends up with a couple of singleton clusters.



.. rst-class:: horizontal


    *

      .. image:: images/plot_digits_linkage_001.png
            :scale: 47

    *

      .. image:: images/plot_digits_linkage_002.png
            :scale: 47

    *

      .. image:: images/plot_digits_linkage_003.png
            :scale: 47


**Script output**::

  Computing embedding
  Done.
  ward : 44.23s
  average : 38.92s
  complete : 40.68s



**Python source code:** :download:`plot_digits_linkage.py <plot_digits_linkage.py>`

.. literalinclude:: plot_digits_linkage.py
    :lines: 18-

**Total running time of the example:**  153.96 seconds
( 2 minutes  33.96 seconds)
    