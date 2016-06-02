

.. _example_cluster_plot_ward_structured_vs_unstructured.py:


===========================================================
Hierarchical clustering: structured vs unstructured ward
===========================================================

Example builds a swiss roll dataset and runs
hierarchical clustering on their position.

For more information, see :ref:`hierarchical_clustering`.

In a first step, the hierarchical clustering is performed without connectivity
constraints on the structure and is solely based on distance, whereas in
a second step the clustering is restricted to the k-Nearest Neighbors
graph: it's a hierarchical clustering with structure prior.

Some of the clusters learned without connectivity constraints do not
respect the structure of the swiss roll and extend across different folds of
the manifolds. On the opposite, when opposing connectivity constraints,
the clusters form a nice parcellation of the swiss roll.



.. rst-class:: horizontal





**Python source code:** :download:`plot_ward_structured_vs_unstructured.py <plot_ward_structured_vs_unstructured.py>`

.. literalinclude:: plot_ward_structured_vs_unstructured.py
    :lines: 21-

**Total running time of the example:**  0.00 seconds
( 0 minutes  0.00 seconds)
    