

.. _example_neighbors_plot_approximate_nearest_neighbors_hyperparameters.py:


=================================================
Hyper-parameters of Approximate Nearest Neighbors
=================================================

This example demonstrates the behaviour of the
accuracy of the nearest neighbor queries of Locality Sensitive Hashing
Forest as the number of candidates and the number of estimators (trees)
vary.

In the first plot, accuracy is measured with the number of candidates. Here,
the term "number of candidates" refers to maximum bound for the number of
distinct points retrieved from each tree to calculate the distances. Nearest
neighbors are selected from this pool of candidates. Number of estimators is
maintained at three fixed levels (1, 5, 10).

In the second plot, the number of candidates is fixed at 50. Number of trees
is varied and the accuracy is plotted against those values. To measure the
accuracy, the true nearest neighbors are required, therefore
:class:`sklearn.neighbors.NearestNeighbors` is used to compute the exact
neighbors.



.. rst-class:: horizontal


    *

      .. image:: images/plot_approximate_nearest_neighbors_hyperparameters_001.png
            :scale: 47

    *

      .. image:: images/plot_approximate_nearest_neighbors_hyperparameters_002.png
            :scale: 47




**Python source code:** :download:`plot_approximate_nearest_neighbors_hyperparameters.py <plot_approximate_nearest_neighbors_hyperparameters.py>`

.. literalinclude:: plot_approximate_nearest_neighbors_hyperparameters.py
    :lines: 23-

**Total running time of the example:**  24.43 seconds
( 0 minutes  24.43 seconds)
    