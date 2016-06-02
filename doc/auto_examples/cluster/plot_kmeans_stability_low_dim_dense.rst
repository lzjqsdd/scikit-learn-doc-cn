

.. _example_cluster_plot_kmeans_stability_low_dim_dense.py:


============================================================
Empirical evaluation of the impact of k-means initialization
============================================================

Evaluate the ability of k-means initializations strategies to make
the algorithm convergence robust as measured by the relative standard
deviation of the inertia of the clustering (i.e. the sum of distances
to the nearest cluster center).

The first plot shows the best inertia reached for each combination
of the model (``KMeans`` or ``MiniBatchKMeans``) and the init method
(``init="random"`` or ``init="kmeans++"``) for increasing values of the
``n_init`` parameter that controls the number of initializations.

The second plot demonstrate one single run of the ``MiniBatchKMeans``
estimator using a ``init="random"`` and ``n_init=1``. This run leads to
a bad convergence (local optimum) with estimated centers stuck
between ground truth clusters.

The dataset used for evaluation is a 2D grid of isotropic Gaussian
clusters widely spaced.



.. rst-class:: horizontal


    *

      .. image:: images/plot_kmeans_stability_low_dim_dense_001.png
            :scale: 47

    *

      .. image:: images/plot_kmeans_stability_low_dim_dense_002.png
            :scale: 47


**Script output**::

  Evaluation of KMeans with k-means++ init
  Evaluation of KMeans with random init
  Evaluation of MiniBatchKMeans with k-means++ init
  Evaluation of MiniBatchKMeans with random init



**Python source code:** :download:`plot_kmeans_stability_low_dim_dense.py <plot_kmeans_stability_low_dim_dense.py>`

.. literalinclude:: plot_kmeans_stability_low_dim_dense.py
    :lines: 24-

**Total running time of the example:**  3.12 seconds
( 0 minutes  3.12 seconds)
    