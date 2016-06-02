

.. _example_cluster_plot_cluster_iris.py:


=========================================================
K-means Clustering
=========================================================

The plots display firstly what a K-means algorithm would yield
using three clusters. It is then shown what the effect of a bad
initialization is on the classification process:
By setting n_init to only 1 (default is 10), the amount of
times that the algorithm will be run with different centroid
seeds is reduced.
The next plot displays what using eight clusters would deliver
and finally the ground truth.




.. rst-class:: horizontal





**Python source code:** :download:`plot_cluster_iris.py <plot_cluster_iris.py>`

.. literalinclude:: plot_cluster_iris.py
    :lines: 19-

**Total running time of the example:**  0.00 seconds
( 0 minutes  0.00 seconds)
    