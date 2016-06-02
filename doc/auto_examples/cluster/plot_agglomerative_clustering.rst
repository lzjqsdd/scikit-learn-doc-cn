

.. _example_cluster_plot_agglomerative_clustering.py:


Agglomerative clustering with and without structure
===================================================

This example shows the effect of imposing a connectivity graph to capture
local structure in the data. The graph is simply the graph of 20 nearest
neighbors.

Two consequences of imposing a connectivity can be seen. First clustering
with a connectivity matrix is much faster.

Second, when using a connectivity matrix, average and complete linkage are
unstable and tend to create a few clusters that grow very quickly. Indeed,
average and complete linkage fight this percolation behavior by considering all
the distances between two clusters when merging them. The connectivity
graph breaks this mechanism. This effect is more pronounced for very
sparse graphs (try decreasing the number of neighbors in
kneighbors_graph) and with complete linkage. In particular, having a very
small number of neighbors in the graph, imposes a geometry that is
close to that of single linkage, which is well known to have this
percolation instability.



.. rst-class:: horizontal


    *

      .. image:: images/plot_agglomerative_clustering_001.png
            :scale: 47

    *

      .. image:: images/plot_agglomerative_clustering_002.png
            :scale: 47

    *

      .. image:: images/plot_agglomerative_clustering_003.png
            :scale: 47

    *

      .. image:: images/plot_agglomerative_clustering_004.png
            :scale: 47




**Python source code:** :download:`plot_agglomerative_clustering.py <plot_agglomerative_clustering.py>`

.. literalinclude:: plot_agglomerative_clustering.py
    :lines: 23-

**Total running time of the example:**  26.87 seconds
( 0 minutes  26.87 seconds)
    