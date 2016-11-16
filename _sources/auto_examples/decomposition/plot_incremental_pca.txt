

.. _example_decomposition_plot_incremental_pca.py:



===============
Incremental PCA
===============

Incremental principal component analysis (IPCA) is typically used as a
replacement for principal component analysis (PCA) when the dataset to be
decomposed is too large to fit in memory. IPCA builds a low-rank approximation
for the input data using an amount of memory which is independent of the
number of input data samples. It is still dependent on the input data features,
but changing the batch size allows for control of memory usage.

This example serves as a visual check that IPCA is able to find a similar
projection of the data to PCA (to a sign flip), while only processing a
few samples at a time. This can be considered a "toy example", as IPCA is
intended for large datasets which do not fit in main memory, requiring
incremental approaches.




.. rst-class:: horizontal


    *

      .. image:: images/plot_incremental_pca_001.png
            :scale: 47

    *

      .. image:: images/plot_incremental_pca_002.png
            :scale: 47




**Python source code:** :download:`plot_incremental_pca.py <plot_incremental_pca.py>`

.. literalinclude:: plot_incremental_pca.py
    :lines: 21-

**Total running time of the example:**  0.15 seconds
( 0 minutes  0.15 seconds)
    