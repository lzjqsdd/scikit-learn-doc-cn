

.. _example_cluster_plot_lena_ward_segmentation.py:


===============================================================
A demo of structured Ward hierarchical clustering on Lena image
===============================================================

Compute the segmentation of a 2D image with Ward hierarchical
clustering. The clustering is spatially constrained in order
for each segmented region to be in one piece.



.. image:: images/plot_lena_ward_segmentation_001.png
    :align: center


**Script output**::

  Compute structured hierarchical clustering...
  Elapsed time:  7.47855305672
  Number of pixels:  65536
  Number of clusters:  15



**Python source code:** :download:`plot_lena_ward_segmentation.py <plot_lena_ward_segmentation.py>`

.. literalinclude:: plot_lena_ward_segmentation.py
    :lines: 10-

**Total running time of the example:**  8.06 seconds
( 0 minutes  8.06 seconds)
    