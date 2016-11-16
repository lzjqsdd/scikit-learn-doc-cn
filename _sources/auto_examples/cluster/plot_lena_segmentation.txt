

.. _example_cluster_plot_lena_segmentation.py:


=========================================
Segmenting the picture of Lena in regions
=========================================

This example uses :ref:`spectral_clustering` on a graph created from
voxel-to-voxel difference on an image to break this image into multiple
partly-homogeneous regions.

This procedure (spectral clustering on an image) is an efficient
approximate solution for finding normalized graph cuts.

There are two options to assign labels:

* with 'kmeans' spectral clustering will cluster samples in the embedding space
  using a kmeans algorithm
* whereas 'discrete' will iteratively search for the closest partition
  space to the embedding space.



.. rst-class:: horizontal





**Python source code:** :download:`plot_lena_segmentation.py <plot_lena_segmentation.py>`

.. literalinclude:: plot_lena_segmentation.py
    :lines: 20-

**Total running time of the example:**  0.00 seconds
( 0 minutes  0.00 seconds)
    