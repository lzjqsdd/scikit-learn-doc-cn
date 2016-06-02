

.. _example_cluster_plot_color_quantization.py:


==================================
Color Quantization using K-Means
==================================

Performs a pixel-wise Vector Quantization (VQ) of an image of the summer palace
(China), reducing the number of colors required to show the image from 96,615
unique colors to 64, while preserving the overall appearance quality.

In this example, pixels are represented in a 3D-space and K-means is used to
find 64 color clusters. In the image processing literature, the codebook
obtained from K-means (the cluster centers) is called the color palette. Using
a single byte, up to 256 colors can be addressed, whereas an RGB encoding
requires 3 bytes per pixel. The GIF file format, for example, uses such a
palette.

For comparison, a quantized image using a random codebook (colors picked up
randomly) is also shown.



.. rst-class:: horizontal


    *

      .. image:: images/plot_color_quantization_001.png
            :scale: 47

    *

      .. image:: images/plot_color_quantization_002.png
            :scale: 47

    *

      .. image:: images/plot_color_quantization_003.png
            :scale: 47


**Script output**::

  Fitting model on a small sub-sample of the data
  done in 0.335s.
  Predicting color indices on the full image (k-means)
  done in 0.341s.
  Predicting color indices on the full image (random)
  done in 0.227s.



**Python source code:** :download:`plot_color_quantization.py <plot_color_quantization.py>`

.. literalinclude:: plot_color_quantization.py
    :lines: 21-

**Total running time of the example:**  1.66 seconds
( 0 minutes  1.66 seconds)
    