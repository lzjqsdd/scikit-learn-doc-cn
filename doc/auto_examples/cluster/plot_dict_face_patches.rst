

.. _example_cluster_plot_dict_face_patches.py:


Online learning of a dictionary of parts of faces
==================================================

This example uses a large dataset of faces to learn a set of 20 x 20
images patches that constitute faces.

From the programming standpoint, it is interesting because it shows how
to use the online API of the scikit-learn to process a very large
dataset by chunks. The way we proceed is that we load an image at a time
and extract randomly 50 patches from this image. Once we have accumulated
500 of these patches (using 10 images), we run the `partial_fit` method
of the online KMeans object, MiniBatchKMeans.

The verbose setting on the MiniBatchKMeans enables us to see that some
clusters are reassigned during the successive calls to
partial-fit. This is because the number of patches that they represent
has become too low, and it is better to choose a random new
cluster.



.. image:: images/plot_dict_face_patches_001.png
    :align: center


**Script output**::

  Learning the dictionary... 
  Partial fit of  100 out of 2400
  Partial fit of  200 out of 2400
  [MiniBatchKMeans] Reassigning 16 cluster centers.
  Partial fit of  300 out of 2400
  Partial fit of  400 out of 2400
  Partial fit of  500 out of 2400
  Partial fit of  600 out of 2400
  Partial fit of  700 out of 2400
  Partial fit of  800 out of 2400
  Partial fit of  900 out of 2400
  Partial fit of 1000 out of 2400
  Partial fit of 1100 out of 2400
  Partial fit of 1200 out of 2400
  Partial fit of 1300 out of 2400
  Partial fit of 1400 out of 2400
  Partial fit of 1500 out of 2400
  Partial fit of 1600 out of 2400
  Partial fit of 1700 out of 2400
  Partial fit of 1800 out of 2400
  Partial fit of 1900 out of 2400
  Partial fit of 2000 out of 2400
  Partial fit of 2100 out of 2400
  Partial fit of 2200 out of 2400
  Partial fit of 2300 out of 2400
  Partial fit of 2400 out of 2400
  done in 7.74s.



**Python source code:** :download:`plot_dict_face_patches.py <plot_dict_face_patches.py>`

.. literalinclude:: plot_dict_face_patches.py
    :lines: 21-

**Total running time of the example:**  12.33 seconds
( 0 minutes  12.33 seconds)
    