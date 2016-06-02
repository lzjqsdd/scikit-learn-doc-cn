

.. _example_decomposition_plot_faces_decomposition.py:


============================
Faces dataset decompositions
============================

This example applies to :ref:`olivetti_faces` different unsupervised
matrix decomposition (dimension reduction) methods from the module
:py:mod:`sklearn.decomposition` (see the documentation chapter
:ref:`decompositions`) .




.. rst-class:: horizontal


    *

      .. image:: images/plot_faces_decomposition_001.png
            :scale: 47

    *

      .. image:: images/plot_faces_decomposition_002.png
            :scale: 47

    *

      .. image:: images/plot_faces_decomposition_003.png
            :scale: 47

    *

      .. image:: images/plot_faces_decomposition_004.png
            :scale: 47

    *

      .. image:: images/plot_faces_decomposition_005.png
            :scale: 47

    *

      .. image:: images/plot_faces_decomposition_006.png
            :scale: 47

    *

      .. image:: images/plot_faces_decomposition_007.png
            :scale: 47

    *

      .. image:: images/plot_faces_decomposition_008.png
            :scale: 47

    *

      .. image:: images/plot_faces_decomposition_009.png
            :scale: 47


**Script output**::

  Dataset consists of 400 faces
  Extracting the top 6 Eigenfaces - RandomizedPCA...
  done in 0.102s
  Extracting the top 6 Non-negative components - NMF...
  done in 0.771s
  Extracting the top 6 Independent components - FastICA...
  done in 0.586s
  Extracting the top 6 Sparse comp. - MiniBatchSparsePCA...
  done in 1.057s
  Extracting the top 6 MiniBatchDictionaryLearning...
  done in 1.202s
  Extracting the top 6 Cluster centers - MiniBatchKMeans...
  done in 0.292s
  Extracting the top 6 Factor Analysis components - FA...
  done in 0.236s



**Python source code:** :download:`plot_faces_decomposition.py <plot_faces_decomposition.py>`

.. literalinclude:: plot_faces_decomposition.py
    :lines: 12-

**Total running time of the example:**  6.74 seconds
( 0 minutes  6.74 seconds)
    