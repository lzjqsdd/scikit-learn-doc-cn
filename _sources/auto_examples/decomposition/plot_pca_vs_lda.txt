

.. _example_decomposition_plot_pca_vs_lda.py:


=======================================================
Comparison of LDA and PCA 2D projection of Iris dataset
=======================================================

The Iris dataset represents 3 kind of Iris flowers (Setosa, Versicolour
and Virginica) with 4 attributes: sepal length, sepal width, petal length
and petal width.

Principal Component Analysis (PCA) applied to this data identifies the
combination of attributes (principal components, or directions in the
feature space) that account for the most variance in the data. Here we
plot the different samples on the 2 first principal components.

Linear Discriminant Analysis (LDA) tries to identify attributes that
account for the most variance *between classes*. In particular,
LDA, in contrast to PCA, is a supervised method, using known class labels.



.. rst-class:: horizontal


    *

      .. image:: images/plot_pca_vs_lda_001.png
            :scale: 47

    *

      .. image:: images/plot_pca_vs_lda_002.png
            :scale: 47


**Script output**::

  explained variance ratio (first two components): [ 0.92461621  0.05301557]



**Python source code:** :download:`plot_pca_vs_lda.py <plot_pca_vs_lda.py>`

.. literalinclude:: plot_pca_vs_lda.py
    :lines: 19-

**Total running time of the example:**  0.13 seconds
( 0 minutes  0.13 seconds)
    