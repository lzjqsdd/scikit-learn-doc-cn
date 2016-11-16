

.. _example_text_mlcomp_sparse_document_classification.py:


========================================================
Classification of text documents: using a MLComp dataset
========================================================

This is an example showing how the scikit-learn can be used to classify
documents by topics using a bag-of-words approach. This example uses
a scipy.sparse matrix to store the features instead of standard numpy arrays.

The dataset used in this example is the 20 newsgroups dataset and should be
downloaded from the http://mlcomp.org (free registration required):

  http://mlcomp.org/datasets/379

Once downloaded unzip the archive somewhere on your filesystem.
For instance in::

  % mkdir -p ~/data/mlcomp
  % cd  ~/data/mlcomp
  % unzip /path/to/dataset-379-20news-18828_XXXXX.zip

You should get a folder ``~/data/mlcomp/379`` with a file named ``metadata``
and subfolders ``raw``, ``train`` and ``test`` holding the text documents
organized by newsgroups.

Then set the ``MLCOMP_DATASETS_HOME`` environment variable pointing to
the root folder holding the uncompressed archive::

  % export MLCOMP_DATASETS_HOME="~/data/mlcomp"

Then you are ready to run this example using your favorite python shell::

  % ipython examples/mlcomp_sparse_document_classification.py



**Python source code:** :download:`mlcomp_sparse_document_classification.py <mlcomp_sparse_document_classification.py>`

.. literalinclude:: mlcomp_sparse_document_classification.py
    :lines: 36-
    