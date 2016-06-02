

.. _example_text_document_classification_20newsgroups.py:


======================================================
Classification of text documents using sparse features
======================================================

This is an example showing how scikit-learn can be used to classify documents
by topics using a bag-of-words approach. This example uses a scipy.sparse
matrix to store the features and demonstrates various classifiers that can
efficiently handle sparse matrices.

The dataset used in this example is the 20 newsgroups dataset. It will be
automatically downloaded, then cached.

The bar plot indicates the accuracy, training time (normalized) and test time
(normalized) of each classifier.



**Python source code:** :download:`document_classification_20newsgroups.py <document_classification_20newsgroups.py>`

.. literalinclude:: document_classification_20newsgroups.py
    :lines: 18-
    