

.. _example_feature_stacker.py:


=================================================
Concatenating multiple feature extraction methods
=================================================

In many real-world examples, there are many ways to extract features from a
dataset. Often it is beneficial to combine several methods to obtain good
performance. This example shows how to use ``FeatureUnion`` to combine
features obtained by PCA and univariate selection.

Combining features using this transformer has the benefit that it allows
cross validation and grid searches over the whole process.

The combination used in this example is not particularly helpful on this
dataset and is only used to illustrate the usage of FeatureUnion.


**Python source code:** :download:`feature_stacker.py <feature_stacker.py>`

.. literalinclude:: feature_stacker.py
    :lines: 17-
    