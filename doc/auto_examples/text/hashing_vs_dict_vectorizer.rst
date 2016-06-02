

.. _example_text_hashing_vs_dict_vectorizer.py:


===========================================
FeatureHasher and DictVectorizer Comparison
===========================================

Compares FeatureHasher and DictVectorizer by using both to vectorize
text documents.

The example demonstrates syntax and speed only; it doesn't actually do
anything useful with the extracted vectors. See the example scripts
{document_classification_20newsgroups,clustering}.py for actual learning
on text documents.

A discrepancy between the number of terms reported for DictVectorizer and
for FeatureHasher is to be expected due to hash collisions.


**Python source code:** :download:`hashing_vs_dict_vectorizer.py <hashing_vs_dict_vectorizer.py>`

.. literalinclude:: hashing_vs_dict_vectorizer.py
    :lines: 17-
    