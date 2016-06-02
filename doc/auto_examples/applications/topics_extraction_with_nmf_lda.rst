

.. _example_applications_topics_extraction_with_nmf_lda.py:


=======================================================================================
Topic extraction with Non-negative Matrix Factorization and Latent Dirichlet Allocation
=======================================================================================

This is an example of applying Non-negative Matrix Factorization
and Latent Dirichlet Allocation on a corpus of documents and
extract additive models of the topic structure of the corpus.
The output is a list of topics, each represented as a list of terms
(weights are not shown).

The default parameters (n_samples / n_features / n_topics) should make
the example runnable in a couple of tens of seconds. You can try to
increase the dimensions of the problem, but be aware that the time
complexity is polynomial in NMF. In LDA, the time complexity is
proportional to (n_samples * iterations).


**Python source code:** :download:`topics_extraction_with_nmf_lda.py <topics_extraction_with_nmf_lda.py>`

.. literalinclude:: topics_extraction_with_nmf_lda.py
    :lines: 18-
    