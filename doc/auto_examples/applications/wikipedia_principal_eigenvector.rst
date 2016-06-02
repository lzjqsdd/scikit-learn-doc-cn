

.. _example_applications_wikipedia_principal_eigenvector.py:


===============================
Wikipedia principal eigenvector
===============================

A classical way to assert the relative importance of vertices in a
graph is to compute the principal eigenvector of the adjacency matrix
so as to assign to each vertex the values of the components of the first
eigenvector as a centrality score:

    http://en.wikipedia.org/wiki/Eigenvector_centrality

On the graph of webpages and links those values are called the PageRank
scores by Google.

The goal of this example is to analyze the graph of links inside
wikipedia articles to rank articles by relative importance according to
this eigenvector centrality.

The traditional way to compute the principal eigenvector is to use the
power iteration method:

    http://en.wikipedia.org/wiki/Power_iteration

Here the computation is achieved thanks to Martinsson's Randomized SVD
algorithm implemented in the scikit.

The graph data is fetched from the DBpedia dumps. DBpedia is an extraction
of the latent structured data of the Wikipedia content.


**Python source code:** :download:`wikipedia_principal_eigenvector.py <wikipedia_principal_eigenvector.py>`

.. literalinclude:: wikipedia_principal_eigenvector.py
    :lines: 31-
    