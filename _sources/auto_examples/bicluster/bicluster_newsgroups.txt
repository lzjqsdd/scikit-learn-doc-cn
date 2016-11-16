

.. _example_bicluster_bicluster_newsgroups.py:


================================================================
Biclustering documents with the Spectral Co-clustering algorithm
================================================================

This example demonstrates the Spectral Co-clustering algorithm on the
twenty newsgroups dataset. The 'comp.os.ms-windows.misc' category is
excluded because it contains many posts containing nothing but data.

The TF-IDF vectorized posts form a word frequency matrix, which is
then biclustered using Dhillon's Spectral Co-Clustering algorithm. The
resulting document-word biclusters indicate subsets words used more
often in those subsets documents.

For a few of the best biclusters, its most common document categories
and its ten most important words get printed. The best biclusters are
determined by their normalized cut. The best words are determined by
comparing their sums inside and outside the bicluster.

For comparison, the documents are also clustered using
MiniBatchKMeans. The document clusters derived from the biclusters
achieve a better V-measure than clusters found by MiniBatchKMeans.

Output::

    Vectorizing...
    Coclustering...
    Done in 9.53s. V-measure: 0.4455
    MiniBatchKMeans...
    Done in 12.00s. V-measure: 0.3309

    Best biclusters:
    ----------------
    bicluster 0 : 1951 documents, 4373 words
    categories   : 23% talk.politics.guns, 19% talk.politics.misc, 14% sci.med
    words        : gun, guns, geb, banks, firearms, drugs, gordon, clinton, cdt, amendment

    bicluster 1 : 1165 documents, 3304 words
    categories   : 29% talk.politics.mideast, 26% soc.religion.christian, 25% alt.atheism
    words        : god, jesus, christians, atheists, kent, sin, morality, belief, resurrection, marriage

    bicluster 2 : 2219 documents, 2830 words
    categories   : 18% comp.sys.mac.hardware, 16% comp.sys.ibm.pc.hardware, 16% comp.graphics
    words        : voltage, dsp, board, receiver, circuit, shipping, packages, stereo, compression, package

    bicluster 3 : 1860 documents, 2745 words
    categories   : 26% rec.motorcycles, 23% rec.autos, 13% misc.forsale
    words        : bike, car, dod, engine, motorcycle, ride, honda, cars, bmw, bikes

    bicluster 4 : 12 documents, 155 words
    categories   : 100% rec.sport.hockey
    words        : scorer, unassisted, reichel, semak, sweeney, kovalenko, ricci, audette, momesso, nedved



**Python source code:** :download:`bicluster_newsgroups.py <bicluster_newsgroups.py>`

.. literalinclude:: bicluster_newsgroups.py
    :lines: 55-
    