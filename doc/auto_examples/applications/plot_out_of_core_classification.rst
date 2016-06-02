

.. _example_applications_plot_out_of_core_classification.py:


======================================================
Out-of-core classification of text documents
======================================================

This is an example showing how scikit-learn can be used for classification
using an out-of-core approach: learning from data that doesn't fit into main
memory. We make use of an online classifier, i.e., one that supports the
partial_fit method, that will be fed with batches of examples. To guarantee
that the features space remains the same over time we leverage a
HashingVectorizer that will project each example into the same feature space.
This is especially useful in the case of text classification where new
features (words) may appear in each batch.

The dataset used in this example is Reuters-21578 as provided by the UCI ML
repository. It will be automatically downloaded and uncompressed on first run.

The plot represents the learning curve of the classifier: the evolution
of classification accuracy over the course of the mini-batches. Accuracy is
measured on the first 1000 samples, held out as a validation set.

To limit the memory consumption, we queue examples up to a fixed amount before
feeding them to the learner.



.. rst-class:: horizontal


    *

      .. image:: images/plot_out_of_core_classification_001.png
            :scale: 47

    *

      .. image:: images/plot_out_of_core_classification_002.png
            :scale: 47

    *

      .. image:: images/plot_out_of_core_classification_003.png
            :scale: 47

    *

      .. image:: images/plot_out_of_core_classification_004.png
            :scale: 47


**Script output**::

  Test set is 982 documents (90 positive)
    Passive-Aggressive classifier :          994 train docs (   121 positive)    982 test docs (    90 positive) accuracy: 0.939 in 1.57s (  631 docs/s)
            Perceptron classifier :          994 train docs (   121 positive)    982 test docs (    90 positive) accuracy: 0.934 in 1.58s (  629 docs/s)
                   SGD classifier :          994 train docs (   121 positive)    982 test docs (    90 positive) accuracy: 0.928 in 1.58s (  628 docs/s)
        NB Multinomial classifier :          994 train docs (   121 positive)    982 test docs (    90 positive) accuracy: 0.908 in 1.61s (  616 docs/s)
  
  
    Passive-Aggressive classifier :         3820 train docs (   520 positive)    982 test docs (    90 positive) accuracy: 0.969 in 4.47s (  853 docs/s)
            Perceptron classifier :         3820 train docs (   520 positive)    982 test docs (    90 positive) accuracy: 0.958 in 4.48s (  852 docs/s)
                   SGD classifier :         3820 train docs (   520 positive)    982 test docs (    90 positive) accuracy: 0.945 in 4.48s (  851 docs/s)
        NB Multinomial classifier :         3820 train docs (   520 positive)    982 test docs (    90 positive) accuracy: 0.919 in 4.52s (  845 docs/s)
  
  
    Passive-Aggressive classifier :         6759 train docs (   902 positive)    982 test docs (    90 positive) accuracy: 0.967 in 7.43s (  909 docs/s)
            Perceptron classifier :         6759 train docs (   902 positive)    982 test docs (    90 positive) accuracy: 0.926 in 7.43s (  909 docs/s)
                   SGD classifier :         6759 train docs (   902 positive)    982 test docs (    90 positive) accuracy: 0.963 in 7.44s (  908 docs/s)
        NB Multinomial classifier :         6759 train docs (   902 positive)    982 test docs (    90 positive) accuracy: 0.931 in 7.47s (  905 docs/s)
  
  
    Passive-Aggressive classifier :         9158 train docs (  1148 positive)    982 test docs (    90 positive) accuracy: 0.960 in 10.63s (  861 docs/s)
            Perceptron classifier :         9158 train docs (  1148 positive)    982 test docs (    90 positive) accuracy: 0.953 in 10.64s (  860 docs/s)
                   SGD classifier :         9158 train docs (  1148 positive)    982 test docs (    90 positive) accuracy: 0.954 in 10.64s (  860 docs/s)
        NB Multinomial classifier :         9158 train docs (  1148 positive)    982 test docs (    90 positive) accuracy: 0.930 in 10.67s (  858 docs/s)
  
  
    Passive-Aggressive classifier :        11953 train docs (  1510 positive)    982 test docs (    90 positive) accuracy: 0.962 in 13.84s (  863 docs/s)
            Perceptron classifier :        11953 train docs (  1510 positive)    982 test docs (    90 positive) accuracy: 0.962 in 13.85s (  863 docs/s)
                   SGD classifier :        11953 train docs (  1510 positive)    982 test docs (    90 positive) accuracy: 0.968 in 13.85s (  863 docs/s)
        NB Multinomial classifier :        11953 train docs (  1510 positive)    982 test docs (    90 positive) accuracy: 0.939 in 13.88s (  861 docs/s)
  
  
    Passive-Aggressive classifier :        14338 train docs (  1778 positive)    982 test docs (    90 positive) accuracy: 0.969 in 16.73s (  857 docs/s)
            Perceptron classifier :        14338 train docs (  1778 positive)    982 test docs (    90 positive) accuracy: 0.959 in 16.73s (  857 docs/s)
                   SGD classifier :        14338 train docs (  1778 positive)    982 test docs (    90 positive) accuracy: 0.958 in 16.73s (  856 docs/s)
        NB Multinomial classifier :        14338 train docs (  1778 positive)    982 test docs (    90 positive) accuracy: 0.937 in 16.76s (  855 docs/s)
  
  
    Passive-Aggressive classifier :        17255 train docs (  2123 positive)    982 test docs (    90 positive) accuracy: 0.965 in 19.40s (  889 docs/s)
            Perceptron classifier :        17255 train docs (  2123 positive)    982 test docs (    90 positive) accuracy: 0.963 in 19.41s (  889 docs/s)
                   SGD classifier :        17255 train docs (  2123 positive)    982 test docs (    90 positive) accuracy: 0.958 in 19.41s (  888 docs/s)
        NB Multinomial classifier :        17255 train docs (  2123 positive)    982 test docs (    90 positive) accuracy: 0.940 in 19.44s (  887 docs/s)



**Python source code:** :download:`plot_out_of_core_classification.py <plot_out_of_core_classification.py>`

.. literalinclude:: plot_out_of_core_classification.py
    :lines: 25-

**Total running time of the example:**  21.05 seconds
( 0 minutes  21.05 seconds)
    