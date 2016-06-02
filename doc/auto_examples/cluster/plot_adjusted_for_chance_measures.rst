

.. _example_cluster_plot_adjusted_for_chance_measures.py:


==========================================================
Adjustment for chance in clustering performance evaluation
==========================================================

The following plots demonstrate the impact of the number of clusters and
number of samples on various clustering performance evaluation metrics.

Non-adjusted measures such as the V-Measure show a dependency between
the number of clusters and the number of samples: the mean V-Measure
of random labeling increases significantly as the number of clusters is
closer to the total number of samples used to compute the measure.

Adjusted for chance measure such as ARI display some random variations
centered around a mean score of 0.0 for any number of samples and
clusters.

Only adjusted measures can hence safely be used as a consensus index
to evaluate the average stability of clustering algorithms for a given
value of k on various overlapping sub-samples of the dataset.




.. rst-class:: horizontal


    *

      .. image:: images/plot_adjusted_for_chance_measures_001.png
            :scale: 47

    *

      .. image:: images/plot_adjusted_for_chance_measures_002.png
            :scale: 47


**Script output**::

  Computing adjusted_rand_score for 10 values of n_clusters and n_samples=100
  done in 0.112s
  Computing v_measure_score for 10 values of n_clusters and n_samples=100
  done in 0.025s
  Computing adjusted_mutual_info_score for 10 values of n_clusters and n_samples=100
  done in 0.593s
  Computing mutual_info_score for 10 values of n_clusters and n_samples=100
  done in 0.015s
  Computing adjusted_rand_score for 10 values of n_clusters and n_samples=1000
  done in 0.127s
  Computing v_measure_score for 10 values of n_clusters and n_samples=1000
  done in 0.044s
  Computing adjusted_mutual_info_score for 10 values of n_clusters and n_samples=1000
  done in 0.304s
  Computing mutual_info_score for 10 values of n_clusters and n_samples=1000
  done in 0.027s



**Python source code:** :download:`plot_adjusted_for_chance_measures.py <plot_adjusted_for_chance_measures.py>`

.. literalinclude:: plot_adjusted_for_chance_measures.py
    :lines: 23-

**Total running time of the example:**  1.39 seconds
( 0 minutes  1.39 seconds)
    