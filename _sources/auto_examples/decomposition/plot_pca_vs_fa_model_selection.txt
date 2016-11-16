

.. _example_decomposition_plot_pca_vs_fa_model_selection.py:


===============================================================
Model selection with Probabilistic PCA and Factor Analysis (FA)
===============================================================

Probabilistic PCA and Factor Analysis are probabilistic models.
The consequence is that the likelihood of new data can be used
for model selection and covariance estimation.
Here we compare PCA and FA with cross-validation on low rank data corrupted
with homoscedastic noise (noise variance
is the same for each feature) or heteroscedastic noise (noise variance
is the different for each feature). In a second step we compare the model
likelihood to the likelihoods obtained from shrinkage covariance estimators.

One can observe that with homoscedastic noise both FA and PCA succeed
in recovering the size of the low rank subspace. The likelihood with PCA
is higher than FA in this case. However PCA fails and overestimates
the rank when heteroscedastic noise is present. Under appropriate
circumstances the low rank models are more likely than shrinkage models.

The automatic estimation from
Automatic Choice of Dimensionality for PCA. NIPS 2000: 598-604
by Thomas P. Minka is also compared.




.. rst-class:: horizontal





**Python source code:** :download:`plot_pca_vs_fa_model_selection.py <plot_pca_vs_fa_model_selection.py>`

.. literalinclude:: plot_pca_vs_fa_model_selection.py
    :lines: 26-

**Total running time of the example:**  0.00 seconds
( 0 minutes  0.00 seconds)
    