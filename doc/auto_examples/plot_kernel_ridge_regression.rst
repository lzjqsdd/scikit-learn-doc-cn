

.. _example_plot_kernel_ridge_regression.py:


=============================================
Comparison of kernel ridge regression and SVR
=============================================

Both kernel ridge regression (KRR) and SVR learn a non-linear function by
employing the kernel trick, i.e., they learn a linear function in the space
induced by the respective kernel which corresponds to a non-linear function in
the original space. They differ in the loss functions (ridge versus
epsilon-insensitive loss). In contrast to SVR, fitting a KRR can be done in
closed-form and is typically faster for medium-sized datasets. On the other
hand, the learned model is non-sparse and thus slower than SVR at
prediction-time.

This example illustrates both methods on an artificial dataset, which
consists of a sinusoidal target function and strong noise added to every fifth
datapoint. The first figure compares the learned model of KRR and SVR when both
complexity/regularization and bandwidth of the RBF kernel are optimized using
grid-search. The learned functions are very similar; however, fitting KRR is
approx. seven times faster than fitting SVR (both with grid-search). However,
prediction of 100000 target values is more than tree times faster with SVR
since it has learned a sparse model using only approx. 1/3 of the 100 training
datapoints as support vectors.

The next figure compares the time for fitting and prediction of KRR and SVR for
different sizes of the training set. Fitting KRR is faster than SVR for medium-
sized training sets (less than 1000 samples); however, for larger training sets
SVR scales better. With regard to prediction time, SVR is faster than
KRR for all sizes of the training set because of the learned sparse
solution. Note that the degree of sparsity and thus the prediction time depends
on the parameters epsilon and C of the SVR.



.. rst-class:: horizontal


    *

      .. image:: images/plot_kernel_ridge_regression_001.png
            :scale: 47

    *

      .. image:: images/plot_kernel_ridge_regression_002.png
            :scale: 47

    *

      .. image:: images/plot_kernel_ridge_regression_003.png
            :scale: 47


**Script output**::

  SVR complexity and bandwidth selected and model fitted in 0.977 s
  KRR complexity and bandwidth selected and model fitted in 0.252 s
  Support vector ratio: 0.320
  SVR prediction for 100000 inputs in 0.115 s
  KRR prediction for 100000 inputs in 0.389 s



**Python source code:** :download:`plot_kernel_ridge_regression.py <plot_kernel_ridge_regression.py>`

.. literalinclude:: plot_kernel_ridge_regression.py
    :lines: 33-

**Total running time of the example:**  63.49 seconds
( 1 minutes  3.49 seconds)
    