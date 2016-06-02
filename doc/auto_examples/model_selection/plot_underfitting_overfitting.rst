

.. _example_model_selection_plot_underfitting_overfitting.py:


============================
Underfitting vs. Overfitting
============================

This example demonstrates the problems of underfitting and overfitting and
how we can use linear regression with polynomial features to approximate
nonlinear functions. The plot shows the function that we want to approximate,
which is a part of the cosine function. In addition, the samples from the
real function and the approximations of different models are displayed. The
models have polynomial features of different degrees. We can see that a
linear function (polynomial with degree 1) is not sufficient to fit the
training samples. This is called **underfitting**. A polynomial of degree 4
approximates the true function almost perfectly. However, for higher degrees
the model will **overfit** the training data, i.e. it learns the noise of the
training data.
We evaluate quantitatively **overfitting** / **underfitting** by using
cross-validation. We calculate the mean squared error (MSE) on the validation
set, the higher, the less likely the model generalizes correctly from the
training data.



.. image:: images/plot_underfitting_overfitting_001.png
    :align: center




**Python source code:** :download:`plot_underfitting_overfitting.py <plot_underfitting_overfitting.py>`

.. literalinclude:: plot_underfitting_overfitting.py
    :lines: 22-

**Total running time of the example:**  0.23 seconds
( 0 minutes  0.23 seconds)
    