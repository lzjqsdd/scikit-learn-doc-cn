

.. _example_linear_model_plot_ols.py:


=========================================================
Linear Regression Example
=========================================================
This example uses the only the first feature of the `diabetes` dataset, in
order to illustrate a two-dimensional plot of this regression technique. The
straight line can be seen in the plot, showing how linear regression attempts
to draw a straight line that will best minimize the residual sum of squares
between the observed responses in the dataset, and the responses predicted by
the linear approximation.

The coefficients, the residual sum of squares and the variance score are also
calculated.




.. image:: images/plot_ols_001.png
    :align: center


**Script output**::

  Coefficients: 
   [ 938.23786125]
  Residual sum of squares: 2548.07
  Variance score: 0.47



**Python source code:** :download:`plot_ols.py <plot_ols.py>`

.. literalinclude:: plot_ols.py
    :lines: 19-

**Total running time of the example:**  0.08 seconds
( 0 minutes  0.08 seconds)
    