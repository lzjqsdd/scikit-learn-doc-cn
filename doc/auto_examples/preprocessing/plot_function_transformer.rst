

.. _example_preprocessing_plot_function_transformer.py:


=========================================================
Using FunctionTransformer to select columns
=========================================================

Shows how to use a function transformer in a pipeline. If you know your
dataset's first principle component is irrelevant for a classification task,
you can use the FunctionTransformer to select all but the first column of the
PCA transformed data.



.. rst-class:: horizontal





**Python source code:** :download:`plot_function_transformer.py <plot_function_transformer.py>`

.. literalinclude:: plot_function_transformer.py
    :lines: 11-

**Total running time of the example:**  0.00 seconds
( 0 minutes  0.00 seconds)
    