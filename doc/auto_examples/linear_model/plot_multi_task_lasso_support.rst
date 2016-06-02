

.. _example_linear_model_plot_multi_task_lasso_support.py:


=============================================
Joint feature selection with multi-task Lasso
=============================================

The multi-task lasso allows to fit multiple regression problems
jointly enforcing the selected features to be the same across
tasks. This example simulates sequential measurements, each task
is a time instant, and the relevant features vary in amplitude
over time while being the same. The multi-task lasso imposes that
features that are selected at one time point are select for all time
point. This makes feature selection by the Lasso more stable.




.. rst-class:: horizontal


    *

      .. image:: images/plot_multi_task_lasso_support_001.png
            :scale: 47

    *

      .. image:: images/plot_multi_task_lasso_support_002.png
            :scale: 47




**Python source code:** :download:`plot_multi_task_lasso_support.py <plot_multi_task_lasso_support.py>`

.. literalinclude:: plot_multi_task_lasso_support.py
    :lines: 16-

**Total running time of the example:**  0.21 seconds
( 0 minutes  0.21 seconds)
    